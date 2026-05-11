use half::f16;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use image::{DynamicImage, GenericImageView, ImageBuffer, Luma, Rgba};
use wgpu::util::DeviceExt;

use crate::ai_processing::CachedDepthMap;
use crate::app_state::AppState;
use crate::cache_utils::GEOMETRY_KEYS;
use crate::gpu_processing::{
    get_or_init_gpu_context, process_and_get_dynamic_image_with_analytics, RenderRequest,
};
use crate::image_processing::{downscale_f32_image, resolve_tonemapper_override_from_handle};
use crate::mask_generation::MaskDefinition;
use crate::{calculate_transform_hash, get_or_load_lut, get_raw_image_path_from_adjustments};

/// Convert DynamicImage (Rgba8) → Rgba16Float (f16 RGBA). Mirrors gpu_processing::to_rgba_f16.
fn to_rgba_f16(img: &DynamicImage) -> Vec<f16> {
    let rgba_f32 = img.to_rgba32f(); // image crate's method → Rgba32F (f32 per channel)
    rgba_f32.into_raw().into_iter().map(f16::from_f32).collect()
}

pub mod bokeh_shader {
    //! Bokeh compute shader constants and types for depth-of-field effect.

    use bytemuck::{Pod, Zeroable};

    #[repr(C)]
    #[derive(Debug, Clone, Copy, Pod, Zeroable)]
    pub struct BokehParams {
        pub image_width: f32,
        pub image_height: f32,
        pub max_blur_radius: f32,
        pub focus_distance: f32,
        pub bokeh_threshold: f32,
        pub num_rings: u32,
        pub samples_per_ring: u32,
        pub shape_mode: u32,
    }

    impl BokehParams {
        pub const fn defaults() -> Self {
            Self {
                image_width: 0.0,
                image_height: 0.0,
                max_blur_radius: 8.0,
                focus_distance: 0.5,
                bokeh_threshold: 0.03,
                num_rings: 3,
                samples_per_ring: 7,
                shape_mode: 0, // circular
            }
        }
    }
}

use bokeh_shader::BokehParams;

/// Parameters for the depth-of-field / portrait blur effect.
#[derive(Debug, Clone)]
pub struct DepthOfFieldConfig {
    /// Normalized depth [0..1] defining the focal plane (1.0 = farthest, 0.0 = nearest).
    pub focus_distance: f32,
    /// Maximum blur radius in pixels at edges of defocus range (0-40).
    pub blur_amount: f32,
    /// Minimum depth offset from focal plane to start blurring (0-0.2).
    pub bokeh_threshold: f32,
    /// Number of concentric rings for disk sampling (3-5).
    pub num_rings: u32,
    /// Base samples per ring (6-10).
    pub samples_per_ring: u32,
    /// Bokeh shape: 0 = circular disk, 1 = hexagonal aperture.
    pub bokeh_shape: u32,
}

impl Default for DepthOfFieldConfig {
    fn default() -> Self {
        Self {
            focus_distance: 0.5,
            blur_amount: 8.0,
            bokeh_threshold: 0.03,
            num_rings: 3,
            samples_per_ring: 7,
            bokeh_shape: 0, // circular
        }
    }
}

impl DepthOfFieldConfig {
    fn to_params(&self, width: u32, height: u32) -> BokehParams {
        BokehParams {
            image_width: width as f32,
            image_height: height as f32,
            max_blur_radius: self.blur_amount,
            focus_distance: self.focus_distance,
            bokeh_threshold: self.bokeh_threshold,
            num_rings: self.num_rings,
            samples_per_ring: self.samples_per_ring,
            shape_mode: self.bokeh_shape,
        }
    }
}

/// Result of applying depth-of-field blur.
pub struct DepthOfFieldResult {
    /// Base64-encoded PNG image data (with "data:image/png;base64," prefix).
    pub preview_url: String,
    pub width: u32,
    pub height: u32,
}

/// Get the cached depth map from AiState for a given image + geometry adjustments.
/// This reuses Depth Anything v2 output — zero ONNX inference calls.
fn get_cached_depth_map(
    state: &AppState,
    js_adjustments: &serde_json::Value,
) -> Result<Option<CachedDepthMap>, String> {
    let path = get_raw_image_path_from_adjustments(js_adjustments).ok_or_else(|| {
        "Could not resolve image path from adjustments".to_string()
    })?;

    // Build the exact same hash that generate_ai_depth_mask uses in ai_commands.rs.
    let mut hasher = blake3::Hasher::new();
    hasher.update(path.as_bytes());
    let mut geo_hasher = DefaultHasher::new();
    for key in GEOMETRY_KEYS {
        if let Some(val) = js_adjustments.get(key) {
            key.hash(&mut geo_hasher);
            val.to_string().hash(&mut geo_hasher);
        }
    }
    hasher.update(geo_hasher.finish().to_le_bytes());
    let path_hash = hasher.finalize().to_hex();

    // The cached depth map is stored as a single Option<CachedDepthMap> in AiState.
    let ai_guard = state.ai_state.lock().map_err(|e| format!("AI state locked: {}", e))?;
    let Some(ai) = &*ai_guard else {
        return Ok(None);
    };

    if let Some(cached) = &ai.depth_map {
        if cached.path_hash == path_hash {
            return Ok(Some(cached.clone()));
        }
        // Fallback: use whatever depth map exists (may have different geometry params).
        log::warn!(
            "Depth map hash mismatch (expected={}, found={}). Using available depth map.",
            path_hash, cached.path_hash
        );
        return Ok(Some(cached.clone()));
    }

    Ok(None)
}

/// Normalize a u8 depth image (0-255) to f32 normalized [0..1] data.
fn normalize_depth(data: &[u8], width: usize, height: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; width * height];
    for i in 0..width * height {
        out[i] = (data[i] as f32 / 255.0).clamp(0.0, 1.0);
    }
    out
}

/// Read back a Rgba16Float texture from GPU to CPU.
fn read_texture_f16(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture_view: &wgpu::TextureView,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, String> {
    let unpadded_bpr = 4 * width; // 4 f16s per pixel (RGBA) × 2 bytes = wait...
                                   // Rgba16Float = 4 × f16 = 8 bytes per pixel.
    let bpr_f16 = width as usize * 4; // 4 channels per pixel in f16
    let unpadded_bpr_bytes = bpr_f16 * 2; // each f16 is 2 bytes
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bpr = (unpadded_bpr_bytes + align - 1) & !(align - 1);

    let output_buffer_size = (padded_bpr * height as usize) as u64;

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Bokeh Readback Buffer"),
        size: output_buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Bokeh Readback Encoder"),
    });

    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: texture_view.as_image_copy(),
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &output_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bpr as u32),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    queue.submit(Some(encoder.finish()));

    let buffer_slice = output_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });

    device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(std::time::Duration::from_secs(60)),
    })?;

    match rx.recv() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => return Err(format!("Buffer map async failed: {}", e)),
        Err(_) => return Err("Buffer map channel closed".to_string()),
    }

    let padded_data = buffer_slice.get_mapped_range().to_vec();
    output_buffer.unmap();

    // Strip padding bytes (f16 data)
    let mut unpadded = Vec::with_capacity(unpadded_bpr_bytes * height as usize);
    for row in 0..height {
        let start = (row as usize) * padded_bpr;
        unpadded.extend_from_slice(&padded_data[start..start + unpadded_bpr_bytes]);
    }

    Ok(unpadded)
}

/// Convert f16 RGBA data to Rgba8 PNG base64.
fn f16_to_png_base64(data: &[u8], width: u32, height: u32) -> Result<String, String> {
    // Parse f16 → f32 → u8 (clamp to [0, 255])
    let f16_slice: &[f16] = bytemuck::cast_slice(data);

    let mut rgba8 = Vec::with_capacity((width * height) as usize * 4);
    for chunk in f16_slice.chunks_exact(4) {
        let r = (chunk[0].to_f32().clamp(0.0, 1.0) * 255.0).round() as u8;
        let g = (chunk[1].to_f32().clamp(0.0, 1.0) * 255.0).round() as u8;
        let b = (chunk[2].to_f32().clamp(0.0, 1.0) * 255.0).round() as u8;
        let a = chunk[3].to_f32(); // keep alpha as-is

        rgba8.extend_from_slice(&[r, g, b, (a.clamp(0.0, 1.0) * 255.0).round() as u8]);
    }

    let img = ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, rgba8)
        .ok_or("Failed to create image buffer from bokeh f16 data")?;

    let mut png_bytes: Vec<u8> = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut png_bytes), image::ImageFormat::Png)
        .map_err(|e| format!("PNG encode failed: {}", e))?;

    Ok(base64::Engine::encode(
        &base64::engine::general_purpose::STANDARD,
        &png_bytes,
    ))
}

/// Build the bokeh compute pipeline (bind group layout + shader module + pipeline).
fn build_bokeh_pipeline(
    device: &wgpu::Device,
) -> Result<
    (
        wgpu::BindGroupLayout,
        wgpu::ComputePipeline,
    ),
    String,
> {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Bokeh Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/bokeh.wgsl").into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Bokeh BGL"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Bokeh Pipeline Layout"),
        bind_group_layouts: &[&bgl],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Bokeh Pipeline"),
        layout: Some(&layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    Ok((bgl, pipeline))
}

/// Apply depth-of-field bokeh blur on GPU given pre-processed RGBA16F data and a depth map.
fn apply_bokeh_pass(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    input_data: &[u8], // Rgba16Float (f16) image data from main pipeline
    depth_data: &[f32], // Normalized [0..1] depth values
    width: u32,
    height: u32,
    config: &DepthOfFieldConfig,
    bgl: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
) -> Result<wgpu::TextureView, String> {
    // Create input texture from main pipeline output (Rgba16Float)
    let input_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Bokeh Input Texture"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &input_texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
        },
        input_data,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(std::mem::size_of::<f16>() as u32 * width * 4), // 4 channels × f16 (2 bytes)
            rows_per_image: Some(height),
        },
        wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
    );

    let input_view = input_texture.create_view(&Default::default());

    // Create depth texture from normalized data
    let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Bokeh Depth Texture"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let depth_view = depth_texture.create_view(&Default::default());
    queue.write_texture(
        wgpu::ImageCopyTexture { texture: &depth_texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        bytemuck::cast_slice(depth_data),
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(std::mem::size_of::<f32>() as u32 * width),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
    );

    // Create output texture (Rgba16Float for HDR accumulation)
    let output_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Bokeh Output Texture"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    let output_view = output_texture.create_view(&Default::default());

    // Uniform buffer for bokeh params (48 bytes — 12 × u32/f32)
    let params = config.to_params(width, height);
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Bokeh Params"),
        contents: bytemuck::cast_slice(std::slice::from_ref(&params)),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Bokeh Bind Group"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(input_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&depth_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(output_view) },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Buffer(params_buffer.as_entire_binding()),
            },
        ],
    });

    // Dispatch compute shader (8×8 workgroups = 64 threads per group)
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Bokeh Encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Bokeh Pass"),
            ..Default::default()
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        let wg_x = (width + 7) / 8;
        let wg_y = (height + 7) / 8;
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    queue.submit(Some(encoder.finish()));

    Ok(output_view)
}

/// Main entry point: apply depth-of-field blur using cached depth map from Depth Anything v2.
/// Main entry point: Tauri command handler for depth-of-field blur.
pub fn apply_depth_blur(
    state: &AppState,
    app_handle: &tauri::AppHandle,
    js_adjustments: serde_json::Value,
    config: &DepthOfFieldConfig,
    is_interactive: bool,
    target_resolution: Option<(u32, u32)>,
    compute_waveform: bool,
    active_waveform_channel: Option<String>,
) -> Result<DepthOfFieldResult, String> {
    let focus_distance = config.focus_distance.clamp(0.0, 1.0);
    let blur_amount = config.blur_amount.clamp(0.0, 40.0);
    let bokeh_threshold = config.bokeh_threshold.clamp(0.0, 0.2).min(focus_distance.max(1.0 - focus_distance) * 0.5);

    // Step 1: Get cached depth map (ZERO ONNX inference — reuses Depth Anything v2 output)
    let cache_entry = match get_cached_depth_map(state, &js_adjustments)? {
        Some(c) => c,
        None => return Err("No cached depth map found. Run generate_ai_depth_mask first to compute a depth map.".to_string()),
    };

    // Step 2: Get GPU context and initialize
    let context = get_or_init_gpu_context(state, app_handle)?;
    let device = &context.device;
    let queue = &context.queue;

    // Step 3: Determine target resolution (downscale for interactive preview)
    let original_w = cache_entry.original_size.0;
    let original_h = cache_entry.original_size.1;
    let (target_w, target_h) = if is_interactive {
        let scale_factor = match target_resolution {
            Some((w, h)) => w as f32 / original_w as f32,
            None => 640.0 / original_w.max(1) as f32,
        };
        (
            (original_w as f32 * scale_factor).round() as u32,
            (original_h as f32 * scale_factor).round() as u32,
        )
    } else if let Some((w, h)) = target_resolution {
        (w.max(1), h.max(1))
    } else {
        (original_w, original_h)
    };

    // Step 4: Run the main GPU pipeline to get pre-adjusted image.
    // This applies geometry corrections, tone mapping, and all standard adjustments.
    let loaded_image_guard = state.original_image.lock().map_err(|e| format!("Image lock failed: {}", e))?;
    let loaded_image = loaded_image_guard.as_ref()
        .ok_or("No original image loaded")?
        .clone();
    drop(loaded_image_guard);

    // Generate the base transformed preview (same as process_preview_job does).
    let dim = target_w.max(target_h).max(1);
    let (base_image, _scale, _offset) = crate::generate_transformed_preview(
        state, &loaded_image, &js_adjustments, dim,
    )?;

    // Downscale if needed for interactive mode.
    let processing_image = if is_interactive {
        let current_w = base_image.width();
        if current_w > target_w {
            downscale_f32_image(&base_image, target_w.max(1), target_h.max(1))
        } else {
            base_image.clone()
        }
    } else {
        base_image.clone()
    };

    let (img_width, img_height) = processing_image.dimensions();

    // Build adjustments for the main pipeline (same as apply_adjustments does — lines 478-482 of lib.rs).
    let is_raw = loaded_image.is_raw;
    let tm_override = resolve_tonemapper_override_from_handle(app_handle, is_raw);
    let final_adjustments = crate::image_processing::get_all_adjustments_from_json(&js_adjustments, is_raw, tm_override);

    // Extract mask bitmaps from adjustments.
    let mask_definitions: Vec<MaskDefinition> = js_adjustments.get("masks")
        .and_then(|m| serde_json::from_value(m.clone()).ok())
        .unwrap_or_default();

    let effective_scale = 1.0;
    let scaled_crop_offset = (0.0_f32, 0.0_f32);
    // Type matches RenderRequest::mask_bitmaps (Vec<ImageBuffer<Luma<u8>, Vec<u8>>>).
    let mask_bitmaps: Vec<ImageBuffer<Luma<u8>, Vec<u8>>> = {
        mask_definitions.iter()
            .filter_map(|def| {
                crate::mask_generation::get_cached_or_generate_mask(
                    state, def, img_width, img_height, effective_scale, scaled_crop_offset, &js_adjustments,
                )
            })
            .collect()
    };

    let transform_hash = calculate_transform_hash(&js_adjustments);

    // Check if there's a LUT to load.
    let lut_path = js_adjustments.get("lutPath").and_then(|v| v.as_str());
    let lut = match lut_path {
        Some(p) => get_or_load_lut(state, p).ok(),
        None => None,
    };

    let request = RenderRequest {
        adjustments: final_adjustments,
        mask_bitmaps: &mask_bitmaps,
        lut,
        roi: None,
    };

    // Run the main GPU processing pipeline (geometry corrections + tone mapping + all standard adjustments).
    let processed_dynamic_image = process_and_get_dynamic_image_with_analytics(
        &context, state, &processing_image, transform_hash, request, "dof_main_pipeline", false, None,
    ).map_err(|e| format!("Main pipeline failed: {}", e))?;

    // Step 5: Convert processed image to Rgba16Float for bokeh shader input.
    let img_rgba_f16 = to_rgba_f16(&processed_dynamic_image);

    // Step 6: Normalize cached depth map to f32 [0..1].
    let depth_normalized = normalize_depth(
        &cache_entry.depth_image,
        cache_entry.original_size.0 as usize,
        cache_entry.original_size.1 as usize,
    );

    // Scale depth data if image dimensions differ from original depth map size.
    // (Depth Anything v2 outputs a fixed-size map; we resize it to match current processing size.)
    let (depth_w, depth_h) = (img_width, img_height);
    let source_depth_data = &depth_normalized;
    let source_dw = cache_entry.original_size.0 as usize;
    let source_dh = cache_entry.original_size.1 as usize;

    let scaled_depth: Vec<f32> = if source_dw != depth_w as usize || source_dh != depth_h as usize {
        // Simple bilinear-ish resize by nearest neighbor (good enough for relative depth).
        let mut resized = vec![0.0f32; (depth_w * depth_h) as usize];
        for y in 0..depth_h {
            for x in 0..depth_w {
                let sx = (x as f32 * source_dw as f32 / depth_w as f32).round() as usize;
                let sy = (y as f32 * source_dh as f32 / depth_h as f32).round() as usize;
                let si = sy.min(source_dh - 1) * source_dw + x.min(source_dw - 1);
                resized[(y * depth_w + x) as usize] = source_depth_data[si];
            }
        }
        resized
    } else {
        source_depth_data.to_vec()
    };

   // Step 7: Build bokeh compute pipeline.
    // In production, this could be cached in AppState to avoid rebuilding per call.
    let (bgl, pipeline) = build_bokeh_pipeline(device)?;

    // Step 8: Apply bokeh compute shader pass.
    let output_view = apply_bokeh_pass(
        device, queue, bytemuck::cast_slice(&img_rgba_f16),
        &scaled_depth, depth_w, depth_h, config, bgl, pipeline,
    )?;

    // Step 9: Read back result and encode as base64 PNG.
    let readback_data = read_texture_f16(device, queue, output_view.as_ref(), img_width, img_height)?;
    let png_base64 = f16_to_png_base64(&readback_data, img_width, img_height)?;

    Ok(DepthOfFieldResult {
        preview_url: format!("data:image/png;base64,{}", png_base64),
        width: img_width,
        height: img_height,
    })
}
