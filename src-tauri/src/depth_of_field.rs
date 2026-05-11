use std::sync::Arc;

use half::f16;
use image::{DynamicImage, GenericImageView, ImageBuffer, Luma, Rgba};
use wgpu::util::DeviceExt;

use crate::ai_processing::CachedDepthMap;
use crate::app_state::AppState;
use crate::gpu_processing::{
    get_or_init_gpu_context, process_and_get_dynamic_image_with_analytics, RenderRequest,
};
use crate::image_processing::{downscale_f32_image, resolve_tonemammer_override_from_handle};
use crate::mask_generation::MaskDefinition;
use crate::{calculate_transform_hash, get_or_load_lut};

/// Convert DynamicImage (Rgba8) → Rgba16Float (f16 RGBA). Mirrors gpu_processing::to_rgba_f16.
fn to_rgba_f16(img: &DynamicImage) -> Vec<f16> {
    let rgba_f32 = img.to_rgba32f();
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

 }

use bokeh_shader::BokehParams;

/// Parameters for the depth-of-field / portrait blur effect.
#[derive(Debug, Clone)]
pub struct DepthOfFieldConfig {
    pub focus_distance: f32,
    pub blur_amount: f32,
    pub bokeh_threshold: f32,
    pub num_rings: u32,
    pub samples_per_ring: u32,
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

/// Get the cached depth map from AiState for a given image + geometry adjustments.
fn get_cached_depth_map(
    state: &tauri::State<AppState>,
    js_adjustments: &serde_json::Value,
) -> Result<Option<CachedDepthMap>, String> {
    // The path comes from the loaded image in AppState, not from adjustments JSON.
    let ai_guard = state.ai_state.lock().map_err(|e| format!("AI state locked: {}", e))?;
    let Some(ai) = &*ai_guard else {
        return Ok(None);
    };

    if let Some(cached) = &ai.depth_map {
        // Build the same hash that generate_ai_depth_mask uses.
        let mut hasher = blake3::Hasher::new();
        hasher.update(cached.path_hash.as_bytes());
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
    let bpr_f16 = (width as usize) * 4; // 4 channels per pixel in f16
    let unpadded_bpr_bytes = bpr_f16 * 2; // each f16 is 2 bytes
    let align: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bpr_u32 = (unpadded_bpr_bytes as u32 + align - 1) & !(align - 1);
    let padded_bpr = padded_bpr_u32 as usize;

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

    // Use TexelCopyTextureInfo (not ImageCopyTexture — wgpu 28.x API)
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: texture_view,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &output_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bpr_u32),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
    );

    queue.submit(Some(encoder.finish()));

    let buffer_slice = output_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });

    match device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(std::time::Duration::from_secs(60)),
    }) {
        Ok(()) => {}
        Err(e) => return Err(format!("GPU poll failed: {}", e)),
    }

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
    let f16_slice: &[f16] = bytemuck::cast_slice(data);

    let mut rgba8 = Vec::with_capacity((width * height) as usize * 4);
    for chunk in f16_slice.chunks_exact(4) {
        let r = (chunk[0].to_f32().clamp(0.0, 1.0) * 255.0).round() as u8;
        let g = (chunk[1].to_f32().clamp(0.0, 1.0) * 255.0).round() as u8;
        let b = (chunk[2].to_f32().clamp(0.0, 1.0) * 255.0).round() as u8;
        let a = chunk[3].to_f32();

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
        push_constant_ranges: &[],
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
    // Create input texture using create_texture_with_data (wgpu 28.x pattern).
    let input_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Bokeh Input Texture"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    // Upload using create_texture_with_data (handles queue write internally).
    device.create_texture_with_data(
        queue,
        &input_texture,
        wgpu::util::TextureDataOrder::MipMajor,
        input_data,
    );

    let input_view = input_texture.create_view(&Default::default());

    // Create depth texture from normalized data.
    let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Bokeh Depth Texture"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let depth_view = depth_texture.create_view(&Default::default());
    // Upload using create_texture_with_data (wgpu 28.x pattern — same as gpu_processing.rs).
    device.create_texture_with_data(
        queue,
        &depth_texture,
        wgpu::util::TextureDataOrder::MipMajor,
        bytemuck::cast_slice(depth_data),
    );

    // Create output texture (Rgba16Float for HDR accumulation).
    let output_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Bokeh Output Texture"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    let output_view = output_texture.create_view(&Default::default());

    // Uniform buffer for bokeh params (32 bytes — 8 × u32/f32).
    let params = config.to_params(width, height);
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Bokeh Params"),
        contents: bytemuck::cast_slice(std::slice::from_ref(&params)),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Bind group — all texture views need & references in wgpu 28.x.
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Bokeh Bind Group"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&input_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&depth_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&output_view) },
             wgpu::BindGroupEntry {
                 binding: 3,
                 resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    // Dispatch compute shader (8×8 workgroups = 64 threads per group).
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Bokeh Encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Bokeh Pass"),
            ..Default::default()
        });
        // Pipeline needs & reference.
        pass.set_pipeline(&pipeline);
        // Bind group needs & reference in wgpu 28.x.
        pass.set_bind_group(0, &bind_group, &[]);
        let wg_x = (width + 7) / 8;
        let wg_y = (height + 7) / 8;
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    queue.submit(Some(encoder.finish()));

    Ok(output_view)
}

/// Main entry point: Tauri command handler for depth-of-field blur.
#[tauri::command]
pub async fn apply_depth_blur(
    js_adjustments: serde_json::Value,
    focus_distance: f32,
    blur_amount: f32,
    bokeh_threshold: f32,
    num_rings: u32,
    samples_per_ring: u32,
    bokeh_shape: u32,
    is_interactive: bool,
    target_resolution: Option<serde_json::Value>,
    _roi: Option<serde_json::Value>,
    compute_waveform: bool,
    state: tauri::State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<String, String> {
    let focus_distance = focus_distance.clamp(0.0, 1.0);
    let blur_amount = blur_amount.clamp(0.0, 40.0);
    let bokeh_threshold = bokeh_threshold.clamp(0.0, 0.2).min(focus_distance.max(1.0 - focus_distance) * 0.5);

    // Step 1: Get cached depth map (ZERO ONNX inference — reuses Depth Anything v2 output).
    let cache_entry = match get_cached_depth_map(&state, &js_adjustments)? {
        Some(c) => c,
        None => return Err("No cached depth map found. Run generate_ai_depth_mask first to compute a depth map.".to_string()),
    };

    // Step 2: Get GPU context and initialize.
    let context = get_or_init_gpu_context(&state, &app_handle)?;
    let device = &context.device;
    let queue = &context.queue;

    // Step 3: Determine target resolution.
    let original_w = cache_entry.original_size.0 as u32;
    let original_h = cache_entry.original_size.1 as u32;
    let (target_w, target_h) = if is_interactive {
        let scale_factor = 640.0 / original_w.max(1) as f32;
        ((original_w as f32 * scale_factor).round() as u32, (original_h as f32 * scale_factor).round() as u32)
    } else if let Some(res) = target_resolution {
        let w = res.get("width").and_then(|v| v.as_u64()).unwrap_or(original_w) as u32;
        let h = res.get("height").and_then(|v| v.as_u64()).unwrap_or(original_h) as u32;
        (w.max(1), h.max(1))
    } else {
        (original_w, original_h)
    };

    // Step 4: Get the loaded image from AppState.
    let loaded_image_guard = state.original_image.lock().map_err(|e| format!("Image lock failed: {}", e))?;
    let loaded_image = loaded_image_guard.as_ref()
        .ok_or("No original image loaded")?
        .clone();
    drop(loaded_image_guard);

    // Generate the base transformed preview (same as process_preview_job does).
    let dim = target_w.max(target_h).max(1);
    let (base_image, _scale, _offset) = crate::generate_transformed_preview(
        &state, &loaded_image, &js_adjustments, dim,
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

    // Build adjustments for the main pipeline.
    let is_raw = loaded_image.is_raw;
    let tm_override = resolve_tonemapper_override_from_handle(&app_handle, is_raw);
    let final_adjustments = crate::image_processing::get_all_adjustments_from_json(&js_adjustments, is_raw, tm_override);

    // Extract mask bitmaps from adjustments.
    let mask_definitions: Vec<MaskDefinition> = js_adjustments.get("masks")
        .and_then(|m| serde_json::from_value(m.clone()).ok())
        .unwrap_or_default();

    let effective_scale = 1.0;
    let scaled_crop_offset = (0.0_f32, 0.0_f32);
    let mask_bitmaps: Vec<ImageBuffer<Luma<u8>, Vec<u8>>> = {
        mask_definitions.iter()
            .filter_map(|def| {
                crate::mask_generation::get_cached_or_generate_mask(
                    &state, def, img_width, img_height, effective_scale, scaled_crop_offset, &js_adjustments,
                )
            })
            .collect()
    };

    // Check if there's a LUT to load.
    let lut_path = js_adjustments.get("lutPath").and_then(|v| v.as_str());
    let lut = match lut_path {
        Some(p) => get_or_load_lut(&state, p).ok(),
        None => None,
    };

    let transform_hash = calculate_transform_hash(&js_adjustments);

    let request = RenderRequest {
        adjustments: final_adjustments,
        mask_bitmaps: &mask_bitmaps,
        lut,
        roi: None,
    };

    // Run the main GPU processing pipeline.
    let processed_dynamic_image = process_and_get_dynamic_image_with_analytics(
        &context, &state, &processing_image, transform_hash, request, "dof_main_pipeline", false, None,
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
    let source_dw = cache_entry.original_size.0;
    let source_dh = cache_entry.original_size.1;
    let target_w_usize = img_width as usize;
    let target_h_usize = img_height as usize;

    let scaled_depth: Vec<f32> = if source_dw != target_w_usize || source_dh != target_h_usize {
        let mut resized = vec![0.0f32; (target_w * target_h) as usize];
        for y in 0..target_h_usize {
            for x in 0..target_w_usize {
                let sx = ((x as f32 * source_dw as f32 / target_w_usize as f32).round() as usize).min(source_dw - 1);
                let sy = ((y as f32 * source_dh as f32 / target_h_usize as f32).round() as usize).min(source_dh - 1);
                resized[y * target_w_usize + x] = depth_normalized[sy * source_dw + sx];
            }
        }
        resized
    } else {
        depth_normalized
    };

    // Step 7: Build bokeh compute pipeline.
    let (bgl, pipeline) = build_bokeh_pipeline(device)?;

    // Step 8: Apply bokeh compute shader pass.
    let output_view = apply_bokeh_pass(
        device, queue, bytemuck::cast_slice(&img_rgba_f16),
        &scaled_depth, img_width, img_height,
        &DepthOfFieldConfig {
            focus_distance, blur_amount, bokeh_threshold,
            num_rings: num_rings.max(1).min(5),
            samples_per_ring: samples_per_ring.max(1).min(10),
            bokeh_shape,
        },
        bgl, pipeline,
    )?;

    // Step 9: Read back result and encode as base64 PNG.
    let readback_data = read_texture_f16(device, queue, &output_view, img_width, img_height)?;
    let png_base64 = f16_to_png_base64(&readback_data, img_width, img_height)?;

    Ok(format!("data:image/png;base64,{}", png_base64))
}
