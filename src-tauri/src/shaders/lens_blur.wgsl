struct LensBlurParams {
    amount: f32,
    size: f32,
    aperture: f32,
    bokeh_shape: u32,
    bokeh_intensity: f32,
    highlight_boost: f32,
    fringe_amount: f32,
    swirl_amount: f32,
    width: u32,
    height: u32,
    tile_offset_x: u32,
    tile_offset_y: u32,
}

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> params: LensBlurParams;
@group(0) @binding(3) var mask_texture: texture_2d<f32>;

const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let cutoff = vec3<f32>(0.04045);
    let higher = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    let lower = c / 12.92;
    return select(higher, lower, c <= cutoff);
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let c_clamped = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
    let cutoff = vec3<f32>(0.0031308);
    let higher = 1.055 * pow(c_clamped, vec3<f32>(1.0 / 2.4)) - 0.055;
    let lower = c_clamped * 12.92;
    return select(higher, lower, c_clamped <= cutoff);
}

fn get_luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Check if a point is inside a regular polygon with N sides
fn point_in_polygon(p: vec2<f32>, n: u32, roundness: f32) -> f32 {
    let angle = atan2(p.y, p.x);
    let r = length(p);

    // Distance to polygon edge
    let segment_angle = TAU / f32(n);
    let half_segment = segment_angle * 0.5;

    // Angle within current segment
    let a = ((angle % segment_angle) + segment_angle) % segment_angle - half_segment;

    // Polygon radius at this angle
    let polygon_r = cos(half_segment) / cos(a);

    // Blend between circle (1.0) and polygon based on roundness
    let shape_r = mix(polygon_r, 1.0, roundness);

    return smoothstep(shape_r, shape_r - 0.05, r);
}

// Rotate a 2D point
fn rotate2d(p: vec2<f32>, angle: f32) -> vec2<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec2<f32>(p.x * c - p.y * s, p.x * s + p.y * c);
}

@compute @workgroup_size(8, 8, 1)
fn lens_blur_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let out_dims = vec2<u32>(params.width, params.height);
    if (id.x >= out_dims.x || id.y >= out_dims.y) { return; }

    let absolute_coord = vec2<u32>(id.x + params.tile_offset_x, id.y + params.tile_offset_y);
    let input_dims = vec2<i32>(textureDimensions(input_texture));

    // Read mask value at this pixel - determines blur intensity
    let mask_value = textureLoad(mask_texture, vec2<i32>(absolute_coord), 0).r;

    // If mask is zero, just pass through the original pixel
    if (mask_value < 0.001) {
        let original = textureLoad(input_texture, absolute_coord, 0).rgb;
        textureStore(output_texture, id.xy, vec4<f32>(original, 1.0));
        return;
    }

    // Calculate effective blur radius based on amount, size, and mask
    let base_radius = params.amount * params.size * 0.5;
    let effective_radius = base_radius * mask_value;

    if (effective_radius < 0.5) {
        let original = textureLoad(input_texture, absolute_coord, 0).rgb;
        textureStore(output_texture, id.xy, vec4<f32>(original, 1.0));
        return;
    }

    let radius_i = i32(ceil(effective_radius));
    let radius_f = effective_radius;

    // Aperture controls roundness: 0 = polygon, 100 = circle
    let roundness = params.aperture;
    let blade_count = params.bokeh_shape;
    let highlight_boost = params.highlight_boost;
    let fringe = params.fringe_amount;
    let swirl = params.swirl_amount;

    var total_color = vec3<f32>(0.0);
    var total_weight = 0.0;

    // Chromatic aberration channels for fringing
    var total_r = 0.0;
    var total_g = 0.0;
    var total_b = 0.0;
    var weight_r = 0.0;
    var weight_g = 0.0;
    var weight_b = 0.0;

    let use_fringe = fringe > 0.001;

    // Sample in a disc/polygon pattern
    for (var dy = -radius_i; dy <= radius_i; dy = dy + 1) {
        for (var dx = -radius_i; dx <= radius_i; dx = dx + 1) {
            let offset = vec2<f32>(f32(dx), f32(dy));
            let dist = length(offset);

            // Skip samples outside the radius
            if (dist > radius_f + 0.5) {
                continue;
            }

            // Normalize to unit disc
            var normalized = offset / radius_f;

            // Apply swirl rotation based on distance from center
            if (abs(swirl) > 0.001) {
                let swirl_angle = dist / radius_f * swirl * PI * 0.5;
                normalized = rotate2d(normalized, swirl_angle);
            }

            // Compute bokeh shape weight
            let shape_weight = point_in_polygon(normalized, blade_count, roundness);

            if (shape_weight < 0.001) {
                continue;
            }

            // Sample coordinate
            let sample_coord = vec2<i32>(absolute_coord) + vec2<i32>(dx, dy);
            let clamped_coord = clamp(sample_coord, vec2<i32>(0), input_dims - vec2<i32>(1));

            let sample_color = textureLoad(input_texture, vec2<u32>(clamped_coord), 0).rgb;

            // Highlight boost: weight brighter pixels more for bokeh disc effect
            var luma_weight = 1.0;
            if (highlight_boost > 0.001) {
                let sample_luma = get_luma(sample_color);
                // Boost highlights to create bright bokeh discs
                let highlight_factor = pow(max(sample_luma, 0.0), 1.0 + highlight_boost * 3.0);
                luma_weight = mix(1.0, 1.0 + highlight_factor * highlight_boost * 4.0, highlight_boost);
            }

            let final_weight = shape_weight * luma_weight;

            if (use_fringe) {
                // Chromatic aberration: shift R and B channels radially
                let fringe_offset = fringe * 0.3;
                let radial_pos = dist / radius_f; // 0 at center, 1 at edge

                // Red channel samples slightly outward
                let r_scale = 1.0 + fringe_offset * radial_pos;
                let r_offset = vec2<i32>(i32(round(f32(dx) * r_scale)), i32(round(f32(dy) * r_scale)));
                let r_coord = clamp(vec2<i32>(absolute_coord) + r_offset, vec2<i32>(0), input_dims - vec2<i32>(1));
                let r_sample = textureLoad(input_texture, vec2<u32>(r_coord), 0).r;

                // Blue channel samples slightly inward
                let b_scale = 1.0 - fringe_offset * radial_pos;
                let b_offset = vec2<i32>(i32(round(f32(dx) * b_scale)), i32(round(f32(dy) * b_scale)));
                let b_coord = clamp(vec2<i32>(absolute_coord) + b_offset, vec2<i32>(0), input_dims - vec2<i32>(1));
                let b_sample = textureLoad(input_texture, vec2<u32>(b_coord), 0).b;

                total_r += r_sample * final_weight;
                total_g += sample_color.g * final_weight;
                total_b += b_sample * final_weight;
                weight_r += final_weight;
                weight_g += final_weight;
                weight_b += final_weight;
            } else {
                total_color += sample_color * final_weight;
                total_weight += final_weight;
            }
        }
    }

    var blurred: vec3<f32>;
    if (use_fringe) {
        blurred = vec3<f32>(
            total_r / max(weight_r, 0.001),
            total_g / max(weight_g, 0.001),
            total_b / max(weight_b, 0.001)
        );
    } else {
        blurred = total_color / max(total_weight, 0.001);
    }

    // Blend between original and blurred based on mask
    let original = textureLoad(input_texture, absolute_coord, 0).rgb;
    let result = mix(original, blurred, mask_value);

    textureStore(output_texture, id.xy, vec4<f32>(result, 1.0));
}
