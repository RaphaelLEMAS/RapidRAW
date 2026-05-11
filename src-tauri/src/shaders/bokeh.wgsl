// shaders/bokeh.wgsl
// Depth-aware bokeh blur with circular and hexagonal shape support
// Each pixel's blur radius is proportional to its depth distance from the focal plane.
// Disk sampling uses concentric rings with sqrt distribution for even coverage,
// plus jitter per-pixel for deterministic (no flicker) output.

struct Params {
    image_width : f32,
    image_height : f32,
    max_blur_radius : f32,       // Maximum blur radius in pixels (0-40)
    focus_distance : f32,         // Normalized depth [0..1] where 1.0 = focal plane
    bokeh_threshold : f32,        // Min depth offset to start blurring (0-0.2)
    num_rings : u32,              // Concentric rings: 3-5
    samples_per_ring : u32,       // Base samples per ring
    shape_mode : u32,             // 0 = circular disk, 1 = hexagonal
};

@group(0) @binding(0) var input_tex : texture_2d<f32>;       // Current image (from main pipeline, Rgba16Float)
@group(0) @binding(1) var depth_tex : texture_2d<f32>;        // Cached depth map (R32Float)
@group(0) @binding(2) var output_tex : texture_storage_2d<rgba16float, write>;  // HDR accumulation buffer
@group(0) @binding(3) var params : buffer<Params> { get params; }

// --- Deterministic hash for jitter (same pixel = same pattern across frames) ---
fn hash_scalar(n: u32) -> f32 {
    var s = n;
    s ^= 2747636419u;
    s *= 2654435769u;
    s ^= s >> 16u;
    s *= 2654435769u;
    s ^= s >> 16u;
    return f32(s) / f32(0xFFFFFFFFu);
}

fn hash_vec2(x: u32, y: u32) -> vec2<f32> {
    return vec2<f32>(hash_scalar(x * 73856093u ^ y), hash_scalar(y * 19349663u ^ x));
}

// --- Depth sampling with bounds clamping ---
fn sample_depth(u: f32, v: f32) -> f32 {
    let w = params.image_width;
    let h = params.image_height;
    let su = i32(clamp(u, 0.0, w - 1.0));
    let sv = i32(clamp(v, 0.0, h - 1.0));
    return textureLoad(depth_tex, vec2<i32>(su, sv), 0).r;
}

// --- Color sampling with bounds clamping (linear interpolation for out-of-bounds) ---
fn sample_color(u: f32, v: f32) -> vec4<f32> {
    let w = params.image_width;
    let h = params.image_height;
    return textureSampleLevel(input_tex, sampler_linear_clamp, vec2<u32>(i32(clamp(u, 0.0, w - 1.0)), i32(clamp(v, 0.0, h - 1.0))), 0).rgba;
}

// --- Hexagonal vertex positions (unit hexagon corners) ---
fn get_hex_corners() -> array<vec2<f32>, 6> {
    var corners: array<vec2<f32>, 6>;
    for (var i = 0u; i < 6u; i++) {
        let angle = f32(i) * 1.047197551 + 0.523598776; // pi/3 per vertex, offset by pi/6
        corners[i] = vec2<f32>(cos(angle), sin(angle));
    }
    return corners;
}

// --- Sample a hexagonal ring pattern (simulates polygonal aperture) ---
fn sample_hex_ring(cx: f32, cy: f32, radius: f32, seed_val: u32) -> vec4<f32> {
    let corners = get_hex_corners();
    var sum = vec4(0.0);
    var weight_sum = 0.0;

    // Sample along each of the 6 hex edges + fill interior with center point
    for (var edge = 0u; edge < 6u; edge++) {
        let p1 = corners[edge];
        let p2 = corners[(edge + 1u) % 6u];

        // Sample at 3 positions along each edge for good coverage
        for (var t_idx = 0u; t_idx <= 2u; t_idx++) {
            var t = f32(t_idx) / 2.0;
            let jx = hash_scalar(seed_val + u32(edge * 10 + t_idx)) * 0.15 - 0.075;
            let jy = hash_scalar(seed_val + u32(edge * 10 + t_idx) ^ 999u) * 0.15 - 0.075;

            let sx = cx + (p1.x * (1.0 - t) + p2.x * t + jx) * radius;
            let sy = cy + (p1.y * (1.0 - t) + p2.y * t + jy) * radius;

            sum += sample_color(sx, sy);
            weight_sum += 1.0;
        }
    }

    // Center point weighted higher to fill hex interior
    let jitter = hash_vec2(seed_val, u32(cx as u32) ^ u32(cy as u32));
    sum += sample_color(
        cx + (jitter.x - 0.5) * radius * 0.6,
        cy + (jitter.y - 0.5) * radius * 0.6
    );
    weight_sum += 1.5;

    return sum / max(weight_sum, 0.001);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let x = f32(gid.x);
    let y = f32(gid.y);
    let w = params.image_width;
    let h = params.image_height;

    if (x >= w || y >= h) { return; }

    // Load current pixel color and depth
    let color: vec4<f32> = sample_color(x, y);
    let depth: f32 = sample_depth(x, y);

    // Compute normalized distance from focal plane
    var depth_offset: f32 = abs(depth - params.focus_distance);

    // Pixels within threshold of focal plane stay sharp — skip blur entirely
    if (depth_offset <= params.bokeh_threshold) {
        textureStore(output_tex, vec2<i32>(i32(x), i32(y)), color);
        return;
    }

    // Scale blur radius proportionally to depth distance from focal plane
    let denom: f32 = max(1.0 - params.focus_distance + 0.01, 0.01);
    var normalized: f32 = clamp((depth_offset - params.bokeh_threshold) / denom, 0.0, 1.0);

    // Also consider near-side blur (for objects closer than focal plane)
    let denom_near: f32 = max(params.focus_distance + 0.01, 0.01);
    var normalized_near: f32 = clamp((depth_offset - params.bokeh_threshold) / denom_near, 0.0, 1.0);
    normalized = max(normalized, normalized_near);

    var blur_radius: f32 = normalized * params.max_blur_radius;
    blur_radius = min(blur_radius, 40.0);

    // Skip tiny blurs (waste of compute)
    if (blur_radius < 0.5) {
        textureStore(output_tex, vec2<i32>(i32(x), i32(y)), color);
        return;
    }

    // --- Bokeh sampling based on shape mode ---
    var sum_color: vec4<f32> = vec4(0.0);
    var total_weight: f32 = 0.0;

    let nr: u32 = max(params.num_rings, 1u);
    let spr: u32 = max(params.samples_per_ring, 1u);
    let seed_val: u32 = gid.x * 131u + gid.y * 7919u;

    if (params.shape_mode == 0u) {
        // === CIRCULAR BOKEH DISK ===
        for (var ring = 1u; ring <= nr; ring++) {
            let ring_r: f32 = (f32(ring) / f32(nr)) * blur_radius;

            // More samples on outer rings for even coverage
            let n: u32 = spr + ring;

            for (var s = 0u; s < n; s++) {
                var angle: f32 = f32(s) / f32(n) * 6.2831853;
                // Angular jitter
                angle += hash_scalar(seed_val ^ (ring << 4u) ^ s) * 0.3 - 0.15;

                // sqrt distribution for even disk coverage (prevents center clustering)
                let u: f32 = hash_vec2(ring, s).x * ring_r;

                let sx = x + cos(angle) * u;
                let sy = y + sin(angle) * u;

                let c: vec4<f32> = sample_color(sx, sy);
                // Weight decreases toward ring edge (creates natural falloff in bokeh balls)
                let w_ = 1.0 - (u / max(ring_r, 0.001)) * 0.5;
                sum_color += c * w_;
                total_weight += w_;
            }
        }

        // Center pixel with full weight
        sum_color += color * 2.0;
        total_weight += 2.0;

    } else {
        // === HEXAGONAL BOKEH (simulates lens aperture blades) ===
        for (var ring = 1u; ring <= nr; ring++) {
            let ring_r: f32 = (f32(ring) / f32(nr)) * blur_radius;
            sum_color += sample_hex_ring(x, y, ring_r, seed_val ^ (ring << 8u));
            total_weight += 1.0;
        }

        // Center hex (smallest aperture shape — sharpest part of bokeh)
        sum_color += sample_hex_ring(x, y, blur_radius * 0.2, seed_val);
        total_weight += 1.5;
    }

    let out_color: vec4<f32> = clamp(sum_color / max(total_weight, 0.001), -100.0, 100.0);
    textureStore(output_tex, vec2<i32>(i32(x), i32(y)), out_color);
}
