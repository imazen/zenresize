//! Brute-force test to find the optimal unpremultiply alpha threshold.
//!
//! For each threshold candidate, compares the final u8 RGBA output
//! against the reference (threshold=0, i.e., always divide), measuring:
//!
//! - How many visible pixels (alpha_u8 > 0) have wrong RGB u8 values
//! - The max u8 delta on visible pixels (the visible artifact)
//! - How many invisible pixels (alpha_u8 = 0) have different RGB values
//!   (cosmetic, changes checksums but not appearance)
//!
//! Run: cargo test --test unpremul_threshold -- --nocapture

use zenresize::{AlphaMode, Filter, PixelDescriptor, ResizeConfig, Resizer};

/// Unpremultiply a premultiplied f32 row with a given threshold, then encode to sRGB u8.
fn unpremul_encode_u8(premul: &[f32], threshold: f32) -> Vec<u8> {
    let mut out = vec![0u8; premul.len()];

    for (px_in, px_out) in premul.chunks_exact(4).zip(out.chunks_exact_mut(4)) {
        let a = px_in[3];
        if a > threshold {
            let inv_a = 1.0 / a;
            for i in 0..3 {
                let unpremul = (px_in[i] * inv_a).clamp(0.0, 1.0);
                // Use linear-srgb's public scalar function for conversion
                px_out[i] = linear_srgb::default::linear_to_srgb_u8(unpremul);
            }
        } else {
            px_out[0] = 0;
            px_out[1] = 0;
            px_out[2] = 0;
        }
        px_out[3] = (a.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    }
    out
}

/// Compare two u8 RGBA outputs pixel by pixel.
/// Returns (visible_wrong, visible_max_delta, invisible_diff)
fn compare_outputs(reference: &[u8], candidate: &[u8]) -> (usize, u8, usize) {
    let mut visible_wrong = 0usize;
    let mut visible_max_delta: u8 = 0;
    let mut invisible_diff = 0usize;

    for (ref_px, cand_px) in reference.chunks_exact(4).zip(candidate.chunks_exact(4)) {
        let ref_a = ref_px[3];
        let cand_a = cand_px[3]; // alpha should be identical (same threshold for alpha encode)
        debug_assert_eq!(ref_a, cand_a);

        let rgb_differs = ref_px[0] != cand_px[0] || ref_px[1] != cand_px[1] || ref_px[2] != cand_px[2];

        if ref_a == 0 {
            if rgb_differs {
                invisible_diff += 1;
            }
        } else if rgb_differs {
            visible_wrong += 1;
            for i in 0..3 {
                let delta = (ref_px[i] as i16 - cand_px[i] as i16).unsigned_abs() as u8;
                visible_max_delta = visible_max_delta.max(delta);
            }
        }
    }

    (visible_wrong, visible_max_delta, invisible_diff)
}

fn resize_to_premul_f32(
    input: &[u8], in_w: u32, in_h: u32, out_w: u32, out_h: u32, filter: Filter,
) -> Vec<f32> {
    let config = ResizeConfig::builder(in_w, in_h, out_w, out_h)
        .filter(filter)
        .input(PixelDescriptor::RGBA8_SRGB)
        .output(PixelDescriptor::RGBAF32_LINEAR.with_alpha(Some(AlphaMode::Premultiplied)))
        .linear()
        .build();
    Resizer::new(&config).resize_u8_to_f32(input)
}

/// Sharp alpha edge: bright white on left, transparent on right.
fn create_sharp_edge(width: u32, height: u32) -> Vec<u8> {
    let mut data = vec![0u8; (width * height * 4) as usize];
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 4) as usize;
            if x < width / 2 {
                data[idx] = 255; data[idx+1] = 255; data[idx+2] = 255; data[idx+3] = 255;
            }
        }
    }
    data
}

/// Alpha gradient from 255 to 0, bright color.
fn create_alpha_gradient(width: u32, height: u32) -> Vec<u8> {
    let mut data = vec![0u8; (width * height * 4) as usize];
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 4) as usize;
            let a = ((1.0 - x as f32 / width as f32) * 255.0) as u8;
            data[idx] = 255; data[idx+1] = 200; data[idx+2] = 100; data[idx+3] = a;
        }
    }
    data
}

/// Single bright pixel surrounded by transparent.
fn create_bright_dot(width: u32, height: u32) -> Vec<u8> {
    let mut data = vec![0u8; (width * height * 4) as usize];
    let cx = width / 2;
    let cy = height / 2;
    let idx = ((cy * width + cx) * 4) as usize;
    data[idx] = 255; data[idx+1] = 255; data[idx+2] = 255; data[idx+3] = 255;
    data
}

/// Checkerboard of opaque white and transparent.
fn create_checkerboard(width: u32, height: u32) -> Vec<u8> {
    let mut data = vec![0u8; (width * height * 4) as usize];
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 4) as usize;
            if (x + y) % 2 == 0 {
                data[idx] = 255; data[idx+1] = 255; data[idx+2] = 255; data[idx+3] = 255;
            }
        }
    }
    data
}

/// Low-alpha pixels: entire image at alpha=2 (barely visible).
fn create_low_alpha(width: u32, height: u32) -> Vec<u8> {
    let mut data = vec![0u8; (width * height * 4) as usize];
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 4) as usize;
            data[idx] = 255; data[idx+1] = 128; data[idx+2] = 64; data[idx+3] = 2;
        }
    }
    data
}

#[test]
fn find_optimal_unpremul_threshold() {
    let thresholds: &[(f32, &str)] = &[
        (0.0,               "0.0"),
        (f32::MIN_POSITIVE, "MIN_POS"),
        (1.0 / 65536.0,     "1/65536"),
        (1.0 / 16384.0,     "1/16384"),
        (1.0 / 4096.0,      "1/4096"),
        (1.0 / 2048.0,      "1/2048"),
        (1.0 / 1024.0,      "1/1024"),
        (1.0 / 512.0,       "1/512"),
        (0.5 / 255.0,       "0.5/255"),
        (1.0 / 256.0,       "1/256"),
        (1.5 / 255.0,       "1.5/255"),
        (1.0 / 128.0,       "1/128"),
        (1.0 / 64.0,        "1/64"),
    ];

    let filters: &[(&str, Filter)] = &[
        ("Lanczos",       Filter::Lanczos),
        ("LanczosSharp",  Filter::LanczosSharp),
        ("Lanczos2",      Filter::Lanczos2),
        ("Lanczos2Sharp", Filter::Lanczos2Sharp),
        ("Ginseng",       Filter::Ginseng),
        ("GinsengSharp",  Filter::GinsengSharp),
        ("Robidoux",      Filter::Robidoux),
        ("RobidouxSharp", Filter::RobidouxSharp),
        ("Mitchell",      Filter::Mitchell),
        ("CatmullRom",    Filter::CatmullRom),
        ("CubicBSpline",  Filter::CubicBSpline),
        ("Hermite",       Filter::Hermite),
        ("Triangle",      Filter::Triangle),
        ("Box",           Filter::Box),
    ];

    struct ImageDef { name: &'static str, data: Vec<u8>, w: u32, h: u32 }
    let images = vec![
        ImageDef { name: "sharp_200", data: create_sharp_edge(200, 200), w: 200, h: 200 },
        ImageDef { name: "sharp_800", data: create_sharp_edge(800, 600), w: 800, h: 600 },
        ImageDef { name: "gradient_200", data: create_alpha_gradient(200, 200), w: 200, h: 200 },
        ImageDef { name: "gradient_800", data: create_alpha_gradient(800, 600), w: 800, h: 600 },
        ImageDef { name: "dot_64", data: create_bright_dot(64, 64), w: 64, h: 64 },
        ImageDef { name: "checker_200", data: create_checkerboard(200, 200), w: 200, h: 200 },
        ImageDef { name: "low_alpha", data: create_low_alpha(200, 200), w: 200, h: 200 },
    ];

    let resize_ops: &[(f32, &str)] = &[
        (0.5,   "2x_down"),
        (0.25,  "4x_down"),
        (0.185, "5.4x_down"),
        (2.0,   "2x_up"),
        (3.0,   "3x_up"),
    ];

    // Accumulate per-threshold stats
    struct ThresholdStats {
        visible_wrong_total: usize,
        visible_max_delta: u8,
        invisible_diff_total: usize,
        worst_case: String,
    }
    let mut stats: Vec<ThresholdStats> = thresholds.iter().map(|_| ThresholdStats {
        visible_wrong_total: 0, visible_max_delta: 0, invisible_diff_total: 0,
        worst_case: String::new(),
    }).collect();

    // Print header for per-case details (only rows with visible_wrong > 0)
    println!();
    println!("{:<10} {:<14} {:<14} {:<10} {:>8} {:>10} {:>10}",
        "threshold", "filter", "image", "resize", "vis_wrong", "vis_maxΔ", "invis_diff");
    println!("{}", "-".repeat(80));

    for (filter_name, filter) in filters {
        for img in &images {
            for &(scale, resize_name) in resize_ops {
                let out_w = (img.w as f32 * scale).round() as u32;
                let out_h = (img.h as f32 * scale).round() as u32;
                if out_w == 0 || out_h == 0 { continue; }

                let premul = resize_to_premul_f32(&img.data, img.w, img.h, out_w, out_h, *filter);

                // Reference: threshold = 0 (always divide)
                let reference = unpremul_encode_u8(&premul, 0.0);

                for (idx, &(threshold, _thr_name)) in thresholds.iter().enumerate() {
                    if threshold == 0.0 { continue; } // skip reference vs itself

                    let candidate = unpremul_encode_u8(&premul, threshold);
                    let (vis_wrong, vis_max_delta, invis_diff) = compare_outputs(&reference, &candidate);

                    stats[idx].visible_wrong_total += vis_wrong;
                    stats[idx].invisible_diff_total += invis_diff;
                    if vis_max_delta > stats[idx].visible_max_delta {
                        stats[idx].visible_max_delta = vis_max_delta;
                        stats[idx].worst_case = format!("{}/{}/{}", filter_name, img.name, resize_name);
                    }

                    if vis_wrong > 0 {
                        println!("{:<10} {:<14} {:<14} {:<10} {:>8} {:>10} {:>10}",
                            _thr_name, filter_name, img.name, resize_name,
                            vis_wrong, vis_max_delta, invis_diff);
                    }
                }
            }
        }
    }

    println!();
    println!("=== SUMMARY ===");
    println!("{:<10} {:>12} {:>10} {:>12} {:<30}",
        "threshold", "vis_wrong", "vis_maxΔ", "invis_diff", "worst_case");
    println!("{}", "-".repeat(80));
    for (idx, &(_, thr_name)) in thresholds.iter().enumerate() {
        let s = &stats[idx];
        println!("{:<10} {:>12} {:>10} {:>12} {:<30}",
            thr_name, s.visible_wrong_total, s.visible_max_delta,
            s.invisible_diff_total, s.worst_case);
    }
}
