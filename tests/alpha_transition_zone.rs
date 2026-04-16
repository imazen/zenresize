//! Analyze the alpha transition zone created by various filters at a hard alpha edge.
//!
//! For a hard opaque→transparent edge, measures:
//! - How many output pixels land in each alpha_u8 bucket (0..=255)
//! - The unpremultiplied color of low-alpha pixels (are they meaningful or garbage?)
//! - How many pixels have alpha_u8 in [1, N] — barely visible, potential trim candidates
//!
//! Run: cargo test --test alpha_transition_zone -- --nocapture

use zenresize::{AlphaMode, Filter, PixelDescriptor, ResizeConfig, Resizer};

/// Create a vertical alpha edge: left half bright white opaque, right half transparent.
/// Single row repeated for all rows (1D edge analysis).
fn create_vertical_edge(width: u32, height: u32) -> Vec<u8> {
    let mut data = vec![0u8; (width * height * 4) as usize];
    let edge = width / 2;
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 4) as usize;
            if x < edge {
                data[idx] = 255; data[idx+1] = 255; data[idx+2] = 255; data[idx+3] = 255;
            }
        }
    }
    data
}

/// Resize to premultiplied linear f32
fn resize_premul(
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

/// Analyze one row of the output across the edge.
/// Returns per-pixel info: (x, alpha_f32, alpha_u8, unpremul_brightness, is_negative_alpha)
fn analyze_edge_row(premul_row: &[f32], out_w: u32) -> Vec<(u32, f32, u8, f32, bool)> {
    let mut results = Vec::new();
    let edge_approx = out_w / 2; // approximate edge location in output

    // Only analyze pixels near the edge (±20 pixels)
    let start = edge_approx.saturating_sub(20);
    let end = (edge_approx + 20).min(out_w);

    for x in start..end {
        let idx = (x * 4) as usize;
        let r = premul_row[idx];
        let g = premul_row[idx + 1];
        let b = premul_row[idx + 2];
        let a = premul_row[idx + 3];

        let alpha_u8 = (a.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        let is_negative = a < 0.0 || r < 0.0 || g < 0.0 || b < 0.0;

        // Unpremultiplied brightness (the color the viewer would see if this pixel is visible)
        let brightness = if a > 1.0 / 1024.0 {
            let inv_a = 1.0 / a;
            let ur = (r * inv_a).clamp(0.0, 1.0);
            let ug = (g * inv_a).clamp(0.0, 1.0);
            let ub = (b * inv_a).clamp(0.0, 1.0);
            (ur + ug + ub) / 3.0
        } else if a > 0.0 {
            // Use clamped divide for tiny alpha
            let inv_a = 1.0 / a;
            let ur = (r * inv_a).clamp(0.0, 1.0);
            let ug = (g * inv_a).clamp(0.0, 1.0);
            let ub = (b * inv_a).clamp(0.0, 1.0);
            (ur + ug + ub) / 3.0
        } else {
            0.0
        };

        // Only include pixels that aren't fully opaque or fully transparent
        if alpha_u8 < 255 || is_negative {
            results.push((x, a, alpha_u8, brightness, is_negative));
        }
    }
    results
}

#[test]
fn analyze_transition_zones() {
    let filters: &[(&str, Filter)] = &[
        ("Lanczos3",      Filter::Lanczos),
        ("LanczosSharp",  Filter::LanczosSharp),
        ("Lanczos2",      Filter::Lanczos2),
        ("Ginseng",       Filter::Ginseng),
        ("Robidoux",      Filter::Robidoux),
        ("RobidouxSharp", Filter::RobidouxSharp),
        ("Mitchell",      Filter::Mitchell),
        ("CatmullRom",    Filter::CatmullRom),
        ("CubicBSpline",  Filter::CubicBSpline),
        ("Hermite",       Filter::Hermite),
        ("Triangle",      Filter::Triangle),
        ("Box",           Filter::Box),
    ];

    let scale_factors: &[(f32, &str)] = &[
        (0.5,    "2x_down"),
        (0.333,  "3x_down"),
        (0.25,   "4x_down"),
        (0.125,  "8x_down"),
        (2.0,    "2x_up"),
    ];

    let in_w = 400u32;
    let in_h = 20u32; // Only need a few rows since the edge is vertical
    let input = create_vertical_edge(in_w, in_h);

    println!();

    for &(scale, scale_name) in scale_factors {
        let out_w = (in_w as f32 * scale).round() as u32;
        let out_h = (in_h as f32 * scale).round() as u32;
        if out_w == 0 || out_h == 0 { continue; }

        println!("============================================================");
        println!("Scale: {} ({}x{} → {}x{})", scale_name, in_w, in_h, out_w, out_h);
        println!("============================================================");

        for &(filter_name, filter) in filters {
            let premul = resize_premul(&input, in_w, in_h, out_w, out_h, filter);

            // Analyze middle row
            let mid_y = out_h / 2;
            let row_start = (mid_y * out_w * 4) as usize;
            let row_end = row_start + (out_w * 4) as usize;
            let row = &premul[row_start..row_end];

            let pixels = analyze_edge_row(row, out_w);

            // Count by alpha bucket
            let mut alpha_histogram = [0u32; 256];
            let mut neg_count = 0u32;
            let mut total_partially_visible = 0u32; // alpha_u8 in [1, 254]
            let mut barely_visible = 0u32; // alpha_u8 in [1, 5]

            for &(_, a, alpha_u8, _, is_neg) in &pixels {
                alpha_histogram[alpha_u8 as usize] += 1;
                if is_neg { neg_count += 1; }
                if alpha_u8 >= 1 && alpha_u8 <= 254 { total_partially_visible += 1; }
                if alpha_u8 >= 1 && alpha_u8 <= 5 { barely_visible += 1; }
            }

            // Find transition zone width (first and last pixel with 0 < alpha_u8 < 255)
            let first_partial = pixels.iter().find(|p| p.2 > 0 && p.2 < 255);
            let last_partial = pixels.iter().rev().find(|p| p.2 > 0 && p.2 < 255);
            let zone_width = match (first_partial, last_partial) {
                (Some(f), Some(l)) => l.0 - f.0 + 1,
                _ => 0,
            };

            // Print summary line
            println!("{:<14} zone={:>2}px  partial={:>2}  barely(1-5)={:>2}  a_u8=0={:>2}  neg={}",
                filter_name, zone_width, total_partially_visible, barely_visible,
                alpha_histogram[0], neg_count);

            // Print per-pixel detail for this filter
            if !pixels.is_empty() {
                print!("  pixels: ");
                for &(x, a, alpha_u8, brightness, is_neg) in &pixels {
                    if alpha_u8 == 0 && !is_neg { continue; } // skip boring transparent pixels
                    let neg_marker = if is_neg { "!" } else { "" };
                    print!("x{}:a={}{:.4}(u8={})br={:.2}  ",
                        x, neg_marker, a, alpha_u8, brightness);
                }
                println!();
            }
        }
        println!();
    }

    // Now print the aggregate: for each filter across all scales,
    // how many alpha_u8=1 pixels exist and what's their typical brightness?
    println!("============================================================");
    println!("AGGREGATE: alpha_u8=1 pixel colors (the barely-visible fringe)");
    println!("============================================================");
    println!("{:<14} {:>6} {:>10} {:>10} {:>10}",
        "filter", "count", "min_br", "avg_br", "max_br");
    println!("{}", "-".repeat(56));

    for &(filter_name, filter) in filters {
        let mut a1_brightnesses = Vec::new();

        for &(scale, _) in scale_factors {
            let out_w = (in_w as f32 * scale).round() as u32;
            let out_h = (in_h as f32 * scale).round() as u32;
            if out_w == 0 || out_h == 0 { continue; }

            let premul = resize_premul(&input, in_w, in_h, out_w, out_h, filter);

            // Check ALL rows, not just middle
            for y in 0..out_h {
                let row_start = (y * out_w * 4) as usize;
                for x in 0..out_w {
                    let idx = row_start + (x * 4) as usize;
                    let a = premul[idx + 3];
                    let alpha_u8 = (a.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                    if alpha_u8 == 1 && a > 0.0 {
                        let inv_a = 1.0 / a;
                        let r = (premul[idx] * inv_a).clamp(0.0, 1.0);
                        let g = (premul[idx+1] * inv_a).clamp(0.0, 1.0);
                        let b = (premul[idx+2] * inv_a).clamp(0.0, 1.0);
                        a1_brightnesses.push((r + g + b) / 3.0);
                    }
                }
            }
        }

        if a1_brightnesses.is_empty() {
            println!("{:<14} {:>6} {:>10} {:>10} {:>10}", filter_name, 0, "-", "-", "-");
        } else {
            let min = a1_brightnesses.iter().cloned().fold(f32::MAX, f32::min);
            let max = a1_brightnesses.iter().cloned().fold(f32::MIN, f32::max);
            let avg: f32 = a1_brightnesses.iter().sum::<f32>() / a1_brightnesses.len() as f32;
            println!("{:<14} {:>6} {:>10.4} {:>10.4} {:>10.4}",
                filter_name, a1_brightnesses.len(), min, avg, max);
        }
    }
}
