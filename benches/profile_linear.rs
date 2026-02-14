//! Phase-by-phase profile of the linear resize path.
//! Usage: cargo bench --bench profile_linear
//!
//! Isolates color conversion cost by comparing:
//! - Full linear resize (sRGB→linear + f32 filter + linear→sRGB)
//! - f32 resize without color conversion (just f32 filter)
//! - Color conversion alone (sRGB↔linear)

use std::time::Instant;

fn mean_ms(times: &[f64]) -> f64 {
    times.iter().sum::<f64>() / times.len() as f64
}

fn main() {
    let (w, h) = (1024u32, 1024u32);
    let out_w = 512u32;
    let out_h = 512u32;

    let mut rgba = vec![0u8; (w as usize) * (h as usize) * 4];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize * 4;
            rgba[i] = (x % 256) as u8;
            rgba[i + 1] = (y % 256) as u8;
            rgba[i + 2] = ((x + y) % 256) as u8;
            rgba[i + 3] = 200;
        }
    }

    let rounds = 30;
    let warmup = 5;

    // --- Full linear resize (sRGB→linear→filter→sRGB) ---
    let config_linear = zenresize::ResizeConfig::builder(w, h, out_w, out_h)
        .filter(zenresize::Filter::Lanczos)
        .format(zenresize::PixelFormat::Srgb8(zenresize::PixelLayout::Rgba))
        .linear()
        .build();

    for _ in 0..warmup {
        let _ = zenresize::resize(&config_linear, &rgba);
    }
    let mut times_linear = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let start = Instant::now();
        let result = zenresize::resize(&config_linear, &rgba);
        times_linear.push(start.elapsed().as_secs_f64() * 1000.0);
        std::hint::black_box(&result);
    }

    // --- sRGB integer fast path (for comparison) ---
    let config_srgb = zenresize::ResizeConfig::builder(w, h, out_w, out_h)
        .filter(zenresize::Filter::Lanczos)
        .format(zenresize::PixelFormat::Srgb8(zenresize::PixelLayout::Rgbx))
        .srgb()
        .build();

    for _ in 0..warmup {
        let _ = zenresize::resize(&config_srgb, &rgba);
    }
    let mut times_srgb = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let start = Instant::now();
        let result = zenresize::resize(&config_srgb, &rgba);
        times_srgb.push(start.elapsed().as_secs_f64() * 1000.0);
        std::hint::black_box(&result);
    }

    // --- f32 path WITHOUT linearization (sRGB space, same f32 codepath) ---
    // This uses the f32 filter path but skips sRGB↔linear conversion.
    // The difference from full linear = color conversion cost.
    // Use 3 channels — no i16 fast path for 3ch, so it forces f32.
    let config_f32_no_linear = zenresize::ResizeConfig::builder(w, h, out_w, out_h)
        .filter(zenresize::Filter::Lanczos)
        .format(zenresize::PixelFormat::Srgb8(zenresize::PixelLayout::Rgb))
        .srgb()
        .build();
    let rgba3: Vec<u8> = rgba
        .chunks_exact(4)
        .flat_map(|c| &c[..3])
        .copied()
        .collect();

    for _ in 0..warmup {
        let _ = zenresize::resize(&config_f32_no_linear, &rgba3);
    }
    let mut times_f32_nolin = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let start = Instant::now();
        let result = zenresize::resize(&config_f32_no_linear, &rgba3);
        times_f32_nolin.push(start.elapsed().as_secs_f64() * 1000.0);
        std::hint::black_box(&result);
    }

    // --- Color conversion alone (using linear-srgb crate directly) ---
    let in_row_len = w as usize * 4;
    let out_row_len = out_w as usize * 4;
    let in_row = &rgba[..in_row_len];
    let mut f32_buf = vec![0.0f32; in_row_len];
    let mut u8_buf = vec![0u8; out_row_len];
    let f32_in = vec![0.5f32; out_row_len]; // dummy linear values

    // Forward: sRGB u8 → linear f32 (1024 rows)
    for _ in 0..warmup {
        for _ in 0..h as usize {
            linear_srgb::default::srgb_u8_to_linear_slice(in_row, &mut f32_buf);
        }
    }
    let mut times_forward = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let start = Instant::now();
        for _ in 0..h as usize {
            linear_srgb::default::srgb_u8_to_linear_slice(in_row, &mut f32_buf);
        }
        times_forward.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    // Inverse: linear f32 → sRGB u8 (512 rows)
    for _ in 0..warmup {
        for _ in 0..out_h as usize {
            linear_srgb::default::linear_to_srgb_u8_slice(&f32_in, &mut u8_buf);
        }
    }
    let mut times_inverse = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let start = Instant::now();
        for _ in 0..out_h as usize {
            linear_srgb::default::linear_to_srgb_u8_slice(&f32_in, &mut u8_buf);
        }
        times_inverse.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    // --- f32 end-to-end (f32 input → f32 output, no conversion) ---
    let f32_rgba: Vec<f32> = rgba.iter().map(|&b| b as f32 / 255.0).collect();
    let config_f32 = zenresize::ResizeConfig::builder(w, h, out_w, out_h)
        .filter(zenresize::Filter::Lanczos)
        .format(zenresize::PixelFormat::LinearF32(
            zenresize::PixelLayout::Rgba,
        ))
        .build();

    for _ in 0..warmup {
        let _ = zenresize::resize_f32(&config_f32, &f32_rgba);
    }
    let mut times_f32_e2e = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let start = Instant::now();
        let result = zenresize::resize_f32(&config_f32, &f32_rgba);
        times_f32_e2e.push(start.elapsed().as_secs_f64() * 1000.0);
        std::hint::black_box(&result);
    }

    // --- f32 end-to-end, no alpha ---
    let config_f32_noalpha = zenresize::ResizeConfig::builder(w, h, out_w, out_h)
        .filter(zenresize::Filter::Lanczos)
        .format(zenresize::PixelFormat::LinearF32(
            zenresize::PixelLayout::Rgbx,
        ))
        .build();

    for _ in 0..warmup {
        let _ = zenresize::resize_f32(&config_f32_noalpha, &f32_rgba);
    }
    let mut times_f32_noalpha = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let start = Instant::now();
        let result = zenresize::resize_f32(&config_f32_noalpha, &f32_rgba);
        times_f32_noalpha.push(start.elapsed().as_secs_f64() * 1000.0);
        std::hint::black_box(&result);
    }

    // --- Results ---
    let linear = mean_ms(&times_linear);
    let srgb = mean_ms(&times_srgb);
    let f32_nolin = mean_ms(&times_f32_nolin);
    let fwd = mean_ms(&times_forward);
    let inv = mean_ms(&times_inverse);
    let f32_e2e = mean_ms(&times_f32_e2e);
    let f32_noalpha = mean_ms(&times_f32_noalpha);

    println!("Linear Path Phase Profile (1024→512, Lanczos, RGBA)");
    println!("=====================================================");
    println!("Full linear resize (u8→u8): {:>7.2} ms", linear);
    println!("sRGB integer fast path:     {:>7.2} ms", srgb);
    println!("f32 u8 path, no lin (3ch):  {:>7.2} ms", f32_nolin);
    println!("f32→f32 (with alpha):       {:>7.2} ms", f32_e2e);
    println!("f32→f32 (no alpha):         {:>7.2} ms", f32_noalpha);
    println!();
    println!("Color conversion:");
    println!("  srgb→linear (1024 rows):  {:>7.2} ms", fwd);
    println!("  linear→srgb (512 rows):   {:>7.2} ms", inv);
    println!("  Total color conv:         {:>7.2} ms", fwd + inv);
    println!();
    println!(
        "Estimated f32 filter time:  {:>7.2} ms (linear - color conv)",
        linear - fwd - inv
    );
    println!(
        "f32/i16 filter ratio:       {:>7.1}x",
        (linear - fwd - inv) / srgb
    );
    println!("f32 e2e / i16 ratio:        {:>7.1}x", f32_e2e / srgb);
}
