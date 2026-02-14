//! Precision analysis: f32 resize vs f64 reference and cross-library comparison.
//! Usage: cargo bench --bench precision

fn main() {
    let (w, h) = (1024u32, 1024u32);
    let out_w = 512u32;
    let out_h = 512u32;

    // Synthetic test image with gradients
    let mut rgba = vec![0u8; (w as usize) * (h as usize) * 4];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize * 4;
            rgba[i] = (x % 256) as u8;
            rgba[i + 1] = (y % 256) as u8;
            rgba[i + 2] = ((x + y) % 256) as u8;
            rgba[i + 3] = 255;
        }
    }

    // f32 input for f32 path
    let f32_rgba: Vec<f32> = rgba.iter().map(|&b| b as f32 / 255.0).collect();

    println!("Precision Analysis (1024→512, Lanczos, RGBA)");
    println!("=============================================\n");

    // === zenresize f32 output ===
    let config_f32 = zenresize::ResizeConfig::builder(w, h, out_w, out_h)
        .filter(zenresize::Filter::Lanczos)
        .format(zenresize::PixelFormat::LinearF32 {
            channels: 4,
            has_alpha: false,
        })
        .build();
    let zen_f32 = zenresize::resize_f32(&config_f32, &f32_rgba);

    // === zenresize linear u8→u8 ===
    let config_linear = zenresize::ResizeConfig::builder(w, h, out_w, out_h)
        .filter(zenresize::Filter::Lanczos)
        .format(zenresize::PixelFormat::Srgb8 {
            channels: 4,
            has_alpha: false,
        })
        .linear()
        .build();
    let zen_linear_u8 = zenresize::resize(&config_linear, &rgba);

    // === zenresize sRGB u8→u8 ===
    let config_srgb = zenresize::ResizeConfig::builder(w, h, out_w, out_h)
        .filter(zenresize::Filter::Lanczos)
        .format(zenresize::PixelFormat::Srgb8 {
            channels: 4,
            has_alpha: false,
        })
        .srgb()
        .build();
    let zen_srgb_u8 = zenresize::resize(&config_srgb, &rgba);

    // === Competitors ===
    // pic-scale-safe f32
    let pss_f32 = {
        use pic_scale_safe::*;
        let src_size = ImageSize::new(w as usize, h as usize);
        let dst_size = ImageSize::new(out_w as usize, out_h as usize);
        resize_rgba_f32(&f32_rgba, src_size, dst_size, ResamplingFunction::Lanczos3).unwrap()
    };

    // pic-scale-safe sRGB u8
    let pss_srgb_u8 = {
        use pic_scale_safe::*;
        let src_size = ImageSize::new(w as usize, h as usize);
        let dst_size = ImageSize::new(out_w as usize, out_h as usize);
        resize_rgba8(&rgba, src_size, dst_size, ResamplingFunction::Lanczos3).unwrap()
    };

    // fir sRGB u8
    let fir_srgb_u8 = {
        use fast_image_resize as fir;
        use fir::images::{Image, ImageRef};
        use fir::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
        let src = ImageRef::new(w, h, &rgba, PixelType::U8x4).unwrap();
        let mut dst = Image::new(out_w, out_h, PixelType::U8x4);
        let mut resizer = Resizer::new();
        let opts = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3));
        resizer.resize(&src, &mut dst, &opts).unwrap();
        dst.into_vec()
    };

    // fir f32
    let fir_f32_result = {
        use fast_image_resize as fir;
        use fir::images::{Image, ImageRef};
        use fir::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
        let f32_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(f32_rgba.as_ptr() as *const u8, f32_rgba.len() * 4)
        };
        let src = ImageRef::new(w, h, f32_bytes, PixelType::F32x4).unwrap();
        let mut dst = Image::new(out_w, out_h, PixelType::F32x4);
        let mut resizer = Resizer::new();
        let opts = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3));
        resizer.resize(&src, &mut dst, &opts).unwrap();
        let result_bytes = dst.into_vec();
        let result_f32: Vec<f32> = unsafe {
            let ptr = result_bytes.as_ptr() as *const f32;
            let len = result_bytes.len() / 4;
            std::slice::from_raw_parts(ptr, len).to_vec()
        };
        result_f32
    };

    // resize crate sRGB u8
    let resize_srgb_u8 = {
        use rgb::FromSlice;
        let mut dst = vec![0u8; out_w as usize * out_h as usize * 4];
        let mut resizer = resize::new(
            w as usize,
            h as usize,
            out_w as usize,
            out_h as usize,
            resize::Pixel::RGBA8P,
            resize::Type::Lanczos3,
        )
        .unwrap();
        resizer.resize(rgba.as_rgba(), dst.as_rgba_mut()).unwrap();
        dst
    };

    // === f32 cross-library comparison ===
    println!("f32 output: zenresize vs competitors");
    println!("------------------------------------");
    print_f32_stats("zenresize_f32 vs pss_f32", &zen_f32, &pss_f32);
    print_f32_stats("zenresize_f32 vs fir_f32", &zen_f32, &fir_f32_result);
    print_f32_stats("pss_f32 vs fir_f32", &pss_f32, &fir_f32_result);
    println!();

    // === u8 cross-library comparison ===
    println!("u8 sRGB output: zenresize vs competitors");
    println!("-----------------------------------------");
    print_u8_stats("zen_srgb vs pss_srgb", &zen_srgb_u8, &pss_srgb_u8);
    print_u8_stats("zen_srgb vs fir_srgb", &zen_srgb_u8, &fir_srgb_u8);
    print_u8_stats("zen_srgb vs resize_crate", &zen_srgb_u8, &resize_srgb_u8);
    print_u8_stats("pss_srgb vs fir_srgb", &pss_srgb_u8, &fir_srgb_u8);
    println!();

    // === zenresize linear vs sRGB ===
    println!("zenresize: linear vs sRGB path (u8 output)");
    println!("-------------------------------------------");
    print_u8_stats("zen_linear vs zen_srgb", &zen_linear_u8, &zen_srgb_u8);
    println!();

    // === f32 range analysis ===
    println!("f32 output range analysis");
    println!("-------------------------");
    print_f32_range("zenresize_f32", &zen_f32);
    print_f32_range("pss_f32", &pss_f32);
    print_f32_range("fir_f32", &fir_f32_result);
}

fn print_f32_stats(label: &str, a: &[f32], b: &[f32]) {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    let mut max_diff: f32 = 0.0;
    let mut sum_diff: f64 = 0.0;
    let mut sum_sq_diff: f64 = 0.0;
    let mut max_diff_idx = 0;
    let mut count_gt_1e4 = 0usize;
    let mut count_gt_1e3 = 0usize;
    let mut count_gt_1e2 = 0usize;

    for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (va - vb).abs();
        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }
        sum_diff += diff as f64;
        sum_sq_diff += (diff as f64).powi(2);
        if diff > 1e-4 {
            count_gt_1e4 += 1;
        }
        if diff > 1e-3 {
            count_gt_1e3 += 1;
        }
        if diff > 1e-2 {
            count_gt_1e2 += 1;
        }
    }

    let mean_diff = sum_diff / a.len() as f64;
    let rms_diff = (sum_sq_diff / a.len() as f64).sqrt();

    // Convert to "u8 levels" for intuition (multiply by 255)
    let max_u8 = max_diff * 255.0;
    let mean_u8 = mean_diff * 255.0;

    println!("  {label}:");
    println!(
        "    max |diff|: {:.2e} ({:.2} u8 levels) at pixel {}",
        max_diff,
        max_u8,
        max_diff_idx / 4
    );
    println!(
        "    mean |diff|: {:.2e} ({:.4} u8 levels), RMS: {:.2e}",
        mean_diff, mean_u8, rms_diff
    );
    println!(
        "    >1e-4: {} ({:.2}%), >1e-3: {} ({:.2}%), >1e-2: {} ({:.2}%)",
        count_gt_1e4,
        count_gt_1e4 as f64 / a.len() as f64 * 100.0,
        count_gt_1e3,
        count_gt_1e3 as f64 / a.len() as f64 * 100.0,
        count_gt_1e2,
        count_gt_1e2 as f64 / a.len() as f64 * 100.0,
    );
}

fn print_u8_stats(label: &str, a: &[u8], b: &[u8]) {
    assert_eq!(a.len(), b.len());
    let mut max_diff: u8 = 0;
    let mut sum_diff: u64 = 0;
    let mut histogram = [0u32; 256]; // diff histogram

    for (&va, &vb) in a.iter().zip(b.iter()) {
        let diff = (va as i16 - vb as i16).unsigned_abs() as u8;
        max_diff = max_diff.max(diff);
        sum_diff += diff as u64;
        histogram[diff as usize] += 1;
    }

    let mean_diff = sum_diff as f64 / a.len() as f64;
    let exact_match = histogram[0] as f64 / a.len() as f64 * 100.0;

    println!("  {label}:");
    println!("    max |diff|: {max_diff} u8 levels");
    println!("    mean |diff|: {mean_diff:.4} u8 levels");
    println!("    exact match: {exact_match:.2}%");
    // Print histogram of non-zero diffs
    for d in 1..=max_diff {
        if histogram[d as usize] > 0 {
            println!(
                "    diff={d}: {} ({:.2}%)",
                histogram[d as usize],
                histogram[d as usize] as f64 / a.len() as f64 * 100.0
            );
        }
    }
}

fn print_f32_range(label: &str, data: &[f32]) {
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    let mut nan_count = 0;
    let mut inf_count = 0;
    let mut neg_count = 0;
    let mut gt1_count = 0;

    for &v in data {
        if v.is_nan() {
            nan_count += 1;
            continue;
        }
        if v.is_infinite() {
            inf_count += 1;
            continue;
        }
        min = min.min(v);
        max = max.max(v);
        if v < 0.0 {
            neg_count += 1;
        }
        if v > 1.0 {
            gt1_count += 1;
        }
    }

    println!(
        "  {label}: [{min:.6}, {max:.6}], NaN: {nan_count}, Inf: {inf_count}, <0: {neg_count}, >1: {gt1_count}"
    );
}
