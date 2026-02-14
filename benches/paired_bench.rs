//! Paired (interleaved) benchmark for fair cross-library comparison.
//!
//! Unlike criterion which runs each benchmark to completion before the next,
//! this interleaves measurements: A, B, A, B, ... so both experience identical
//! system conditions (thermal state, OS scheduling, cache pressure). The paired
//! difference (A_i - B_i) cancels systematic drift, giving much tighter confidence
//! intervals for the *relative* performance.
//!
//! Usage:
//!   cargo bench --bench paired_bench                              # pic-scale scalar
//!   cargo bench --bench paired_bench --features bench-simd-competitors  # pic-scale with SSE+AVX
//!
//! For zenresize self-regression detection, use tango_bench instead.

use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

fn mean(data: &[f64]) -> f64 {
    data.iter().sum::<f64>() / data.len() as f64
}

fn stddev(data: &[f64]) -> f64 {
    let m = mean(data);
    let var = data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
    var.sqrt()
}

/// 95% confidence interval half-width using t-distribution approximation.
fn ci95(data: &[f64]) -> f64 {
    // For n>=30, t ≈ 1.96. For smaller n, use a conservative 2.0.
    let t = if data.len() >= 30 { 1.96 } else { 2.0 };
    t * stddev(data) / (data.len() as f64).sqrt()
}

/// Trim outliers beyond 2 stddev.
fn trim_outliers(data: &mut Vec<f64>) {
    let m = mean(data);
    let s = stddev(data);
    let lo = m - 2.0 * s;
    let hi = m + 2.0 * s;
    data.retain(|x| *x >= lo && *x <= hi);
}

// ---------------------------------------------------------------------------
// Image loading
// ---------------------------------------------------------------------------

struct TestImage {
    name: &'static str,
    width: u32,
    height: u32,
    rgba: Vec<u8>,
    rgba_f32: Vec<f32>,
}

fn test_image() -> TestImage {
    let (w, h) = (1024u32, 1024u32);
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
    let rgba_f32: Vec<f32> = rgba.iter().map(|&b| b as f32 / 255.0).collect();
    TestImage {
        name: "synth_1024",
        width: w,
        height: h,
        rgba,
        rgba_f32,
    }
}

// ---------------------------------------------------------------------------
// Library wrappers — each returns the output to prevent dead-code elimination
// ---------------------------------------------------------------------------

fn run_zenresize_srgb(img: &TestImage, out_w: u32, out_h: u32) -> Vec<u8> {
    let config = zenresize::ResizeConfig::builder(img.width, img.height, out_w, out_h)
        .filter(zenresize::Filter::Lanczos)
        .format(zenresize::PixelFormat::Srgb8 {
            channels: 4,
            has_alpha: false,
        })
        .srgb()
        .build();
    zenresize::resize(&config, &img.rgba)
}

fn run_zenresize_linear(img: &TestImage, out_w: u32, out_h: u32) -> Vec<u8> {
    let config = zenresize::ResizeConfig::builder(img.width, img.height, out_w, out_h)
        .filter(zenresize::Filter::Lanczos)
        .format(zenresize::PixelFormat::Srgb8 {
            channels: 4,
            has_alpha: true,
        })
        .linear()
        .build();
    zenresize::resize(&config, &img.rgba)
}

fn run_picscale_srgb(img: &TestImage, out_w: u32, out_h: u32) -> Vec<u8> {
    use pic_scale::*;
    let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
    scaler.set_threading_policy(ThreadingPolicy::Single);
    let store = ImageStore::<u8, 4>::from_slice(&img.rgba, img.width as usize, img.height as usize)
        .unwrap();
    let mut dst = ImageStoreMut::<u8, 4>::alloc(out_w as usize, out_h as usize);
    let _ = scaler.resize_rgba(&store, &mut dst, true);
    dst.as_bytes().to_vec()
}

fn run_picscale_linear(img: &TestImage, out_w: u32, out_h: u32) -> Vec<u8> {
    use pic_scale::*;
    let mut scaler = LinearScaler::new(ResamplingFunction::Lanczos3);
    scaler.set_threading_policy(ThreadingPolicy::Single);
    let store = ImageStore::<u8, 4>::from_slice(&img.rgba, img.width as usize, img.height as usize)
        .unwrap();
    let mut dst = ImageStoreMut::<u8, 4>::alloc(out_w as usize, out_h as usize);
    let _ = scaler.resize_rgba(&store, &mut dst, true);
    dst.as_bytes().to_vec()
}

fn run_fir_srgb(img: &TestImage, out_w: u32, out_h: u32) -> Vec<u8> {
    use fast_image_resize as fir;
    use fir::images::{Image, ImageRef};
    use fir::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
    let src = ImageRef::new(img.width, img.height, &img.rgba, PixelType::U8x4).unwrap();
    let mut dst = Image::new(out_w, out_h, PixelType::U8x4);
    let mut resizer = Resizer::new();
    let opts = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3));
    resizer.resize(&src, &mut dst, &opts).unwrap();
    dst.into_vec()
}

fn run_fir_linear(img: &TestImage, out_w: u32, out_h: u32) -> Vec<u8> {
    use fast_image_resize as fir;
    use fir::images::{Image, ImageRef};
    use fir::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
    let mapper = fir::create_srgb_mapper();
    let src = ImageRef::new(img.width, img.height, &img.rgba, PixelType::U8x4).unwrap();
    let mut linear_src = Image::new(img.width, img.height, PixelType::U16x4);
    mapper.forward_map(&src, &mut linear_src).unwrap();
    let mut linear_dst = Image::new(out_w, out_h, PixelType::U16x4);
    let mut resizer = Resizer::new();
    let opts = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3));
    resizer.resize(&linear_src, &mut linear_dst, &opts).unwrap();
    let mut dst = Image::new(out_w, out_h, PixelType::U8x4);
    mapper.backward_map(&linear_dst, &mut dst).unwrap();
    dst.into_vec()
}

fn run_picscale_safe_srgb(img: &TestImage, out_w: u32, out_h: u32) -> Vec<u8> {
    use pic_scale_safe::*;
    let src_size = ImageSize::new(img.width as usize, img.height as usize);
    let dst_size = ImageSize::new(out_w as usize, out_h as usize);
    // sRGB-aware: linearize, resize, re-gamma
    let mut src = img.rgba.clone();
    image_to_linear::<4>(&mut src, TransferFunction::Srgb);
    let mut result = resize_rgba8(&src, src_size, dst_size, ResamplingFunction::Lanczos3).unwrap();
    linear_to_gamma_image::<4>(&mut result, TransferFunction::Srgb);
    result
}

fn run_picscale_safe_linear(img: &TestImage, out_w: u32, out_h: u32) -> Vec<u8> {
    use pic_scale_safe::*;
    let src_size = ImageSize::new(img.width as usize, img.height as usize);
    let dst_size = ImageSize::new(out_w as usize, out_h as usize);
    resize_rgba8(&img.rgba, src_size, dst_size, ResamplingFunction::Lanczos3).unwrap()
}

fn run_picscale_safe_f32(img: &TestImage, out_w: u32, out_h: u32) -> Vec<u8> {
    use pic_scale_safe::*;
    let src_size = ImageSize::new(img.width as usize, img.height as usize);
    let dst_size = ImageSize::new(out_w as usize, out_h as usize);
    let result_f32 =
        resize_rgba_f32(&img.rgba_f32, src_size, dst_size, ResamplingFunction::Lanczos3).unwrap();
    // Return f32 bytes as u8 slice (black_box prevents elision)
    let bytes: Vec<u8> = result_f32
        .iter()
        .flat_map(|v| v.to_ne_bytes())
        .collect();
    bytes
}

fn run_zenresize_f32(img: &TestImage, out_w: u32, out_h: u32) -> Vec<u8> {
    let config = zenresize::ResizeConfig::builder(img.width, img.height, out_w, out_h)
        .filter(zenresize::Filter::Lanczos)
        .format(zenresize::PixelFormat::LinearF32 {
            channels: 4,
            has_alpha: false,
        })
        .build();
    let result_f32 = zenresize::resize_f32(&config, &img.rgba_f32);
    let bytes: Vec<u8> = result_f32
        .iter()
        .flat_map(|v| v.to_ne_bytes())
        .collect();
    bytes
}

fn run_fir_f32(img: &TestImage, out_w: u32, out_h: u32) -> Vec<u8> {
    use fast_image_resize as fir;
    use fir::images::{Image, ImageRef};
    use fir::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
    // Safety: f32 and u8 are both Pod types, this is just a reinterpret cast
    let f32_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            img.rgba_f32.as_ptr() as *const u8,
            img.rgba_f32.len() * 4,
        )
    };
    let src = ImageRef::new(img.width, img.height, f32_bytes, PixelType::F32x4).unwrap();
    let mut dst = Image::new(out_w, out_h, PixelType::F32x4);
    let mut resizer = Resizer::new();
    let opts = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3));
    resizer.resize(&src, &mut dst, &opts).unwrap();
    dst.into_vec()
}

fn run_resize_crate(img: &TestImage, out_w: u32, out_h: u32) -> Vec<u8> {
    use rgb::FromSlice;
    let mut dst = vec![0u8; out_w as usize * out_h as usize * 4];
    let mut resizer = resize::new(
        img.width as usize,
        img.height as usize,
        out_w as usize,
        out_h as usize,
        resize::Pixel::RGBA8P,
        resize::Type::Lanczos3,
    )
    .unwrap();
    resizer
        .resize(img.rgba.as_rgba(), dst.as_rgba_mut())
        .unwrap();
    dst
}

// ---------------------------------------------------------------------------
// Paired measurement core
// ---------------------------------------------------------------------------

type ResizeFn = fn(&TestImage, u32, u32) -> Vec<u8>;

struct Contender {
    name: &'static str,
    func: ResizeFn,
}

/// Measure a single call, returning elapsed duration.
fn measure_one(f: ResizeFn, img: &TestImage, out_w: u32, out_h: u32) -> Duration {
    let start = Instant::now();
    let result = f(img, out_w, out_h);
    let elapsed = start.elapsed();
    // Prevent dead-code elimination
    std::hint::black_box(&result);
    elapsed
}

/// Run paired interleaved benchmark between two contenders.
/// Returns (mean_a_ms, mean_b_ms, diff_mean_ms, diff_ci95_ms, ratio_a_over_b).
fn paired_bench(
    a: &Contender,
    b: &Contender,
    img: &TestImage,
    out_w: u32,
    out_h: u32,
    rounds: usize,
) -> (f64, f64, f64, f64, f64) {
    let warmup = 3;

    // Warmup
    for _ in 0..warmup {
        let _ = (a.func)(img, out_w, out_h);
        let _ = (b.func)(img, out_w, out_h);
    }

    let mut times_a = Vec::with_capacity(rounds);
    let mut times_b = Vec::with_capacity(rounds);
    let mut diffs = Vec::with_capacity(rounds);

    for round in 0..rounds {
        let (ta, tb) = if round % 2 == 0 {
            // A first, then B
            let ta = measure_one(a.func, img, out_w, out_h);
            let tb = measure_one(b.func, img, out_w, out_h);
            (ta, tb)
        } else {
            // B first, then A
            let tb = measure_one(b.func, img, out_w, out_h);
            let ta = measure_one(a.func, img, out_w, out_h);
            (ta, tb)
        };

        let ta_ms = ta.as_secs_f64() * 1000.0;
        let tb_ms = tb.as_secs_f64() * 1000.0;
        times_a.push(ta_ms);
        times_b.push(tb_ms);
        diffs.push(ta_ms - tb_ms);
    }

    // Trim outliers from diffs
    trim_outliers(&mut diffs);

    let mean_a = mean(&times_a);
    let mean_b = mean(&times_b);
    let diff_mean = mean(&diffs);
    let diff_ci = ci95(&diffs);
    let ratio = mean_a / mean_b;

    (mean_a, mean_b, diff_mean, diff_ci, ratio)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let img = test_image();
    let out_w = img.width / 2;
    let out_h = img.height / 2;
    let megapixels = (img.width as f64 * img.height as f64) / 1e6;
    let rounds = 50;

    println!("Paired Interleaved Resize Benchmark");
    println!("====================================");
    println!(
        "Image: {} ({}x{}, {:.2} MP)",
        img.name, img.width, img.height, megapixels
    );
    println!("Output: {}x{} (50% downscale)", out_w, out_h);
    println!("Rounds: {} (interleaved A/B/B/A pattern)", rounds);
    println!();

    let ps_label = if cfg!(feature = "bench-simd-competitors") {
        "simd"
    } else {
        "scalar"
    };

    let contenders: Vec<Contender> = vec![
        Contender {
            name: "zenresize_srgb",
            func: run_zenresize_srgb,
        },
        Contender {
            name: "zenresize_linear",
            func: run_zenresize_linear,
        },
        Contender {
            name: Box::leak(format!("pic_scale_{ps_label}_srgb").into_boxed_str()),
            func: run_picscale_srgb,
        },
        Contender {
            name: Box::leak(format!("pic_scale_{ps_label}_lin").into_boxed_str()),
            func: run_picscale_linear,
        },
        Contender {
            name: "pic_scale_safe_srgb",
            func: run_picscale_safe_srgb,
        },
        Contender {
            name: "pic_scale_safe_lin",
            func: run_picscale_safe_linear,
        },
        Contender {
            name: "fir_srgb",
            func: run_fir_srgb,
        },
        Contender {
            name: "fir_linear",
            func: run_fir_linear,
        },
        Contender {
            name: "resize_crate_srgb",
            func: run_resize_crate,
        },
        Contender {
            name: "zenresize_f32",
            func: run_zenresize_f32,
        },
        Contender {
            name: "pic_scale_safe_f32",
            func: run_picscale_safe_f32,
        },
        Contender {
            name: "fir_f32",
            func: run_fir_f32,
        },
    ];

    // --- Absolute times ---
    println!("Absolute times (mean of {} rounds):", rounds);
    println!("{:<24} {:>10} {:>10}", "Library", "Time (ms)", "MP/s");
    println!("{:-<24} {:-^10} {:-^10}", "", "", "");

    let mut abs_times: Vec<(&str, f64)> = Vec::new();
    for c in &contenders {
        // Warmup
        for _ in 0..3 {
            let _ = (c.func)(&img, out_w, out_h);
        }
        let mut times = Vec::with_capacity(rounds);
        for _ in 0..rounds {
            let t = measure_one(c.func, &img, out_w, out_h);
            times.push(t.as_secs_f64() * 1000.0);
        }
        trim_outliers(&mut times);
        let m = mean(&times);
        let ci = ci95(&times);
        let mps = megapixels / (m / 1000.0);
        println!("{:<24} {:>7.2} ±{:.2} {:>10.0}", c.name, m, ci, mps);
        abs_times.push((c.name, m));
    }

    // --- Pairwise comparisons ---
    // In paired_bench(a, b), diff = a - b, ratio = a / b.
    // We always put the contender as 'a' and baseline as 'b',
    // so positive diff = contender is slower, ratio > 1 = contender is slower.

    println!();
    println!("Paired comparisons (each row vs zenresize_srgb):");
    println!(
        "{:<24} {:>12} {:>12} {:>14}",
        "Library", "Diff (ms)", "95% CI", "vs zenresize"
    );
    println!("{:-<24} {:-^12} {:-^12} {:-^14}", "", "", "", "");

    let baseline = &contenders[0]; // zenresize_srgb
    for c in &contenders[1..] {
        // contender = a, baseline = b → diff = contender - baseline
        let (_mean_a, _mean_b, diff_mean, diff_ci, ratio) =
            paired_bench(c, baseline, &img, out_w, out_h, rounds);
        let label = if ratio > 1.0 {
            format!("{:.2}x slower", ratio)
        } else {
            format!("{:.2}x faster", 1.0 / ratio)
        };
        println!(
            "{:<24} {:>+9.3} ms {:>9.3} ms {:>14}",
            c.name, diff_mean, diff_ci, label,
        );
    }

    println!();
    println!("Paired comparisons (each row vs pic_scale_srgb):");
    println!(
        "{:<24} {:>12} {:>12} {:>14}",
        "Library", "Diff (ms)", "95% CI", "vs pic-scale"
    );
    println!("{:-<24} {:-^12} {:-^12} {:-^14}", "", "", "", "");

    let baseline_ps = &contenders[2]; // pic_scale_srgb
    for c in &contenders {
        if std::ptr::eq(c, baseline_ps) {
            continue;
        }
        // contender = a, baseline = b → diff = contender - baseline
        let (_mean_a, _mean_b, diff_mean, diff_ci, ratio) =
            paired_bench(c, baseline_ps, &img, out_w, out_h, rounds);
        let label = if ratio > 1.0 {
            format!("{:.2}x slower", ratio)
        } else {
            format!("{:.2}x faster", 1.0 / ratio)
        };
        println!(
            "{:<24} {:>+9.3} ms {:>9.3} ms {:>14}",
            c.name, diff_mean, diff_ci, label,
        );
    }

    // --- Resizer (cached weights) benchmark ---
    println!();
    println!("Resizer (cached weights) vs one-shot:");
    {
        let config = zenresize::ResizeConfig::builder(img.width, img.height, out_w, out_h)
            .filter(zenresize::Filter::Lanczos)
            .format(zenresize::PixelFormat::Srgb8 {
                channels: 4,
                has_alpha: false,
            })
            .srgb()
            .build();
        let mut resizer = zenresize::Resizer::new(&config);

        // Warmup
        for _ in 0..5 {
            let _ = resizer.resize(&img.rgba);
        }

        let mut times = Vec::with_capacity(rounds);
        for _ in 0..rounds {
            let start = Instant::now();
            let result = resizer.resize(&img.rgba);
            let elapsed = start.elapsed();
            std::hint::black_box(&result);
            times.push(elapsed.as_secs_f64() * 1000.0);
        }
        trim_outliers(&mut times);
        let m = mean(&times);
        let ci = ci95(&times);
        let mps = megapixels / (m / 1000.0);
        println!(
            "{:<24} {:>7.2} ±{:.2} {:>10.0} MP/s",
            "zenresize_resizer", m, ci, mps
        );
    }
}
