//! Head-to-head benchmark: fullframe Resizer vs StreamingResize.
//!
//! Tests both performance and output quality across:
//! - Multiple image sizes and scale factors
//! - All three internal paths (sRGB i16, linear i16, f32)
//! - Different channel counts (3ch, 4ch)
//!
//! Quality section runs first to verify parity, then performance benchmarks follow.

use std::time::Instant;

// ---------------------------------------------------------------------------
// Stats (paired difference testing)
// ---------------------------------------------------------------------------

fn mean(data: &[f64]) -> f64 {
    data.iter().sum::<f64>() / data.len() as f64
}

fn stddev(data: &[f64]) -> f64 {
    let m = mean(data);
    let var = data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
    var.sqrt()
}

fn ci95(data: &[f64]) -> f64 {
    let t = if data.len() >= 30 { 1.96 } else { 2.0 };
    t * stddev(data) / (data.len() as f64).sqrt()
}

fn trim_outliers(data: &mut Vec<f64>) {
    let m = mean(data);
    let s = stddev(data);
    let lo = m - 2.0 * s;
    let hi = m + 2.0 * s;
    data.retain(|x| *x >= lo && *x <= hi);
}

// ---------------------------------------------------------------------------
// Test image generation
// ---------------------------------------------------------------------------

fn make_gradient(w: u32, h: u32, channels: usize) -> Vec<u8> {
    let mut buf = vec![0u8; w as usize * h as usize * channels];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize * channels;
            buf[i] = (x % 256) as u8;
            if channels >= 2 {
                buf[i + 1] = (y % 256) as u8;
            }
            if channels >= 3 {
                buf[i + 2] = ((x + y) % 256) as u8;
            }
            if channels >= 4 {
                buf[i + 3] = 255; // opaque alpha
            }
        }
    }
    buf
}

// ---------------------------------------------------------------------------
// Resize helpers
// ---------------------------------------------------------------------------

fn fullframe_resize(config: &zenresize::ResizeConfig, input: &[u8]) -> Vec<u8> {
    zenresize::Resizer::new(config).resize(input)
}

fn streaming_resize(config: &zenresize::ResizeConfig, input: &[u8]) -> Vec<u8> {
    let in_w = config.in_width as usize;
    let channels = config.input.channels();
    let row_len = in_w * channels;
    let in_h = config.in_height as usize;

    let mut resizer = zenresize::StreamingResize::new(config);
    let mut output =
        Vec::with_capacity(config.out_width as usize * config.out_height as usize * channels);

    for y in 0..in_h {
        resizer
            .push_row(&input[y * row_len..(y + 1) * row_len])
            .unwrap();
        while let Some(row) = resizer.next_output_row() {
            output.extend_from_slice(row);
        }
    }
    resizer.finish();
    while let Some(row) = resizer.next_output_row() {
        output.extend_from_slice(row);
    }
    output
}

fn hfirst_streaming_resize(config: &zenresize::ResizeConfig, input: &[u8]) -> Vec<u8> {
    let ch = config.input.channels();
    if ch == 4 {
        // Use i16 path for 4ch sRGB
        zenresize::resize_hfirst_streaming(config, input)
    } else {
        // Use f32 path for 3ch / other
        zenresize::resize_hfirst_streaming_f32(config, input)
    }
}

fn streaming_resize_batch(config: &zenresize::ResizeConfig, input: &[u8], batch: usize) -> Vec<u8> {
    let in_w = config.in_width as usize;
    let channels = config.input.channels();
    let row_len = in_w * channels;
    let in_h = config.in_height as usize;

    let mut resizer = zenresize::StreamingResize::with_batch_hint(config, batch as u32);
    let mut output =
        Vec::with_capacity(config.out_width as usize * config.out_height as usize * channels);

    let mut pushed = 0usize;
    while pushed < in_h {
        let count = batch.min(in_h - pushed);
        let buf = &input[pushed * row_len..(pushed + count) * row_len];
        let available = resizer.push_rows(buf, row_len, count as u32).unwrap();
        pushed += count;
        for _ in 0..available {
            output.extend_from_slice(resizer.next_output_row().unwrap());
        }
    }
    let remaining = resizer.finish();
    for _ in 0..remaining {
        output.extend_from_slice(resizer.next_output_row().unwrap());
    }
    output
}

// ---------------------------------------------------------------------------
// Quality comparison
// ---------------------------------------------------------------------------

struct QualityResult {
    label: String,
    max_diff: u8,
    mean_diff: f64,
    psnr_db: f64,
    fullframe_path: &'static str,
    streaming_path: String,
}

fn compare_quality(label: &str, config: &zenresize::ResizeConfig, input: &[u8]) -> QualityResult {
    let full = fullframe_resize(config, input);
    let stream = streaming_resize(config, input);

    assert_eq!(
        full.len(),
        stream.len(),
        "{label}: output length mismatch ({} vs {})",
        full.len(),
        stream.len()
    );

    let mut max_diff = 0u8;
    let mut sum_sq: f64 = 0.0;
    let mut sum_abs: f64 = 0.0;
    let n = full.len() as f64;

    for (&a, &b) in full.iter().zip(stream.iter()) {
        let d = (a as i16 - b as i16).unsigned_abs() as u8;
        max_diff = max_diff.max(d);
        sum_sq += (d as f64) * (d as f64);
        sum_abs += d as f64;
    }

    let mse = sum_sq / n;
    let psnr = if mse > 0.0 {
        10.0 * (255.0_f64 * 255.0 / mse).log10()
    } else {
        f64::INFINITY
    };

    // Determine paths from config
    let channels = config.input.channels();
    let is_u8 = config.input.channel_type() == zenresize::ChannelType::U8;
    let input_tf = config.effective_input_transfer();
    let output_tf = config.effective_output_transfer();
    let needs_premul = config.input.alpha == Some(zenresize::AlphaMode::Straight);
    let linearize = input_tf == zenresize::TransferFunction::Srgb
        && output_tf == zenresize::TransferFunction::Srgb;

    let actual_linearize = config.needs_linearization();
    // Path selection mirrors resize.rs: path 0 = sRGB i16 (4ch, no linearize),
    // path 1 = linear i16 (4ch, linearize, no premul), path 2 = f32.
    let ff_path = if is_u8 && channels == 4 && !actual_linearize {
        "sRGB-i16"
    } else if is_u8 && channels == 4 && actual_linearize && !needs_premul {
        "linear-i16"
    } else if is_u8 {
        "f32(u8)"
    } else {
        "f32"
    };
    let streamer = zenresize::StreamingResize::new(config);
    let st_path = format!("{:?}", streamer.working_format());

    QualityResult {
        label: label.to_string(),
        max_diff,
        mean_diff: sum_abs / n,
        psnr_db: psnr,
        fullframe_path: ff_path,
        streaming_path: st_path,
    }
}

// ---------------------------------------------------------------------------
// Benchmark scenarios
// ---------------------------------------------------------------------------

struct Scenario {
    label: &'static str,
    in_w: u32,
    in_h: u32,
    out_w: u32,
    out_h: u32,
    config: zenresize::ResizeConfig,
    input: Vec<u8>,
}

fn scenarios() -> Vec<Scenario> {
    let mut out = Vec::new();

    // Helper to create scenarios with a builder closure
    let mut add = |label: &'static str,
                   in_w: u32,
                   in_h: u32,
                   out_w: u32,
                   out_h: u32,
                   build: fn(u32, u32, u32, u32) -> zenresize::ResizeConfig| {
        let config = build(in_w, in_h, out_w, out_h);
        let channels = config.input.channels();
        let input = make_gradient(in_w, in_h, channels);
        out.push(Scenario {
            label,
            in_w,
            in_h,
            out_w,
            out_h,
            config,
            input,
        });
    };

    // --- sRGB i16 path (4ch, srgb, no linearization) ---
    fn srgb4(in_w: u32, in_h: u32, out_w: u32, out_h: u32) -> zenresize::ResizeConfig {
        zenresize::ResizeConfig::builder(in_w, in_h, out_w, out_h)
            .filter(zenresize::Filter::Lanczos)
            .format(zenresize::PixelDescriptor::RGBA8_SRGB)
            .srgb()
            .build()
    }

    // --- linear i16 path (4ch Rgbx, srgb input, linearized) ---
    fn linear4(in_w: u32, in_h: u32, out_w: u32, out_h: u32) -> zenresize::ResizeConfig {
        zenresize::ResizeConfig::builder(in_w, in_h, out_w, out_h)
            .filter(zenresize::Filter::Lanczos)
            .format(zenresize::PixelDescriptor::RGBX8_SRGB)
            .linear()
            .build()
    }

    // --- f32 path (3ch, linearized) ---
    fn f32_3ch(in_w: u32, in_h: u32, out_w: u32, out_h: u32) -> zenresize::ResizeConfig {
        zenresize::ResizeConfig::builder(in_w, in_h, out_w, out_h)
            .filter(zenresize::Filter::Lanczos)
            .format(zenresize::PixelDescriptor::RGB8_SRGB)
            .linear()
            .build()
    }

    // --- f32 path (4ch RGBA straight alpha, linearized) ---
    fn f32_4ch_alpha(in_w: u32, in_h: u32, out_w: u32, out_h: u32) -> zenresize::ResizeConfig {
        zenresize::ResizeConfig::builder(in_w, in_h, out_w, out_h)
            .filter(zenresize::Filter::Lanczos)
            .format(zenresize::PixelDescriptor::RGBA8_SRGB)
            .linear()
            .build()
    }

    // Small (dominates by setup/overhead)
    add("srgb4_256_2x", 256, 256, 128, 128, srgb4);
    add("linear4_256_2x", 256, 256, 128, 128, linear4);
    add("f32_3ch_256_2x", 256, 256, 128, 128, f32_3ch);

    // Medium (typical web thumbnails)
    add("srgb4_1024_2x", 1024, 1024, 512, 512, srgb4);
    add("linear4_1024_2x", 1024, 1024, 512, 512, linear4);
    add("f32_3ch_1024_2x", 1024, 1024, 512, 512, f32_3ch);

    // Large downscale (4K → 1080p)
    add("srgb4_4k_2x", 3840, 2160, 1920, 1080, srgb4);
    add("linear4_4k_2x", 3840, 2160, 1920, 1080, linear4);
    add("f32_3ch_4k_2x", 3840, 2160, 1920, 1080, f32_3ch);

    // Heavy downscale (4K → 400p)
    add("srgb4_4k_10x", 4000, 3000, 400, 300, srgb4);
    add("linear4_4k_10x", 4000, 3000, 400, 300, linear4);
    add("f32_3ch_4k_10x", 4000, 3000, 400, 300, f32_3ch);

    // Upscale
    add("srgb4_512_up2x", 512, 512, 1024, 1024, srgb4);
    add("linear4_512_up2x", 512, 512, 1024, 1024, linear4);
    add("f32_3ch_512_up2x", 512, 512, 1024, 1024, f32_3ch);

    // Alpha path (forces f32 streaming)
    add("f32_alpha_1024_2x", 1024, 1024, 512, 512, f32_4ch_alpha);
    add("f32_alpha_4k_2x", 3840, 2160, 1920, 1080, f32_4ch_alpha);

    out
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let all = scenarios();

    // ===== QUALITY COMPARISON =====
    println!(
        "╔══════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║ QUALITY: Fullframe vs Streaming                                                    ║"
    );
    println!(
        "╠══════════════════════════════════════════════════════════════════════════════════════╣"
    );
    println!(
        "║ {:<24} {:>8} {:>8} {:>8} {:>12} {:>12} ║",
        "Scenario", "MaxDiff", "MeanDif", "PSNR", "FF Path", "ST Path"
    );
    println!(
        "╠══════════════════════════════════════════════════════════════════════════════════════╣"
    );

    for s in &all {
        let q = compare_quality(s.label, &s.config, &s.input);
        let psnr_str = if q.psnr_db.is_infinite() {
            "inf".to_string()
        } else {
            format!("{:.1}", q.psnr_db)
        };
        println!(
            "║ {:<24} {:>8} {:>8.4} {:>8} {:>12} {:>12} ║",
            q.label, q.max_diff, q.mean_diff, psnr_str, q.fullframe_path, q.streaming_path
        );
    }
    println!(
        "╚══════════════════════════════════════════════════════════════════════════════════════╝"
    );
    println!();

    // ===== PERFORMANCE COMPARISON =====
    println!("╔════════════════════════════════════════════════════════════════════════════════╗");
    println!("║ PERFORMANCE: Fullframe vs Streaming (paired interleaved)                      ║");
    println!("╠════════════════════════════════════════════════════════════════════════════════╣");
    println!(
        "║ {:<24} {:>10} {:>10} {:>10} {:>10} {:>7} {:>7} ║",
        "Scenario", "Fullframe", "VFirst", "HFirst", "Best", "VF/FF", "HF/FF"
    );
    println!(
        "╠══════════════════════════════════════════════════════════════════════════════════════╣"
    );

    for s in &all {
        let has_hfirst = true; // now available for all channel counts

        // Warmup
        for _ in 0..3 {
            std::hint::black_box(fullframe_resize(&s.config, &s.input));
            std::hint::black_box(streaming_resize(&s.config, &s.input));
            if has_hfirst {
                std::hint::black_box(hfirst_streaming_resize(&s.config, &s.input));
            }
        }

        let iters = if s.in_w * s.in_h > 2_000_000 { 20 } else { 40 };

        let mut ff_times = Vec::with_capacity(iters);
        let mut vf_times = Vec::with_capacity(iters);
        let mut hf_times = Vec::with_capacity(iters);

        // Interleaved measurement
        for _ in 0..iters {
            let t0 = Instant::now();
            std::hint::black_box(fullframe_resize(&s.config, &s.input));
            ff_times.push(t0.elapsed().as_secs_f64() * 1e6);

            let t1 = Instant::now();
            std::hint::black_box(streaming_resize(&s.config, &s.input));
            vf_times.push(t1.elapsed().as_secs_f64() * 1e6);

            if has_hfirst {
                let t2 = Instant::now();
                std::hint::black_box(hfirst_streaming_resize(&s.config, &s.input));
                hf_times.push(t2.elapsed().as_secs_f64() * 1e6);
            }
        }

        trim_outliers(&mut ff_times);
        trim_outliers(&mut vf_times);
        if has_hfirst {
            trim_outliers(&mut hf_times);
        }

        let ff_mean = mean(&ff_times);
        let vf_mean = mean(&vf_times);
        let hf_mean = if has_hfirst {
            mean(&hf_times)
        } else {
            f64::NAN
        };

        let ff_ci = ci95(&ff_times);
        let vf_ci = ci95(&vf_times);
        let hf_ci = if has_hfirst { ci95(&hf_times) } else { 0.0 };

        let vf_ratio = vf_mean / ff_mean;
        let hf_ratio = if has_hfirst {
            hf_mean / ff_mean
        } else {
            f64::NAN
        };

        let fmt_time = |us: f64, ci: f64| -> String {
            if us.is_nan() {
                "n/a".to_string()
            } else if us > 1000.0 {
                format!("{:.2}±{:.1}ms", us / 1000.0, ci / 1000.0)
            } else {
                format!("{:.0}±{:.0}µs", us, ci)
            }
        };

        let best = if has_hfirst {
            if ff_mean < vf_mean && ff_mean < hf_mean {
                "FF"
            } else if hf_mean < vf_mean {
                "HF"
            } else {
                "VF"
            }
        } else if ff_mean < vf_mean {
            "FF"
        } else {
            "VF"
        };

        let hf_ratio_str = if has_hfirst {
            format!("{:.2}x", hf_ratio)
        } else {
            "n/a".to_string()
        };

        println!(
            "║ {:<24} {:>10} {:>10} {:>10} {:>10} {:>7.2}x {:>7} ║",
            s.label,
            fmt_time(ff_mean, ff_ci),
            fmt_time(vf_mean, vf_ci),
            fmt_time(hf_mean, hf_ci),
            best,
            vf_ratio,
            hf_ratio_str,
        );
    }
    println!("╚════════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // ===== MEMORY COMPARISON =====
    println!("╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║ MEMORY: Approximate buffer sizes                                         ║");
    println!("╠════════════════════════════════════════════════════════════════════════════╣");
    println!(
        "║ {:<24} {:>10} {:>10} {:>8} {:>10} {:>10} ║",
        "Scenario", "Input", "FF Intm", "FF Out", "ST Ring", "ST Total"
    );
    println!("╠════════════════════════════════════════════════════════════════════════════╣");

    for s in &all {
        let ch = s.config.input.channels();
        let in_bytes = s.in_w as usize * s.in_h as usize * ch;
        let h_row_len = s.out_w as usize * ch;

        // Fullframe intermediate: h_row_len × in_h (path-dependent element size)
        let is_u8 = s.config.input.channel_type() == zenresize::ChannelType::U8;
        let input_tf = s.config.effective_input_transfer();
        let output_tf = s.config.effective_output_transfer();
        let needs_premul = s.config.input.alpha == Some(zenresize::AlphaMode::Straight);
        let linearize = input_tf == zenresize::TransferFunction::Srgb
            && output_tf == zenresize::TransferFunction::Srgb;

        let elem_size = if is_u8 && ch == 4 && !linearize && !needs_premul {
            1 // u8 sRGB path
        } else if is_u8 && ch == 4 && linearize && !needs_premul {
            2 // i16 linear path
        } else {
            2 // f16 intermediate
        };
        let ff_intermediate = h_row_len * s.in_h as usize * elem_size;
        let ff_output = h_row_len * s.out_h as usize;

        // Streaming ring buffer: (max_taps + 2) × in_width × ch × elem_size
        let streamer = zenresize::StreamingResize::new(&s.config);
        let st_format = streamer.working_format();
        let st_elem = match st_format {
            zenresize::WorkingFormat::F32 => 2,       // f16 ring
            zenresize::WorkingFormat::I16Srgb => 1,   // u8 ring
            zenresize::WorkingFormat::I16Linear => 2, // i16 ring
            _ => 4,
        };
        // Ring: v_cache_size × in_width × ch × elem + temp buffers
        // Approximate: Lanczos3 window=3.0, max_taps ≈ ceil(window * 2 * scale_ratio)
        let scale_ratio = (s.in_h as f64 / s.out_h as f64).max(1.0);
        let v_max_taps = (3.0 * 2.0 * scale_ratio).ceil();
        let ring_slots = v_max_taps.ceil() as usize + 3;
        let ring_row_len = s.in_w as usize * ch;
        let st_ring = ring_slots * ring_row_len * st_elem;
        // Total streaming = ring + temp_v + temp_h + output row
        let st_total = st_ring + ring_row_len * 4 + h_row_len * 4 + h_row_len;

        let fmt_bytes = |b: usize| -> String {
            if b >= 1_048_576 {
                format!("{:.1}MB", b as f64 / 1_048_576.0)
            } else {
                format!("{:.0}KB", b as f64 / 1024.0)
            }
        };

        println!(
            "║ {:<24} {:>10} {:>10} {:>8} {:>10} {:>10} ║",
            s.label,
            fmt_bytes(in_bytes),
            fmt_bytes(ff_intermediate),
            fmt_bytes(ff_output),
            fmt_bytes(st_ring),
            fmt_bytes(st_total),
        );
    }
    println!("╚════════════════════════════════════════════════════════════════════════════╝");
}
