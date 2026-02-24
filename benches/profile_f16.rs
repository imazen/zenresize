//! F16 pipeline profiling binary.
//!
//! Exercises fullframe and streaming paths at multiple sizes to measure
//! throughput with f16 intermediates. Use with callgrind/cachegrind/perf.
//!
//! Usage:
//!   cargo build --release --bench profile_f16
//!   valgrind --tool=cachegrind target/release/deps/profile_f16-*
//!   # or: perf stat target/release/deps/profile_f16-*

use std::time::Instant;

fn mean_ms(times: &[f64]) -> f64 {
    times.iter().sum::<f64>() / times.len() as f64
}

fn main() {
    let sizes: &[(u32, u32, u32, u32)] = &[
        (1024, 1024, 512, 512),   // small
        (2048, 2048, 1024, 1024), // medium
        (4000, 3000, 800, 600),   // large (typical photo)
    ];

    let warmup = 2;
    let rounds = 5;

    for &(in_w, in_h, out_w, out_h) in sizes {
        let channels = 4usize;
        let in_row_len = in_w as usize * channels;

        // Synthetic RGBA gradient input
        let input: Vec<u8> = (0..in_w as usize * in_h as usize * channels)
            .map(|i| (i % 256) as u8)
            .collect();

        let config = zenresize::ResizeConfig::builder(in_w, in_h, out_w, out_h)
            .filter(zenresize::Filter::Lanczos)
            .format(zenresize::PixelFormat::Srgb8(zenresize::PixelLayout::Rgba))
            .linear()
            .build();

        // --- Fullframe ---
        for _ in 0..warmup {
            let mut r = zenresize::Resizer::new(&config);
            std::hint::black_box(r.resize(&input));
        }
        let mut times_ff = Vec::with_capacity(rounds);
        for _ in 0..rounds {
            let mut r = zenresize::Resizer::new(&config);
            let start = Instant::now();
            std::hint::black_box(r.resize(&input));
            times_ff.push(start.elapsed().as_secs_f64() * 1000.0);
        }

        // --- Streaming ---
        let batch = 8usize;
        let buf: Vec<u8> = input[..in_row_len * batch].to_vec();

        for _ in 0..warmup {
            let mut resizer = zenresize::StreamingResize::with_batch_hint(&config, batch as u32);
            let mut pushed = 0u32;
            while pushed < in_h {
                let count = batch.min((in_h - pushed) as usize) as u32;
                let avail = resizer
                    .push_rows(&buf[..in_row_len * count as usize], in_row_len, count)
                    .unwrap();
                pushed += count;
                for _ in 0..avail {
                    std::hint::black_box(resizer.next_output_row().unwrap());
                }
            }
            let rem = resizer.finish();
            for _ in 0..rem {
                std::hint::black_box(resizer.next_output_row().unwrap());
            }
        }

        let mut times_st = Vec::with_capacity(rounds);
        for _ in 0..rounds {
            let mut resizer = zenresize::StreamingResize::with_batch_hint(&config, batch as u32);
            let mut pushed = 0u32;
            let start = Instant::now();
            while pushed < in_h {
                let count = batch.min((in_h - pushed) as usize) as u32;
                let avail = resizer
                    .push_rows(&buf[..in_row_len * count as usize], in_row_len, count)
                    .unwrap();
                pushed += count;
                for _ in 0..avail {
                    std::hint::black_box(resizer.next_output_row().unwrap());
                }
            }
            let rem = resizer.finish();
            for _ in 0..rem {
                std::hint::black_box(resizer.next_output_row().unwrap());
            }
            times_st.push(start.elapsed().as_secs_f64() * 1000.0);
        }

        let mpix = (in_w as f64 * in_h as f64) / 1e6;
        println!(
            "{in_w}x{in_h} -> {out_w}x{out_h}  fullframe: {:>7.2} ms ({:.1} MP/s)  streaming: {:>7.2} ms ({:.1} MP/s)",
            mean_ms(&times_ff),
            mpix / (mean_ms(&times_ff) / 1000.0),
            mean_ms(&times_st),
            mpix / (mean_ms(&times_st) / 1000.0),
        );
    }
}
