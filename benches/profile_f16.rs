//! F16 pipeline profiling binary.
//!
//! Exercises fullframe and streaming paths at multiple sizes to measure
//! throughput with f16 intermediates. Use with callgrind/cachegrind/perf.
//!
//! Usage:
//!   cargo build --release --bench profile_f16
//!   valgrind --tool=cachegrind target/release/deps/profile_f16-*
//!   # or: perf stat target/release/deps/profile_f16-*

fn main() {
    let sizes: &[(u32, u32, u32, u32)] = &[
        (1024, 1024, 512, 512),   // small
        (2048, 2048, 1024, 1024), // medium
        (4000, 3000, 800, 600),   // large (typical photo)
    ];

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
        println!("Fullframe {in_w}x{in_h} -> {out_w}x{out_h} (RGBA, linear, f16 intermediate)");
        let mut resizer = zenresize::Resizer::new(&config);
        for _ in 0..3 {
            let result = resizer.resize(&input);
            std::hint::black_box(&result);
        }

        // --- Streaming ---
        println!("Streaming {in_w}x{in_h} -> {out_w}x{out_h} (RGBA, linear, f16 ring buffer)");
        let batch = 8usize;
        let buf: Vec<u8> = input[..in_row_len * batch].to_vec();

        for _ in 0..3 {
            let mut resizer =
                zenresize::StreamingResize::with_batch_hint(&config, batch as u32);
            let mut input_pushed = 0u32;
            let mut output_rows = 0u32;

            while input_pushed < in_h {
                let count = batch.min((in_h - input_pushed) as usize) as u32;
                let available = resizer
                    .push_rows(&buf[..in_row_len * count as usize], in_row_len, count)
                    .unwrap();
                input_pushed += count;
                for _ in 0..available {
                    let row = resizer.next_output_row().unwrap();
                    std::hint::black_box(row);
                    output_rows += 1;
                }
            }
            let remaining = resizer.finish();
            for _ in 0..remaining {
                let row = resizer.next_output_row().unwrap();
                std::hint::black_box(row);
                output_rows += 1;
            }
            assert_eq!(output_rows, out_h);
        }
    }

    println!("Done.");
}
