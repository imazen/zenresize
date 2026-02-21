//! Streaming resize profiling binary for heaptrack / valgrind.
//!
//! Exercises the streaming resize pipeline with batch push/drain,
//! mirroring zenjpeg's lossy layout path.
//!
//! Usage:
//!   cargo build --release --bench profile_streaming
//!   heaptrack target/release/deps/profile_streaming-*

fn main() {
    let (in_w, in_h) = (4000u32, 3000u32);
    let (out_w, out_h) = (800u32, 600u32);
    let channels = 3usize; // RGB like zenjpeg

    // Synthetic input: gradient pattern
    let in_row_len = in_w as usize * channels;
    let mut input_row = vec![0u8; in_row_len];
    for x in 0..in_w as usize {
        let val = ((x * 255) / in_w as usize) as u8;
        for c in 0..channels {
            input_row[x * channels + c] = val.wrapping_add(c as u8 * 60);
        }
    }

    println!("Streaming resize: {in_w}×{in_h} → {out_w}×{out_h} (RGB, Robidoux, linear)");

    // --- Streaming resize (batch=8) ---
    let config = zenresize::ResizeConfig::builder(in_w, in_h, out_w, out_h)
        .filter(zenresize::Filter::Robidoux)
        .format(zenresize::PixelFormat::Srgb8(zenresize::PixelLayout::Rgb))
        .linear()
        .build();

    let batch = 8usize;
    let buf: Vec<u8> = input_row
        .iter()
        .copied()
        .cycle()
        .take(in_row_len * batch)
        .collect();

    for round in 0..5 {
        let mut resizer = zenresize::StreamingResize::with_batch_hint(&config, batch as u32);

        let mut output_rows = 0u32;
        let mut input_pushed = 0u32;

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

        assert_eq!(output_rows, out_h, "round {round}: wrong output row count");
    }

    // --- I16Srgb path: RGBA, no linearization ---
    let channels4 = 4usize;
    let in_row_len4 = in_w as usize * channels4;
    let config_i16_srgb = zenresize::ResizeConfig::builder(in_w, in_h, out_w, out_h)
        .filter(zenresize::Filter::Robidoux)
        .format(zenresize::PixelFormat::Srgb8(zenresize::PixelLayout::Rgba))
        .srgb()
        .build();

    let buf4: Vec<u8> = (0..in_row_len4 * batch).map(|i| (i % 256) as u8).collect();

    println!("Streaming I16Srgb: {in_w}×{in_h} → {out_w}×{out_h} (RGBA, Robidoux, srgb)");
    for round in 0..5 {
        let mut resizer =
            zenresize::StreamingResize::with_batch_hint(&config_i16_srgb, batch as u32);
        let mut output_rows = 0u32;
        let mut input_pushed = 0u32;

        while input_pushed < in_h {
            let count = batch.min((in_h - input_pushed) as usize) as u32;
            let available = resizer
                .push_rows(&buf4[..in_row_len4 * count as usize], in_row_len4, count)
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
        assert_eq!(
            output_rows, out_h,
            "i16srgb round {round}: wrong output row count"
        );
    }

    // --- I16Linear path: RGBA (Rgbx), linearized ---
    let config_i16_linear = zenresize::ResizeConfig::builder(in_w, in_h, out_w, out_h)
        .filter(zenresize::Filter::Robidoux)
        .format(zenresize::PixelFormat::Srgb8(zenresize::PixelLayout::Rgba))
        .linear()
        .build();

    println!("Streaming I16Linear: {in_w}×{in_h} → {out_w}×{out_h} (RGBA, Robidoux, linear)");
    for round in 0..5 {
        let mut resizer =
            zenresize::StreamingResize::with_batch_hint(&config_i16_linear, batch as u32);
        let mut output_rows = 0u32;
        let mut input_pushed = 0u32;

        while input_pushed < in_h {
            let count = batch.min((in_h - input_pushed) as usize) as u32;
            let available = resizer
                .push_rows(&buf4[..in_row_len4 * count as usize], in_row_len4, count)
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
        assert_eq!(
            output_rows, out_h,
            "i16linear round {round}: wrong output row count"
        );
    }

    // --- Fullframe resize (for comparison) ---
    println!("Fullframe resize: {in_w}×{in_h} → {out_w}×{out_h} (RGB, Robidoux, linear)");

    let full_input: Vec<u8> = input_row
        .iter()
        .copied()
        .cycle()
        .take(in_row_len * in_h as usize)
        .collect();

    for _ in 0..5 {
        let mut resizer = zenresize::Resizer::new(&config);
        let output = resizer.resize(&full_input);
        assert_eq!(output.len(), out_w as usize * out_h as usize * channels);
        std::hint::black_box(&output);
    }

    println!("Done.");
}
