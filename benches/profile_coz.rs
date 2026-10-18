//! Causal-profiling driver for `coz` (dev-only, never shipped).
//!
//! Loops completed resize units across the streaming paths (I16Srgb,
//! I16Linear, F32) plus fullframe, bumping a coz throughput counter per
//! produced output row and per finished resize. Progress counters resolve
//! to no-ops outside `coz run`, so the binary also works standalone as a
//! timing smoke test.
//!
//! Usage:
//!   CARGO_PROFILE_RELEASE_DEBUG=1 cargo build --release --bench profile_coz
//!   coz run --output ~/tmp/profile.jsonl --- ./target/release/deps/profile_coz-<hash> [rounds]
//!
//! `coz` is a Linux-only dev-dependency (it does not build on Windows). On
//! other targets the counters compile to nothing and the same resize loop
//! still runs.

use zenresize::{Filter, PixelDescriptor, ResizeConfig, Resizer, StreamingResize};

/// `coz::progress!` on Linux; a no-op where the coz dev-dependency is absent.
#[cfg(target_os = "linux")]
macro_rules! progress {
    ($name:literal) => {
        coz::progress!($name)
    };
}
#[cfg(not(target_os = "linux"))]
macro_rules! progress {
    ($name:literal) => {};
}

fn make_row(w: u32, channels: usize) -> Vec<u8> {
    let mut row = vec![0u8; w as usize * channels];
    for x in 0..w as usize {
        let v = ((x * 255) / w as usize) as u8;
        for c in 0..channels {
            row[x * channels + c] = v.wrapping_add((c as u8) * 60).wrapping_add(x as u8);
        }
    }
    row
}

/// Push `in_h` rows in batches, draining output rows as they appear.
/// Returns the number of output rows produced.
fn run_streaming(config: &ResizeConfig, row: &[u8], in_h: u32, out_h: u32) -> u32 {
    let batch = 8usize;
    let in_row_len = row.len();
    let buf: Vec<u8> = row
        .iter()
        .copied()
        .cycle()
        .take(in_row_len * batch)
        .collect();

    let mut resizer = StreamingResize::with_batch_hint(config, batch as u32);
    let mut output_rows = 0u32;
    let mut pushed = 0u32;
    while pushed < in_h {
        let count = batch.min((in_h - pushed) as usize) as u32;
        let available = resizer
            .push_rows(&buf[..in_row_len * count as usize], in_row_len, count)
            .unwrap();
        pushed += count;
        for _ in 0..available {
            let r = resizer.next_output_row().unwrap();
            std::hint::black_box(r);
            output_rows += 1;
            progress!("row");
        }
    }
    let remaining = resizer.finish();
    for _ in 0..remaining {
        let r = resizer.next_output_row().unwrap();
        std::hint::black_box(r);
        output_rows += 1;
        progress!("row");
    }
    assert_eq!(output_rows, out_h);
    progress!("resize");
    output_rows
}

fn main() {
    #[cfg(target_os = "linux")]
    coz::thread_init();
    let rounds: u32 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(3);

    // I16Srgb: 4ch sRGB, identity transfer, 5x downscale (tap guard < 50).
    let cfg_srgb = ResizeConfig::builder(4000, 3000, 800, 600)
        .filter(Filter::Lanczos)
        .format(PixelDescriptor::RGBA8_SRGB)
        .srgb()
        .build();
    // I16Linear: 4ch Rgbx, sRGB->linear, 2x downscale (tap guard <= 14).
    let cfg_linear = ResizeConfig::builder(2000, 1500, 1000, 750)
        .filter(Filter::Lanczos)
        .format(PixelDescriptor::RGBX8_SRGB)
        .linear()
        .build();
    // F32: 3ch forces the f32 path.
    let cfg_f32 = ResizeConfig::builder(2000, 1500, 1000, 750)
        .filter(Filter::Lanczos)
        .format(PixelDescriptor::RGB8_SRGB)
        .linear()
        .build();
    // Fullframe for the H-first pipeline.
    let cfg_ff = ResizeConfig::builder(2000, 1500, 1000, 750)
        .filter(Filter::Lanczos)
        .format(PixelDescriptor::RGBA8_SRGB)
        .srgb()
        .build();

    let row4k = make_row(4000, 4);
    let row2k4 = make_row(2000, 4);
    let row2k3 = make_row(2000, 3);
    let ff_input: Vec<u8> = row2k4
        .iter()
        .copied()
        .cycle()
        .take(2000 * 1500 * 4)
        .collect();

    eprintln!("profile_coz: {rounds} rounds x (i16srgb + i16linear + f32 streaming + fullframe)");
    for round in 0..rounds {
        run_streaming(&cfg_srgb, &row4k, 3000, 600);
        run_streaming(&cfg_linear, &row2k4, 1500, 750);
        run_streaming(&cfg_f32, &row2k3, 1500, 750);
        let out = Resizer::new(&cfg_ff).resize(&ff_input);
        assert_eq!(out.len(), 1000 * 750 * 4);
        std::hint::black_box(&out);
        progress!("resize");
        eprintln!("  round {round} done");
    }
    eprintln!("done.");
}
