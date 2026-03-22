//! Cachegrind profiling binary: compare tiled vs non-tiled V-filter.
//!
//! Usage:
//!   cargo build --release --bench cachegrind_vfilter
//!   valgrind --tool=cachegrind target/release/deps/cachegrind_vfilter-*

use zenresize::Filter;
use zenresize::bench_internals::{I16WeightTable, InterpolationDetails, filter_v_all_u8_i16, filter_v_all_u8_i16_tiled};

fn main() {
    let in_w = 3840u32;
    let in_h = 2160u32;
    let out_w = 1920u32;
    let out_h = 1080u32;
    let channels = 4usize;
    let h_row_len = out_w as usize * channels;

    let filter = InterpolationDetails::create(Filter::Lanczos);
    let v_weights = I16WeightTable::new(in_h, out_h, &filter);

    let intermediate_len = h_row_len * in_h as usize;
    let mut intermediate = vec![0u8; intermediate_len];
    for (i, v) in intermediate.iter_mut().enumerate() {
        *v = (i % 256) as u8;
    }

    let out_len = h_row_len * out_h as usize;

    let mode = std::env::args().nth(1).unwrap_or_else(|| "baseline".into());

    match mode.as_str() {
        "baseline" => {
            let mut output = vec![0u8; out_len];
            for _ in 0..3 {
                filter_v_all_u8_i16(
                    &intermediate,
                    &mut output,
                    h_row_len,
                    in_h as usize,
                    out_h as usize,
                    &v_weights,
                );
                std::hint::black_box(&output);
            }
        }
        "tiled" => {
            let mut output = vec![0u8; out_len];
            for _ in 0..3 {
                filter_v_all_u8_i16_tiled(
                    &intermediate,
                    &mut output,
                    h_row_len,
                    in_h as usize,
                    out_h as usize,
                    &v_weights,
                    256, // 256 chunks × 16 bytes = 4KB per tile
                );
                std::hint::black_box(&output);
            }
        }
        other => {
            eprintln!("Usage: {} [baseline|tiled]", other);
            std::process::exit(1);
        }
    }
}
