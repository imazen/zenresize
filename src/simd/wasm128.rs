//! WASM SIMD128 convolution kernels.
//!
//! Most implementations delegate to `wide_kernels` (magetypes portable SIMD).
//! The 4-channel i16 horizontal filter uses native wasm128 intrinsics via
//! `#[rite(import_intrinsics)]` for widening loads and `i32x4_dot_i16x8`
//! (paired multiply-add), matching the x86 AVX2 `madd_epi16` strategy.

use crate::proven::sub;
use crate::weights::{F32WeightTable, I16_PRECISION, I16WeightTable};
use archmage::Wasm128Token;

#[archmage::arcane]
pub(crate) fn filter_h_row_f32_wasm128(
    _token: Wasm128Token,
    input: &[f32],
    output: &mut [f32],
    weights: &F32WeightTable,
    channels: usize,
) {
    super::wide_kernels::filter_h_row_f32_impl_wasm128(_token, input, output, weights, channels)
}

#[archmage::arcane]
pub(crate) fn filter_v_row_f32_wasm128(
    _token: Wasm128Token,
    rows: &[&[f32]],
    output: &mut [f32],
    weights: &[f32],
) {
    super::wide_kernels::filter_v_row_f32_impl_wasm128(_token, rows, output, weights)
}

#[archmage::arcane]
pub(crate) fn u8_to_f32_row_wasm128(_token: Wasm128Token, input: &[u8], output: &mut [f32]) {
    super::wide_kernels::u8_to_f32_row_impl_wasm128(_token, input, output)
}

#[archmage::arcane]
pub(crate) fn f32_to_u8_row_wasm128(_token: Wasm128Token, input: &[f32], output: &mut [u8]) {
    super::wide_kernels::f32_to_u8_row_impl_wasm128(_token, input, output)
}

#[archmage::arcane]
pub(crate) fn premultiply_alpha_row_wasm128(_token: Wasm128Token, row: &mut [f32]) {
    super::wide_kernels::premultiply_alpha_row_impl_wasm128(_token, row)
}

#[archmage::arcane]
pub(crate) fn unpremultiply_alpha_row_wasm128(_token: Wasm128Token, row: &mut [f32]) {
    super::wide_kernels::unpremultiply_alpha_row_impl_wasm128(_token, row)
}

#[archmage::arcane]
pub(crate) fn filter_h_u8_i16_wasm128(
    _token: Wasm128Token,
    input: &[u8],
    output: &mut [u8],
    weights: &I16WeightTable,
    channels: usize,
) {
    match channels {
        4 => filter_h_u8_i16_4ch_intrinsic(_token, input, output, weights),
        _ => super::wide_kernels::filter_h_u8_i16_impl_wasm128(
            _token, input, output, weights, channels,
        ),
    }
}

/// 4-channel i16 horizontal convolution using native wasm128 intrinsics.
///
/// Uses `i32x4_dot_i16x8` (paired multiply-add) with the pre-expanded
/// weight table. Each group of 4 taps is processed as two 128-bit dot
/// products — the lower 8 i16s handle taps 0,1 and the upper 8 handle
/// taps 2,3 — mirroring the AVX2 `madd_epi16` strategy at half width.
///
/// Widening loads (`u16x8_extend_low/high_u8x16`) replace the scalar
/// byte-at-a-time loads in the magetypes fallback.
#[allow(clippy::needless_range_loop)]
#[archmage::rite(import_intrinsics)]
fn filter_h_u8_i16_4ch_intrinsic(
    _token: Wasm128Token,
    input: &[u8],
    output: &mut [u8],
    weights: &I16WeightTable,
) {
    let out_width = weights.len();
    let groups4 = weights.groups4;
    let in_pixels = input.len() / 4;

    // Need at least 4 pixels for the 16-byte loads
    if in_pixels < 4 {
        filter_h_u8_i16_4ch_scalar(input, output, weights);
        return;
    }

    // Check if the input slice has enough padding for all SIMD reads.
    // The +3 accounts for the 16-byte (4-pixel) SIMD load width: the last
    // group's load at pixel max_left + (groups4-1)*4 reads 4 pixels forward.
    let max_left = weights.left.iter().map(|&l| l as usize).max().unwrap_or(0);
    let has_full_padding = max_left + groups4 * 4 + 3 < in_pixels;

    // Shuffle mask: interleave channels from two consecutive RGBA pixels.
    // Input  i16 lanes: [p0r, p0g, p0b, p0a, p1r, p1g, p1b, p1a]
    // Output i16 lanes: [p0r, p1r, p0g, p1g, p0b, p1b, p0a, p1a]
    // Byte indices:      [0,1, 8,9, 2,3, 10,11, 4,5, 12,13, 6,7, 14,15]

    let half = i32x4_splat(1 << (I16_PRECISION - 1));
    let zero = i32x4_splat(0);
    let max_val = i32x4_splat(255);

    // Weight data: contiguous stride-16 i16 array per group.
    let ew_all = weights.expanded_4ch_all();
    let (ew_chunks, _) = ew_all.as_chunks::<8>();

    // Output: contiguous stride-4 u8 pixels.
    let (out_pixels, _) = output.as_chunks_mut::<4>();

    if has_full_padding {
        // Fast path: all pixels have enough input padding for SIMD loads
        for out_x in 0..out_width {
            let left = weights.left[out_x] as usize;
            let byte_start = left * 4;

            // Pre-slice input window and weight window
            let input_window = sub(input, byte_start..byte_start + groups4 * 16);
            let (in_chunks, _) = input_window.as_chunks::<16>();
            let ew_base = out_x * groups4 * 2; // 2 chunks of 8 i16 per group

            let mut acc = i32x4_splat(0);

            for (g, chunk) in in_chunks.iter().enumerate() {
                // Load 16 bytes (4 pixels × 4 channels)
                let raw = v128_load(chunk);

                // Widen low 8 bytes to i16: pixels for taps 0,1
                let lo_i16 = u16x8_extend_low_u8x16(raw);
                // Shuffle to interleave channels: [p0r,p1r, p0g,p1g, p0b,p1b, p0a,p1a]
                let lo_shuffled =
                    i8x16_shuffle::<0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15>(
                        lo_i16, lo_i16,
                    );
                // Load lower weight half: [w0,w1, w0,w1, w0,w1, w0,w1]
                let w_lo = v128_load(&ew_chunks[ew_base + g * 2]);
                // Paired multiply-add: R=p0r*w0+p1r*w1, G=p0g*w0+p1g*w1, ...
                acc = i32x4_add(acc, i32x4_dot_i16x8(lo_shuffled, w_lo));

                // Widen high 8 bytes to i16: pixels for taps 2,3
                let hi_i16 = u16x8_extend_high_u8x16(raw);
                // Same channel interleave shuffle
                let hi_shuffled =
                    i8x16_shuffle::<0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15>(
                        hi_i16, hi_i16,
                    );
                // Load upper weight half: [w2,w3, w2,w3, w2,w3, w2,w3]
                let w_hi = v128_load(&ew_chunks[ew_base + g * 2 + 1]);
                acc = i32x4_add(acc, i32x4_dot_i16x8(hi_shuffled, w_hi));
            }

            // Round, shift right, clamp to [0, 255], store
            let rounded = i32x4_add(acc, half);
            let shifted = i32x4_shr(rounded, I16_PRECISION as u32);
            let clamped = i32x4_min(i32x4_max(shifted, zero), max_val);

            // Pack i32x4 → i16x8 → u8x16, extract low 4 bytes
            let packed16 = i16x8_narrow_i32x4(clamped, zero);
            let packed8 = u8x16_narrow_i16x8(packed16, packed16);
            let pixel_val = u32x4_extract_lane::<0>(packed8);
            out_pixels[out_x].copy_from_slice(&pixel_val.to_ne_bytes());
        }
    } else {
        // Edge-aware path: per-pixel bounds check
        let safe_end = (0..out_width)
            .position(|x| (weights.left[x] as usize) + groups4 * 4 > in_pixels)
            .unwrap_or(out_width);

        // SIMD for interior pixels
        for out_x in 0..safe_end {
            let left = weights.left[out_x] as usize;
            let byte_start = left * 4;

            let input_window = &input[byte_start..byte_start + groups4 * 16];
            let (in_chunks, _) = input_window.as_chunks::<16>();
            let ew_base = out_x * groups4 * 2;

            let mut acc = i32x4_splat(0);

            for (g, chunk) in in_chunks.iter().enumerate() {
                let raw = v128_load(chunk);

                let lo_i16 = u16x8_extend_low_u8x16(raw);
                let lo_shuffled =
                    i8x16_shuffle::<0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15>(
                        lo_i16, lo_i16,
                    );
                let w_lo = v128_load(&ew_chunks[ew_base + g * 2]);
                acc = i32x4_add(acc, i32x4_dot_i16x8(lo_shuffled, w_lo));

                let hi_i16 = u16x8_extend_high_u8x16(raw);
                let hi_shuffled =
                    i8x16_shuffle::<0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15>(
                        hi_i16, hi_i16,
                    );
                let w_hi = v128_load(&ew_chunks[ew_base + g * 2 + 1]);
                acc = i32x4_add(acc, i32x4_dot_i16x8(hi_shuffled, w_hi));
            }

            let rounded = i32x4_add(acc, half);
            let shifted = i32x4_shr(rounded, I16_PRECISION as u32);
            let clamped = i32x4_min(i32x4_max(shifted, zero), max_val);

            let packed16 = i16x8_narrow_i32x4(clamped, zero);
            let packed8 = u8x16_narrow_i16x8(packed16, packed16);
            let pixel_val = u32x4_extract_lane::<0>(packed8);
            out_pixels[out_x].copy_from_slice(&pixel_val.to_ne_bytes());
        }

        // Scalar for edge pixels
        for out_x in safe_end..out_width {
            let left = weights.left[out_x] as usize;
            let w = weights.weights(out_x);
            let out_base = out_x * 4;
            for c in 0..4 {
                let mut acc: i32 = 0;
                for (t, &weight) in w.iter().enumerate() {
                    acc += input[(left + t) * 4 + c] as i32 * weight as i32;
                }
                let rounded = (acc + (1 << (I16_PRECISION - 1))) >> I16_PRECISION;
                output[out_base + c] = rounded.clamp(0, 255) as u8;
            }
        }
    }
}

/// Scalar fallback for 4ch when input is too small for SIMD.
#[inline(always)]
fn filter_h_u8_i16_4ch_scalar(input: &[u8], output: &mut [u8], weights: &I16WeightTable) {
    for out_x in 0..weights.len() {
        let left = weights.left[out_x] as usize;
        let w = weights.weights(out_x);
        let out_base = out_x * 4;
        for c in 0..4 {
            let mut acc: i32 = 0;
            for (t, &weight) in w.iter().enumerate() {
                acc += input[(left + t) * 4 + c] as i32 * weight as i32;
            }
            let rounded = (acc + (1 << (I16_PRECISION - 1))) >> I16_PRECISION;
            output[out_base + c] = rounded.clamp(0, 255) as u8;
        }
    }
}

#[archmage::arcane]
pub(crate) fn filter_h_u8_to_i16_wasm128(
    _token: Wasm128Token,
    input: &[u8],
    output: &mut [i16],
    weights: &I16WeightTable,
    channels: usize,
) {
    match channels {
        4 => filter_h_u8_to_i16_4ch_intrinsic(_token, input, output, weights),
        _ => super::wide_kernels::filter_h_u8_to_i16_impl_wasm128(
            _token, input, output, weights, channels,
        ),
    }
}

/// 4-channel u8→i16 horizontal convolution using native wasm128 intrinsics.
///
/// Same `i32x4_dot_i16x8` paired multiply-add approach as the u8→u8 variant,
/// but outputs unclamped i16 to preserve Lanczos ringing for the next filter stage.
#[allow(clippy::needless_range_loop)]
#[archmage::rite(import_intrinsics)]
fn filter_h_u8_to_i16_4ch_intrinsic(
    _token: Wasm128Token,
    input: &[u8],
    output: &mut [i16],
    weights: &I16WeightTable,
) {
    let out_width = weights.len();
    let groups4 = weights.groups4;
    let in_pixels = input.len() / 4;

    if in_pixels < 4 {
        filter_h_u8_to_i16_4ch_scalar(input, output, weights);
        return;
    }

    let max_left = weights.left.iter().map(|&l| l as usize).max().unwrap_or(0);
    let has_full_padding = max_left + groups4 * 4 + 3 < in_pixels;

    let half = i32x4_splat(1 << (I16_PRECISION - 1));
    let ew_all = weights.expanded_4ch_all();
    let (ew_chunks, _) = ew_all.as_chunks::<8>();

    if has_full_padding {
        for out_x in 0..out_width {
            let left = weights.left[out_x] as usize;
            let byte_start = left * 4;
            let input_window = sub(input, byte_start..byte_start + groups4 * 16);
            let (in_chunks, _) = input_window.as_chunks::<16>();
            let ew_base = out_x * groups4 * 2;

            let mut acc = i32x4_splat(0);

            for (g, chunk) in in_chunks.iter().enumerate() {
                let raw = v128_load(chunk);

                let lo_i16 = u16x8_extend_low_u8x16(raw);
                let lo_shuffled =
                    i8x16_shuffle::<0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15>(
                        lo_i16, lo_i16,
                    );
                let w_lo = v128_load(&ew_chunks[ew_base + g * 2]);
                acc = i32x4_add(acc, i32x4_dot_i16x8(lo_shuffled, w_lo));

                let hi_i16 = u16x8_extend_high_u8x16(raw);
                let hi_shuffled =
                    i8x16_shuffle::<0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15>(
                        hi_i16, hi_i16,
                    );
                let w_hi = v128_load(&ew_chunks[ew_base + g * 2 + 1]);
                acc = i32x4_add(acc, i32x4_dot_i16x8(hi_shuffled, w_hi));
            }

            // Round, shift — output i16 without clamping to [0,255]
            let rounded = i32x4_shr(i32x4_add(acc, half), I16_PRECISION as u32);
            let out_base = out_x * 4;
            let arr: [i32; 4] = {
                let mut a = [0i32; 4];
                v128_store(&mut a, rounded);
                a
            };
            output[out_base] = arr[0] as i16;
            output[out_base + 1] = arr[1] as i16;
            output[out_base + 2] = arr[2] as i16;
            output[out_base + 3] = arr[3] as i16;
        }
    } else {
        // Edge-aware: SIMD for interior, scalar for edges
        let safe_end = (0..out_width)
            .position(|x| (weights.left[x] as usize) + groups4 * 4 > in_pixels)
            .unwrap_or(out_width);

        for out_x in 0..safe_end {
            let left = weights.left[out_x] as usize;
            let byte_start = left * 4;
            let input_window = &input[byte_start..byte_start + groups4 * 16];
            let (in_chunks, _) = input_window.as_chunks::<16>();
            let ew_base = out_x * groups4 * 2;

            let mut acc = i32x4_splat(0);

            for (g, chunk) in in_chunks.iter().enumerate() {
                let raw = v128_load(chunk);

                let lo_i16 = u16x8_extend_low_u8x16(raw);
                let lo_shuffled =
                    i8x16_shuffle::<0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15>(
                        lo_i16, lo_i16,
                    );
                let w_lo = v128_load(&ew_chunks[ew_base + g * 2]);
                acc = i32x4_add(acc, i32x4_dot_i16x8(lo_shuffled, w_lo));

                let hi_i16 = u16x8_extend_high_u8x16(raw);
                let hi_shuffled =
                    i8x16_shuffle::<0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15>(
                        hi_i16, hi_i16,
                    );
                let w_hi = v128_load(&ew_chunks[ew_base + g * 2 + 1]);
                acc = i32x4_add(acc, i32x4_dot_i16x8(hi_shuffled, w_hi));
            }

            let rounded = i32x4_shr(i32x4_add(acc, half), I16_PRECISION as u32);
            let out_base = out_x * 4;
            let arr: [i32; 4] = {
                let mut a = [0i32; 4];
                v128_store(&mut a, rounded);
                a
            };
            output[out_base] = arr[0] as i16;
            output[out_base + 1] = arr[1] as i16;
            output[out_base + 2] = arr[2] as i16;
            output[out_base + 3] = arr[3] as i16;
        }

        // Scalar for edge pixels
        for out_x in safe_end..out_width {
            let left = weights.left[out_x] as usize;
            let w = weights.weights(out_x);
            let out_base = out_x * 4;
            for c in 0..4 {
                let mut acc: i32 = 0;
                for (t, &weight) in w.iter().enumerate() {
                    acc += input[(left + t) * 4 + c] as i32 * weight as i32;
                }
                let rounded = (acc + (1 << (I16_PRECISION - 1))) >> I16_PRECISION;
                output[out_base + c] = rounded as i16;
            }
        }
    }
}

/// Scalar fallback for 4ch u8→i16 when input is too small for SIMD.
#[inline(always)]
fn filter_h_u8_to_i16_4ch_scalar(input: &[u8], output: &mut [i16], weights: &I16WeightTable) {
    for out_x in 0..weights.len() {
        let left = weights.left[out_x] as usize;
        let w = weights.weights(out_x);
        let out_base = out_x * 4;
        for c in 0..4 {
            let mut acc: i32 = 0;
            for (t, &weight) in w.iter().enumerate() {
                acc += input[(left + t) * 4 + c] as i32 * weight as i32;
            }
            let rounded = (acc + (1 << (I16_PRECISION - 1))) >> I16_PRECISION;
            output[out_base + c] = rounded as i16;
        }
    }
}

#[archmage::arcane]
pub(crate) fn filter_h_u8_to_i16_4rows_wasm128(
    _token: Wasm128Token,
    in0: &[u8],
    in1: &[u8],
    in2: &[u8],
    in3: &[u8],
    out0: &mut [i16],
    out1: &mut [i16],
    out2: &mut [i16],
    out3: &mut [i16],
    weights: &I16WeightTable,
) {
    super::wide_kernels::filter_h_u8_to_i16_4rows_impl_wasm128(
        _token, in0, in1, in2, in3, out0, out1, out2, out3, weights,
    )
}

#[archmage::arcane]
pub(crate) fn filter_h_u8_i16_4rows_wasm128(
    _token: Wasm128Token,
    in0: &[u8],
    in1: &[u8],
    in2: &[u8],
    in3: &[u8],
    out0: &mut [u8],
    out1: &mut [u8],
    out2: &mut [u8],
    out3: &mut [u8],
    weights: &I16WeightTable,
) {
    super::wide_kernels::filter_h_u8_i16_4rows_impl_wasm128(
        _token, in0, in1, in2, in3, out0, out1, out2, out3, weights,
    )
}

#[archmage::arcane]
pub(crate) fn filter_v_all_u8_i16_wasm128(
    _token: Wasm128Token,
    intermediate: &[u8],
    output: &mut [u8],
    h_row_len: usize,
    in_h: usize,
    out_h: usize,
    weights: &crate::weights::I16WeightTable,
) {
    super::wide_kernels::filter_v_all_u8_i16_impl_wasm128(
        _token,
        intermediate,
        output,
        h_row_len,
        in_h,
        out_h,
        weights,
    )
}

#[archmage::arcane]
pub(crate) fn filter_v_all_u8_i16_tiled_wasm128(
    _token: Wasm128Token,
    intermediate: &[u8],
    output: &mut [u8],
    h_row_len: usize,
    in_h: usize,
    out_h: usize,
    weights: &crate::weights::I16WeightTable,
    tile_chunks: usize,
) {
    super::wide_kernels::filter_v_all_u8_i16_tiled_impl_wasm128(
        _token,
        intermediate,
        output,
        h_row_len,
        in_h,
        out_h,
        weights,
        tile_chunks,
    )
}

#[archmage::arcane]
pub(crate) fn premultiply_u8_row_wasm128(_token: Wasm128Token, input: &[u8], output: &mut [u8]) {
    super::wide_kernels::premultiply_u8_row_impl_wasm128(_token, input, output)
}

#[archmage::arcane]
pub(crate) fn unpremultiply_u8_row_wasm128(_token: Wasm128Token, row: &mut [u8]) {
    super::wide_kernels::unpremultiply_u8_row_impl_wasm128(_token, row)
}

#[archmage::arcane]
pub(crate) fn filter_h_i16_i16_wasm128(
    _token: Wasm128Token,
    input: &[i16],
    output: &mut [i16],
    weights: &I16WeightTable,
    channels: usize,
) {
    super::wide_kernels::filter_h_i16_i16_impl_wasm128(_token, input, output, weights, channels)
}

#[archmage::arcane]
pub(crate) fn filter_v_all_i16_i16_wasm128(
    _token: Wasm128Token,
    intermediate: &[i16],
    output: &mut [i16],
    h_row_len: usize,
    in_h: usize,
    out_h: usize,
    weights: &crate::weights::I16WeightTable,
) {
    super::wide_kernels::filter_v_all_i16_i16_impl_wasm128(
        _token,
        intermediate,
        output,
        h_row_len,
        in_h,
        out_h,
        weights,
    )
}

#[archmage::arcane]
pub(crate) fn filter_v_row_u8_i16_wasm128(
    _token: Wasm128Token,
    rows: &[&[u8]],
    output: &mut [u8],
    weights: &[i16],
) {
    filter_v_row_u8_i16_intrinsic(_token, rows, output, weights)
}

/// Streaming V-filter: u8 rows → u8 output using native wasm128 intrinsics.
///
/// Processes 16 bytes per iteration using widening loads (`u16x8_extend_low/high_u8x16`)
/// to go u8→i16→i32 for accumulation, then narrowing stores (`i16x8_narrow_i32x4` +
/// `u8x16_narrow_i16x8`) to pack back to u8. This replaces the magetypes generic path
/// which constructs i32x4 from scalar arrays (4 scalar loads per vector).
#[archmage::rite(import_intrinsics)]
fn filter_v_row_u8_i16_intrinsic(
    _token: Wasm128Token,
    rows: &[&[u8]],
    output: &mut [u8],
    weights: &[i16],
) {
    let width = output.len();
    debug_assert_eq!(rows.len(), weights.len());

    let half = i32x4_splat(1 << (I16_PRECISION - 1));

    // Process 16 bytes at a time (4 × i32x4 accumulators)
    let chunks16 = width / 16;

    for chunk_idx in 0..chunks16 {
        let x = chunk_idx * 16;
        let mut acc0 = i32x4_splat(0);
        let mut acc1 = i32x4_splat(0);
        let mut acc2 = i32x4_splat(0);
        let mut acc3 = i32x4_splat(0);

        for (row, &weight) in rows.iter().zip(weights.iter()) {
            let w = i32x4_splat(weight as i32);
            // Load 16 bytes as u8x16
            let raw: &[u8; 16] = row[x..x + 16].try_into().unwrap();
            let bytes = v128_load(raw);

            // Widen low 8 bytes: u8→u16
            let lo16 = u16x8_extend_low_u8x16(bytes);
            // Widen to i32: low 4 and high 4 of the low 8
            let lo_lo32 = u32x4_extend_low_u16x8(lo16);
            let lo_hi32 = u32x4_extend_high_u16x8(lo16);
            acc0 = i32x4_add(acc0, i32x4_mul(lo_lo32, w));
            acc1 = i32x4_add(acc1, i32x4_mul(lo_hi32, w));

            // Widen high 8 bytes: u8→u16
            let hi16 = u16x8_extend_high_u8x16(bytes);
            let hi_lo32 = u32x4_extend_low_u16x8(hi16);
            let hi_hi32 = u32x4_extend_high_u16x8(hi16);
            acc2 = i32x4_add(acc2, i32x4_mul(hi_lo32, w));
            acc3 = i32x4_add(acc3, i32x4_mul(hi_hi32, w));
        }

        // Round, shift, and pack to u8
        let r0 = i32x4_shr(i32x4_add(acc0, half), I16_PRECISION as u32);
        let r1 = i32x4_shr(i32x4_add(acc1, half), I16_PRECISION as u32);
        let r2 = i32x4_shr(i32x4_add(acc2, half), I16_PRECISION as u32);
        let r3 = i32x4_shr(i32x4_add(acc3, half), I16_PRECISION as u32);

        // Narrow i32x4 → i16x8 (signed saturate), then i16x8 → u8x16 (unsigned saturate)
        let lo16 = i16x8_narrow_i32x4(r0, r1);
        let hi16 = i16x8_narrow_i32x4(r2, r3);
        let packed = u8x16_narrow_i16x8(lo16, hi16);

        let out_arr: &mut [u8; 16] = (&mut output[x..x + 16]).try_into().unwrap();
        v128_store(out_arr, packed);
    }

    // Scalar tail for remaining bytes
    let base16 = chunks16 * 16;
    for x in base16..width {
        let mut acc: i32 = 0;
        for (row, &weight) in rows.iter().zip(weights.iter()) {
            acc += row[x] as i32 * weight as i32;
        }
        let rounded = (acc + (1 << (I16_PRECISION - 1))) >> I16_PRECISION;
        output[x] = rounded.clamp(0, 255) as u8;
    }
}

#[archmage::arcane]
pub(crate) fn filter_v_row_i16_wasm128(
    _token: Wasm128Token,
    rows: &[&[i16]],
    output: &mut [i16],
    weights: &[i16],
) {
    filter_v_row_i16_intrinsic(_token, rows, output, weights)
}

/// Streaming V-filter: i16 rows → i16 output using native wasm128 intrinsics.
///
/// Processes 8 i16 per iteration using widening loads (`i32x4_extend_low/high_i16x8`)
/// for i16→i32 accumulation, then narrowing stores (`i16x8_narrow_i32x4`).
#[archmage::rite(import_intrinsics)]
fn filter_v_row_i16_intrinsic(
    _token: Wasm128Token,
    rows: &[&[i16]],
    output: &mut [i16],
    weights: &[i16],
) {
    let width = output.len();
    debug_assert_eq!(rows.len(), weights.len());

    let half = i32x4_splat(1 << (I16_PRECISION - 1));

    // Process 8 i16 values at a time (2 × i32x4 accumulators)
    let chunks8 = width / 8;

    for chunk_idx in 0..chunks8 {
        let x = chunk_idx * 8;
        let mut acc0 = i32x4_splat(0);
        let mut acc1 = i32x4_splat(0);

        for (row, &weight) in rows.iter().zip(weights.iter()) {
            let w = i32x4_splat(weight as i32);
            // Load 8 i16 values
            let vals: &[i16; 8] = row[x..x + 8].try_into().unwrap();
            let v16 = v128_load(vals);

            // Widen i16→i32
            let lo32 = i32x4_extend_low_i16x8(v16);
            let hi32 = i32x4_extend_high_i16x8(v16);
            acc0 = i32x4_add(acc0, i32x4_mul(lo32, w));
            acc1 = i32x4_add(acc1, i32x4_mul(hi32, w));
        }

        // Round, shift
        let r0 = i32x4_shr(i32x4_add(acc0, half), I16_PRECISION as u32);
        let r1 = i32x4_shr(i32x4_add(acc1, half), I16_PRECISION as u32);

        // Narrow i32x4 → i16x8 (signed saturate)
        let packed = i16x8_narrow_i32x4(r0, r1);
        let out_arr: &mut [i16; 8] = (&mut output[x..x + 8]).try_into().unwrap();
        v128_store(out_arr, packed);
    }

    // Scalar tail
    let base8 = chunks8 * 8;
    for x in base8..width {
        let mut acc: i32 = 0;
        for (row, &weight) in rows.iter().zip(weights.iter()) {
            acc += row[x] as i32 * weight as i32;
        }
        let rounded = (acc + (1 << (I16_PRECISION - 1))) >> I16_PRECISION;
        output[x] = rounded as i16;
    }
}

// =========================================================================
// f16 ↔ f32 vectorized conversion using wasm128 integer SIMD
// =========================================================================

/// Branchless f16 → f32 conversion, 4 values at a time.
///
/// Handles normal, zero, subnormal, and inf/nan cases via SIMD comparison
/// masks and `v128_bitselect`. Subnormals use the "magic float" denormalization
/// trick: place mantissa bits into a float with exponent 113, then subtract
/// `f32::from_bits(113 << 23)` — the float subtraction renormalizes the
/// mantissa, producing the correct subnormal value.
///
/// Bit-exact with `f16_to_f32_soft` for all normal, zero, inf, and nan inputs.
/// Subnormals may differ by at most 1 ULP due to the float subtraction trick
/// vs the scalar's iterative shift approach, but both produce correct IEEE 754
/// results and neither produces NaN for valid f16 inputs.
#[archmage::rite(import_intrinsics)]
fn f16_to_f32_x4(_token: Wasm128Token, h: &[u16; 4]) -> [f32; 4] {
    // Load 4 u16 values, zero-extend to i32x4
    let bits = u32x4_load_extend_u16x4(h);

    // Extract fields
    let sign = i32x4_shl(v128_and(bits, i32x4_splat(0x8000_u32 as i32)), 16);
    let exp = v128_and(u32x4_shr(bits, 10), i32x4_splat(0x1F));
    let mant = v128_and(bits, i32x4_splat(0x03FF));

    // Normal case: sign | ((exp + 112) << 23) | (mant << 13)
    let normal_exp = i32x4_shl(i32x4_add(exp, i32x4_splat(112)), 23);
    let normal_mant = i32x4_shl(mant, 13);
    let normal_result = v128_or(sign, v128_or(normal_exp, normal_mant));

    // Inf/NaN case: sign | 0x7F800000 | (mant << 13)
    let inf_nan_result = v128_or(
        sign,
        v128_or(i32x4_splat(0x7F80_0000_u32 as i32), normal_mant),
    );

    // Subnormal case: magic float denormalization trick
    // Place mantissa into a float with exponent 113: (113 << 23) | (mant << 13)
    // Subtract f32::from_bits(113 << 23) to renormalize.
    // This works because the float subtraction shifts the mantissa left and adjusts
    // the exponent, effectively normalizing the subnormal.
    let magic = i32x4_splat(113_i32 << 23); // 0x38800000
    let subnorm_float = v128_or(magic, normal_mant);
    let subnorm_result = v128_or(sign, f32x4_sub(subnorm_float, magic));

    // Zero case: just sign
    let zero_result = sign;

    // Build masks
    let exp_zero = i32x4_eq(exp, i32x4_splat(0));
    let mant_zero = i32x4_eq(mant, i32x4_splat(0));
    let exp_max = i32x4_eq(exp, i32x4_splat(31));
    let is_zero = v128_and(exp_zero, mant_zero);
    let is_subnorm = v128_and(exp_zero, v128_not(mant_zero));

    // Select result: zero → zero_result, subnorm → subnorm_result,
    //                inf/nan → inf_nan_result, normal → normal_result
    let mut result = normal_result;
    result = v128_bitselect(inf_nan_result, result, exp_max);
    result = v128_bitselect(subnorm_result, result, is_subnorm);
    result = v128_bitselect(zero_result, result, is_zero);

    let mut out = [0.0f32; 4];
    v128_store(&mut out, result);
    out
}

/// Branchless f32 → f16 conversion, 4 values at a time.
///
/// Uses round-to-nearest-even for the normal case. Handles overflow to inf,
/// f32 zero/subnormal to f16 zero, f32 inf/nan, and f16 subnormal output.
/// Values too small for f16 subnormals flush to zero.
#[archmage::rite(import_intrinsics)]
fn f32_to_f16_x4(_token: Wasm128Token, f: &[f32; 4]) -> [u16; 4] {
    let bits = v128_load(f);

    // Extract f32 fields
    let f32_sign = v128_and(u32x4_shr(bits, 16), i32x4_splat(0x8000_u32 as i32));
    let f32_exp = v128_and(u32x4_shr(bits, 23), i32x4_splat(0xFF));
    let f32_mant = v128_and(bits, i32x4_splat(0x007F_FFFF_u32 as i32));

    // f16 exponent = f32 exponent - 112 (bias difference: 127 - 15)
    let f16_exp = i32x4_sub(f32_exp, i32x4_splat(112));

    // Normal case: round mantissa from 23 to 10 bits with round-to-nearest-even
    let shifted_mant = u32x4_shr(f32_mant, 13);
    let remainder = v128_and(f32_mant, i32x4_splat(0x1FFF));
    // Round up if remainder > 0x1000, or remainder == 0x1000 and shifted_mant is odd
    let half = i32x4_splat(0x1000);
    let round_up = v128_or(
        i32x4_gt(remainder, half),
        v128_and(
            i32x4_eq(remainder, half),
            i32x4_eq(v128_and(shifted_mant, i32x4_splat(1)), i32x4_splat(1)),
        ),
    );
    // round_up is a mask (all-ones or all-zeros), extract 1 or 0
    let round_bit = v128_and(round_up, i32x4_splat(1));
    let normal_result = i32x4_add(
        v128_or(f32_sign, v128_or(i32x4_shl(f16_exp, 10), shifted_mant)),
        round_bit,
    );

    // Overflow case: f16_exp >= 31 → ±Inf (0x7C00)
    let overflow_result = v128_or(f32_sign, i32x4_splat(0x7C00));

    // Inf/NaN case: f32_exp == 255
    let nan_mant = i32x4_max(u32x4_shr(f32_mant, 13), i32x4_splat(1));
    let nan_mant_or_zero = v128_bitselect(
        nan_mant,
        i32x4_splat(0),
        // mask: mant != 0
        v128_not(i32x4_eq(f32_mant, i32x4_splat(0))),
    );
    let inf_nan_result = v128_or(f32_sign, v128_or(i32x4_splat(0x7C00), nan_mant_or_zero));

    // Zero/too-small case: f32_exp == 0 or f16_exp < -10 → f16 zero
    let zero_result = f32_sign;

    // Subnormal f16 output: 0 < f16_exp <= 0 (i.e., f32_exp in [103..112])
    // Prepend implicit 1 bit, shift right by (1 - f16_exp + 13) = (14 - f16_exp)
    // We compute this per-lane via scalar fallback since variable-count shifts
    // aren't available in wasm simd128.
    // For the subnormal range we fall back to scalar since variable per-lane
    // shifts aren't available in wasm simd128.

    // Build selection masks
    let exp_is_255 = i32x4_eq(f32_exp, i32x4_splat(255));
    let exp_is_0 = i32x4_eq(f32_exp, i32x4_splat(0));
    let f16_exp_ge_31 = i32x4_gt(f16_exp, i32x4_splat(30));
    let f16_exp_le_0 = i32x4_lt(f16_exp, i32x4_splat(1));
    let too_small = i32x4_lt(f16_exp, i32x4_splat(-10));
    let is_subnorm = v128_and(f16_exp_le_0, v128_not(v128_or(exp_is_0, too_small)));

    // Start with normal result, then override special cases
    let mut result = normal_result;
    result = v128_bitselect(overflow_result, result, f16_exp_ge_31);
    result = v128_bitselect(inf_nan_result, result, exp_is_255);
    result = v128_bitselect(zero_result, result, v128_or(exp_is_0, too_small));

    // Check if any lane needs subnormal handling
    let any_subnorm = v128_any_true(is_subnorm);

    // Extract result as u16. Narrow i32x4 → i16x8, then extract 4 lanes.
    if any_subnorm {
        // Slow path: handle subnormal lanes via scalar
        let result_arr: [i32; 4] = {
            let mut a = [0i32; 4];
            v128_store(&mut a, result);
            a
        };
        let sub_mask: [i32; 4] = {
            let mut a = [0i32; 4];
            v128_store(&mut a, is_subnorm);
            a
        };
        let f_arr: [f32; 4] = {
            let mut a = [0.0f32; 4];
            v128_store(&mut a, v128_load(f));
            a
        };
        let mut out = [0u16; 4];
        for i in 0..4 {
            if sub_mask[i] != 0 {
                out[i] = super::scalar::f32_to_f16_soft(f_arr[i]);
            } else {
                out[i] = result_arr[i] as u16;
            }
        }
        out
    } else {
        // Fast path: no subnormals, just narrow and extract
        let narrow = u16x8_narrow_i32x4(result, i32x4_splat(0));
        let mut out = [0u16; 4];
        out[0] = u16x8_extract_lane::<0>(narrow);
        out[1] = u16x8_extract_lane::<1>(narrow);
        out[2] = u16x8_extract_lane::<2>(narrow);
        out[3] = u16x8_extract_lane::<3>(narrow);
        out
    }
}

/// Bulk convert f16 → f32 row using wasm128 SIMD.
/// Processes 4 f16 values at a time via branchless integer SIMD.
#[archmage::arcane]
pub(crate) fn f16_to_f32_row_wasm128(_token: Wasm128Token, input: &[u16], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());
    let len = input.len();

    let (in_chunks, _) = input.as_chunks::<4>();
    let (out_chunks, _) = output.as_chunks_mut::<4>();

    for (in_chunk, out_chunk) in in_chunks.iter().zip(out_chunks.iter_mut()) {
        *out_chunk = f16_to_f32_x4(_token, in_chunk);
    }

    // Scalar tail
    let base4 = in_chunks.len() * 4;
    for i in base4..len {
        output[i] = super::scalar::f16_to_f32_soft(input[i]);
    }
}

/// Bulk convert f32 → f16 row using wasm128 SIMD.
/// Processes 4 f32 values at a time.
#[archmage::arcane]
pub(crate) fn f32_to_f16_row_wasm128(_token: Wasm128Token, input: &[f32], output: &mut [u16]) {
    debug_assert_eq!(input.len(), output.len());
    let len = input.len();

    let (in_chunks, _) = input.as_chunks::<4>();
    let (out_chunks, _) = output.as_chunks_mut::<4>();

    for (in_chunk, out_chunk) in in_chunks.iter().zip(out_chunks.iter_mut()) {
        *out_chunk = f32_to_f16_x4(_token, in_chunk);
    }

    // Scalar tail
    let base4 = in_chunks.len() * 4;
    for i in base4..len {
        output[i] = super::scalar::f32_to_f16_soft(input[i]);
    }
}

/// Horizontal filter: f32 input → f16 (u16) output.
/// Accumulates in f32, converts to f16 on store.
#[archmage::arcane]
pub(crate) fn filter_h_row_f32_to_f16_wasm128(
    _token: Wasm128Token,
    input: &[f32],
    output: &mut [u16],
    weights: &F32WeightTable,
    channels: usize,
) {
    // H-filter accumulates in f32 and converts per output pixel.
    // The conversion is not the bottleneck here (one per output pixel),
    // so delegate to the portable implementation.
    super::wide_kernels::filter_h_row_f32_to_f16_impl_wasm128(
        _token, input, output, weights, channels,
    )
}

/// Streaming V-filter: f16 rows → f32 output via f32 weights.
///
/// This is the primary hotspot: converts f16 → f32 for every pixel in every
/// tap row, then multiplies by weight and accumulates. Vectorized to process
/// 4 pixels at a time with inline f16 → f32 conversion.
#[allow(clippy::needless_range_loop)]
#[archmage::arcane]
pub(crate) fn filter_v_row_f16_wasm128(
    _token: Wasm128Token,
    rows: &[&[u16]],
    output: &mut [f32],
    weights: &[f32],
) {
    filter_v_row_f16_simd(_token, rows, output, weights)
}

/// Vectorized V-filter inner loop: load 4 u16 f16 values, convert to f32x4
/// inline, multiply by broadcast weight, accumulate into f32 output.
#[archmage::rite(import_intrinsics)]
fn filter_v_row_f16_simd(
    _token: Wasm128Token,
    rows: &[&[u16]],
    output: &mut [f32],
    weights: &[f32],
) {
    let width = output.len();
    debug_assert_eq!(rows.len(), weights.len());

    let chunks4 = width / 4;
    let base4 = chunks4 * 4;

    // Zero the output buffer — 4 floats at a time
    let zero = f32x4_splat(0.0);
    {
        let (out_chunks, out_tail) = output.as_chunks_mut::<4>();
        for chunk in out_chunks.iter_mut() {
            v128_store(chunk, zero);
        }
        for v in out_tail.iter_mut() {
            *v = 0.0;
        }
    }

    // Constants for branchless f16 → f32 conversion
    let mask_sign = i32x4_splat(0x8000_u32 as i32);
    let mask_exp = i32x4_splat(0x1F);
    let mask_mant = i32x4_splat(0x03FF);
    let exp_bias = i32x4_splat(112);
    let zero_const = i32x4_splat(0);
    let exp_max_const = i32x4_splat(31);
    let magic = i32x4_splat(113_i32 << 23); // 0x38800000
    let inf_bits = i32x4_splat(0x7F80_0000_u32 as i32);

    // Row-major accumulation: broadcast weight once, sweep entire row
    for (row, &weight) in rows.iter().zip(weights.iter()) {
        debug_assert!(row.len() >= width);
        let w = f32x4_splat(weight);

        // SIMD: 4 pixels at a time
        let (row_chunks, _) = row[..base4].as_chunks::<4>();
        let (out_chunks, _) = output[..base4].as_chunks_mut::<4>();

        for (out_chunk, row_chunk) in out_chunks.iter_mut().zip(row_chunks.iter()) {
            // Load 4 u16 f16 values, zero-extend to i32x4
            let bits = u32x4_load_extend_u16x4(row_chunk);

            // Inline branchless f16 → f32 conversion
            let sign = i32x4_shl(v128_and(bits, mask_sign), 16);
            let exp = v128_and(u32x4_shr(bits, 10), mask_exp);
            let mant = v128_and(bits, mask_mant);

            let normal_exp = i32x4_shl(i32x4_add(exp, exp_bias), 23);
            let mant13 = i32x4_shl(mant, 13);
            let normal_result = v128_or(sign, v128_or(normal_exp, mant13));

            let inf_nan_result = v128_or(sign, v128_or(inf_bits, mant13));

            let subnorm_float = v128_or(magic, mant13);
            let subnorm_result = v128_or(sign, f32x4_sub(subnorm_float, magic));

            let exp_is_zero = i32x4_eq(exp, zero_const);
            let mant_is_zero = i32x4_eq(mant, zero_const);
            let exp_is_max = i32x4_eq(exp, exp_max_const);
            let is_zero = v128_and(exp_is_zero, mant_is_zero);
            let is_subnorm = v128_and(exp_is_zero, v128_not(mant_is_zero));

            let mut f32_val = normal_result;
            f32_val = v128_bitselect(inf_nan_result, f32_val, exp_is_max);
            f32_val = v128_bitselect(subnorm_result, f32_val, is_subnorm);
            f32_val = v128_bitselect(sign, f32_val, is_zero);

            // Multiply by weight and accumulate
            let acc = v128_load(out_chunk);
            v128_store(out_chunk, f32x4_add(acc, f32x4_mul(f32_val, w)));
        }

        // Scalar tail
        for x in base4..width {
            output[x] += super::scalar::f16_to_f32_soft(row[x]) * weight;
        }
    }
}

/// Batch V-filter: f16 intermediate → f32 output.
///
/// Vectorized with inline f16 → f32 conversion, 4 values at a time.
#[allow(clippy::needless_range_loop)]
#[archmage::arcane]
pub(crate) fn filter_v_all_f16_wasm128(
    _token: Wasm128Token,
    intermediate: &[u16],
    output: &mut [f32],
    h_row_len: usize,
    in_h: usize,
    out_h: usize,
    weights: &F32WeightTable,
) {
    filter_v_all_f16_simd(
        _token,
        intermediate,
        output,
        h_row_len,
        in_h,
        out_h,
        weights,
    )
}

/// Vectorized batch V-filter inner loop.
#[allow(clippy::needless_range_loop)]
#[allow(clippy::too_many_arguments)]
#[archmage::rite(import_intrinsics)]
fn filter_v_all_f16_simd(
    _token: Wasm128Token,
    intermediate: &[u16],
    output: &mut [f32],
    h_row_len: usize,
    in_h: usize,
    out_h: usize,
    weights: &F32WeightTable,
) {
    // Constants for branchless f16 → f32 conversion
    let mask_sign = i32x4_splat(0x8000_u32 as i32);
    let mask_exp = i32x4_splat(0x1F);
    let mask_mant = i32x4_splat(0x03FF);
    let exp_bias = i32x4_splat(112);
    let zero_const = i32x4_splat(0);
    let exp_max_const = i32x4_splat(31);
    let magic = i32x4_splat(113_i32 << 23);
    let inf_bits = i32x4_splat(0x7F80_0000_u32 as i32);
    let f32_zero = f32x4_splat(0.0);

    let chunks4 = h_row_len / 4;
    let base4 = chunks4 * 4;

    for out_y in 0..out_h {
        let left = weights.left[out_y];
        let tap_count = weights.tap_count(out_y);
        let w = weights.weights(out_y);
        let out_start = out_y * h_row_len;

        // Zero the output row — 4 floats at a time (scoped borrow)
        {
            let (out_row_chunks, out_row_tail) =
                output[out_start..out_start + h_row_len].as_chunks_mut::<4>();
            for chunk in out_row_chunks.iter_mut() {
                v128_store(chunk, f32_zero);
            }
            for v in out_row_tail.iter_mut() {
                *v = 0.0;
            }
        }

        for (t, &weight) in w[..tap_count].iter().enumerate() {
            let in_y = (left + t as i32).clamp(0, in_h as i32 - 1) as usize;
            let row_start = in_y * h_row_len;
            let wv = f32x4_splat(weight);

            // SIMD: 4 pixels at a time (scoped borrow)
            {
                let (row_chunks, _) = intermediate[row_start..row_start + base4].as_chunks::<4>();
                let (out_chunks_4, _) = output[out_start..out_start + base4].as_chunks_mut::<4>();

                for (row_chunk, out_chunk) in row_chunks.iter().zip(out_chunks_4.iter_mut()) {
                    // Load 4 u16 f16 values, zero-extend to i32x4
                    let bits = u32x4_load_extend_u16x4(row_chunk);

                    // Inline branchless f16 → f32
                    let sign = i32x4_shl(v128_and(bits, mask_sign), 16);
                    let exp = v128_and(u32x4_shr(bits, 10), mask_exp);
                    let mant = v128_and(bits, mask_mant);

                    let normal_exp = i32x4_shl(i32x4_add(exp, exp_bias), 23);
                    let mant13 = i32x4_shl(mant, 13);
                    let normal_result = v128_or(sign, v128_or(normal_exp, mant13));

                    let inf_nan_result = v128_or(sign, v128_or(inf_bits, mant13));

                    let subnorm_float = v128_or(magic, mant13);
                    let subnorm_result = v128_or(sign, f32x4_sub(subnorm_float, magic));

                    let exp_is_zero = i32x4_eq(exp, zero_const);
                    let mant_is_zero = i32x4_eq(mant, zero_const);
                    let exp_is_max = i32x4_eq(exp, exp_max_const);
                    let is_zero = v128_and(exp_is_zero, mant_is_zero);
                    let is_subnorm = v128_and(exp_is_zero, v128_not(mant_is_zero));

                    let mut f32_val = normal_result;
                    f32_val = v128_bitselect(inf_nan_result, f32_val, exp_is_max);
                    f32_val = v128_bitselect(subnorm_result, f32_val, is_subnorm);
                    f32_val = v128_bitselect(sign, f32_val, is_zero);

                    // Multiply by weight and accumulate
                    let acc = v128_load(out_chunk);
                    v128_store(out_chunk, f32x4_add(acc, f32x4_mul(f32_val, wv)));
                }
            }

            // Scalar tail
            for x in base4..h_row_len {
                output[out_start + x] +=
                    super::scalar::f16_to_f32_soft(intermediate[row_start + x]) * weight;
            }
        }
    }
}

// Transfer function batch processors — wrap linear-srgb rites via closures.

use magetypes::simd::generic::f32x4;

macro_rules! tf_wasm {
    ($name_wasm:ident, $rite_fn:path, $scalar_fn:path) => {
        #[archmage::arcane]
        pub(crate) fn $name_wasm(
            _token: Wasm128Token,
            row: &mut [f32],
            channels: usize,
            has_alpha: bool,
        ) {
            super::tf_portable::tf_row_inplace(
                _token,
                row,
                channels,
                has_alpha,
                |t, v: f32x4<Wasm128Token>| f32x4::from_array(t, $rite_fn(t, v.to_array())),
                $scalar_fn,
            );
        }
    };
}

tf_wasm!(
    srgb_to_linear_row_wasm128,
    linear_srgb::tokens::x4::tf_srgb_to_linear_wasm128,
    linear_srgb::tf::srgb_to_linear
);
tf_wasm!(
    srgb_from_linear_row_wasm128,
    linear_srgb::tokens::x4::tf_linear_to_srgb_wasm128,
    linear_srgb::tf::linear_to_srgb
);
tf_wasm!(
    bt709_to_linear_row_wasm128,
    linear_srgb::tokens::x4::bt709_to_linear_wasm128,
    linear_srgb::tf::bt709_to_linear
);
tf_wasm!(
    bt709_from_linear_row_wasm128,
    linear_srgb::tokens::x4::linear_to_bt709_wasm128,
    linear_srgb::tf::linear_to_bt709
);
tf_wasm!(
    pq_to_linear_row_wasm128,
    linear_srgb::tokens::x4::pq_to_linear_wasm128,
    linear_srgb::tf::pq_to_linear
);
tf_wasm!(
    pq_from_linear_row_wasm128,
    linear_srgb::tokens::x4::linear_to_pq_wasm128,
    linear_srgb::tf::linear_to_pq
);
tf_wasm!(
    hlg_to_linear_row_wasm128,
    linear_srgb::tokens::x4::hlg_to_linear_wasm128,
    linear_srgb::tf::hlg_to_linear
);
tf_wasm!(
    hlg_from_linear_row_wasm128,
    linear_srgb::tokens::x4::linear_to_hlg_wasm128,
    linear_srgb::tf::linear_to_hlg
);

#[archmage::arcane]
pub(crate) fn srgb_u8_to_linear_f32_wasm128(
    _token: Wasm128Token,
    input: &[u8],
    output: &mut [f32],
    channels: usize,
    has_alpha: bool,
) {
    crate::color::srgb_u8_to_linear_f32_impl(input, output, channels, has_alpha);
}

#[archmage::arcane]
pub(crate) fn linear_f32_to_srgb_u8_wasm128(
    _token: Wasm128Token,
    input: &[f32],
    output: &mut [u8],
    channels: usize,
    has_alpha: bool,
) {
    crate::color::linear_f32_to_srgb_u8_impl(input, output, channels, has_alpha);
}
