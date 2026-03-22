//! x86-64 AVX2+FMA convolution and conversion kernels.
// Range loops in SIMD kernels index into multiple arrays (weights, pixels, chunks)
// simultaneously. Iterator refactoring would hurt readability and risk codegen regressions.
#![allow(clippy::needless_range_loop)]
// 4-row batch functions naturally have many parameters (4 in + 4 out + weights + token).
#![allow(clippy::too_many_arguments)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::fastmath;
use crate::proven::{idx, idx_mut, sub};
use crate::weights::{F32WeightTable, I16_PRECISION, I16WeightTable};
use archmage::X64V3Token;
#[cfg(feature = "avx512")]
use archmage::X64V4Token;

// Safe unaligned SIMD load/store — takes references instead of raw pointers.
// Explicit imports because names overlap with core::arch intrinsics.
#[cfg(target_arch = "x86_64")]
use safe_unaligned_simd::x86_64::{
    _mm_loadu_ps, _mm_loadu_si32, _mm_loadu_si64, _mm_loadu_si128, _mm_storeu_ps, _mm_storeu_si64,
    _mm_storeu_si128, _mm256_loadu_ps, _mm256_loadu_si256, _mm256_storeu_ps,
};
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
use safe_unaligned_simd::x86_64::{_mm256_storeu_si256, _mm512_loadu_si512, _mm512_storeu_si512};

// =============================================================================
// Color conversion kernels
// =============================================================================

/// Convert u8 → f32 (÷ 255) using AVX2.
/// Processes 8 bytes at a time: load 8 u8 → zero-extend to 8×i32 → cvt to 8×f32 → mul by 1/255.
#[archmage::arcane]
pub(crate) fn u8_to_f32_row_v3(_token: X64V3Token, input: &[u8], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());
    let len = input.len();
    let scale = _mm256_set1_ps(1.0 / 255.0);

    let (in_chunks, _) = input.as_chunks::<8>();
    let (out_chunks, _) = output.as_chunks_mut::<8>();

    for (in_chunk, out_chunk) in in_chunks.iter().zip(out_chunks.iter_mut()) {
        let bytes = _mm_loadu_si64(in_chunk);
        let ints = _mm256_cvtepu8_epi32(bytes);
        let floats = _mm256_cvtepi32_ps(ints);
        let result = _mm256_mul_ps(floats, scale);
        _mm256_storeu_ps(out_chunk, result);
    }

    // Scalar tail
    let chunks8 = in_chunks.len();
    for i in (chunks8 * 8)..len {
        output[i] = input[i] as f32 * (1.0 / 255.0);
    }
}

/// Convert f32 → u8 (× 255 + 0.5, clamp) using AVX2.
#[archmage::arcane]
pub(crate) fn f32_to_u8_row_v3(_token: X64V3Token, input: &[f32], output: &mut [u8]) {
    debug_assert_eq!(input.len(), output.len());
    let len = input.len();
    let scale = _mm256_set1_ps(255.0);
    let half = _mm256_set1_ps(0.5);
    let zero = _mm256_setzero_ps();
    let max_val = _mm256_set1_ps(255.0);

    let (in_chunks, _) = input.as_chunks::<8>();
    let (out_chunks, _) = output.as_chunks_mut::<8>();

    for (in_chunk, out_chunk) in in_chunks.iter().zip(out_chunks.iter_mut()) {
        let floats = _mm256_loadu_ps(in_chunk);
        let scaled = _mm256_fmadd_ps(floats, scale, half);
        let clamped = _mm256_min_ps(_mm256_max_ps(scaled, zero), max_val);
        let ints = _mm256_cvttps_epi32(clamped);
        // Pack 8×i32 → 8×u8 via two-stage pack
        let packed16 = _mm256_packus_epi32(ints, ints);
        let packed8 = _mm256_packus_epi16(packed16, packed16);
        // AVX2 pack works in 128-bit lanes. Extract bytes from each lane.
        let lo = _mm256_extracti128_si256::<0>(packed8);
        let hi = _mm256_extracti128_si256::<1>(packed8);
        let lo_val = _mm_cvtsi128_si32(lo) as u32;
        let hi_val = _mm_cvtsi128_si32(hi) as u32;
        out_chunk[..4].copy_from_slice(&lo_val.to_ne_bytes());
        out_chunk[4..8].copy_from_slice(&hi_val.to_ne_bytes());
    }

    // Scalar tail
    let chunks8 = in_chunks.len();
    for i in (chunks8 * 8)..len {
        output[i] = (input[i] * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
    }
}

/// Premultiply alpha in-place using AVX2 (processes 2 RGBA pixels = 8 floats at a time).
#[archmage::arcane]
pub(crate) fn premultiply_alpha_row_v3(_token: X64V3Token, row: &mut [f32]) {
    let (chunks, tail) = row.as_chunks_mut::<8>();

    for chunk in chunks.iter_mut() {
        let px = _mm256_loadu_ps(chunk);
        let alpha = _mm256_permutevar8x32_ps(px, _mm256_set_epi32(7, 7, 7, 7, 3, 3, 3, 3));
        let result = _mm256_mul_ps(px, alpha);
        let mask = _mm256_blend_ps::<0b10001000>(result, px);
        _mm256_storeu_ps(chunk, mask);
    }
    // Scalar tail for remaining pixels
    for pixel in tail.chunks_exact_mut(4) {
        let a = pixel[3];
        pixel[0] *= a;
        pixel[1] *= a;
        pixel[2] *= a;
    }
}

/// Unpremultiply alpha in-place using SSE (1 pixel at a time due to division).
#[archmage::arcane]
pub(crate) fn unpremultiply_alpha_row_v3(_token: X64V3Token, row: &mut [f32]) {
    let threshold = _mm_set1_ps(1.0 / 1024.0);
    let one = _mm_set1_ps(1.0);

    let (chunks, _) = row.as_chunks_mut::<4>();

    for chunk in chunks.iter_mut() {
        let px = _mm_loadu_ps(chunk);
        let alpha = _mm_shuffle_ps::<0xFF>(px, px);
        let mask = _mm_cmpgt_ps(alpha, threshold);
        let inv_alpha = _mm_div_ps(one, alpha);
        let inv_alpha_masked = _mm_and_ps(inv_alpha, mask);
        let unpremul = _mm_mul_ps(px, inv_alpha_masked);
        let result = _mm_blend_ps::<0b1000>(unpremul, px);
        _mm_storeu_ps(chunk, result);
    }
}

/// Horizontal convolution using AVX2+FMA.
///
/// For each output pixel, accumulate weighted input pixels.
/// Dispatches to specialized channel-count implementations.
#[archmage::arcane]
pub(crate) fn filter_h_row_f32_v3(
    _token: X64V3Token,
    input: &[f32],
    output: &mut [f32],
    weights: &F32WeightTable,
    channels: usize,
) {
    match channels {
        4 => filter_h_4ch(_token, input, output, weights),
        3 => filter_h_3ch(_token, input, output, weights),
        _ => filter_h_generic(_token, input, output, weights, channels),
    }
}

/// Horizontal filter for 4-channel (RGBA) data using AVX2 256-bit.
///
/// Each AVX2 accumulator processes 2 taps simultaneously: the lower 128 bits
/// accumulate `pixel[t] * weight[t]`, the upper 128 bits accumulate
/// `pixel[t+1] * weight[t+1]`. Four accumulators process 8 taps per iteration.
///
/// Weight broadcasting uses `vpermps` (cross-lane permute) to extract per-tap
/// broadcasts from a single 8-weight load, avoiding individual `vbroadcastss`
/// + `vinsertf128` overhead.
///
/// The final reduction adds upper 128 bits to lower 128 bits across all
/// accumulators, collapsing into one RGBA pixel result.
///
/// Remaining taps (0..7) are handled by SSE 128-bit (1 tap at a time).
#[archmage::rite]
fn filter_h_4ch(_token: X64V3Token, input: &[f32], output: &mut [f32], weights: &F32WeightTable) {
    let out_width = weights.len();
    let max_taps = weights.max_taps;

    // View input as per-pixel [f32; 4] chunks (for SSE remainder loop).
    let in_pixels_arr: &[[f32; 4]] = input.as_chunks().0;
    let (out_pixels, _) = output.as_chunks_mut::<4>();

    let chunks8 = max_taps / 8;
    let remainder = max_taps - chunks8 * 8;

    // Permutation indices for broadcasting weights from a single 8-weight load.
    // vpermps: each element index selects from the 8-element source.
    // _mm256_set_epi32(e7, e6, e5, e4, e3, e2, e1, e0) — high to low.
    let perm01 = _mm256_set_epi32(1, 1, 1, 1, 0, 0, 0, 0);
    let perm23 = _mm256_set_epi32(3, 3, 3, 3, 2, 2, 2, 2);
    let perm45 = _mm256_set_epi32(5, 5, 5, 5, 4, 4, 4, 4);
    let perm67 = _mm256_set_epi32(7, 7, 7, 7, 6, 6, 6, 6);

    for out_x in 0..out_width {
        let left = weights.left[out_x] as usize;
        let w = weights.weights_padded(out_x);

        // Pre-slice input and weights for AVX2 main loop.
        // Input window: max_taps pixels × 4 channels = max_taps * 4 f32.
        // Pairs: 2 consecutive pixels = 8 f32 per pair.
        let flat_start = left * 4;
        let input_window = &input[flat_start..flat_start + max_taps * 4];
        let (pairs, _) = input_window.as_chunks::<8>();
        let (w_chunks, _) = w.as_chunks::<8>();

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        for c in 0..chunks8 {
            // Load 8 weights at once, permute to per-tap broadcasts
            let w_vec = _mm256_loadu_ps(idx(w_chunks, c));
            let w01 = _mm256_permutevar8x32_ps(w_vec, perm01);
            let w23 = _mm256_permutevar8x32_ps(w_vec, perm23);
            let w45 = _mm256_permutevar8x32_ps(w_vec, perm45);
            let w67 = _mm256_permutevar8x32_ps(w_vec, perm67);

            // Load pairs of pixels (8 f32 = 2 RGBA pixels each)
            let pi = c * 4;
            let p01 = _mm256_loadu_ps(idx(pairs, pi));
            let p23 = _mm256_loadu_ps(idx(pairs, pi + 1));
            let p45 = _mm256_loadu_ps(idx(pairs, pi + 2));
            let p67 = _mm256_loadu_ps(idx(pairs, pi + 3));

            acc0 = _mm256_fmadd_ps(p01, w01, acc0);
            acc1 = _mm256_fmadd_ps(p23, w23, acc1);
            acc2 = _mm256_fmadd_ps(p45, w45, acc2);
            acc3 = _mm256_fmadd_ps(p67, w67, acc3);
        }

        // Reduce 256-bit accumulators: add upper 128 to lower 128
        let sum01 = _mm256_add_ps(acc0, acc1);
        let sum23 = _mm256_add_ps(acc2, acc3);
        let sum = _mm256_add_ps(sum01, sum23);
        let lo = _mm256_castps256_ps128(sum);
        let hi = _mm256_extractf128_ps::<1>(sum);
        let mut acc_128 = _mm_add_ps(lo, hi);

        // SSE remainder for leftover taps (0..7)
        let t_start = chunks8 * 8;
        for t in 0..remainder {
            let tt = t_start + t;
            let w_val = _mm_set1_ps(*idx(w, tt));
            let pixel = _mm_loadu_ps(idx(in_pixels_arr, left + tt));
            acc_128 = _mm_fmadd_ps(pixel, w_val, acc_128);
        }

        _mm_storeu_ps(idx_mut(out_pixels, out_x), acc_128);
    }
}

/// Horizontal filter for 3-channel (RGB) data.
/// Scalar with compiler auto-vectorization.
#[archmage::rite]
fn filter_h_3ch(_token: X64V3Token, input: &[f32], output: &mut [f32], weights: &F32WeightTable) {
    let out_width = weights.len();

    for out_x in 0..out_width {
        let left = weights.left[out_x] as usize;
        let w = weights.weights(out_x);
        let out_offset = out_x * 3;

        let mut acc0 = 0.0f32;
        let mut acc1 = 0.0f32;
        let mut acc2 = 0.0f32;

        for (t, &weight) in w.iter().enumerate() {
            let in_offset = (left + t) * 3;
            acc0 += input[in_offset] * weight;
            acc1 += input[in_offset + 1] * weight;
            acc2 += input[in_offset + 2] * weight;
        }

        output[out_offset] = acc0;
        output[out_offset + 1] = acc1;
        output[out_offset + 2] = acc2;
    }
}

/// Horizontal filter for arbitrary channel count.
#[archmage::rite]
fn filter_h_generic(
    _token: X64V3Token,
    input: &[f32],
    output: &mut [f32],
    weights: &F32WeightTable,
    channels: usize,
) {
    let out_width = weights.len();

    for out_x in 0..out_width {
        let left = weights.left[out_x] as usize;
        let w = weights.weights(out_x);
        let out_offset = out_x * channels;

        for c in 0..channels {
            output[out_offset + c] = 0.0;
        }

        for (t, &weight) in w.iter().enumerate() {
            let in_offset = (left + t) * channels;
            for c in 0..channels {
                output[out_offset + c] += input[in_offset + c] * weight;
            }
        }
    }
}

/// Vertical convolution using AVX2+FMA.
///
/// Row-major loop order: outer loop over rows (amortize weight broadcast),
/// inner loop over width in blocks of 4×8 floats for instruction-level parallelism.
#[archmage::arcane]
pub(crate) fn filter_v_row_f32_v3(
    _token: X64V3Token,
    rows: &[&[f32]],
    output: &mut [f32],
    weights: &[f32],
) {
    let width = output.len();
    debug_assert_eq!(rows.len(), weights.len());

    let (out_chunks, out_tail) = output.as_chunks_mut::<8>();

    // Zero the output buffer using AVX2
    let zero = _mm256_setzero_ps();
    for chunk in out_chunks.iter_mut() {
        _mm256_storeu_ps(chunk, zero);
    }
    for v in out_tail.iter_mut() {
        *v = 0.0;
    }

    let base8 = out_chunks.len() * 8;

    // Row-major accumulation: broadcast weight once, sweep entire row
    for (row, &weight) in rows.iter().zip(weights.iter()) {
        let w = _mm256_set1_ps(weight);
        let (row_chunks, _) = row.as_chunks::<8>();

        // Process 32 floats (4×8) at a time for ILP
        let (out_blocks, out_rem) = output[..base8].as_chunks_mut::<32>();

        for (out_block, row_block) in out_blocks.iter_mut().zip(row_chunks.chunks_exact(4)) {
            let s0 = _mm256_loadu_ps(&row_block[0]);
            let s1 = _mm256_loadu_ps(&row_block[1]);
            let s2 = _mm256_loadu_ps(&row_block[2]);
            let s3 = _mm256_loadu_ps(&row_block[3]);

            let oc: &mut [[f32; 8]] = out_block.as_mut_slice().as_chunks_mut().0;
            let a0 = _mm256_loadu_ps(&oc[0]);
            let a1 = _mm256_loadu_ps(&oc[1]);
            let a2 = _mm256_loadu_ps(&oc[2]);
            let a3 = _mm256_loadu_ps(&oc[3]);

            _mm256_storeu_ps(&mut oc[0], _mm256_fmadd_ps(s0, w, a0));
            _mm256_storeu_ps(&mut oc[1], _mm256_fmadd_ps(s1, w, a1));
            _mm256_storeu_ps(&mut oc[2], _mm256_fmadd_ps(s2, w, a2));
            _mm256_storeu_ps(&mut oc[3], _mm256_fmadd_ps(s3, w, a3));
        }

        // Remaining 8-float chunks (0..3 of them)
        let blocks4 = out_blocks.len();
        let rem_row_chunks = &row_chunks[blocks4 * 4..];
        let (rem_out_chunks, _) = out_rem.as_chunks_mut::<8>();
        for (out_chunk, row_chunk) in rem_out_chunks.iter_mut().zip(rem_row_chunks.iter()) {
            let src = _mm256_loadu_ps(row_chunk);
            let acc = _mm256_loadu_ps(out_chunk);
            _mm256_storeu_ps(out_chunk, _mm256_fmadd_ps(src, w, acc));
        }

        // Scalar tail
        let w_scalar = weight;
        for x in base8..width {
            output[x] += row[x] * w_scalar;
        }
    }
}

// =============================================================================
// u8 alpha premultiply / unpremultiply
// =============================================================================

/// Premultiply alpha on RGBA u8 row using SSE4.1.
///
/// For each pixel: `C' = (C * A + 127) / 255`, alpha preserved.
/// Processes 2 pixels (8 bytes) at a time using i16 multiplication.
#[archmage::arcane]
pub(crate) fn premultiply_u8_row_v3(_token: X64V3Token, input: &[u8], output: &mut [u8]) {
    debug_assert_eq!(input.len(), output.len());

    let alpha_bcast = _mm_set_epi8(15, 14, 15, 14, 15, 14, 15, 14, 7, 6, 7, 6, 7, 6, 7, 6);
    let alpha_blend = _mm_set_epi16(-1, 0, 0, 0, -1, 0, 0, 0);
    let bias = _mm_set1_epi16(127);

    let (in_chunks, _) = input.as_chunks::<8>();
    let (out_chunks, _) = output.as_chunks_mut::<8>();

    for (in_chunk, out_chunk) in in_chunks.iter().zip(out_chunks.iter_mut()) {
        let bytes = _mm_loadu_si64(in_chunk);
        let ext = _mm_cvtepu8_epi16(bytes);
        let alpha = _mm_shuffle_epi8(ext, alpha_bcast);
        let product = _mm_mullo_epi16(ext, alpha);
        let biased = _mm_add_epi16(product, bias);
        let approx = _mm_srli_epi16::<8>(_mm_add_epi16(biased, _mm_srli_epi16::<8>(biased)));
        let result = _mm_blendv_epi8(approx, ext, alpha_blend);
        let packed = _mm_packus_epi16(result, result);
        _mm_storeu_si64(out_chunk, packed);
    }

    // Scalar tail
    let tail = in_chunks.len() * 8;
    for pixel in input[tail..]
        .chunks_exact(4)
        .zip(output[tail..].chunks_exact_mut(4))
    {
        let (inp, out) = pixel;
        let a = inp[3] as u16;
        out[0] = ((inp[0] as u16 * a + 127) / 255) as u8;
        out[1] = ((inp[1] as u16 * a + 127) / 255) as u8;
        out[2] = ((inp[2] as u16 * a + 127) / 255) as u8;
        out[3] = inp[3];
    }
}

/// Unpremultiply alpha in-place on RGBA u8 row using SSE4.1.
///
/// For each pixel: `C = min(C' * 255 / A, 255)` where A > 0.
/// Uses float reciprocal (_mm_rcp_ps) for throughput.
#[archmage::arcane]
pub(crate) fn unpremultiply_u8_row_v3(_token: X64V3Token, row: &mut [u8]) {
    let scale = _mm_set1_ps(255.0);
    let zero_f = _mm_setzero_ps();
    let max_val = _mm_set1_ps(255.0);
    let half = _mm_set1_ps(0.5);

    let (chunks, _) = row.as_chunks_mut::<4>();

    for chunk in chunks.iter_mut() {
        let a = chunk[3];
        if a == 0 {
            chunk[0] = 0;
            chunk[1] = 0;
            chunk[2] = 0;
            continue;
        }
        if a == 255 {
            continue;
        }

        let bytes = _mm_loadu_si32(chunk);
        let ext = _mm_cvtepu8_epi32(bytes);
        let fpixel = _mm_cvtepi32_ps(ext);

        let fa = _mm_set1_ps(a as f32);
        let inv_a = _mm_rcp_ps(fa);
        let refined = _mm_mul_ps(inv_a, _mm_sub_ps(_mm_set1_ps(2.0), _mm_mul_ps(fa, inv_a)));
        let result = _mm_add_ps(_mm_mul_ps(_mm_mul_ps(fpixel, scale), refined), half);
        let clamped = _mm_min_ps(_mm_max_ps(result, zero_f), max_val);

        let ints = _mm_cvttps_epi32(clamped);
        let packed16 = _mm_packs_epi32(ints, ints);
        let packed8 = _mm_packus_epi16(packed16, packed16);
        let val = _mm_cvtsi128_si32(packed8) as u32;

        let val_with_alpha = (val & 0x00FF_FFFF) | ((a as u32) << 24);
        chunk.copy_from_slice(&val_with_alpha.to_ne_bytes());
    }
}

// =============================================================================
// Integer convolution kernels (sRGB fast path)
// =============================================================================

/// Integer horizontal convolution: u8 input → u8 output via i32 accumulator.
///
/// Uses `_mm_madd_epi16` to process 2 taps per instruction: loads a pair of
/// RGBA u8 pixels, shuffles channels into pairs, and multiplies by paired i16
/// weights with implicit i32 accumulation.
///
/// For sRGB-space resize only (no linearization needed).
#[archmage::arcane]
pub(crate) fn filter_h_u8_i16_v3(
    _token: X64V3Token,
    input: &[u8],
    output: &mut [u8],
    weights: &I16WeightTable,
    channels: usize,
) {
    match channels {
        4 => filter_h_u8_4ch(_token, input, output, weights),
        _ => filter_h_u8_generic(_token, input, output, weights, channels),
    }
}

/// Integer horizontal filter for 4-channel (RGBA) u8 data.
///
/// Uses 256-bit (ymm) registers to process 4 taps per madd instruction.
/// Loads pre-expanded weights from the weight table (computed once at table
/// construction) to eliminate per-pixel broadcasts.
///
/// Each ymm holds 2 pairs of taps (lo lane: taps i,i+1; hi lane: taps i+2,i+3)
/// and one madd processes all 4 taps for all 4 channels simultaneously.
///
/// Fixed `groups4` loop count (from weight table) enables LLVM unrolling.
/// The input slice should include padding beyond the actual pixel data
/// (from adjacent rows) so all SIMD reads are in bounds.
/// Edge pixels where padding is insufficient fall back to scalar.
#[archmage::rite]
fn filter_h_u8_4ch(_token: X64V3Token, input: &[u8], output: &mut [u8], weights: &I16WeightTable) {
    let out_width = weights.len();
    let groups4 = weights.groups4;
    let in_pixels = input.len() / 4;

    // Need at least 4 pixels for the 16-byte loads
    if in_pixels < 4 {
        filter_h_u8_generic(_token, input, output, weights, 4);
        return;
    }

    // Check if the input slice has enough padding for all SIMD reads.
    // The +3 accounts for the 16-byte (4-pixel) SIMD load width: the last
    // group's load at pixel max_left + (groups4-1)*4 reads 4 pixels forward.
    let max_left = weights.left.iter().map(|&l| l as usize).max().unwrap_or(0);
    let has_full_padding = max_left + groups4 * 4 + 3 < in_pixels;

    if !has_full_padding {
        // No padding: per-pixel edge check
        filter_h_u8_4ch_with_edge_fallback(_token, input, output, weights);
        return;
    }

    let ymm_shuffle = _mm256_set_epi8(
        15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0, 15, 14, 7, 6, 13, 12, 5, 4, 11, 10,
        3, 2, 9, 8, 1, 0,
    );

    let half = _mm_set1_epi32(1 << (I16_PRECISION - 1));
    let zero = _mm_setzero_si128();

    // Weight chunks: contiguous stride-16 i16 array, perfect for as_chunks.
    let ew_all = weights.expanded_4ch_all();
    let (ew_chunks, _) = ew_all.as_chunks::<16>();

    // Output: contiguous stride-4 u8 pixels.
    let (out_pixels, _) = output.as_chunks_mut::<4>();

    for out_x in 0..out_width {
        let left = weights.left[out_x] as usize;
        let byte_start = left * 4;

        // Pre-slice input window and weight window: bounds proven by
        // the `has_full_padding` guard above.
        let input_window = sub(input, byte_start..byte_start + groups4 * 16);
        let (in_chunks, _) = input_window.as_chunks::<16>();
        let ew_base = out_x * groups4;
        let ew_window = sub(ew_chunks, ew_base..ew_base + groups4);

        let mut acc = _mm256_setzero_si256();

        for (chunk, ew) in in_chunks.iter().zip(ew_window.iter()) {
            let pixels = _mm_loadu_si128(chunk);
            let ext = _mm256_cvtepu8_epi16(pixels);
            let shuffled = _mm256_shuffle_epi8(ext, ymm_shuffle);
            let w = _mm256_loadu_si256(ew);
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(shuffled, w));
        }

        let lo = _mm256_castsi256_si128(acc);
        let hi = _mm256_extracti128_si256::<1>(acc);
        let final_acc = _mm_add_epi32(lo, hi);

        let rounded = _mm_add_epi32(final_acc, half);
        let shifted = _mm_srai_epi32::<{ I16_PRECISION }>(rounded);
        let packed16 = _mm_packs_epi32(shifted, zero);
        let packed8 = _mm_packus_epi16(packed16, zero);

        let pixel_val = _mm_cvtsi128_si32(packed8) as u32;
        out_pixels[out_x].copy_from_slice(&pixel_val.to_ne_bytes());
    }
}

/// Fallback for filter_h_u8_4ch when input lacks SIMD padding.
/// Uses per-pixel edge checks for safety.
#[archmage::rite]
fn filter_h_u8_4ch_with_edge_fallback(
    _token: X64V3Token,
    input: &[u8],
    output: &mut [u8],
    weights: &I16WeightTable,
) {
    let out_width = weights.len();
    let groups4 = weights.groups4;
    let in_pixels = input.len() / 4;

    let ymm_shuffle = _mm256_set_epi8(
        15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0, 15, 14, 7, 6, 13, 12, 5, 4, 11, 10,
        3, 2, 9, 8, 1, 0,
    );
    let half = _mm_set1_epi32(1 << (I16_PRECISION - 1));
    let zero = _mm_setzero_si128();

    // Find the contiguous range of output pixels where SIMD reads are in-bounds.
    // Scan forward to find first failure — for monotonic left values, this gives
    // the exact boundary. For any non-monotonic edge cases, it's conservative.
    let safe_end = (0..out_width)
        .position(|x| (weights.left[x] as usize) + groups4 * 4 > in_pixels)
        .unwrap_or(out_width);

    // Edge pixels: scalar
    for out_x in safe_end..out_width {
        let left = weights.left[out_x] as usize;
        filter_h_u8_4ch_edge(
            _token,
            input,
            &mut output[out_x * 4..],
            out_x,
            left,
            weights,
        );
    }

    // Interior pixels: SIMD
    if safe_end > 0 {
        let (out_pixels, _) = output.as_chunks_mut::<4>();

        for out_x in 0..safe_end {
            let left = weights.left[out_x] as usize;
            let byte_start = left * 4;

            let ew = weights.weights_expanded_4ch(out_x);
            let (ew_chunks, _) = ew.as_chunks::<16>();

            // Pre-slice input window: ONE bounds check per pixel.
            let input_window = &input[byte_start..byte_start + groups4 * 16];
            let (in_chunks, _) = input_window.as_chunks::<16>();

            let mut acc = _mm256_setzero_si256();

            for (chunk, ew) in in_chunks.iter().zip(ew_chunks.iter()) {
                let pixels = _mm_loadu_si128(chunk);
                let ext = _mm256_cvtepu8_epi16(pixels);
                let shuffled = _mm256_shuffle_epi8(ext, ymm_shuffle);
                let w = _mm256_loadu_si256(ew);
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(shuffled, w));
            }

            let lo = _mm256_castsi256_si128(acc);
            let hi = _mm256_extracti128_si256::<1>(acc);
            let final_acc = _mm_add_epi32(lo, hi);

            let rounded = _mm_add_epi32(final_acc, half);
            let shifted = _mm_srai_epi32::<{ I16_PRECISION }>(rounded);
            let packed16 = _mm_packs_epi32(shifted, zero);
            let packed8 = _mm_packus_epi16(packed16, zero);

            let pixel_val = _mm_cvtsi128_si32(packed8) as u32;
            out_pixels[out_x].copy_from_slice(&pixel_val.to_ne_bytes());
        }
    }
}

/// 4-row batch horizontal filter for 4-channel (RGBA) u8 data.
///
/// Processes 4 input rows simultaneously with shared weight computation.
/// Amortizes outer loop overhead (left offset lookup, weight pointer setup)
/// across 4 rows. Each row gets its own accumulator chain.
#[archmage::arcane]
pub(crate) fn filter_h_u8_i16_4rows_v3(
    _token: X64V3Token,
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
    filter_h_u8_4ch_4rows(_token, in0, in1, in2, in3, out0, out1, out2, out3, weights);
}

/// Inner implementation of 4-row batch horizontal convolution.
///
/// Fixed `groups4` loop count (from weight table) enables LLVM unrolling.
/// Input slices should include SIMD padding from adjacent row data.
/// Falls back to single-row kernel when padding is insufficient.
#[archmage::rite]
fn filter_h_u8_4ch_4rows(
    _token: X64V3Token,
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
    let out_width = weights.len();
    let groups4 = weights.groups4;
    let in_pixels = [in0.len(), in1.len(), in2.len(), in3.len()]
        .iter()
        .copied()
        .min()
        .unwrap()
        / 4;

    // Check if all rows have padding for SIMD reads.
    // +3 accounts for 16-byte (4-pixel) SIMD load width.
    let max_left = weights.left.iter().map(|&l| l as usize).max().unwrap_or(0);
    if max_left + groups4 * 4 + 3 >= in_pixels || in_pixels < 4 {
        // No padding: process rows individually (handles edge fallback internally)
        filter_h_u8_4ch(_token, in0, out0, weights);
        filter_h_u8_4ch(_token, in1, out1, weights);
        filter_h_u8_4ch(_token, in2, out2, weights);
        filter_h_u8_4ch(_token, in3, out3, weights);
        return;
    }

    let ymm_shuffle = _mm256_set_epi8(
        15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0, 15, 14, 7, 6, 13, 12, 5, 4, 11, 10,
        3, 2, 9, 8, 1, 0,
    );
    let half = _mm_set1_epi32(1 << (I16_PRECISION - 1));

    // Weight chunks: contiguous stride-16 i16 array.
    let ew_all = weights.expanded_4ch_all();
    let (ew_chunks, _) = ew_all.as_chunks::<16>();

    // Output: contiguous stride-4 u8 pixels per row.
    let (op0, _) = out0.as_chunks_mut::<4>();
    let (op1, _) = out1.as_chunks_mut::<4>();
    let (op2, _) = out2.as_chunks_mut::<4>();
    let (op3, _) = out3.as_chunks_mut::<4>();

    for out_x in 0..out_width {
        let left = weights.left[out_x] as usize;
        let byte_start = left * 4;
        let byte_end = byte_start + groups4 * 16;
        let ew_base = out_x * groups4;

        // Pre-slice all 4 input windows + weight window: bounds proven by
        // the `max_left + groups4*4 + 3 < in_pixels` guard above.
        let (c0, _) = sub(in0, byte_start..byte_end).as_chunks::<16>();
        let (c1, _) = sub(in1, byte_start..byte_end).as_chunks::<16>();
        let (c2, _) = sub(in2, byte_start..byte_end).as_chunks::<16>();
        let (c3, _) = sub(in3, byte_start..byte_end).as_chunks::<16>();
        let ew_window = sub(ew_chunks, ew_base..ew_base + groups4);

        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        let mut acc2 = _mm256_setzero_si256();
        let mut acc3 = _mm256_setzero_si256();

        for g in 0..groups4 {
            let w = _mm256_loadu_si256(idx(ew_window, g));

            let p0 = _mm256_cvtepu8_epi16(_mm_loadu_si128(idx(c0, g)));
            acc0 = _mm256_add_epi32(
                acc0,
                _mm256_madd_epi16(_mm256_shuffle_epi8(p0, ymm_shuffle), w),
            );

            let p1 = _mm256_cvtepu8_epi16(_mm_loadu_si128(idx(c1, g)));
            acc1 = _mm256_add_epi32(
                acc1,
                _mm256_madd_epi16(_mm256_shuffle_epi8(p1, ymm_shuffle), w),
            );

            let p2 = _mm256_cvtepu8_epi16(_mm_loadu_si128(idx(c2, g)));
            acc2 = _mm256_add_epi32(
                acc2,
                _mm256_madd_epi16(_mm256_shuffle_epi8(p2, ymm_shuffle), w),
            );

            let p3 = _mm256_cvtepu8_epi16(_mm_loadu_si128(idx(c3, g)));
            acc3 = _mm256_add_epi32(
                acc3,
                _mm256_madd_epi16(_mm256_shuffle_epi8(p3, ymm_shuffle), w),
            );
        }

        let f0 = _mm_add_epi32(
            _mm256_castsi256_si128(acc0),
            _mm256_extracti128_si256::<1>(acc0),
        );
        let f1 = _mm_add_epi32(
            _mm256_castsi256_si128(acc1),
            _mm256_extracti128_si256::<1>(acc1),
        );
        let f2 = _mm_add_epi32(
            _mm256_castsi256_si128(acc2),
            _mm256_extracti128_si256::<1>(acc2),
        );
        let f3 = _mm_add_epi32(
            _mm256_castsi256_si128(acc3),
            _mm256_extracti128_si256::<1>(acc3),
        );

        let s0 = _mm_srai_epi32::<{ I16_PRECISION }>(_mm_add_epi32(f0, half));
        let s1 = _mm_srai_epi32::<{ I16_PRECISION }>(_mm_add_epi32(f1, half));
        let s2 = _mm_srai_epi32::<{ I16_PRECISION }>(_mm_add_epi32(f2, half));
        let s3 = _mm_srai_epi32::<{ I16_PRECISION }>(_mm_add_epi32(f3, half));

        let pack01 = _mm_packs_epi32(s0, s1);
        let pack23 = _mm_packs_epi32(s2, s3);
        let result = _mm_packus_epi16(pack01, pack23);

        idx_mut(op0, out_x).copy_from_slice(&(_mm_cvtsi128_si32(result) as u32).to_ne_bytes());
        idx_mut(op1, out_x).copy_from_slice(&(_mm_extract_epi32::<1>(result) as u32).to_ne_bytes());
        idx_mut(op2, out_x).copy_from_slice(&(_mm_extract_epi32::<2>(result) as u32).to_ne_bytes());
        idx_mut(op3, out_x).copy_from_slice(&(_mm_extract_epi32::<3>(result) as u32).to_ne_bytes());
    }
}

/// Scalar fallback for a single edge pixel in 4ch integer horizontal filter.
///
/// Called for pixels where SIMD 16-byte reads would extend past the input buffer.
#[inline]
fn filter_h_u8_4ch_edge(
    _token: X64V3Token,
    input: &[u8],
    output: &mut [u8],
    out_x: usize,
    left: usize,
    weights: &I16WeightTable,
) {
    let w = weights.weights(out_x);
    for c in 0..4 {
        let mut acc: i32 = 0;
        for (t, &weight) in w.iter().enumerate() {
            acc += input[(left + t) * 4 + c] as i32 * weight as i32;
        }
        let rounded = (acc + (1 << (I16_PRECISION - 1))) >> I16_PRECISION;
        output[c] = rounded.clamp(0, 255) as u8;
    }
}

/// Scalar fallback for integer horizontal filter (arbitrary channel count).
#[archmage::rite]
fn filter_h_u8_generic(
    _token: X64V3Token,
    input: &[u8],
    output: &mut [u8],
    weights: &I16WeightTable,
    channels: usize,
) {
    let out_width = weights.len();

    for out_x in 0..out_width {
        let left = weights.left[out_x] as usize;
        let w = weights.weights(out_x);
        let out_base = out_x * channels;

        for c in 0..channels {
            let mut acc: i32 = 0;
            for (t, &weight) in w.iter().enumerate() {
                acc += input[(left + t) * channels + c] as i32 * weight as i32;
            }
            // Round and shift
            let rounded = (acc + (1 << (I16_PRECISION - 1))) >> I16_PRECISION;
            output[out_base + c] = rounded.clamp(0, 255) as u8;
        }
    }
}

// =============================================================================
// Integer convolution kernels: u8 input → i16 output (unclamped)
//
// Identical SIMD computation to filter_h_u8_i16 but output stores i16 values
// instead of clamping to [0,255] u8. This preserves Lanczos ringing in the
// intermediate buffer for higher quality when a V-filter follows.
// =============================================================================

/// Integer horizontal convolution: u8 input → i16 output via i32 accumulator.
///
/// Same madd_epi16 computation as `filter_h_u8_i16_v3`. Only the output pack
/// step differs: packs_epi32 → i16 store instead of packs_epi32 → packus_epi16 → u8.
#[archmage::arcane]
pub(crate) fn filter_h_u8_to_i16_v3(
    _token: X64V3Token,
    input: &[u8],
    output: &mut [i16],
    weights: &I16WeightTable,
    channels: usize,
) {
    match channels {
        4 => filter_h_u8_to_i16_4ch(_token, input, output, weights),
        _ => filter_h_u8_to_i16_generic(_token, input, output, weights, channels),
    }
}

/// Integer horizontal filter for 4-channel (RGBA) u8 → i16 output.
///
/// Same 256-bit madd_epi16 accumulation as `filter_h_u8_4ch`. Output step:
/// packs_epi32 produces 4 i16 values → storeu_si64 writes 8 bytes (4 i16).
#[archmage::rite]
fn filter_h_u8_to_i16_4ch(
    _token: X64V3Token,
    input: &[u8],
    output: &mut [i16],
    weights: &I16WeightTable,
) {
    let out_width = weights.len();
    let groups4 = weights.groups4;
    let in_pixels = input.len() / 4;

    // Need at least 4 pixels for the 16-byte loads
    if in_pixels < 4 {
        filter_h_u8_to_i16_generic(_token, input, output, weights, 4);
        return;
    }

    // Check if the input slice has enough padding for all SIMD reads.
    let max_left = weights.left.iter().map(|&l| l as usize).max().unwrap_or(0);
    let has_full_padding = max_left + groups4 * 4 + 3 < in_pixels;

    if !has_full_padding {
        filter_h_u8_to_i16_4ch_with_edge_fallback(_token, input, output, weights);
        return;
    }

    let ymm_shuffle = _mm256_set_epi8(
        15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0, 15, 14, 7, 6, 13, 12, 5, 4, 11, 10,
        3, 2, 9, 8, 1, 0,
    );

    let half = _mm_set1_epi32(1 << (I16_PRECISION - 1));
    let zero = _mm_setzero_si128();

    let ew_all = weights.expanded_4ch_all();
    let (ew_chunks, _) = ew_all.as_chunks::<16>();

    let (out_pixels, _) = output.as_chunks_mut::<4>();

    for out_x in 0..out_width {
        let left = weights.left[out_x] as usize;
        let byte_start = left * 4;

        let input_window = sub(input, byte_start..byte_start + groups4 * 16);
        let (in_chunks, _) = input_window.as_chunks::<16>();
        let ew_base = out_x * groups4;
        let ew_window = sub(ew_chunks, ew_base..ew_base + groups4);

        let mut acc = _mm256_setzero_si256();

        for (chunk, ew) in in_chunks.iter().zip(ew_window.iter()) {
            let pixels = _mm_loadu_si128(chunk);
            let ext = _mm256_cvtepu8_epi16(pixels);
            let shuffled = _mm256_shuffle_epi8(ext, ymm_shuffle);
            let w = _mm256_loadu_si256(ew);
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(shuffled, w));
        }

        let lo = _mm256_castsi256_si128(acc);
        let hi = _mm256_extracti128_si256::<1>(acc);
        let final_acc = _mm_add_epi32(lo, hi);

        let rounded = _mm_add_epi32(final_acc, half);
        let shifted = _mm_srai_epi32::<{ I16_PRECISION }>(rounded);
        // Pack i32 → i16 (signed saturation), store 4 i16 values (8 bytes)
        let packed16 = _mm_packs_epi32(shifted, zero);
        _mm_storeu_si64(idx_mut(out_pixels, out_x), packed16);
    }
}

/// Fallback for filter_h_u8_to_i16_4ch when input lacks SIMD padding.
/// Uses per-pixel edge checks for safety.
#[archmage::rite]
fn filter_h_u8_to_i16_4ch_with_edge_fallback(
    _token: X64V3Token,
    input: &[u8],
    output: &mut [i16],
    weights: &I16WeightTable,
) {
    let out_width = weights.len();
    let groups4 = weights.groups4;
    let in_pixels = input.len() / 4;

    let ymm_shuffle = _mm256_set_epi8(
        15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0, 15, 14, 7, 6, 13, 12, 5, 4, 11, 10,
        3, 2, 9, 8, 1, 0,
    );
    let half = _mm_set1_epi32(1 << (I16_PRECISION - 1));
    let zero = _mm_setzero_si128();

    let safe_end = (0..out_width)
        .position(|x| (weights.left[x] as usize) + groups4 * 4 > in_pixels)
        .unwrap_or(out_width);

    // Edge pixels: scalar
    for out_x in safe_end..out_width {
        let left = weights.left[out_x] as usize;
        filter_h_u8_to_i16_4ch_edge(
            _token,
            input,
            &mut output[out_x * 4..],
            out_x,
            left,
            weights,
        );
    }

    // Interior pixels: SIMD
    if safe_end > 0 {
        let (out_pixels, _) = output.as_chunks_mut::<4>();

        for out_x in 0..safe_end {
            let left = weights.left[out_x] as usize;
            let byte_start = left * 4;

            let ew = weights.weights_expanded_4ch(out_x);
            let (ew_chunks, _) = ew.as_chunks::<16>();

            let input_window = &input[byte_start..byte_start + groups4 * 16];
            let (in_chunks, _) = input_window.as_chunks::<16>();

            let mut acc = _mm256_setzero_si256();

            for (chunk, ew) in in_chunks.iter().zip(ew_chunks.iter()) {
                let pixels = _mm_loadu_si128(chunk);
                let ext = _mm256_cvtepu8_epi16(pixels);
                let shuffled = _mm256_shuffle_epi8(ext, ymm_shuffle);
                let w = _mm256_loadu_si256(ew);
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(shuffled, w));
            }

            let lo = _mm256_castsi256_si128(acc);
            let hi = _mm256_extracti128_si256::<1>(acc);
            let final_acc = _mm_add_epi32(lo, hi);

            let rounded = _mm_add_epi32(final_acc, half);
            let shifted = _mm_srai_epi32::<{ I16_PRECISION }>(rounded);
            let packed16 = _mm_packs_epi32(shifted, zero);
            _mm_storeu_si64(idx_mut(out_pixels, out_x), packed16);
        }
    }
}

/// Scalar fallback for a single edge pixel in 4ch u8→i16 horizontal filter.
#[inline]
fn filter_h_u8_to_i16_4ch_edge(
    _token: X64V3Token,
    input: &[u8],
    output: &mut [i16],
    out_x: usize,
    left: usize,
    weights: &I16WeightTable,
) {
    let w = weights.weights(out_x);
    for c in 0..4 {
        let mut acc: i32 = 0;
        for (t, &weight) in w.iter().enumerate() {
            acc += input[(left + t) * 4 + c] as i32 * weight as i32;
        }
        let rounded = (acc + (1 << (I16_PRECISION - 1))) >> I16_PRECISION;
        output[c] = rounded as i16;
    }
}

/// Scalar fallback for u8→i16 integer horizontal filter (arbitrary channel count).
#[archmage::rite]
fn filter_h_u8_to_i16_generic(
    _token: X64V3Token,
    input: &[u8],
    output: &mut [i16],
    weights: &I16WeightTable,
    channels: usize,
) {
    let out_width = weights.len();

    for out_x in 0..out_width {
        let left = weights.left[out_x] as usize;
        let w = weights.weights(out_x);
        let out_base = out_x * channels;

        for c in 0..channels {
            let mut acc: i32 = 0;
            for (t, &weight) in w.iter().enumerate() {
                acc += input[(left + t) * channels + c] as i32 * weight as i32;
            }
            let rounded = (acc + (1 << (I16_PRECISION - 1))) >> I16_PRECISION;
            output[out_base + c] = rounded as i16;
        }
    }
}

/// 4-row batch horizontal filter for 4-channel (RGBA) u8 → i16 output.
///
/// Same shared-weight amortization as `filter_h_u8_i16_4rows_v3`. Output stores
/// i16 values via packs_epi32 → storeu_si64 instead of u8 via packus_epi16.
#[archmage::arcane]
pub(crate) fn filter_h_u8_to_i16_4rows_v3(
    _token: X64V3Token,
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
    filter_h_u8_to_i16_4ch_4rows(_token, in0, in1, in2, in3, out0, out1, out2, out3, weights);
}

/// Inner implementation of 4-row batch u8→i16 horizontal convolution.
#[archmage::rite]
fn filter_h_u8_to_i16_4ch_4rows(
    _token: X64V3Token,
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
    let out_width = weights.len();
    let groups4 = weights.groups4;
    let in_pixels = [in0.len(), in1.len(), in2.len(), in3.len()]
        .iter()
        .copied()
        .min()
        .unwrap()
        / 4;

    let max_left = weights.left.iter().map(|&l| l as usize).max().unwrap_or(0);
    if max_left + groups4 * 4 + 3 >= in_pixels || in_pixels < 4 {
        filter_h_u8_to_i16_4ch(_token, in0, out0, weights);
        filter_h_u8_to_i16_4ch(_token, in1, out1, weights);
        filter_h_u8_to_i16_4ch(_token, in2, out2, weights);
        filter_h_u8_to_i16_4ch(_token, in3, out3, weights);
        return;
    }

    let ymm_shuffle = _mm256_set_epi8(
        15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0, 15, 14, 7, 6, 13, 12, 5, 4, 11, 10,
        3, 2, 9, 8, 1, 0,
    );
    let half = _mm_set1_epi32(1 << (I16_PRECISION - 1));
    let zero = _mm_setzero_si128();

    let ew_all = weights.expanded_4ch_all();
    let (ew_chunks, _) = ew_all.as_chunks::<16>();

    let (op0, _) = out0.as_chunks_mut::<4>();
    let (op1, _) = out1.as_chunks_mut::<4>();
    let (op2, _) = out2.as_chunks_mut::<4>();
    let (op3, _) = out3.as_chunks_mut::<4>();

    for out_x in 0..out_width {
        let left = weights.left[out_x] as usize;
        let byte_start = left * 4;
        let byte_end = byte_start + groups4 * 16;
        let ew_base = out_x * groups4;

        let (c0, _) = sub(in0, byte_start..byte_end).as_chunks::<16>();
        let (c1, _) = sub(in1, byte_start..byte_end).as_chunks::<16>();
        let (c2, _) = sub(in2, byte_start..byte_end).as_chunks::<16>();
        let (c3, _) = sub(in3, byte_start..byte_end).as_chunks::<16>();
        let ew_window = sub(ew_chunks, ew_base..ew_base + groups4);

        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        let mut acc2 = _mm256_setzero_si256();
        let mut acc3 = _mm256_setzero_si256();

        for g in 0..groups4 {
            let w = _mm256_loadu_si256(idx(ew_window, g));

            let p0 = _mm256_cvtepu8_epi16(_mm_loadu_si128(idx(c0, g)));
            acc0 = _mm256_add_epi32(
                acc0,
                _mm256_madd_epi16(_mm256_shuffle_epi8(p0, ymm_shuffle), w),
            );

            let p1 = _mm256_cvtepu8_epi16(_mm_loadu_si128(idx(c1, g)));
            acc1 = _mm256_add_epi32(
                acc1,
                _mm256_madd_epi16(_mm256_shuffle_epi8(p1, ymm_shuffle), w),
            );

            let p2 = _mm256_cvtepu8_epi16(_mm_loadu_si128(idx(c2, g)));
            acc2 = _mm256_add_epi32(
                acc2,
                _mm256_madd_epi16(_mm256_shuffle_epi8(p2, ymm_shuffle), w),
            );

            let p3 = _mm256_cvtepu8_epi16(_mm_loadu_si128(idx(c3, g)));
            acc3 = _mm256_add_epi32(
                acc3,
                _mm256_madd_epi16(_mm256_shuffle_epi8(p3, ymm_shuffle), w),
            );
        }

        let f0 = _mm_add_epi32(
            _mm256_castsi256_si128(acc0),
            _mm256_extracti128_si256::<1>(acc0),
        );
        let f1 = _mm_add_epi32(
            _mm256_castsi256_si128(acc1),
            _mm256_extracti128_si256::<1>(acc1),
        );
        let f2 = _mm_add_epi32(
            _mm256_castsi256_si128(acc2),
            _mm256_extracti128_si256::<1>(acc2),
        );
        let f3 = _mm_add_epi32(
            _mm256_castsi256_si128(acc3),
            _mm256_extracti128_si256::<1>(acc3),
        );

        let s0 = _mm_srai_epi32::<{ I16_PRECISION }>(_mm_add_epi32(f0, half));
        let s1 = _mm_srai_epi32::<{ I16_PRECISION }>(_mm_add_epi32(f1, half));
        let s2 = _mm_srai_epi32::<{ I16_PRECISION }>(_mm_add_epi32(f2, half));
        let s3 = _mm_srai_epi32::<{ I16_PRECISION }>(_mm_add_epi32(f3, half));

        // Pack i32 → i16 (signed saturation), store 4 i16 per row
        let pack0 = _mm_packs_epi32(s0, zero);
        let pack1 = _mm_packs_epi32(s1, zero);
        let pack2 = _mm_packs_epi32(s2, zero);
        let pack3 = _mm_packs_epi32(s3, zero);

        _mm_storeu_si64(idx_mut(op0, out_x), pack0);
        _mm_storeu_si64(idx_mut(op1, out_x), pack1);
        _mm_storeu_si64(idx_mut(op2, out_x), pack2);
        _mm_storeu_si64(idx_mut(op3, out_x), pack3);
    }
}

/// Batch vertical filter: process all output rows from the intermediate buffer.
///
/// Uses 2D guards on the contiguous intermediate buffer to eliminate unsafe.
/// Pre-computed row offsets with monotonic-in-tap ordering enable O(1) guard
/// construction per output row.
///
/// When consecutive output rows share the same `left` (common for upscale),
/// processes them in pairs: loads each input chunk once, accumulates into two
/// output rows with different weights. This halves memory traffic for the
/// shared taps.
#[archmage::arcane]
pub(crate) fn filter_v_all_u8_i16_v3(
    _token: X64V3Token,
    intermediate: &[u8],
    output: &mut [u8],
    h_row_len: usize,
    in_h: usize,
    out_h: usize,
    weights: &I16WeightTable,
) {
    let half = _mm256_set1_epi32(1 << (I16_PRECISION - 1));
    let chunks16 = h_row_len / 16;
    let in_h_i32 = in_h as i32;

    // Pre-chunk all intermediate rows for direct indexing.
    // Each row's chunked view is stored once, reused across output rows.
    let mut int_row_chunks: Vec<&[[u8; 16]]> = Vec::with_capacity(in_h);
    for y in 0..in_h {
        let row = &intermediate[y * h_row_len..y * h_row_len + h_row_len];
        int_row_chunks.push(row.as_chunks::<16>().0);
    }

    // Allocate working buffers once, sized for max tap count.
    let max_taps = weights.max_taps;
    let mut row_indices = vec![0usize; max_taps];
    let mut paired_wts_a = vec![_mm256_setzero_si256(); max_taps.div_ceil(2)];
    let mut paired_wts_b = vec![_mm256_setzero_si256(); max_taps.div_ceil(2)];
    // Pre-allocated buffer for per-output-row tap chunk slices (avoids per-row Vec alloc).
    let empty_chunks: &[[u8; 16]] = &[];
    let mut tap_rows = vec![empty_chunks; max_taps];

    let mut out_y = 0;
    while out_y < out_h {
        let left = weights.left[out_y];
        let tap_count = weights.tap_count(out_y);
        let w_a = weights.weights(out_y);

        // Check if next output row shares the same input rows (same left + tap_count).
        // Common for upscale where multiple output rows map to the same input window.
        let batch2 = out_y + 1 < out_h
            && weights.left[out_y + 1] == left
            && weights.tap_count(out_y + 1) == tap_count;

        // Pre-compute shared clamped row indices.
        let pairs = tap_count / 2;
        let odd = !tap_count.is_multiple_of(2);

        for t in 0..tap_count {
            row_indices[t] = (left + t as i32).clamp(0, in_h_i32 - 1) as usize;
        }

        // Weights for row A (always needed).
        for p in 0..pairs {
            let w0 = w_a[p * 2] as i32;
            let w1 = w_a[p * 2 + 1] as i32;
            paired_wts_a[p] = _mm256_set1_epi32((w1 << 16) | (w0 & 0xFFFF));
        }
        let odd_wt_a = if odd {
            _mm256_set1_epi32(w_a[tap_count - 1] as i32 & 0xFFFF)
        } else {
            _mm256_setzero_si256()
        };

        // Pre-fetch row chunk slices for this output row's taps.
        // Slicing to [..chunks16] proves inner-loop [ci] accesses are in bounds.
        for (t, &ri) in row_indices[..tap_count].iter().enumerate() {
            tap_rows[t] = &int_row_chunks[ri][..chunks16];
        }

        if batch2 {
            // === Paired output rows: shared input loads, dual accumulation ===
            let w_b = weights.weights(out_y + 1);

            // Weights for row B.
            for p in 0..pairs {
                let w0 = w_b[p * 2] as i32;
                let w1 = w_b[p * 2 + 1] as i32;
                paired_wts_b[p] = _mm256_set1_epi32((w1 << 16) | (w0 & 0xFFFF));
            }
            let odd_wt_b = if odd {
                _mm256_set1_epi32(w_b[tap_count - 1] as i32 & 0xFFFF)
            } else {
                _mm256_setzero_si256()
            };

            let out_base_a = out_y * h_row_len;
            // Split output into two non-overlapping mutable row slices.
            let (row_a, rest) = output[out_base_a..].split_at_mut(h_row_len);
            let row_b = &mut rest[..h_row_len];
            let (chunks_a, tail_a) = row_a.as_chunks_mut::<16>();
            let (chunks_b, tail_b) = row_b.as_chunks_mut::<16>();

            for ci in 0..chunks16 {
                let mut acc_a_lo = _mm256_setzero_si256();
                let mut acc_a_hi = _mm256_setzero_si256();
                let mut acc_b_lo = _mm256_setzero_si256();
                let mut acc_b_hi = _mm256_setzero_si256();

                for p in 0..pairs {
                    let src0 = _mm_loadu_si128(idx(tap_rows[p * 2], ci));
                    let src1 = _mm_loadu_si128(idx(tap_rows[p * 2 + 1], ci));

                    let il_lo = _mm_unpacklo_epi8(src0, src1);
                    let il_hi = _mm_unpackhi_epi8(src0, src1);
                    let ext_lo = _mm256_cvtepu8_epi16(il_lo);
                    let ext_hi = _mm256_cvtepu8_epi16(il_hi);

                    acc_a_lo =
                        _mm256_add_epi32(acc_a_lo, _mm256_madd_epi16(ext_lo, paired_wts_a[p]));
                    acc_a_hi =
                        _mm256_add_epi32(acc_a_hi, _mm256_madd_epi16(ext_hi, paired_wts_a[p]));
                    acc_b_lo =
                        _mm256_add_epi32(acc_b_lo, _mm256_madd_epi16(ext_lo, paired_wts_b[p]));
                    acc_b_hi =
                        _mm256_add_epi32(acc_b_hi, _mm256_madd_epi16(ext_hi, paired_wts_b[p]));
                }

                if odd {
                    let src = _mm_loadu_si128(idx(tap_rows[tap_count - 1], ci));
                    let zero_src = _mm_setzero_si128();
                    let il_lo = _mm_unpacklo_epi8(src, zero_src);
                    let il_hi = _mm_unpackhi_epi8(src, zero_src);
                    let ext_lo = _mm256_cvtepu8_epi16(il_lo);
                    let ext_hi = _mm256_cvtepu8_epi16(il_hi);

                    acc_a_lo = _mm256_add_epi32(acc_a_lo, _mm256_madd_epi16(ext_lo, odd_wt_a));
                    acc_a_hi = _mm256_add_epi32(acc_a_hi, _mm256_madd_epi16(ext_hi, odd_wt_a));
                    acc_b_lo = _mm256_add_epi32(acc_b_lo, _mm256_madd_epi16(ext_lo, odd_wt_b));
                    acc_b_hi = _mm256_add_epi32(acc_b_hi, _mm256_madd_epi16(ext_hi, odd_wt_b));
                }

                // Pack and store row A.
                let ra_lo =
                    _mm256_srai_epi32::<{ I16_PRECISION }>(_mm256_add_epi32(acc_a_lo, half));
                let ra_hi =
                    _mm256_srai_epi32::<{ I16_PRECISION }>(_mm256_add_epi32(acc_a_hi, half));
                let a_ll = _mm256_castsi256_si128(ra_lo);
                let a_lh = _mm256_extracti128_si256::<1>(ra_lo);
                let a_hl = _mm256_castsi256_si128(ra_hi);
                let a_hh = _mm256_extracti128_si256::<1>(ra_hi);
                _mm_storeu_si128(
                    idx_mut(chunks_a, ci),
                    _mm_packus_epi16(_mm_packs_epi32(a_ll, a_lh), _mm_packs_epi32(a_hl, a_hh)),
                );

                // Pack and store row B.
                let rb_lo =
                    _mm256_srai_epi32::<{ I16_PRECISION }>(_mm256_add_epi32(acc_b_lo, half));
                let rb_hi =
                    _mm256_srai_epi32::<{ I16_PRECISION }>(_mm256_add_epi32(acc_b_hi, half));
                let b_ll = _mm256_castsi256_si128(rb_lo);
                let b_lh = _mm256_extracti128_si256::<1>(rb_lo);
                let b_hl = _mm256_castsi256_si128(rb_hi);
                let b_hh = _mm256_extracti128_si256::<1>(rb_hi);
                _mm_storeu_si128(
                    idx_mut(chunks_b, ci),
                    _mm_packus_epi16(_mm_packs_epi32(b_ll, b_lh), _mm_packs_epi32(b_hl, b_hh)),
                );
            }

            // Scalar tails.
            let tail_start = chunks16 * 16;
            for (x, out_byte) in tail_a.iter_mut().enumerate() {
                let mut acc: i32 = 0;
                for t in 0..tap_count {
                    acc += intermediate[row_indices[t] * h_row_len + tail_start + x] as i32
                        * w_a[t] as i32;
                }
                *out_byte =
                    ((acc + (1 << (I16_PRECISION - 1))) >> I16_PRECISION).clamp(0, 255) as u8;
            }
            for (x, out_byte) in tail_b.iter_mut().enumerate() {
                let mut acc: i32 = 0;
                for t in 0..tap_count {
                    acc += intermediate[row_indices[t] * h_row_len + tail_start + x] as i32
                        * w_b[t] as i32;
                }
                *out_byte =
                    ((acc + (1 << (I16_PRECISION - 1))) >> I16_PRECISION).clamp(0, 255) as u8;
            }
            out_y += 2;
        } else {
            // === Single output row ===
            let out_base = out_y * h_row_len;
            let (out_chunks, out_tail) =
                output[out_base..out_base + h_row_len].as_chunks_mut::<16>();

            for (ci, out_chunk) in out_chunks.iter_mut().enumerate() {
                let mut acc_lo = _mm256_setzero_si256();
                let mut acc_hi = _mm256_setzero_si256();

                for (pw, ri_pair) in paired_wts_a[..pairs].iter().zip(tap_rows.chunks_exact(2)) {
                    let src0 = _mm_loadu_si128(idx(ri_pair[0], ci));
                    let src1 = _mm_loadu_si128(idx(ri_pair[1], ci));

                    let il_lo = _mm_unpacklo_epi8(src0, src1);
                    let il_hi = _mm_unpackhi_epi8(src0, src1);
                    let ext_lo = _mm256_cvtepu8_epi16(il_lo);
                    let ext_hi = _mm256_cvtepu8_epi16(il_hi);

                    acc_lo = _mm256_add_epi32(acc_lo, _mm256_madd_epi16(ext_lo, *pw));
                    acc_hi = _mm256_add_epi32(acc_hi, _mm256_madd_epi16(ext_hi, *pw));
                }

                if odd {
                    let src = _mm_loadu_si128(idx(tap_rows[tap_count - 1], ci));
                    let zero_src = _mm_setzero_si128();
                    let il_lo = _mm_unpacklo_epi8(src, zero_src);
                    let il_hi = _mm_unpackhi_epi8(src, zero_src);
                    let ext_lo = _mm256_cvtepu8_epi16(il_lo);
                    let ext_hi = _mm256_cvtepu8_epi16(il_hi);
                    acc_lo = _mm256_add_epi32(acc_lo, _mm256_madd_epi16(ext_lo, odd_wt_a));
                    acc_hi = _mm256_add_epi32(acc_hi, _mm256_madd_epi16(ext_hi, odd_wt_a));
                }

                let rounded_lo = _mm256_add_epi32(acc_lo, half);
                let rounded_hi = _mm256_add_epi32(acc_hi, half);
                let shifted_lo = _mm256_srai_epi32::<{ I16_PRECISION }>(rounded_lo);
                let shifted_hi = _mm256_srai_epi32::<{ I16_PRECISION }>(rounded_hi);

                let lo_lo = _mm256_castsi256_si128(shifted_lo);
                let lo_hi = _mm256_extracti128_si256::<1>(shifted_lo);
                let hi_lo = _mm256_castsi256_si128(shifted_hi);
                let hi_hi = _mm256_extracti128_si256::<1>(shifted_hi);

                let pack01 = _mm_packs_epi32(lo_lo, lo_hi);
                let pack23 = _mm_packs_epi32(hi_lo, hi_hi);
                _mm_storeu_si128(out_chunk, _mm_packus_epi16(pack01, pack23));
            }

            // Scalar tail
            let tail_start = chunks16 * 16;
            for (x, out_byte) in out_tail.iter_mut().enumerate() {
                let mut acc: i32 = 0;
                for t in 0..tap_count {
                    let in_y = row_indices[t];
                    acc += intermediate[in_y * h_row_len + tail_start + x] as i32 * w_a[t] as i32;
                }
                let rounded = (acc + (1 << (I16_PRECISION - 1))) >> I16_PRECISION;
                *out_byte = rounded.clamp(0, 255) as u8;
            }
            out_y += 1;
        }
    }
}

/// Tiled batch vertical filter: u8 intermediate → u8 output via i16 weights.
///
/// Processes column tiles to keep `max_taps × tile_chunks × 16` bytes in L1 cache,
/// improving reuse of shared input rows across consecutive output rows.
///
/// `tile_chunks` is the number of 16-byte chunks per tile. For best results,
/// choose so that `max_taps × tile_chunks × 16 ≤ L1_SIZE / 2`.
#[archmage::arcane]
pub(crate) fn filter_v_all_u8_i16_tiled_v3(
    _token: X64V3Token,
    intermediate: &[u8],
    output: &mut [u8],
    h_row_len: usize,
    in_h: usize,
    out_h: usize,
    weights: &I16WeightTable,
    tile_chunks: usize,
) {
    let half = _mm256_set1_epi32(1 << (I16_PRECISION - 1));
    let chunks16 = h_row_len / 16;
    let in_h_i32 = in_h as i32;

    // Pre-chunk all intermediate rows for direct indexing.
    let mut int_row_chunks: Vec<&[[u8; 16]]> = Vec::with_capacity(in_h);
    for y in 0..in_h {
        let row = &intermediate[y * h_row_len..y * h_row_len + h_row_len];
        int_row_chunks.push(row.as_chunks::<16>().0);
    }

    let max_taps = weights.max_taps;
    let mut row_indices = vec![0usize; max_taps];
    let mut paired_wts_a = vec![_mm256_setzero_si256(); max_taps.div_ceil(2)];
    let mut paired_wts_b = vec![_mm256_setzero_si256(); max_taps.div_ceil(2)];
    let empty_chunks: &[[u8; 16]] = &[];
    let mut tap_rows = vec![empty_chunks; max_taps];

    // Process column tiles.
    for tile_ci in (0..chunks16).step_by(tile_chunks) {
        let tile_end = (tile_ci + tile_chunks).min(chunks16);

        let mut out_y = 0;
        while out_y < out_h {
            let left = weights.left[out_y];
            let tap_count = weights.tap_count(out_y);
            let w_a = weights.weights(out_y);

            let batch2 = out_y + 1 < out_h
                && weights.left[out_y + 1] == left
                && weights.tap_count(out_y + 1) == tap_count;

            let pairs = tap_count / 2;
            let odd = !tap_count.is_multiple_of(2);

            for t in 0..tap_count {
                row_indices[t] = (left + t as i32).clamp(0, in_h_i32 - 1) as usize;
            }

            for p in 0..pairs {
                let w0 = w_a[p * 2] as i32;
                let w1 = w_a[p * 2 + 1] as i32;
                paired_wts_a[p] = _mm256_set1_epi32((w1 << 16) | (w0 & 0xFFFF));
            }
            let odd_wt_a = if odd {
                _mm256_set1_epi32(w_a[tap_count - 1] as i32 & 0xFFFF)
            } else {
                _mm256_setzero_si256()
            };

            for (t, &ri) in row_indices[..tap_count].iter().enumerate() {
                tap_rows[t] = &int_row_chunks[ri][..chunks16];
            }

            if batch2 {
                let w_b = weights.weights(out_y + 1);
                for p in 0..pairs {
                    let w0 = w_b[p * 2] as i32;
                    let w1 = w_b[p * 2 + 1] as i32;
                    paired_wts_b[p] = _mm256_set1_epi32((w1 << 16) | (w0 & 0xFFFF));
                }
                let odd_wt_b = if odd {
                    _mm256_set1_epi32(w_b[tap_count - 1] as i32 & 0xFFFF)
                } else {
                    _mm256_setzero_si256()
                };

                let out_base_a = out_y * h_row_len;
                // Split output into two non-overlapping mutable row slices.
                let (row_a, rest) = output[out_base_a..].split_at_mut(h_row_len);
                let row_b = &mut rest[..h_row_len];
                let (chunks_a, _) = row_a.as_chunks_mut::<16>();
                let (chunks_b, _) = row_b.as_chunks_mut::<16>();

                for ci in tile_ci..tile_end {
                    let mut acc_a_lo = _mm256_setzero_si256();
                    let mut acc_a_hi = _mm256_setzero_si256();
                    let mut acc_b_lo = _mm256_setzero_si256();
                    let mut acc_b_hi = _mm256_setzero_si256();

                    for p in 0..pairs {
                        let src0 = _mm_loadu_si128(idx(tap_rows[p * 2], ci));
                        let src1 = _mm_loadu_si128(idx(tap_rows[p * 2 + 1], ci));

                        let il_lo = _mm_unpacklo_epi8(src0, src1);
                        let il_hi = _mm_unpackhi_epi8(src0, src1);
                        let ext_lo = _mm256_cvtepu8_epi16(il_lo);
                        let ext_hi = _mm256_cvtepu8_epi16(il_hi);

                        acc_a_lo =
                            _mm256_add_epi32(acc_a_lo, _mm256_madd_epi16(ext_lo, paired_wts_a[p]));
                        acc_a_hi =
                            _mm256_add_epi32(acc_a_hi, _mm256_madd_epi16(ext_hi, paired_wts_a[p]));
                        acc_b_lo =
                            _mm256_add_epi32(acc_b_lo, _mm256_madd_epi16(ext_lo, paired_wts_b[p]));
                        acc_b_hi =
                            _mm256_add_epi32(acc_b_hi, _mm256_madd_epi16(ext_hi, paired_wts_b[p]));
                    }

                    if odd {
                        let src = _mm_loadu_si128(idx(tap_rows[tap_count - 1], ci));
                        let zero_src = _mm_setzero_si128();
                        let il_lo = _mm_unpacklo_epi8(src, zero_src);
                        let il_hi = _mm_unpackhi_epi8(src, zero_src);
                        let ext_lo = _mm256_cvtepu8_epi16(il_lo);
                        let ext_hi = _mm256_cvtepu8_epi16(il_hi);

                        acc_a_lo = _mm256_add_epi32(acc_a_lo, _mm256_madd_epi16(ext_lo, odd_wt_a));
                        acc_a_hi = _mm256_add_epi32(acc_a_hi, _mm256_madd_epi16(ext_hi, odd_wt_a));
                        acc_b_lo = _mm256_add_epi32(acc_b_lo, _mm256_madd_epi16(ext_lo, odd_wt_b));
                        acc_b_hi = _mm256_add_epi32(acc_b_hi, _mm256_madd_epi16(ext_hi, odd_wt_b));
                    }

                    // Pack and store row A.
                    let ra_lo =
                        _mm256_srai_epi32::<{ I16_PRECISION }>(_mm256_add_epi32(acc_a_lo, half));
                    let ra_hi =
                        _mm256_srai_epi32::<{ I16_PRECISION }>(_mm256_add_epi32(acc_a_hi, half));
                    let a_ll = _mm256_castsi256_si128(ra_lo);
                    let a_lh = _mm256_extracti128_si256::<1>(ra_lo);
                    let a_hl = _mm256_castsi256_si128(ra_hi);
                    let a_hh = _mm256_extracti128_si256::<1>(ra_hi);
                    _mm_storeu_si128(
                        idx_mut(chunks_a, ci),
                        _mm_packus_epi16(_mm_packs_epi32(a_ll, a_lh), _mm_packs_epi32(a_hl, a_hh)),
                    );

                    // Pack and store row B.
                    let rb_lo =
                        _mm256_srai_epi32::<{ I16_PRECISION }>(_mm256_add_epi32(acc_b_lo, half));
                    let rb_hi =
                        _mm256_srai_epi32::<{ I16_PRECISION }>(_mm256_add_epi32(acc_b_hi, half));
                    let b_ll = _mm256_castsi256_si128(rb_lo);
                    let b_lh = _mm256_extracti128_si256::<1>(rb_lo);
                    let b_hl = _mm256_castsi256_si128(rb_hi);
                    let b_hh = _mm256_extracti128_si256::<1>(rb_hi);
                    _mm_storeu_si128(
                        idx_mut(chunks_b, ci),
                        _mm_packus_epi16(_mm_packs_epi32(b_ll, b_lh), _mm_packs_epi32(b_hl, b_hh)),
                    );
                }

                out_y += 2;
            } else {
                let out_base = out_y * h_row_len;
                let out_row = &mut output[out_base..out_base + h_row_len];
                let (out_chunks, _) = out_row.as_chunks_mut::<16>();

                for ci in tile_ci..tile_end {
                    let mut acc_lo = _mm256_setzero_si256();
                    let mut acc_hi = _mm256_setzero_si256();

                    for (pw, ri_pair) in paired_wts_a[..pairs].iter().zip(tap_rows.chunks_exact(2))
                    {
                        let src0 = _mm_loadu_si128(idx(ri_pair[0], ci));
                        let src1 = _mm_loadu_si128(idx(ri_pair[1], ci));

                        let il_lo = _mm_unpacklo_epi8(src0, src1);
                        let il_hi = _mm_unpackhi_epi8(src0, src1);
                        let ext_lo = _mm256_cvtepu8_epi16(il_lo);
                        let ext_hi = _mm256_cvtepu8_epi16(il_hi);

                        acc_lo = _mm256_add_epi32(acc_lo, _mm256_madd_epi16(ext_lo, *pw));
                        acc_hi = _mm256_add_epi32(acc_hi, _mm256_madd_epi16(ext_hi, *pw));
                    }

                    if odd {
                        let src = _mm_loadu_si128(idx(tap_rows[tap_count - 1], ci));
                        let zero_src = _mm_setzero_si128();
                        let il_lo = _mm_unpacklo_epi8(src, zero_src);
                        let il_hi = _mm_unpackhi_epi8(src, zero_src);
                        let ext_lo = _mm256_cvtepu8_epi16(il_lo);
                        let ext_hi = _mm256_cvtepu8_epi16(il_hi);
                        acc_lo = _mm256_add_epi32(acc_lo, _mm256_madd_epi16(ext_lo, odd_wt_a));
                        acc_hi = _mm256_add_epi32(acc_hi, _mm256_madd_epi16(ext_hi, odd_wt_a));
                    }

                    let rounded_lo = _mm256_add_epi32(acc_lo, half);
                    let rounded_hi = _mm256_add_epi32(acc_hi, half);
                    let shifted_lo = _mm256_srai_epi32::<{ I16_PRECISION }>(rounded_lo);
                    let shifted_hi = _mm256_srai_epi32::<{ I16_PRECISION }>(rounded_hi);

                    let lo_lo = _mm256_castsi256_si128(shifted_lo);
                    let lo_hi = _mm256_extracti128_si256::<1>(shifted_lo);
                    let hi_lo = _mm256_castsi256_si128(shifted_hi);
                    let hi_hi = _mm256_extracti128_si256::<1>(shifted_hi);

                    let pack01 = _mm_packs_epi32(lo_lo, lo_hi);
                    let pack23 = _mm_packs_epi32(hi_lo, hi_hi);
                    _mm_storeu_si128(idx_mut(out_chunks, ci), _mm_packus_epi16(pack01, pack23));
                }

                out_y += 1;
            }
        }
    }

    // Scalar tail for remaining bytes not covered by 16-byte chunks.
    let tail_start = chunks16 * 16;
    if tail_start < h_row_len {
        let in_h_i32_s = in_h as i32 - 1;
        for out_y in 0..out_h {
            let left = weights.left[out_y];
            let tap_count = weights.tap_count(out_y);
            let w = weights.weights(out_y);
            let out_start = out_y * h_row_len;
            for x in tail_start..h_row_len {
                let mut acc: i32 = 0;
                for (t, &weight) in w[..tap_count].iter().enumerate() {
                    let in_y = (left + t as i32).clamp(0, in_h_i32_s) as usize;
                    acc += intermediate[in_y * h_row_len + x] as i32 * weight as i32;
                }
                output[out_start + x] =
                    ((acc + (1 << (I16_PRECISION - 1))) >> I16_PRECISION).clamp(0, 255) as u8;
            }
        }
    }
}

/// Convert sRGB u8 → linear f32 using LUT.
#[archmage::arcane]
pub(crate) fn srgb_u8_to_linear_f32_v3(
    _token: X64V3Token,
    input: &[u8],
    output: &mut [f32],
    channels: usize,
    has_alpha: bool,
) {
    crate::color::srgb_u8_to_linear_f32_impl(input, output, channels, has_alpha);
}

/// Convert linear f32 → sRGB u8 using LUT.
#[archmage::arcane]
pub(crate) fn linear_f32_to_srgb_u8_v3(
    _token: X64V3Token,
    input: &[f32],
    output: &mut [u8],
    channels: usize,
    has_alpha: bool,
) {
    crate::color::linear_f32_to_srgb_u8_impl(input, output, channels, has_alpha);
}

// =============================================================================
// Linear-light i16→i16 convolution kernels (i12 values 0-4095)
// =============================================================================

/// Integer horizontal convolution: i16 input → i16 output via i32 accumulator.
///
/// Adapted from `filter_h_u8_i16_v3` — the key difference is loading i16 data
/// directly (no `cvtepu8_epi16` needed). The shuffle + madd_epi16 are identical.
#[archmage::arcane]
pub(crate) fn filter_h_i16_i16_v3(
    _token: X64V3Token,
    input: &[i16],
    output: &mut [i16],
    weights: &I16WeightTable,
    channels: usize,
) {
    match channels {
        4 => filter_h_i16_4ch(_token, input, output, weights),
        _ => {
            // Scalar fallback for non-4ch
            let out_width = weights.len();
            for out_x in 0..out_width {
                let left = weights.left[out_x] as usize;
                let w = weights.weights(out_x);
                let out_base = out_x * channels;
                for c in 0..channels {
                    let mut acc: i32 = 0;
                    for (t, &weight) in w.iter().enumerate() {
                        acc += input[(left + t) * channels + c] as i32 * weight as i32;
                    }
                    let rounded = (acc + (1 << (I16_PRECISION - 1))) >> I16_PRECISION;
                    output[out_base + c] = rounded as i16;
                }
            }
        }
    }
}

/// Integer horizontal filter for 4-channel i16 data (linear-light i12 path).
///
/// Input: i16 values (0-4095) laid out as [R,G,B,A, R,G,B,A, ...]
/// Output: i16 values (0-4095) clamped after convolution.
///
/// Since i16 data is already in i16 format, we load directly with
/// `_mm256_loadu_si256` — no `cvtepu8_epi16` needed. Each 4 RGBA pixels
/// (16 i16 values) fit in one ymm register.
///
/// Uses `as_chunks` for safe bounds-hoisted access.
#[archmage::rite]
fn filter_h_i16_4ch(
    _token: X64V3Token,
    input: &[i16],
    output: &mut [i16],
    weights: &I16WeightTable,
) {
    let out_width = weights.len();
    let groups4 = weights.groups4;
    let in_pixels = input.len() / 4;

    // Need at least 4 pixels for the 32-byte (4 RGBA pixel) i16 loads
    if in_pixels < 4 {
        // Scalar fallback for tiny inputs
        for out_x in 0..out_width {
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
        return;
    }

    // Same shuffle as u8 path: within each 128-bit lane, rearrange
    // [R0,G0,B0,A0,R1,G1,B1,A1] → [R0,R1,G0,G1,B0,B1,A0,A1]
    let ymm_shuffle = _mm256_set_epi8(
        15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0, 15, 14, 7, 6, 13, 12, 5, 4, 11, 10,
        3, 2, 9, 8, 1, 0,
    );

    let xmm_shuffle = _mm_set_epi8(15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0);
    let half = _mm_set1_epi32(1 << (I16_PRECISION - 1));
    let zero = _mm_setzero_si128();

    // Weight chunks: contiguous stride-16 i16 array.
    let ew_all = weights.expanded_4ch_all();
    let (ew_chunks, _) = ew_all.as_chunks::<16>();

    // Output as 4-i16 pixels (8 bytes each = 4 values)
    let (out_pixels, _) = output.as_chunks_mut::<4>();

    // Check if input has enough elements for global-guard path.
    // Each group loads 4 RGBA pixels (16 i16 values).
    let max_left = weights.left.iter().map(|&l| l as usize).max().unwrap_or(0);
    let has_full_padding = max_left * 4 + groups4 * 16 <= input.len();

    let rem_taps = weights.max_taps % 4;

    if has_full_padding {
        for out_x in 0..out_width {
            let left = weights.left[out_x] as usize;
            let in_base = left * 4; // in i16 elements

            // Pre-slice input window: 16 i16 per group
            let input_window = &input[in_base..in_base + groups4 * 16];
            let (in_chunks, _) = input_window.as_chunks::<16>();
            let ew_base = out_x * groups4;
            let ew_window = &ew_chunks[ew_base..ew_base + groups4];

            let mut acc = _mm256_setzero_si256();

            for (chunk, ew) in in_chunks.iter().zip(ew_window.iter()) {
                // Load 16 i16 directly (4 RGBA pixels)
                let pixels = _mm256_loadu_si256(chunk);
                let shuffled = _mm256_shuffle_epi8(pixels, ymm_shuffle);
                let w = _mm256_loadu_si256(ew);
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(shuffled, w));
            }

            let lo = _mm256_castsi256_si128(acc);
            let hi = _mm256_extracti128_si256::<1>(acc);
            let final_acc = _mm_add_epi32(lo, hi);

            let rounded = _mm_add_epi32(final_acc, half);
            let shifted = _mm_srai_epi32::<{ I16_PRECISION }>(rounded);

            // Pack i32 → i16 (signed saturation to i16 range, no [0,4095] clamp)
            let packed16 = _mm_packs_epi32(shifted, zero);

            // Store 4 × i16 (8 bytes)
            _mm_storeu_si64(&mut out_pixels[out_x], packed16);
        }
    } else {
        // Per-pixel edge check path
        for out_x in 0..out_width {
            let left = weights.left[out_x] as usize;
            let in_base = left * 4;

            // Check if this pixel has enough data for SIMD
            if in_base + groups4 * 16 <= input.len() {
                let input_window = &input[in_base..in_base + groups4 * 16];
                let (in_chunks, _) = input_window.as_chunks::<16>();

                let ew = weights.weights_expanded_4ch(out_x);
                let (ew_c, _) = ew.as_chunks::<16>();

                let mut acc = _mm256_setzero_si256();
                for (chunk, ewc) in in_chunks.iter().zip(ew_c.iter()) {
                    let pixels = _mm256_loadu_si256(chunk);
                    let shuffled = _mm256_shuffle_epi8(pixels, ymm_shuffle);
                    let w = _mm256_loadu_si256(ewc);
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(shuffled, w));
                }

                let lo = _mm256_castsi256_si128(acc);
                let hi = _mm256_extracti128_si256::<1>(acc);
                let mut final_acc = _mm_add_epi32(lo, hi);

                // Handle remaining 2 taps
                if rem_taps >= 2 {
                    let t = (groups4 - 1) * 4 + 4; // first tap after full groups
                    let tap_off = in_base + t * 4;
                    if tap_off + 8 <= input.len() {
                        let chunk: &[i16; 8] = input[tap_off..tap_off + 8].try_into().unwrap();
                        let pixels = _mm_loadu_si128(chunk);
                        let shuffled = _mm_shuffle_epi8(pixels, xmm_shuffle);
                        let wp = weights.weights_padded(out_x);
                        let w_val = (wp[t] as i32 & 0xFFFF) | ((wp[t + 1] as i32) << 16);
                        let w = _mm_set1_epi32(w_val);
                        final_acc = _mm_add_epi32(final_acc, _mm_madd_epi16(shuffled, w));
                    }
                }

                let rounded = _mm_add_epi32(final_acc, half);
                let shifted = _mm_srai_epi32::<{ I16_PRECISION }>(rounded);
                let packed16 = _mm_packs_epi32(shifted, zero);
                _mm_storeu_si64(&mut out_pixels[out_x], packed16);
            } else {
                // Scalar fallback for edge pixels
                let w = weights.weights(out_x);
                for c in 0..4 {
                    let mut acc: i32 = 0;
                    for (t, &weight) in w.iter().enumerate() {
                        acc += input[(left + t) * 4 + c] as i32 * weight as i32;
                    }
                    let rounded = (acc + (1 << (I16_PRECISION - 1))) >> I16_PRECISION;
                    out_pixels[out_x][c] = rounded as i16;
                }
            }
        }
    }
}

/// Batch vertical filter for all output rows (i16→i16), with paired output row batching.
///
/// Same dual-accumulation strategy as `filter_v_all_u8_i16_v3`: consecutive output
/// rows sharing the same `left` are processed together, halving input loads.
///
/// Adapted from `filter_v_all_u8_i16_v3` — loads i16 directly instead of u8.
/// Paired row interleaving uses `unpacklo_epi16` instead of `unpacklo_epi8` + `cvtepu8_epi16`.
/// Processes 8 i16 values (16 bytes) per chunk.
#[archmage::arcane]
pub(crate) fn filter_v_all_i16_i16_v3(
    _token: X64V3Token,
    intermediate: &[i16],
    output: &mut [i16],
    h_row_len: usize,
    in_h: usize,
    out_h: usize,
    weights: &I16WeightTable,
) {
    let half = _mm256_set1_epi32(1 << (I16_PRECISION - 1));
    let chunks8 = h_row_len / 8;
    let in_h_i32 = in_h as i32;

    // Pre-chunk all intermediate rows for direct indexing.
    let mut int_row_chunks: Vec<&[[i16; 8]]> = Vec::with_capacity(in_h);
    for y in 0..in_h {
        let row = &intermediate[y * h_row_len..y * h_row_len + h_row_len];
        int_row_chunks.push(row.as_chunks::<8>().0);
    }

    let max_taps = weights.max_taps;
    let mut row_indices = vec![0usize; max_taps];
    let mut paired_wts_a = vec![_mm256_setzero_si256(); max_taps.div_ceil(2)];
    let mut paired_wts_b = vec![_mm256_setzero_si256(); max_taps.div_ceil(2)];
    let empty_chunks: &[[i16; 8]] = &[];
    let mut tap_rows = vec![empty_chunks; max_taps];

    let mut out_y = 0;
    while out_y < out_h {
        let left = weights.left[out_y];
        let tap_count = weights.tap_count(out_y);
        let w_a = weights.weights(out_y);

        let batch2 = out_y + 1 < out_h
            && weights.left[out_y + 1] == left
            && weights.tap_count(out_y + 1) == tap_count;

        let pairs = tap_count / 2;
        let odd = !tap_count.is_multiple_of(2);

        for t in 0..tap_count {
            row_indices[t] = (left + t as i32).clamp(0, in_h_i32 - 1) as usize;
        }

        for p in 0..pairs {
            let w0 = w_a[p * 2] as i32;
            let w1 = w_a[p * 2 + 1] as i32;
            paired_wts_a[p] = _mm256_set1_epi32((w1 << 16) | (w0 & 0xFFFF));
        }
        let odd_wt_a = if odd {
            _mm256_set1_epi32(w_a[tap_count - 1] as i32 & 0xFFFF)
        } else {
            _mm256_setzero_si256()
        };

        for (t, &ri) in row_indices[..tap_count].iter().enumerate() {
            tap_rows[t] = &int_row_chunks[ri][..chunks8];
        }

        if batch2 {
            let w_b = weights.weights(out_y + 1);

            for p in 0..pairs {
                let w0 = w_b[p * 2] as i32;
                let w1 = w_b[p * 2 + 1] as i32;
                paired_wts_b[p] = _mm256_set1_epi32((w1 << 16) | (w0 & 0xFFFF));
            }
            let odd_wt_b = if odd {
                _mm256_set1_epi32(w_b[tap_count - 1] as i32 & 0xFFFF)
            } else {
                _mm256_setzero_si256()
            };

            let out_base_a = out_y * h_row_len;
            let (row_a, rest) = output[out_base_a..].split_at_mut(h_row_len);
            let row_b = &mut rest[..h_row_len];
            let (chunks_a, tail_a) = row_a.as_chunks_mut::<8>();
            let (chunks_b, tail_b) = row_b.as_chunks_mut::<8>();

            for ci in 0..chunks8 {
                let mut acc_a = _mm256_setzero_si256();
                let mut acc_b = _mm256_setzero_si256();

                for p in 0..pairs {
                    let src0 = _mm_loadu_si128(idx(tap_rows[p * 2], ci));
                    let src1 = _mm_loadu_si128(idx(tap_rows[p * 2 + 1], ci));

                    let il_lo = _mm_unpacklo_epi16(src0, src1);
                    let il_hi = _mm_unpackhi_epi16(src0, src1);
                    let combined = _mm256_set_m128i(il_hi, il_lo);

                    acc_a = _mm256_add_epi32(acc_a, _mm256_madd_epi16(combined, paired_wts_a[p]));
                    acc_b = _mm256_add_epi32(acc_b, _mm256_madd_epi16(combined, paired_wts_b[p]));
                }

                if odd {
                    let src = _mm_loadu_si128(idx(tap_rows[tap_count - 1], ci));
                    let zero_src = _mm_setzero_si128();
                    let il_lo = _mm_unpacklo_epi16(src, zero_src);
                    let il_hi = _mm_unpackhi_epi16(src, zero_src);
                    let combined = _mm256_set_m128i(il_hi, il_lo);
                    acc_a = _mm256_add_epi32(acc_a, _mm256_madd_epi16(combined, odd_wt_a));
                    acc_b = _mm256_add_epi32(acc_b, _mm256_madd_epi16(combined, odd_wt_b));
                }

                // Pack and store row A.
                let shifted_a =
                    _mm256_srai_epi32::<{ I16_PRECISION }>(_mm256_add_epi32(acc_a, half));
                let a_lo = _mm256_castsi256_si128(shifted_a);
                let a_hi = _mm256_extracti128_si256::<1>(shifted_a);
                _mm_storeu_si128(idx_mut(chunks_a, ci), _mm_packs_epi32(a_lo, a_hi));

                // Pack and store row B.
                let shifted_b =
                    _mm256_srai_epi32::<{ I16_PRECISION }>(_mm256_add_epi32(acc_b, half));
                let b_lo = _mm256_castsi256_si128(shifted_b);
                let b_hi = _mm256_extracti128_si256::<1>(shifted_b);
                _mm_storeu_si128(idx_mut(chunks_b, ci), _mm_packs_epi32(b_lo, b_hi));
            }

            // Scalar tails.
            let tail_start = chunks8 * 8;
            for (x, out_val) in tail_a.iter_mut().enumerate() {
                let mut acc: i32 = 0;
                for t in 0..tap_count {
                    acc += intermediate[row_indices[t] * h_row_len + tail_start + x] as i32
                        * w_a[t] as i32;
                }
                *out_val = ((acc + (1 << (I16_PRECISION - 1))) >> I16_PRECISION) as i16;
            }
            for (x, out_val) in tail_b.iter_mut().enumerate() {
                let mut acc: i32 = 0;
                for t in 0..tap_count {
                    acc += intermediate[row_indices[t] * h_row_len + tail_start + x] as i32
                        * w_b[t] as i32;
                }
                *out_val = ((acc + (1 << (I16_PRECISION - 1))) >> I16_PRECISION) as i16;
            }
            out_y += 2;
        } else {
            let out_base = out_y * h_row_len;
            let (out_chunks, out_tail) =
                output[out_base..out_base + h_row_len].as_chunks_mut::<8>();

            for (ci, out_chunk) in out_chunks.iter_mut().enumerate() {
                let mut acc_lo = _mm256_setzero_si256();

                for (pw, ri_pair) in paired_wts_a[..pairs].iter().zip(tap_rows.chunks_exact(2)) {
                    let src0 = _mm_loadu_si128(idx(ri_pair[0], ci));
                    let src1 = _mm_loadu_si128(idx(ri_pair[1], ci));
                    let il_lo = _mm_unpacklo_epi16(src0, src1);
                    let il_hi = _mm_unpackhi_epi16(src0, src1);
                    let combined = _mm256_set_m128i(il_hi, il_lo);
                    acc_lo = _mm256_add_epi32(acc_lo, _mm256_madd_epi16(combined, *pw));
                }

                if odd {
                    let src = _mm_loadu_si128(idx(tap_rows[tap_count - 1], ci));
                    let zero_src = _mm_setzero_si128();
                    let il_lo = _mm_unpacklo_epi16(src, zero_src);
                    let il_hi = _mm_unpackhi_epi16(src, zero_src);
                    let combined = _mm256_set_m128i(il_hi, il_lo);
                    acc_lo = _mm256_add_epi32(acc_lo, _mm256_madd_epi16(combined, odd_wt_a));
                }

                let rounded = _mm256_add_epi32(acc_lo, half);
                let shifted = _mm256_srai_epi32::<{ I16_PRECISION }>(rounded);
                let lo_128 = _mm256_castsi256_si128(shifted);
                let hi_128 = _mm256_extracti128_si256::<1>(shifted);
                let packed = _mm_packs_epi32(lo_128, hi_128);
                _mm_storeu_si128(out_chunk, packed);
            }

            let tail_start = chunks8 * 8;
            for (x, out_val) in out_tail.iter_mut().enumerate() {
                let mut acc: i32 = 0;
                for t in 0..tap_count {
                    acc += intermediate[row_indices[t] * h_row_len + tail_start + x] as i32
                        * w_a[t] as i32;
                }
                *out_val = ((acc + (1 << (I16_PRECISION - 1))) >> I16_PRECISION) as i16;
            }
            out_y += 1;
        }
    }
}

// =============================================================================
// Streaming single-row V-filter kernels (for StreamingResize i16 paths)
// =============================================================================

/// Streaming V-filter: u8 rows → u8 output via i16 weights using AVX2.
///
/// For sRGB gamma i16 streaming path. Processes row references from the ring
/// buffer (not a contiguous intermediate). Uses paired interleaving + madd_epi16
/// same as the batch kernel, but for a single output row.
#[archmage::arcane]
pub(crate) fn filter_v_row_u8_i16_v3(
    _token: X64V3Token,
    rows: &[&[u8]],
    output: &mut [u8],
    weights: &[i16],
) {
    let width = output.len();
    let tap_count = rows.len();
    debug_assert_eq!(tap_count, weights.len());

    let half = _mm256_set1_epi32(1 << (I16_PRECISION - 1));
    let chunks16 = width / 16;

    // Pre-chunk row slices for direct indexing (stack array, no heap alloc).
    const MAX_TAPS: usize = 128;
    debug_assert!(tap_count <= MAX_TAPS);
    let effective_taps = tap_count.min(MAX_TAPS);
    let empty_chunks: &[[u8; 16]] = &[];
    let mut row_chunks = [empty_chunks; MAX_TAPS];
    for (t, slot) in row_chunks.iter_mut().enumerate().take(effective_taps) {
        *slot = rows[t].as_chunks::<16>().0;
    }
    let row_chunks = &row_chunks[..effective_taps];

    // Pre-compute paired weights (stack array, no heap alloc).
    let pairs = tap_count / 2;
    let odd = !tap_count.is_multiple_of(2);

    let zero_ymm = _mm256_setzero_si256();
    let mut paired_wts = [zero_ymm; MAX_TAPS / 2];
    for p in 0..pairs {
        let w0 = weights[p * 2] as i32;
        let w1 = weights[p * 2 + 1] as i32;
        paired_wts[p] = _mm256_set1_epi32((w1 << 16) | (w0 & 0xFFFF));
    }
    let paired_wts = &paired_wts[..pairs];
    let odd_weight = if odd {
        _mm256_set1_epi32(weights[tap_count - 1] as i32 & 0xFFFF)
    } else {
        _mm256_setzero_si256()
    };

    let (out_chunks, out_tail) = output.as_chunks_mut::<16>();

    for (ci, out_chunk) in out_chunks.iter_mut().enumerate() {
        let mut acc_lo = _mm256_setzero_si256();
        let mut acc_hi = _mm256_setzero_si256();

        for (pw, row_pair) in paired_wts.iter().zip(row_chunks.chunks_exact(2)) {
            let src0 = _mm_loadu_si128(idx(row_pair[0], ci));
            let src1 = _mm_loadu_si128(idx(row_pair[1], ci));

            let il_lo = _mm_unpacklo_epi8(src0, src1);
            let il_hi = _mm_unpackhi_epi8(src0, src1);

            let ext_lo = _mm256_cvtepu8_epi16(il_lo);
            let ext_hi = _mm256_cvtepu8_epi16(il_hi);

            acc_lo = _mm256_add_epi32(acc_lo, _mm256_madd_epi16(ext_lo, *pw));
            acc_hi = _mm256_add_epi32(acc_hi, _mm256_madd_epi16(ext_hi, *pw));
        }

        if odd {
            let src = _mm_loadu_si128(idx(row_chunks[tap_count - 1], ci));
            let zero_src = _mm_setzero_si128();
            let il_lo = _mm_unpacklo_epi8(src, zero_src);
            let il_hi = _mm_unpackhi_epi8(src, zero_src);
            let ext_lo = _mm256_cvtepu8_epi16(il_lo);
            let ext_hi = _mm256_cvtepu8_epi16(il_hi);
            acc_lo = _mm256_add_epi32(acc_lo, _mm256_madd_epi16(ext_lo, odd_weight));
            acc_hi = _mm256_add_epi32(acc_hi, _mm256_madd_epi16(ext_hi, odd_weight));
        }

        let rounded_lo = _mm256_add_epi32(acc_lo, half);
        let rounded_hi = _mm256_add_epi32(acc_hi, half);
        let shifted_lo = _mm256_srai_epi32::<{ I16_PRECISION }>(rounded_lo);
        let shifted_hi = _mm256_srai_epi32::<{ I16_PRECISION }>(rounded_hi);

        let lo_lo = _mm256_castsi256_si128(shifted_lo);
        let lo_hi = _mm256_extracti128_si256::<1>(shifted_lo);
        let hi_lo = _mm256_castsi256_si128(shifted_hi);
        let hi_hi = _mm256_extracti128_si256::<1>(shifted_hi);

        let pack01 = _mm_packs_epi32(lo_lo, lo_hi);
        let pack23 = _mm_packs_epi32(hi_lo, hi_hi);
        let result = _mm_packus_epi16(pack01, pack23);

        _mm_storeu_si128(out_chunk, result);
    }

    // Scalar tail
    let tail_start = chunks16 * 16;
    for (x, out_byte) in out_tail.iter_mut().enumerate() {
        let mut acc: i32 = 0;
        for (row, &weight) in rows.iter().zip(weights.iter()) {
            acc += row[tail_start + x] as i32 * weight as i32;
        }
        let rounded = (acc + (1 << (I16_PRECISION - 1))) >> I16_PRECISION;
        *out_byte = rounded.clamp(0, 255) as u8;
    }
}

/// Streaming V-filter: i16 rows → i16 output via i16 weights using AVX2.
///
/// For linear i12 streaming path. Processes row references from the ring
/// buffer. Uses paired interleaving + madd_epi16, clamping output to [0, 4095].
#[archmage::arcane]
pub(crate) fn filter_v_row_i16_v3(
    _token: X64V3Token,
    rows: &[&[i16]],
    output: &mut [i16],
    weights: &[i16],
) {
    let width = output.len();
    let tap_count = rows.len();
    debug_assert_eq!(tap_count, weights.len());

    // Fall back to scalar for extreme upscale (>128 taps exceeds stack arrays).
    const MAX_TAPS: usize = 128;
    if tap_count > MAX_TAPS {
        super::scalar::filter_v_row_i16_scalar(archmage::ScalarToken, rows, output, weights);
        return;
    }

    let half = _mm256_set1_epi32(1 << (I16_PRECISION - 1));
    let chunks8 = width / 8;

    let effective_taps = tap_count;
    let empty_chunks: &[[i16; 8]] = &[];
    let mut row_chunks = [empty_chunks; MAX_TAPS];
    for (t, slot) in row_chunks.iter_mut().enumerate().take(effective_taps) {
        *slot = rows[t].as_chunks::<8>().0;
    }
    let row_chunks = &row_chunks[..effective_taps];

    // Pre-compute paired weights (stack array, no heap alloc).
    let pairs = tap_count / 2;
    let odd = !tap_count.is_multiple_of(2);

    let zero_ymm = _mm256_setzero_si256();
    let mut paired_wts = [zero_ymm; MAX_TAPS / 2];
    for p in 0..pairs {
        let w0 = weights[p * 2] as i32;
        let w1 = weights[p * 2 + 1] as i32;
        paired_wts[p] = _mm256_set1_epi32((w1 << 16) | (w0 & 0xFFFF));
    }
    let paired_wts = &paired_wts[..pairs];
    let odd_weight = if odd {
        _mm256_set1_epi32(weights[tap_count - 1] as i32 & 0xFFFF)
    } else {
        _mm256_setzero_si256()
    };

    let (out_chunks, out_tail) = output.as_chunks_mut::<8>();

    for (ci, out_chunk) in out_chunks.iter_mut().enumerate() {
        let mut acc = _mm256_setzero_si256();

        for (pw, row_pair) in paired_wts.iter().zip(row_chunks.chunks_exact(2)) {
            let src0 = _mm_loadu_si128(idx(row_pair[0], ci));
            let src1 = _mm_loadu_si128(idx(row_pair[1], ci));

            let il_lo = _mm_unpacklo_epi16(src0, src1);
            let il_hi = _mm_unpackhi_epi16(src0, src1);

            let combined = _mm256_set_m128i(il_hi, il_lo);
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(combined, *pw));
        }

        if odd {
            let src = _mm_loadu_si128(idx(row_chunks[tap_count - 1], ci));
            let zero_src = _mm_setzero_si128();
            let il_lo = _mm_unpacklo_epi16(src, zero_src);
            let il_hi = _mm_unpackhi_epi16(src, zero_src);
            let combined = _mm256_set_m128i(il_hi, il_lo);
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(combined, odd_weight));
        }

        let rounded = _mm256_add_epi32(acc, half);
        let shifted = _mm256_srai_epi32::<{ I16_PRECISION }>(rounded);

        let lo_128 = _mm256_castsi256_si128(shifted);
        let hi_128 = _mm256_extracti128_si256::<1>(shifted);
        let packed = _mm_packs_epi32(lo_128, hi_128);

        _mm_storeu_si128(out_chunk, packed);
    }

    // Scalar tail
    let tail_start = chunks8 * 8;
    for (x, out_val) in out_tail.iter_mut().enumerate() {
        let mut acc: i32 = 0;
        for (row, &weight) in rows.iter().zip(weights.iter()) {
            acc += row[tail_start + x] as i32 * weight as i32;
        }
        let rounded = (acc + (1 << (I16_PRECISION - 1))) >> I16_PRECISION;
        *out_val = rounded as i16;
    }
}

// =============================================================================
// f16 (IEEE 754 half-precision) kernels — F16C is guaranteed by X64V3Token
// =============================================================================

/// Bulk convert f32 → f16 row using F16C (vcvtps2ph).
/// Processes 8 f32 at a time → 8 f16 (stored as u16).
#[archmage::arcane]
pub(crate) fn f32_to_f16_row_v3(_token: X64V3Token, input: &[f32], output: &mut [u16]) {
    debug_assert_eq!(input.len(), output.len());
    let len = input.len();

    let (in_chunks, _) = input.as_chunks::<8>();
    let (out_chunks, _) = output.as_chunks_mut::<8>();

    for (in_chunk, out_chunk) in in_chunks.iter().zip(out_chunks.iter_mut()) {
        let floats = _mm256_loadu_ps(in_chunk);
        let halfs = _mm256_cvtps_ph::<0>(floats); // round-to-nearest-even
        _mm_storeu_si128(out_chunk, halfs);
    }

    // Scalar tail
    let chunks8 = in_chunks.len();
    for i in (chunks8 * 8)..len {
        output[i] = super::scalar::f32_to_f16_soft(input[i]);
    }
}

/// Bulk convert f16 → f32 row using F16C (vcvtph2ps).
/// Processes 8 f16 at a time → 8 f32.
#[archmage::arcane]
pub(crate) fn f16_to_f32_row_v3(_token: X64V3Token, input: &[u16], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());
    let len = input.len();

    let (in_chunks, _) = input.as_chunks::<8>();
    let (out_chunks, _) = output.as_chunks_mut::<8>();

    for (in_chunk, out_chunk) in in_chunks.iter().zip(out_chunks.iter_mut()) {
        let halfs = _mm_loadu_si128(in_chunk);
        let floats = _mm256_cvtph_ps(halfs);
        _mm256_storeu_ps(out_chunk, floats);
    }

    // Scalar tail
    let chunks8 = in_chunks.len();
    for i in (chunks8 * 8)..len {
        output[i] = super::scalar::f16_to_f32_soft(input[i]);
    }
}

/// Horizontal filter: f32 input → f16 (u16) output using AVX2+F16C.
/// Accumulates in f32, converts to f16 on store.
#[archmage::arcane]
pub(crate) fn filter_h_row_f32_to_f16_v3(
    _token: X64V3Token,
    input: &[f32],
    output: &mut [u16],
    weights: &F32WeightTable,
    channels: usize,
) {
    match channels {
        4 => filter_h_4ch_to_f16(_token, input, output, weights),
        3 => filter_h_3ch_to_f16(_token, input, output, weights),
        _ => filter_h_generic_to_f16(_token, input, output, weights, channels),
    }
}

/// 4-channel horizontal filter: f32 → f16 output.
/// Same accumulation as filter_h_4ch but stores via vcvtps2ph.
#[archmage::rite]
fn filter_h_4ch_to_f16(
    _token: X64V3Token,
    input: &[f32],
    output: &mut [u16],
    weights: &F32WeightTable,
) {
    let out_width = weights.len();
    let max_taps = weights.max_taps;

    let in_pixels_arr: &[[f32; 4]] = input.as_chunks().0;
    let (out_pixels, _) = output.as_chunks_mut::<4>();

    let chunks8 = max_taps / 8;
    let remainder = max_taps - chunks8 * 8;

    let perm01 = _mm256_set_epi32(1, 1, 1, 1, 0, 0, 0, 0);
    let perm23 = _mm256_set_epi32(3, 3, 3, 3, 2, 2, 2, 2);
    let perm45 = _mm256_set_epi32(5, 5, 5, 5, 4, 4, 4, 4);
    let perm67 = _mm256_set_epi32(7, 7, 7, 7, 6, 6, 6, 6);

    for out_x in 0..out_width {
        let left = weights.left[out_x] as usize;
        let w = weights.weights_padded(out_x);

        let flat_start = left * 4;
        let input_window = &input[flat_start..flat_start + max_taps * 4];
        let (pairs, _) = input_window.as_chunks::<8>();
        let (w_chunks, _) = w.as_chunks::<8>();

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        for c in 0..chunks8 {
            let w_vec = _mm256_loadu_ps(idx(w_chunks, c));
            let w01 = _mm256_permutevar8x32_ps(w_vec, perm01);
            let w23 = _mm256_permutevar8x32_ps(w_vec, perm23);
            let w45 = _mm256_permutevar8x32_ps(w_vec, perm45);
            let w67 = _mm256_permutevar8x32_ps(w_vec, perm67);

            let pi = c * 4;
            let p01 = _mm256_loadu_ps(idx(pairs, pi));
            let p23 = _mm256_loadu_ps(idx(pairs, pi + 1));
            let p45 = _mm256_loadu_ps(idx(pairs, pi + 2));
            let p67 = _mm256_loadu_ps(idx(pairs, pi + 3));

            acc0 = _mm256_fmadd_ps(p01, w01, acc0);
            acc1 = _mm256_fmadd_ps(p23, w23, acc1);
            acc2 = _mm256_fmadd_ps(p45, w45, acc2);
            acc3 = _mm256_fmadd_ps(p67, w67, acc3);
        }

        // Reduce 256→128
        let sum01 = _mm256_add_ps(acc0, acc1);
        let sum23 = _mm256_add_ps(acc2, acc3);
        let sum = _mm256_add_ps(sum01, sum23);
        let lo = _mm256_castps256_ps128(sum);
        let hi = _mm256_extractf128_ps::<1>(sum);
        let mut acc_128 = _mm_add_ps(lo, hi);

        // SSE remainder
        let t_start = chunks8 * 8;
        for t in 0..remainder {
            let tt = t_start + t;
            let w_val = _mm_set1_ps(*idx(w, tt));
            let pixel = _mm_loadu_ps(idx(in_pixels_arr, left + tt));
            acc_128 = _mm_fmadd_ps(pixel, w_val, acc_128);
        }

        // Convert 4 f32 → 4 f16 and store as 8 bytes (4 u16)
        let f16_vec = _mm_cvtps_ph::<0>(acc_128);
        _mm_storeu_si64(idx_mut(out_pixels, out_x), f16_vec);
    }
}

/// 3-channel horizontal filter: f32 → f16 output (scalar accumulation).
#[archmage::rite]
fn filter_h_3ch_to_f16(
    _token: X64V3Token,
    input: &[f32],
    output: &mut [u16],
    weights: &F32WeightTable,
) {
    for out_x in 0..weights.len() {
        let left = weights.left[out_x] as usize;
        let w = weights.weights(out_x);
        let out_offset = out_x * 3;

        let mut acc0 = 0.0f32;
        let mut acc1 = 0.0f32;
        let mut acc2 = 0.0f32;

        for (t, &weight) in w.iter().enumerate() {
            let in_offset = (left + t) * 3;
            acc0 += input[in_offset] * weight;
            acc1 += input[in_offset + 1] * weight;
            acc2 += input[in_offset + 2] * weight;
        }

        output[out_offset] = super::scalar::f32_to_f16_soft(acc0);
        output[out_offset + 1] = super::scalar::f32_to_f16_soft(acc1);
        output[out_offset + 2] = super::scalar::f32_to_f16_soft(acc2);
    }
}

/// Generic-channel horizontal filter: f32 → f16 output (scalar).
#[archmage::rite]
fn filter_h_generic_to_f16(
    _token: X64V3Token,
    input: &[f32],
    output: &mut [u16],
    weights: &F32WeightTable,
    channels: usize,
) {
    for out_x in 0..weights.len() {
        let left = weights.left[out_x] as usize;
        let w = weights.weights(out_x);
        let out_offset = out_x * channels;

        for c in 0..channels {
            let mut acc = 0.0f32;
            for (t, &weight) in w.iter().enumerate() {
                acc += input[(left + t) * channels + c] * weight;
            }
            output[out_offset + c] = super::scalar::f32_to_f16_soft(acc);
        }
    }
}

/// Streaming V-filter: f16 rows → f32 output using AVX2+F16C.
///
/// Per-tap: load 8 f16 → vcvtph2ps → fmadd_ps. Same 4×8 ILP structure
/// as filter_v_row_f32_v3 but with f16→f32 conversion per row load.
#[archmage::arcane]
pub(crate) fn filter_v_row_f16_v3(
    _token: X64V3Token,
    rows: &[&[u16]],
    output: &mut [f32],
    weights: &[f32],
) {
    let width = output.len();
    debug_assert_eq!(rows.len(), weights.len());

    // Zero the output buffer
    let (out_chunks, out_tail) = output.as_chunks_mut::<8>();
    let zero = _mm256_setzero_ps();
    for chunk in out_chunks.iter_mut() {
        _mm256_storeu_ps(chunk, zero);
    }
    for v in out_tail.iter_mut() {
        *v = 0.0;
    }

    let base8 = out_chunks.len() * 8;

    // Row-major accumulation: broadcast weight once, sweep entire row
    for (row, &weight) in rows.iter().zip(weights.iter()) {
        let w = _mm256_set1_ps(weight);
        let (row_chunks, _) = row.as_chunks::<8>();

        // Process 32 f16 values (4×8) at a time for ILP
        let (out_blocks, out_rem) = output[..base8].as_chunks_mut::<32>();

        for (out_block, row_block) in out_blocks.iter_mut().zip(row_chunks.chunks_exact(4)) {
            // Load 4 groups of 8 f16, convert to f32, FMA accumulate
            let h0 = _mm_loadu_si128(&row_block[0]);
            let h1 = _mm_loadu_si128(&row_block[1]);
            let h2 = _mm_loadu_si128(&row_block[2]);
            let h3 = _mm_loadu_si128(&row_block[3]);

            let s0 = _mm256_cvtph_ps(h0);
            let s1 = _mm256_cvtph_ps(h1);
            let s2 = _mm256_cvtph_ps(h2);
            let s3 = _mm256_cvtph_ps(h3);

            let oc: &mut [[f32; 8]] = out_block.as_mut_slice().as_chunks_mut().0;
            let a0 = _mm256_loadu_ps(&oc[0]);
            let a1 = _mm256_loadu_ps(&oc[1]);
            let a2 = _mm256_loadu_ps(&oc[2]);
            let a3 = _mm256_loadu_ps(&oc[3]);

            _mm256_storeu_ps(&mut oc[0], _mm256_fmadd_ps(s0, w, a0));
            _mm256_storeu_ps(&mut oc[1], _mm256_fmadd_ps(s1, w, a1));
            _mm256_storeu_ps(&mut oc[2], _mm256_fmadd_ps(s2, w, a2));
            _mm256_storeu_ps(&mut oc[3], _mm256_fmadd_ps(s3, w, a3));
        }

        // Remaining 8-element chunks
        let blocks4 = out_blocks.len();
        let rem_row_chunks = &row_chunks[blocks4 * 4..];
        let (rem_out_chunks, _) = out_rem.as_chunks_mut::<8>();
        for (out_chunk, row_chunk) in rem_out_chunks.iter_mut().zip(rem_row_chunks.iter()) {
            let halfs = _mm_loadu_si128(row_chunk);
            let src = _mm256_cvtph_ps(halfs);
            let acc = _mm256_loadu_ps(out_chunk);
            _mm256_storeu_ps(out_chunk, _mm256_fmadd_ps(src, w, acc));
        }

        // Scalar tail
        let w_scalar = weight;
        for x in base8..width {
            output[x] += super::scalar::f16_to_f32_soft(row[x]) * w_scalar;
        }
    }
}

/// Batch V-filter for fullframe: f16 intermediate → f32 output using AVX2+F16C.
///
/// Reads f16 intermediate buffer, accumulates in f32, outputs f32.
/// Same structure as filter_v_all_u8_i16 but with f16→f32 conversion.
#[archmage::arcane]
pub(crate) fn filter_v_all_f16_v3(
    _token: X64V3Token,
    intermediate: &[u16],
    output: &mut [f32],
    h_row_len: usize,
    in_h: usize,
    out_h: usize,
    weights: &F32WeightTable,
) {
    for out_y in 0..out_h {
        let left = weights.left[out_y];
        let tap_count = weights.tap_count(out_y);
        let w = weights.weights(out_y);
        let out_start = out_y * h_row_len;

        let in_h_i32 = in_h as i32 - 1;

        // Zero the output row
        let out_row = &mut output[out_start..out_start + h_row_len];
        let (out_chunks, out_tail) = out_row.as_chunks_mut::<8>();
        let zero = _mm256_setzero_ps();
        for chunk in out_chunks.iter_mut() {
            _mm256_storeu_ps(chunk, zero);
        }
        for v in out_tail.iter_mut() {
            *v = 0.0;
        }

        let base8 = out_chunks.len() * 8;

        // Accumulate each tap row
        for (t, &weight) in w[..tap_count].iter().enumerate() {
            let in_y = (left + t as i32).clamp(0, in_h_i32) as usize;
            let row_off = in_y * h_row_len;
            let row = &intermediate[row_off..row_off + h_row_len];
            let wv = _mm256_set1_ps(weight);
            let (row_chunks, _) = row.as_chunks::<8>();

            // Process 32 elements (4×8) at a time for ILP
            let (out_blocks, out_rem) = output[out_start..out_start + base8].as_chunks_mut::<32>();

            for (out_block, row_block) in out_blocks.iter_mut().zip(row_chunks.chunks_exact(4)) {
                let h0 = _mm_loadu_si128(&row_block[0]);
                let h1 = _mm_loadu_si128(&row_block[1]);
                let h2 = _mm_loadu_si128(&row_block[2]);
                let h3 = _mm_loadu_si128(&row_block[3]);

                let s0 = _mm256_cvtph_ps(h0);
                let s1 = _mm256_cvtph_ps(h1);
                let s2 = _mm256_cvtph_ps(h2);
                let s3 = _mm256_cvtph_ps(h3);

                let oc: &mut [[f32; 8]] = out_block.as_mut_slice().as_chunks_mut().0;
                let a0 = _mm256_loadu_ps(&oc[0]);
                let a1 = _mm256_loadu_ps(&oc[1]);
                let a2 = _mm256_loadu_ps(&oc[2]);
                let a3 = _mm256_loadu_ps(&oc[3]);

                _mm256_storeu_ps(&mut oc[0], _mm256_fmadd_ps(s0, wv, a0));
                _mm256_storeu_ps(&mut oc[1], _mm256_fmadd_ps(s1, wv, a1));
                _mm256_storeu_ps(&mut oc[2], _mm256_fmadd_ps(s2, wv, a2));
                _mm256_storeu_ps(&mut oc[3], _mm256_fmadd_ps(s3, wv, a3));
            }

            // Remaining 8-element chunks
            let blocks4 = out_blocks.len();
            let rem_row_chunks = &row_chunks[blocks4 * 4..];
            let (rem_out_chunks, _) = out_rem.as_chunks_mut::<8>();
            for (out_chunk, row_chunk) in rem_out_chunks.iter_mut().zip(rem_row_chunks.iter()) {
                let halfs = _mm_loadu_si128(row_chunk);
                let src = _mm256_cvtph_ps(halfs);
                let acc = _mm256_loadu_ps(out_chunk);
                _mm256_storeu_ps(out_chunk, _mm256_fmadd_ps(src, wv, acc));
            }

            // Scalar tail
            let w_scalar = weight;
            for x in base8..h_row_len {
                output[out_start + x] += super::scalar::f16_to_f32_soft(row[x]) * w_scalar;
            }
        }
    }
}

// =============================================================================
// Transfer function row processors — delegate to linear-srgb rites
// =============================================================================

/// Thin x8 wrappers: convert __m256 ↔ [f32; 8] and call linear-srgb rites.
/// The array conversions are optimized away since the rites are #[inline(always)].
macro_rules! tf_x8_wrapper {
    ($name:ident, $rite:path) => {
        #[archmage::rite]
        fn $name(_token: X64V3Token, v: __m256) -> __m256 {
            let mut arr = [0.0f32; 8];
            safe_unaligned_simd::x86_64::_mm256_storeu_ps(&mut arr, v);
            let result = $rite(_token, arr);
            safe_unaligned_simd::x86_64::_mm256_loadu_ps(&result)
        }
    };
}

tf_x8_wrapper!(
    srgb_to_linear_x8,
    linear_srgb::tokens::x8::tf_srgb_to_linear_v3
);
tf_x8_wrapper!(
    srgb_from_linear_x8,
    linear_srgb::tokens::x8::tf_linear_to_srgb_v3
);
tf_x8_wrapper!(
    bt709_to_linear_x8,
    linear_srgb::tokens::x8::bt709_to_linear_v3
);
tf_x8_wrapper!(
    bt709_from_linear_x8,
    linear_srgb::tokens::x8::linear_to_bt709_v3
);
tf_x8_wrapper!(pq_to_linear_x8, linear_srgb::tokens::x8::pq_to_linear_v3);
tf_x8_wrapper!(pq_from_linear_x8, linear_srgb::tokens::x8::linear_to_pq_v3);
tf_x8_wrapper!(hlg_to_linear_x8, linear_srgb::tokens::x8::hlg_to_linear_v3);
tf_x8_wrapper!(
    hlg_from_linear_x8,
    linear_srgb::tokens::x8::linear_to_hlg_v3
);

// --- Batch transfer function row processors ---

macro_rules! tf_row_v3 {
    ($name:ident, $tf_x8:ident, $tf_scalar:path) => {
        #[archmage::arcane]
        pub(crate) fn $name(_token: X64V3Token, row: &mut [f32], channels: usize, has_alpha: bool) {
            tf_row_inplace(_token, row, channels, has_alpha, $tf_x8, $tf_scalar);
        }
    };
}

tf_row_v3!(
    srgb_to_linear_row_v3,
    srgb_to_linear_x8,
    fastmath::srgb_to_linear
);
tf_row_v3!(
    srgb_from_linear_row_v3,
    srgb_from_linear_x8,
    fastmath::srgb_from_linear
);
tf_row_v3!(
    bt709_to_linear_row_v3,
    bt709_to_linear_x8,
    fastmath::bt709_to_linear
);
tf_row_v3!(
    bt709_from_linear_row_v3,
    bt709_from_linear_x8,
    fastmath::bt709_from_linear
);
tf_row_v3!(pq_to_linear_row_v3, pq_to_linear_x8, fastmath::pq_to_linear);
tf_row_v3!(
    pq_from_linear_row_v3,
    pq_from_linear_x8,
    fastmath::pq_from_linear
);
tf_row_v3!(
    hlg_to_linear_row_v3,
    hlg_to_linear_x8,
    fastmath::hlg_to_linear
);
tf_row_v3!(
    hlg_from_linear_row_v3,
    hlg_from_linear_x8,
    fastmath::hlg_from_linear
);

/// Common implementation for applying a SIMD transfer function to a row.
///
/// For RGBA (4ch, has_alpha): processes 8 floats (2 pixels) at a time,
/// restoring alpha channels after TF application.
///
/// For no-alpha: processes 8 floats at a time, all values through TF.
#[archmage::rite]
fn tf_row_inplace(
    _token: X64V3Token,
    row: &mut [f32],
    channels: usize,
    has_alpha: bool,
    tf_x8: fn(X64V3Token, __m256) -> __m256,
    tf_scalar: fn(f32) -> f32,
) {
    if has_alpha && channels == 4 {
        // RGBA: process 8 floats (2 RGBA pixels) at a time, restore alpha
        let alpha_mask = _mm256_castsi256_ps(_mm256_set_epi32(
            -1, 0, 0, 0, // pixel 1: alpha lane mask
            -1, 0, 0, 0, // pixel 0: alpha lane mask
        ));

        let (chunks, tail) = row.as_chunks_mut::<8>();
        for chunk in chunks.iter_mut() {
            let v = _mm256_loadu_ps(chunk);
            let converted = tf_x8(_token, v);
            // Blend: keep original alpha (lanes 3, 7), use converted for RGB
            let result = _mm256_blendv_ps(converted, v, alpha_mask);
            _mm256_storeu_ps(chunk, result);
        }

        // Scalar tail: 0 or 1 pixel (4 floats)
        for pixel in tail.chunks_exact_mut(4) {
            for v in &mut pixel[..3] {
                *v = tf_scalar(*v);
            }
        }
    } else if has_alpha && channels >= 2 {
        // Non-4ch with alpha: per-pixel, skip last channel
        for pixel in row.chunks_exact_mut(channels) {
            for v in &mut pixel[..channels - 1] {
                *v = tf_scalar(*v);
            }
        }
    } else {
        // No alpha: process all values flat
        let (chunks, tail) = row.as_chunks_mut::<8>();
        for chunk in chunks.iter_mut() {
            let v = _mm256_loadu_ps(chunk);
            let converted = tf_x8(_token, v);
            _mm256_storeu_ps(chunk, converted);
        }
        for v in tail.iter_mut() {
            *v = tf_scalar(*v);
        }
    }
}

// =============================================================================
// AVX-512 (x86v4) kernels
// =============================================================================

// Most kernels delegate to AVX2+FMA (v3) since X64V4Token ⊃ X64V3Token.
// Only kernels with real 512-bit implementations are written natively.
#[cfg(feature = "avx512")]
macro_rules! v4_delegate_v3 {
    ($(fn $name:ident($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty)?;)*) => {$(
        #[archmage::arcane]
        #[allow(dead_code)]
        pub(crate) fn $name(_token: X64V4Token, $($arg: $ty),*) $(-> $ret)? {
            paste_v3!($name)(_token.v3(), $($arg),*)
        }
    )*};
}

// Helper: call the _v3 variant by replacing _v4 suffix.
// We can't do suffix manipulation in declarative macros, so we use a second
// macro that takes the original _v4 name and generates the _v3 call.
#[cfg(feature = "avx512")]
macro_rules! paste_v3 {
    (u8_to_f32_row_v4) => {
        u8_to_f32_row_v3
    };
    (f32_to_u8_row_v4) => {
        f32_to_u8_row_v3
    };
    (premultiply_alpha_row_v4) => {
        premultiply_alpha_row_v3
    };
    (unpremultiply_alpha_row_v4) => {
        unpremultiply_alpha_row_v3
    };
    (filter_h_row_f32_v4) => {
        filter_h_row_f32_v3
    };
    (filter_v_row_f32_v4) => {
        filter_v_row_f32_v3
    };
    (filter_h_u8_i16_v4) => {
        filter_h_u8_i16_v3
    };
    (filter_h_u8_i16_4rows_v4) => {
        filter_h_u8_i16_4rows_v3
    };
    (premultiply_u8_row_v4) => {
        premultiply_u8_row_v3
    };
    (unpremultiply_u8_row_v4) => {
        unpremultiply_u8_row_v3
    };
    (srgb_u8_to_linear_f32_v4) => {
        srgb_u8_to_linear_f32_v3
    };
    (linear_f32_to_srgb_u8_v4) => {
        linear_f32_to_srgb_u8_v3
    };
    (filter_h_i16_i16_v4) => {
        filter_h_i16_i16_v3
    };
    (filter_v_all_i16_i16_v4) => {
        filter_v_all_i16_i16_v3
    };
    (filter_v_row_u8_i16_v4) => {
        filter_v_row_u8_i16_v3
    };
    (filter_v_row_i16_v4) => {
        filter_v_row_i16_v3
    };
    (filter_h_u8_to_i16_v4) => {
        filter_h_u8_to_i16_v3
    };
    (filter_h_u8_to_i16_4rows_v4) => {
        filter_h_u8_to_i16_4rows_v3
    };
    (filter_v_all_u8_i16_tiled_v4) => {
        filter_v_all_u8_i16_tiled_v3
    };
    (f32_to_f16_row_v4) => {
        f32_to_f16_row_v3
    };
    (f16_to_f32_row_v4) => {
        f16_to_f32_row_v3
    };
    (filter_h_row_f32_to_f16_v4) => {
        filter_h_row_f32_to_f16_v3
    };
    (filter_v_row_f16_v4) => {
        filter_v_row_f16_v3
    };
    (filter_v_all_f16_v4) => {
        filter_v_all_f16_v3
    };
    (srgb_to_linear_row_v4) => {
        srgb_to_linear_row_v3
    };
    (srgb_from_linear_row_v4) => {
        srgb_from_linear_row_v3
    };
    (bt709_to_linear_row_v4) => {
        bt709_to_linear_row_v3
    };
    (bt709_from_linear_row_v4) => {
        bt709_from_linear_row_v3
    };
    (pq_to_linear_row_v4) => {
        pq_to_linear_row_v3
    };
    (pq_from_linear_row_v4) => {
        pq_from_linear_row_v3
    };
    (hlg_to_linear_row_v4) => {
        hlg_to_linear_row_v3
    };
    (hlg_from_linear_row_v4) => {
        hlg_from_linear_row_v3
    };
}

#[cfg(feature = "avx512")]
v4_delegate_v3! {
    fn u8_to_f32_row_v4(input: &[u8], output: &mut [f32]);
    fn f32_to_u8_row_v4(input: &[f32], output: &mut [u8]);
    fn premultiply_alpha_row_v4(row: &mut [f32]);
    fn unpremultiply_alpha_row_v4(row: &mut [f32]);
    fn filter_h_row_f32_v4(input: &[f32], output: &mut [f32], weights: &F32WeightTable, channels: usize);
    fn filter_v_row_f32_v4(rows: &[&[f32]], output: &mut [f32], weights: &[f32]);
    fn filter_h_u8_i16_v4(input: &[u8], output: &mut [u8], weights: &I16WeightTable, channels: usize);
    fn filter_h_u8_i16_4rows_v4(in0: &[u8], in1: &[u8], in2: &[u8], in3: &[u8], out0: &mut [u8], out1: &mut [u8], out2: &mut [u8], out3: &mut [u8], weights: &I16WeightTable);
    fn premultiply_u8_row_v4(input: &[u8], output: &mut [u8]);
    fn unpremultiply_u8_row_v4(row: &mut [u8]);
    fn srgb_u8_to_linear_f32_v4(input: &[u8], output: &mut [f32], channels: usize, has_alpha: bool);
    fn linear_f32_to_srgb_u8_v4(input: &[f32], output: &mut [u8], channels: usize, has_alpha: bool);
    fn filter_h_i16_i16_v4(input: &[i16], output: &mut [i16], weights: &I16WeightTable, channels: usize);
    fn filter_v_all_i16_i16_v4(intermediate: &[i16], output: &mut [i16], h_row_len: usize, in_h: usize, out_h: usize, weights: &I16WeightTable);
    fn filter_v_row_u8_i16_v4(rows: &[&[u8]], output: &mut [u8], weights: &[i16]);
    fn filter_h_u8_to_i16_v4(input: &[u8], output: &mut [i16], weights: &I16WeightTable, channels: usize);
    fn filter_h_u8_to_i16_4rows_v4(in0: &[u8], in1: &[u8], in2: &[u8], in3: &[u8], out0: &mut [i16], out1: &mut [i16], out2: &mut [i16], out3: &mut [i16], weights: &I16WeightTable);
    fn filter_v_all_u8_i16_tiled_v4(intermediate: &[u8], output: &mut [u8], h_row_len: usize, in_h: usize, out_h: usize, weights: &I16WeightTable, tile_chunks: usize);
    fn f32_to_f16_row_v4(input: &[f32], output: &mut [u16]);
    fn f16_to_f32_row_v4(input: &[u16], output: &mut [f32]);
    fn filter_h_row_f32_to_f16_v4(input: &[f32], output: &mut [u16], weights: &F32WeightTable, channels: usize);
    fn filter_v_row_f16_v4(rows: &[&[u16]], output: &mut [f32], weights: &[f32]);
    fn filter_v_all_f16_v4(intermediate: &[u16], output: &mut [f32], h_row_len: usize, in_h: usize, out_h: usize, weights: &F32WeightTable);
    fn srgb_to_linear_row_v4(row: &mut [f32], channels: usize, has_alpha: bool);
    fn srgb_from_linear_row_v4(row: &mut [f32], channels: usize, has_alpha: bool);
    fn bt709_to_linear_row_v4(row: &mut [f32], channels: usize, has_alpha: bool);
    fn bt709_from_linear_row_v4(row: &mut [f32], channels: usize, has_alpha: bool);
    fn pq_to_linear_row_v4(row: &mut [f32], channels: usize, has_alpha: bool);
    fn pq_from_linear_row_v4(row: &mut [f32], channels: usize, has_alpha: bool);
    fn hlg_to_linear_row_v4(row: &mut [f32], channels: usize, has_alpha: bool);
    fn hlg_from_linear_row_v4(row: &mut [f32], channels: usize, has_alpha: bool);
}

/// Batch vertical filter: u8→u8 via i16 weights, AVX-512.
///
/// Processes 32 bytes per inner-loop iteration (vs 16 for AVX2).
/// Uses paired output row batching when consecutive rows share the same `left`.
#[cfg(feature = "avx512")]
#[archmage::arcane]
pub(crate) fn filter_v_all_u8_i16_v4(
    _token: X64V4Token,
    intermediate: &[u8],
    output: &mut [u8],
    h_row_len: usize,
    in_h: usize,
    out_h: usize,
    weights: &I16WeightTable,
) {
    const SHIFT: u32 = I16_PRECISION as u32;
    let half = _mm512_set1_epi32(1 << (I16_PRECISION - 1));
    // After packs_epi32 + packus_epi16 (both per-128-bit-lane), and
    // _mm256_unpacklo/hi lane scrambling, the 32 useful result bytes
    // sit at dword positions [0,4,1,5,8,12,9,13]. This permute gathers them.
    let dword_perm = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 13, 9, 12, 8, 5, 1, 4, 0);
    let chunks32 = h_row_len / 32;
    let in_h_i32 = in_h as i32;

    // Pre-chunk all intermediate rows for 32-byte blocks.
    let mut int_row_chunks: Vec<&[[u8; 32]]> = Vec::with_capacity(in_h);
    for y in 0..in_h {
        let row = &intermediate[y * h_row_len..y * h_row_len + h_row_len];
        int_row_chunks.push(row.as_chunks::<32>().0);
    }

    let max_taps = weights.max_taps;
    let mut row_indices = vec![0usize; max_taps];
    let mut paired_wts_a = vec![_mm512_setzero_si512(); max_taps.div_ceil(2)];
    let mut paired_wts_b = vec![_mm512_setzero_si512(); max_taps.div_ceil(2)];
    let empty_chunks: &[[u8; 32]] = &[];
    let mut tap_rows = vec![empty_chunks; max_taps];

    let mut out_y = 0;
    while out_y < out_h {
        let left = weights.left[out_y];
        let tap_count = weights.tap_count(out_y);
        let w_a = weights.weights(out_y);

        let batch2 = out_y + 1 < out_h
            && weights.left[out_y + 1] == left
            && weights.tap_count(out_y + 1) == tap_count;

        let pairs = tap_count / 2;
        let odd = !tap_count.is_multiple_of(2);

        for t in 0..tap_count {
            row_indices[t] = (left + t as i32).clamp(0, in_h_i32 - 1) as usize;
        }

        for p in 0..pairs {
            let w0 = w_a[p * 2] as i32;
            let w1 = w_a[p * 2 + 1] as i32;
            paired_wts_a[p] = _mm512_set1_epi32((w1 << 16) | (w0 & 0xFFFF));
        }
        let odd_wt_a = if odd {
            _mm512_set1_epi32(w_a[tap_count - 1] as i32 & 0xFFFF)
        } else {
            _mm512_setzero_si512()
        };

        for (t, &ri) in row_indices[..tap_count].iter().enumerate() {
            tap_rows[t] = &int_row_chunks[ri][..chunks32];
        }

        if batch2 {
            let w_b = weights.weights(out_y + 1);
            for p in 0..pairs {
                let w0 = w_b[p * 2] as i32;
                let w1 = w_b[p * 2 + 1] as i32;
                paired_wts_b[p] = _mm512_set1_epi32((w1 << 16) | (w0 & 0xFFFF));
            }
            let odd_wt_b = if odd {
                _mm512_set1_epi32(w_b[tap_count - 1] as i32 & 0xFFFF)
            } else {
                _mm512_setzero_si512()
            };

            let out_base_a = out_y * h_row_len;
            let (row_a, rest) = output[out_base_a..].split_at_mut(h_row_len);
            let row_b = &mut rest[..h_row_len];
            let (chunks_a, _) = row_a.as_chunks_mut::<32>();
            let (chunks_b, _) = row_b.as_chunks_mut::<32>();

            for ci in 0..chunks32 {
                let mut acc_a_lo = _mm512_setzero_si512();
                let mut acc_a_hi = _mm512_setzero_si512();
                let mut acc_b_lo = _mm512_setzero_si512();
                let mut acc_b_hi = _mm512_setzero_si512();

                for p in 0..pairs {
                    // Load 32 bytes from each of two tap rows.
                    let src0 = _mm256_loadu_si256(idx(tap_rows[p * 2], ci));
                    let src1 = _mm256_loadu_si256(idx(tap_rows[p * 2 + 1], ci));

                    // Interleave and extend to i16 in 512-bit registers.
                    let il_lo = _mm256_unpacklo_epi8(src0, src1);
                    let il_hi = _mm256_unpackhi_epi8(src0, src1);
                    let ext_lo = _mm512_cvtepu8_epi16(il_lo);
                    let ext_hi = _mm512_cvtepu8_epi16(il_hi);

                    acc_a_lo =
                        _mm512_add_epi32(acc_a_lo, _mm512_madd_epi16(ext_lo, paired_wts_a[p]));
                    acc_a_hi =
                        _mm512_add_epi32(acc_a_hi, _mm512_madd_epi16(ext_hi, paired_wts_a[p]));
                    acc_b_lo =
                        _mm512_add_epi32(acc_b_lo, _mm512_madd_epi16(ext_lo, paired_wts_b[p]));
                    acc_b_hi =
                        _mm512_add_epi32(acc_b_hi, _mm512_madd_epi16(ext_hi, paired_wts_b[p]));
                }

                if odd {
                    let src = _mm256_loadu_si256(idx(tap_rows[tap_count - 1], ci));
                    let zero = _mm256_setzero_si256();
                    let il_lo = _mm256_unpacklo_epi8(src, zero);
                    let il_hi = _mm256_unpackhi_epi8(src, zero);
                    let ext_lo = _mm512_cvtepu8_epi16(il_lo);
                    let ext_hi = _mm512_cvtepu8_epi16(il_hi);

                    acc_a_lo = _mm512_add_epi32(acc_a_lo, _mm512_madd_epi16(ext_lo, odd_wt_a));
                    acc_a_hi = _mm512_add_epi32(acc_a_hi, _mm512_madd_epi16(ext_hi, odd_wt_a));
                    acc_b_lo = _mm512_add_epi32(acc_b_lo, _mm512_madd_epi16(ext_lo, odd_wt_b));
                    acc_b_hi = _mm512_add_epi32(acc_b_hi, _mm512_madd_epi16(ext_hi, odd_wt_b));
                }

                // Pack row A: i32→i16→u8 via 512-bit pack, then dword permute.
                //
                // Both pack instructions operate per 128-bit lane, and
                // _mm256_unpacklo/hi_epi8 scrambles lanes (bytes 0-7 in lane 0,
                // bytes 16-23 in lane 1). After packs_epi32 + packus_epi16, the
                // 32 useful result bytes sit at dword positions [0,4,1,5,8,12,9,13].
                // A single vpermutexvar_epi32 gathers them contiguously.
                let ra_lo = _mm512_srai_epi32::<SHIFT>(_mm512_add_epi32(acc_a_lo, half));
                let ra_hi = _mm512_srai_epi32::<SHIFT>(_mm512_add_epi32(acc_a_hi, half));
                let packed_a16 = _mm512_packs_epi32(ra_lo, ra_hi);
                let packed_a8 = _mm512_packus_epi16(packed_a16, packed_a16);
                let perm_a = _mm512_permutexvar_epi32(dword_perm, packed_a8);
                _mm256_storeu_si256(idx_mut(chunks_a, ci), _mm512_castsi512_si256(perm_a));

                // Pack row B.
                let rb_lo = _mm512_srai_epi32::<SHIFT>(_mm512_add_epi32(acc_b_lo, half));
                let rb_hi = _mm512_srai_epi32::<SHIFT>(_mm512_add_epi32(acc_b_hi, half));
                let packed_b16 = _mm512_packs_epi32(rb_lo, rb_hi);
                let packed_b8 = _mm512_packus_epi16(packed_b16, packed_b16);
                let perm_b = _mm512_permutexvar_epi32(dword_perm, packed_b8);
                _mm256_storeu_si256(idx_mut(chunks_b, ci), _mm512_castsi512_si256(perm_b));
            }

            // Scalar tail for remaining bytes.
            let tail_start = chunks32 * 32;
            for x in tail_start..h_row_len {
                let mut acc_a: i32 = 0;
                let mut acc_b: i32 = 0;
                for t in 0..tap_count {
                    let v = intermediate[row_indices[t] * h_row_len + x] as i32;
                    acc_a += v * w_a[t] as i32;
                    acc_b += v * w_b[t] as i32;
                }
                output[out_base_a + x] =
                    ((acc_a + (1 << (I16_PRECISION - 1))) >> I16_PRECISION).clamp(0, 255) as u8;
                output[out_base_a + h_row_len + x] =
                    ((acc_b + (1 << (I16_PRECISION - 1))) >> I16_PRECISION).clamp(0, 255) as u8;
            }
            out_y += 2;
        } else {
            let out_base = out_y * h_row_len;
            let (out_chunks, _) = output[out_base..out_base + h_row_len].as_chunks_mut::<32>();

            for ci in 0..chunks32 {
                let mut acc_lo = _mm512_setzero_si512();
                let mut acc_hi = _mm512_setzero_si512();

                for p in 0..pairs {
                    let src0 = _mm256_loadu_si256(idx(tap_rows[p * 2], ci));
                    let src1 = _mm256_loadu_si256(idx(tap_rows[p * 2 + 1], ci));
                    let il_lo = _mm256_unpacklo_epi8(src0, src1);
                    let il_hi = _mm256_unpackhi_epi8(src0, src1);
                    let ext_lo = _mm512_cvtepu8_epi16(il_lo);
                    let ext_hi = _mm512_cvtepu8_epi16(il_hi);
                    acc_lo = _mm512_add_epi32(acc_lo, _mm512_madd_epi16(ext_lo, paired_wts_a[p]));
                    acc_hi = _mm512_add_epi32(acc_hi, _mm512_madd_epi16(ext_hi, paired_wts_a[p]));
                }

                if odd {
                    let src = _mm256_loadu_si256(idx(tap_rows[tap_count - 1], ci));
                    let zero = _mm256_setzero_si256();
                    let il_lo = _mm256_unpacklo_epi8(src, zero);
                    let il_hi = _mm256_unpackhi_epi8(src, zero);
                    let ext_lo = _mm512_cvtepu8_epi16(il_lo);
                    let ext_hi = _mm512_cvtepu8_epi16(il_hi);
                    acc_lo = _mm512_add_epi32(acc_lo, _mm512_madd_epi16(ext_lo, odd_wt_a));
                    acc_hi = _mm512_add_epi32(acc_hi, _mm512_madd_epi16(ext_hi, odd_wt_a));
                }

                let r_lo = _mm512_srai_epi32::<SHIFT>(_mm512_add_epi32(acc_lo, half));
                let r_hi = _mm512_srai_epi32::<SHIFT>(_mm512_add_epi32(acc_hi, half));
                let packed16 = _mm512_packs_epi32(r_lo, r_hi);
                let packed8 = _mm512_packus_epi16(packed16, packed16);
                let perm = _mm512_permutexvar_epi32(dword_perm, packed8);
                _mm256_storeu_si256(idx_mut(out_chunks, ci), _mm512_castsi512_si256(perm));
            }

            // Scalar tail.
            let tail_start = chunks32 * 32;
            for x in tail_start..h_row_len {
                let mut acc: i32 = 0;
                for t in 0..tap_count {
                    acc += intermediate[row_indices[t] * h_row_len + x] as i32 * w_a[t] as i32;
                }
                output[out_base + x] =
                    ((acc + (1 << (I16_PRECISION - 1))) >> I16_PRECISION).clamp(0, 255) as u8;
            }
            out_y += 1;
        }
    }
}

/// Streaming V-filter: i16 rows → i16 output via i16 weights, AVX-512.
///
/// Processes 32 i16 values per inner-loop iteration (vs 8 for AVX2).
/// Uses 512-bit `vpmaddwd` for paired tap accumulation.
#[cfg(feature = "avx512")]
#[archmage::arcane]
pub(crate) fn filter_v_row_i16_v4(
    _token: X64V4Token,
    rows: &[&[i16]],
    output: &mut [i16],
    weights: &[i16],
) {
    let width = output.len();
    let tap_count = rows.len();
    debug_assert_eq!(tap_count, weights.len());

    // Fall back to AVX2 for extreme upscale (>128 taps exceeds stack arrays).
    const MAX_TAPS: usize = 128;
    if tap_count > MAX_TAPS {
        filter_v_row_i16_v3(_token.v3(), rows, output, weights);
        return;
    }

    let half = _mm512_set1_epi32(1 << (I16_PRECISION - 1));
    let chunks32 = width / 32;

    // Pre-chunk row slices for direct indexing.
    let effective_taps = tap_count;
    let empty_chunks: &[[i16; 32]] = &[];
    let mut row_chunks = [empty_chunks; MAX_TAPS];
    for (t, slot) in row_chunks.iter_mut().enumerate().take(effective_taps) {
        *slot = rows[t].as_chunks::<32>().0;
    }
    let row_chunks = &row_chunks[..effective_taps];

    // Pre-compute paired weights.
    let pairs = tap_count / 2;
    let odd = !tap_count.is_multiple_of(2);

    let zero_zmm = _mm512_setzero_si512();
    let mut paired_wts = [zero_zmm; MAX_TAPS / 2];
    for p in 0..pairs {
        let w0 = weights[p * 2] as i32;
        let w1 = weights[p * 2 + 1] as i32;
        paired_wts[p] = _mm512_set1_epi32((w1 << 16) | (w0 & 0xFFFF));
    }
    let paired_wts = &paired_wts[..pairs];
    let odd_weight = if odd {
        _mm512_set1_epi32(weights[tap_count - 1] as i32 & 0xFFFF)
    } else {
        _mm512_setzero_si512()
    };

    let (out_chunks, out_tail) = output.as_chunks_mut::<32>();

    for (ci, out_chunk) in out_chunks.iter_mut().enumerate() {
        let mut acc_lo = _mm512_setzero_si512();
        let mut acc_hi = _mm512_setzero_si512();

        for (pw, row_pair) in paired_wts.iter().zip(row_chunks.chunks_exact(2)) {
            let src0 = _mm512_loadu_si512(idx(row_pair[0], ci));
            let src1 = _mm512_loadu_si512(idx(row_pair[1], ci));

            let il_lo = _mm512_unpacklo_epi16(src0, src1);
            let il_hi = _mm512_unpackhi_epi16(src0, src1);

            acc_lo = _mm512_add_epi32(acc_lo, _mm512_madd_epi16(il_lo, *pw));
            acc_hi = _mm512_add_epi32(acc_hi, _mm512_madd_epi16(il_hi, *pw));
        }

        if odd {
            let src = _mm512_loadu_si512(idx(row_chunks[tap_count - 1], ci));
            let il_lo = _mm512_unpacklo_epi16(src, zero_zmm);
            let il_hi = _mm512_unpackhi_epi16(src, zero_zmm);
            acc_lo = _mm512_add_epi32(acc_lo, _mm512_madd_epi16(il_lo, odd_weight));
            acc_hi = _mm512_add_epi32(acc_hi, _mm512_madd_epi16(il_hi, odd_weight));
        }

        let shifted_lo =
            _mm512_srai_epi32::<{ I16_PRECISION as u32 }>(_mm512_add_epi32(acc_lo, half));
        let shifted_hi =
            _mm512_srai_epi32::<{ I16_PRECISION as u32 }>(_mm512_add_epi32(acc_hi, half));

        // Pack i32→i16: _mm512_packs_epi32 operates per 128-bit lane.
        // acc_lo has positions [0-3, 8-11, 16-19, 24-27] (low halves).
        // acc_hi has positions [4-7, 12-15, 20-23, 28-31] (high halves).
        // packs_epi32(lo, hi) per lane gives [lo[0..3], hi[0..3]] = [0..7] per lane.
        // Result is already sequential: lanes 0-3 = positions 0..31.
        let packed = _mm512_packs_epi32(shifted_lo, shifted_hi);

        _mm512_storeu_si512(out_chunk, packed);
    }

    // Scalar tail
    let tail_start = chunks32 * 32;
    for (x, out_val) in out_tail.iter_mut().enumerate() {
        let mut acc: i32 = 0;
        for (row, &weight) in rows.iter().zip(weights.iter()) {
            acc += row[tail_start + x] as i32 * weight as i32;
        }
        let rounded = (acc + (1 << (I16_PRECISION - 1))) >> I16_PRECISION;
        *out_val = rounded as i16;
    }
}
