//! x86-64 AVX2+FMA convolution and conversion kernels.
#![cfg_attr(feature = "unsafe_kernels", allow(unsafe_code))]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::weights::{F32WeightTable, I16_PRECISION, I16WeightTable};
use archmage::X64V3Token;

// Safe unaligned SIMD load/store — takes references instead of raw pointers.
// Explicit imports because names overlap with core::arch intrinsics.
#[cfg(target_arch = "x86_64")]
use safe_unaligned_simd::x86_64::{
    _mm256_loadu_ps, _mm256_loadu_si256, _mm256_storeu_ps, _mm_loadu_ps, _mm_loadu_si128,
    _mm_loadu_si32, _mm_loadu_si64, _mm_storeu_ps, _mm_storeu_si128, _mm_storeu_si64,
};

/// Load 16 bytes from `slice` at `offset` as `__m128i`.
///
/// With the default (safe) build, this uses slice bounds checking + safe_unaligned_simd.
/// With the `unsafe_kernels` feature, this uses unchecked pointer access for maximum
/// performance in hot loops where bounds have been pre-validated.
#[archmage::rite]
#[allow(unsafe_code)]
fn load_si128_at(_token: X64V3Token, slice: &[u8], offset: usize) -> __m128i {
    #[cfg(feature = "unsafe_kernels")]
    {
        // SAFETY: Caller must ensure offset + 16 <= slice.len().
        // The H kernel functions verify this with the has_full_padding / safe_end checks.
        unsafe {
            core::arch::x86_64::_mm_loadu_si128(slice.as_ptr().add(offset) as *const __m128i)
        }
    }
    #[cfg(not(feature = "unsafe_kernels"))]
    {
        _mm_loadu_si128(
            <&[u8; 16]>::try_from(&slice[offset..offset + 16]).unwrap(),
        )
    }
}

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

/// Horizontal filter for 4-channel (RGBA) data.
///
/// Uses 4 independent accumulators to break the serial FMA dependency chain.
/// FMA latency is 4 cycles, throughput is 2/cycle — with 4 independent chains
/// the CPU's out-of-order engine can keep both FMA ports busy.
///
/// Fixed `max_taps` loop count enables LLVM unrolling. Edge pixels that would
/// read beyond the input are handled by clamping to the last valid pixel;
/// the zero-padded weights ensure these clamped reads don't affect the result.
#[archmage::rite]
fn filter_h_4ch(_token: X64V3Token, input: &[f32], output: &mut [f32], weights: &F32WeightTable) {
    let out_width = weights.len();
    let max_taps = weights.max_taps;
    let in_pixels = input.len() / 4;
    let max_pixel = in_pixels - 1;

    // View input/output as per-pixel [f32; 4] chunks for bounds-proven access.
    let in_pixels_arr: &[[f32; 4]] = input.as_chunks().0;
    let (out_pixels, _) = output.as_chunks_mut::<4>();

    let chunks4 = max_taps / 4;
    let remainder = max_taps % 4;

    for out_x in 0..out_width {
        let left = weights.left[out_x] as usize;
        let w = weights.weights_padded(out_x);

        let mut acc0 = _mm_setzero_ps();
        let mut acc1 = _mm_setzero_ps();
        let mut acc2 = _mm_setzero_ps();
        let mut acc3 = _mm_setzero_ps();

        for c in 0..chunks4 {
            let t = c * 4;
            let w0 = _mm_set1_ps(w[t]);
            let w1 = _mm_set1_ps(w[t + 1]);
            let w2 = _mm_set1_ps(w[t + 2]);
            let w3 = _mm_set1_ps(w[t + 3]);

            // Clamp to last valid pixel; zero-padded weights make clamped reads inert.
            let p0 = _mm_loadu_ps(&in_pixels_arr[(left + t).min(max_pixel)]);
            let p1 = _mm_loadu_ps(&in_pixels_arr[(left + t + 1).min(max_pixel)]);
            let p2 = _mm_loadu_ps(&in_pixels_arr[(left + t + 2).min(max_pixel)]);
            let p3 = _mm_loadu_ps(&in_pixels_arr[(left + t + 3).min(max_pixel)]);

            acc0 = _mm_fmadd_ps(p0, w0, acc0);
            acc1 = _mm_fmadd_ps(p1, w1, acc1);
            acc2 = _mm_fmadd_ps(p2, w2, acc2);
            acc3 = _mm_fmadd_ps(p3, w3, acc3);
        }

        let t_start = chunks4 * 4;
        for t in 0..remainder {
            let tt = t_start + t;
            let w_val = _mm_set1_ps(w[tt]);
            let pixel = _mm_loadu_ps(&in_pixels_arr[(left + tt).min(max_pixel)]);
            acc0 = _mm_fmadd_ps(pixel, w_val, acc0);
        }

        let sum01 = _mm_add_ps(acc0, acc1);
        let sum23 = _mm_add_ps(acc2, acc3);
        let acc = _mm_add_ps(sum01, sum23);

        _mm_storeu_ps(&mut out_pixels[out_x], acc);
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

    let alpha_bcast = _mm_set_epi8(
        15, 14, 15, 14, 15, 14, 15, 14,
        7, 6, 7, 6, 7, 6, 7, 6,
    );
    let alpha_blend = _mm_set_epi16(
        -1, 0, 0, 0,
        -1, 0, 0, 0,
    );
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
        15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0,
        15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0,
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
        let ew_base = out_x * groups4;

        let mut acc = _mm256_setzero_si256();

        // groups4 is constant across all pixels — enables LLVM unrolling.
        for g in 0..groups4 {
            let byte_offset = (left + g * 4) * 4;
            let pixels = load_si128_at(_token, input, byte_offset);
            let ext = _mm256_cvtepu8_epi16(pixels);
            let shuffled = _mm256_shuffle_epi8(ext, ymm_shuffle);
            let w = _mm256_loadu_si256(&ew_chunks[ew_base + g]);
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
        15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0,
        15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0,
    );
    let half = _mm_set1_epi32(1 << (I16_PRECISION - 1));
    let zero = _mm_setzero_si128();

    let safe_end = (0..out_width)
        .rev()
        .find(|&x| (weights.left[x] as usize) + groups4 * 4 <= in_pixels)
        .map_or(0, |x| x + 1);

    // Edge pixels: scalar
    for out_x in safe_end..out_width {
        let left = weights.left[out_x] as usize;
        filter_h_u8_4ch_edge(_token, input, &mut output[out_x * 4..], out_x, left, weights);
    }

    // Interior pixels: SIMD
    if safe_end > 0 {
        let (out_pixels, _) = output.as_chunks_mut::<4>();

        for out_x in 0..safe_end {
            let left = weights.left[out_x] as usize;

            let ew = weights.weights_expanded_4ch(out_x);
            let (ew_chunks, _) = ew.as_chunks::<16>();

            let mut acc = _mm256_setzero_si256();

            for g in 0..groups4 {
                let byte_offset = left * 4 + g * 16;
                let pixels = load_si128_at(_token, input, byte_offset);
                let ext = _mm256_cvtepu8_epi16(pixels);
                let shuffled = _mm256_shuffle_epi8(ext, ymm_shuffle);
                let w = _mm256_loadu_si256(&ew_chunks[g]);
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
    let in_pixels = [in0.len(), in1.len(), in2.len(), in3.len()].iter().copied().min().unwrap() / 4;

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
        15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0,
        15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0,
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
        let ew_base = out_x * groups4;

        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        let mut acc2 = _mm256_setzero_si256();
        let mut acc3 = _mm256_setzero_si256();

        for g in 0..groups4 {
            let w = _mm256_loadu_si256(&ew_chunks[ew_base + g]);
            let byte_off = (left + g * 4) * 4;

            let p0 = _mm256_cvtepu8_epi16(load_si128_at(_token, in0, byte_off));
            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(_mm256_shuffle_epi8(p0, ymm_shuffle), w));

            let p1 = _mm256_cvtepu8_epi16(load_si128_at(_token, in1, byte_off));
            acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(_mm256_shuffle_epi8(p1, ymm_shuffle), w));

            let p2 = _mm256_cvtepu8_epi16(load_si128_at(_token, in2, byte_off));
            acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(_mm256_shuffle_epi8(p2, ymm_shuffle), w));

            let p3 = _mm256_cvtepu8_epi16(load_si128_at(_token, in3, byte_off));
            acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(_mm256_shuffle_epi8(p3, ymm_shuffle), w));
        }

        let f0 = _mm_add_epi32(_mm256_castsi256_si128(acc0), _mm256_extracti128_si256::<1>(acc0));
        let f1 = _mm_add_epi32(_mm256_castsi256_si128(acc1), _mm256_extracti128_si256::<1>(acc1));
        let f2 = _mm_add_epi32(_mm256_castsi256_si128(acc2), _mm256_extracti128_si256::<1>(acc2));
        let f3 = _mm_add_epi32(_mm256_castsi256_si128(acc3), _mm256_extracti128_si256::<1>(acc3));

        let s0 = _mm_srai_epi32::<{ I16_PRECISION }>(_mm_add_epi32(f0, half));
        let s1 = _mm_srai_epi32::<{ I16_PRECISION }>(_mm_add_epi32(f1, half));
        let s2 = _mm_srai_epi32::<{ I16_PRECISION }>(_mm_add_epi32(f2, half));
        let s3 = _mm_srai_epi32::<{ I16_PRECISION }>(_mm_add_epi32(f3, half));

        let pack01 = _mm_packs_epi32(s0, s1);
        let pack23 = _mm_packs_epi32(s2, s3);
        let result = _mm_packus_epi16(pack01, pack23);

        op0[out_x].copy_from_slice(&(_mm_cvtsi128_si32(result) as u32).to_ne_bytes());
        op1[out_x].copy_from_slice(&(_mm_extract_epi32::<1>(result) as u32).to_ne_bytes());
        op2[out_x].copy_from_slice(&(_mm_extract_epi32::<2>(result) as u32).to_ne_bytes());
        op3[out_x].copy_from_slice(&(_mm_extract_epi32::<3>(result) as u32).to_ne_bytes());
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

/// Integer vertical convolution: u8 rows → u8 output via i32 accumulator.
///
/// Uses AVX2 `_mm256_madd_epi16` to process pairs of rows efficiently.
/// Per-load guards eliminate unsafe — acceptable for this non-hot-path kernel
/// (the batch variant `filter_v_all_u8_i16_v3` handles the critical path).
#[archmage::arcane]
pub(crate) fn filter_v_u8_i16_v3(
    _token: X64V3Token,
    rows: &[&[u8]],
    output: &mut [u8],
    weights: &[i16],
) {
    let num_rows = rows.len();
    debug_assert_eq!(num_rows, weights.len());

    let half = _mm256_set1_epi32(1 << (I16_PRECISION - 1));

    // Pre-compute all paired weight vectors before chunk loop.
    let pairs = num_rows / 2;
    let odd = num_rows % 2 != 0;

    let mut paired_weights = vec![_mm256_setzero_si256(); pairs];
    for p in 0..pairs {
        let w0 = weights[p * 2] as i32;
        let w1 = weights[p * 2 + 1] as i32;
        paired_weights[p] = _mm256_set1_epi32((w1 << 16) | (w0 & 0xFFFF));
    }
    let odd_weight = if odd {
        _mm256_set1_epi32(weights[num_rows - 1] as i32 & 0xFFFF)
    } else {
        _mm256_setzero_si256()
    };

    let (out_chunks, out_tail) = output.as_chunks_mut::<16>();

    // Pre-chunk all rows for direct indexing
    let row_chunks_arr: Vec<&[[u8; 16]]> = rows.iter().map(|r| r.as_chunks::<16>().0).collect();

    for (ci, out_chunk) in out_chunks.iter_mut().enumerate() {
        let mut acc_lo = _mm256_setzero_si256();
        let mut acc_hi = _mm256_setzero_si256();

        for p in 0..pairs {
            let pw = paired_weights[p];
            let src0 = _mm_loadu_si128(&row_chunks_arr[p * 2][ci]);
            let src1 = _mm_loadu_si128(&row_chunks_arr[p * 2 + 1][ci]);

            let interleaved_lo = _mm_unpacklo_epi8(src0, src1);
            let interleaved_hi = _mm_unpackhi_epi8(src0, src1);

            let ext_lo = _mm256_cvtepu8_epi16(interleaved_lo);
            let ext_hi = _mm256_cvtepu8_epi16(interleaved_hi);

            acc_lo = _mm256_add_epi32(acc_lo, _mm256_madd_epi16(ext_lo, pw));
            acc_hi = _mm256_add_epi32(acc_hi, _mm256_madd_epi16(ext_hi, pw));
        }

        if odd {
            let src = _mm_loadu_si128(&row_chunks_arr[num_rows - 1][ci]);
            let zero_src = _mm_setzero_si128();

            let interleaved_lo = _mm_unpacklo_epi8(src, zero_src);
            let interleaved_hi = _mm_unpackhi_epi8(src, zero_src);

            let ext_lo = _mm256_cvtepu8_epi16(interleaved_lo);
            let ext_hi = _mm256_cvtepu8_epi16(interleaved_hi);

            acc_lo = _mm256_add_epi32(acc_lo, _mm256_madd_epi16(ext_lo, odd_weight));
            acc_hi = _mm256_add_epi32(acc_hi, _mm256_madd_epi16(ext_hi, odd_weight));
        }

        // Round and shift
        let rounded_lo = _mm256_add_epi32(acc_lo, half);
        let rounded_hi = _mm256_add_epi32(acc_hi, half);
        let shifted_lo = _mm256_srai_epi32::<{ I16_PRECISION }>(rounded_lo);
        let shifted_hi = _mm256_srai_epi32::<{ I16_PRECISION }>(rounded_hi);

        // Pack i32 → i16 → u8 using 128-bit ops to avoid lane-crossing
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
    let tail_start = out_chunks.len() * 16;
    for (x, out_byte) in out_tail.iter_mut().enumerate() {
        let mut acc: i32 = 0;
        for (row, &w) in rows.iter().zip(weights.iter()) {
            acc += row[tail_start + x] as i32 * w as i32;
        }
        let rounded = (acc + (1 << (I16_PRECISION - 1))) >> I16_PRECISION;
        *out_byte = rounded.clamp(0, 255) as u8;
    }
}

/// Batch vertical filter: process all output rows from the intermediate buffer.
///
/// Uses 2D guards on the contiguous intermediate buffer to eliminate unsafe.
/// Pre-computed row offsets with monotonic-in-tap ordering enable O(1) guard
/// construction per output row.
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
    let mut paired_wts = vec![_mm256_setzero_si256(); (max_taps + 1) / 2];

    for out_y in 0..out_h {
        let left = weights.left[out_y];
        let tap_count = weights.tap_count(out_y);
        let w = weights.weights(out_y);
        let out_base = out_y * h_row_len;

        // Pre-compute paired weights and clamped row indices for this output row.
        let pairs = tap_count / 2;
        let odd = tap_count % 2 != 0;

        for t in 0..tap_count {
            row_indices[t] = (left + t as i32).clamp(0, in_h_i32 - 1) as usize;
        }
        for p in 0..pairs {
            let w0 = w[p * 2] as i32;
            let w1 = w[p * 2 + 1] as i32;
            paired_wts[p] = _mm256_set1_epi32((w1 << 16) | (w0 & 0xFFFF));
        }
        let odd_weight = if odd {
            _mm256_set1_epi32(w[tap_count - 1] as i32 & 0xFFFF)
        } else {
            _mm256_setzero_si256()
        };

        let (out_chunks, out_tail) =
            output[out_base..out_base + h_row_len].as_chunks_mut::<16>();

        for (ci, out_chunk) in out_chunks.iter_mut().enumerate() {
            let mut acc_lo = _mm256_setzero_si256();
            let mut acc_hi = _mm256_setzero_si256();

            for p in 0..pairs {
                let pw = paired_wts[p];
                let src0 = _mm_loadu_si128(&int_row_chunks[row_indices[p * 2]][ci]);
                let src1 = _mm_loadu_si128(&int_row_chunks[row_indices[p * 2 + 1]][ci]);

                let il_lo = _mm_unpacklo_epi8(src0, src1);
                let il_hi = _mm_unpackhi_epi8(src0, src1);

                let ext_lo = _mm256_cvtepu8_epi16(il_lo);
                let ext_hi = _mm256_cvtepu8_epi16(il_hi);

                acc_lo = _mm256_add_epi32(acc_lo, _mm256_madd_epi16(ext_lo, pw));
                acc_hi = _mm256_add_epi32(acc_hi, _mm256_madd_epi16(ext_hi, pw));
            }

            if odd {
                let src = _mm_loadu_si128(&int_row_chunks[row_indices[tap_count - 1]][ci]);
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
            for t in 0..tap_count {
                let in_y = row_indices[t];
                acc += intermediate[in_y * h_row_len + tail_start + x] as i32 * w[t] as i32;
            }
            let rounded = (acc + (1 << (I16_PRECISION - 1))) >> I16_PRECISION;
            *out_byte = rounded.clamp(0, 255) as u8;
        }
    }
}

