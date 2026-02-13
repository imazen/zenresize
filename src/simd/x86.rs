//! x86-64 AVX2+FMA convolution and conversion kernels.
#![allow(unsafe_code)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::weights::{F32WeightTable, I16_PRECISION, I16WeightTable};
use archmage::X64V3Token;
use hoisted_bounds::{GuardedSlice, GuardedSlice2D, GuardedSliceMut};

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

    let chunks8 = len / 8;
    // Guard: reads 8 bytes at i*8 for i in 0..chunks8
    let in_guard = GuardedSlice::<u8, _, 8>::new(input, |i| i * 8, 0..chunks8);
    // Guard: writes 8 f32 at i*8 for i in 0..chunks8
    let mut out_guard = GuardedSliceMut::<f32, _, 8>::new(output, |i| i * 8, 0..chunks8);

    for i in 0..chunks8 {
        // Load 8 bytes into low 64 bits of __m128i
        let bytes = in_guard.loadl_epi64(i, _token);
        // Zero-extend 8×u8 → 8×i32
        let ints = _mm256_cvtepu8_epi32(bytes);
        // Convert 8×i32 → 8×f32
        let floats = _mm256_cvtepi32_ps(ints);
        // Multiply by 1/255
        let result = _mm256_mul_ps(floats, scale);
        out_guard.store_ps256(i, result, _token);
    }

    // Scalar tail
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

    let chunks8 = len / 8;
    // Guard: reads 8 f32 at i*8 for i in 0..chunks8
    let in_guard = GuardedSlice::<f32, _, 8>::new(input, |i| i * 8, 0..chunks8);
    // Guard: writes 8 u8 at i*8 for i in 0..chunks8
    let mut out_guard = GuardedSliceMut::<u8, _, 8>::new(output, |i| i * 8, 0..chunks8);

    for i in 0..chunks8 {
        let floats = in_guard.load_ps256(i, _token);
        // val * 255 + 0.5
        let scaled = _mm256_fmadd_ps(floats, scale, half);
        // Clamp to [0, 255]
        let clamped = _mm256_min_ps(_mm256_max_ps(scaled, zero), max_val);
        // Convert to i32 (truncation)
        let ints = _mm256_cvttps_epi32(clamped);
        // Pack 8×i32 → 8×u8 via two-stage pack
        // packus_epi32: 8×i32 → 8×u16 (saturating)
        let packed16 = _mm256_packus_epi32(ints, ints); // [a0..a3,a0..a3,a4..a7,a4..a7] as u16
        // packus_epi16: 8×u16 → 8×u8 (saturating)
        let packed8 = _mm256_packus_epi16(packed16, packed16);
        // AVX2 pack works in 128-bit lanes. Extract the bytes we need.
        // After double-pack: lane0=[a0,a1,a2,a3,a0,a1,a2,a3,...], lane1=[a4,a5,a6,a7,a4,a5,a6,a7,...]
        // We need the first 4 bytes from lane 0 and first 4 from lane 1.
        let lo = _mm256_extracti128_si256::<0>(packed8);
        let hi = _mm256_extracti128_si256::<1>(packed8);
        // Write 4 bytes from each lane (safe value extraction + guard write)
        let lo_val = _mm_cvtsi128_si32(lo) as u32;
        let hi_val = _mm_cvtsi128_si32(hi) as u32;
        let arr = out_guard.write(i);
        arr[..4].copy_from_slice(&lo_val.to_ne_bytes());
        arr[4..8].copy_from_slice(&hi_val.to_ne_bytes());
    }

    // Scalar tail
    for i in (chunks8 * 8)..len {
        output[i] = (input[i] * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
    }
}

/// Premultiply alpha in-place using AVX2 (processes 2 RGBA pixels = 8 floats at a time).
#[archmage::arcane]
pub(crate) fn premultiply_alpha_row_v3(_token: X64V3Token, row: &mut [f32]) {
    let chunks2 = row.len() / 8; // 2 pixels per 8 floats
    // Guard: reads/writes 8 f32 at i*8 for i in 0..chunks2
    let mut guard = GuardedSliceMut::<f32, _, 8>::new(row, |i| i * 8, 0..chunks2);

    for i in 0..chunks2 {
        let px = guard.load_ps256(i, _token);
        // Extract alpha values: indices 3 and 7
        // Shuffle to broadcast alpha within each pixel
        // For pixel 0: [r,g,b,a] → need [a,a,a,a]
        // For pixel 1: [r,g,b,a] → need [a,a,a,a]
        let alpha = _mm256_permutevar8x32_ps(px, _mm256_set_epi32(7, 7, 7, 7, 3, 3, 3, 3));
        let result = _mm256_mul_ps(px, alpha);
        // Restore original alpha values (don't multiply alpha by itself)
        let mask = _mm256_blend_ps::<0b10001000>(result, px); // bits 3,7 from px
        guard.store_ps256(i, mask, _token);
    }
    // Scalar tail for remaining pixels
    let remaining = &mut row[chunks2 * 8..];
    for pixel in remaining.chunks_exact_mut(4) {
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

    let pixels = row.len() / 4;
    let mut guard = GuardedSliceMut::<f32, _, 4>::new(row, |i| i * 4, 0..pixels);

    for i in 0..pixels {
        let px = guard.load_ps(i, _token);
        let alpha = _mm_shuffle_ps::<0xFF>(px, px); // broadcast alpha
        let mask = _mm_cmpgt_ps(alpha, threshold);
        let inv_alpha = _mm_div_ps(one, alpha);
        let inv_alpha_masked = _mm_and_ps(inv_alpha, mask); // zero if alpha <= threshold
        let unpremul = _mm_mul_ps(px, inv_alpha_masked);
        // Restore alpha channel
        let result = _mm_blend_ps::<0b1000>(unpremul, px);
        guard.store_ps(i, result, _token);
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
    let mut out_guard = GuardedSliceMut::<f32, _, 4>::new(output, |x| x * 4, 0..out_width);

    // Single guard for all pixel loads across all output pixels.
    // Clamping ensures edge pixels read from the last valid position;
    // zero-padded weights make these clamped reads contribute nothing.
    let in_guard = GuardedSlice::<f32, _, 4>::new(
        input,
        |px| px.min(max_pixel) * 4,
        0..in_pixels + max_taps,
    );

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

            let p0 = in_guard.load_ps(left + t, _token);
            let p1 = in_guard.load_ps(left + t + 1, _token);
            let p2 = in_guard.load_ps(left + t + 2, _token);
            let p3 = in_guard.load_ps(left + t + 3, _token);

            acc0 = _mm_fmadd_ps(p0, w0, acc0);
            acc1 = _mm_fmadd_ps(p1, w1, acc1);
            acc2 = _mm_fmadd_ps(p2, w2, acc2);
            acc3 = _mm_fmadd_ps(p3, w3, acc3);
        }

        let t_start = chunks4 * 4;
        for t in 0..remainder {
            let tt = t_start + t;
            let w_val = _mm_set1_ps(w[tt]);
            let pixel = in_guard.load_ps(left + tt, _token);
            acc0 = _mm_fmadd_ps(pixel, w_val, acc0);
        }

        let sum01 = _mm_add_ps(acc0, acc1);
        let sum23 = _mm_add_ps(acc2, acc3);
        let acc = _mm_add_ps(sum01, sum23);

        out_guard.store_ps(out_x, acc, _token);
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

    let chunks8 = width / 8;
    let base8 = chunks8 * 8;

    // Zero the output buffer using AVX2
    let zero = _mm256_setzero_ps();
    {
        // Guard: writes 8 f32 at i*8 for i in 0..chunks8
        let mut out_guard = GuardedSliceMut::<f32, _, 8>::new(output, |i| i * 8, 0..chunks8);
        for i in 0..chunks8 {
            out_guard.store_ps256(i, zero, _token);
        }
    }
    for v in &mut output[base8..width] {
        *v = 0.0;
    }

    // Row-major accumulation: broadcast weight once, sweep entire row
    for (row, &weight) in rows.iter().zip(weights.iter()) {
        let w = _mm256_set1_ps(weight);
        // Guard: reads 8 f32 at i*8 for i in 0..chunks8
        let row_guard = GuardedSlice::<f32, _, 8>::new(row, |i| i * 8, 0..chunks8);

        // Process 32 floats (4×8) at a time for ILP
        let blocks4 = chunks8 / 4;
        let block_rem = chunks8 % 4;

        {
            // Guard scoped to SIMD block; dropped before scalar tail accesses output
            let mut out_guard = GuardedSliceMut::<f32, _, 8>::new(output, |i| i * 8, 0..chunks8);

            for b in 0..blocks4 {
                let bi = b * 4;
                let s0 = row_guard.load_ps256(bi, _token);
                let s1 = row_guard.load_ps256(bi + 1, _token);
                let s2 = row_guard.load_ps256(bi + 2, _token);
                let s3 = row_guard.load_ps256(bi + 3, _token);

                let a0 = out_guard.load_ps256(bi, _token);
                let a1 = out_guard.load_ps256(bi + 1, _token);
                let a2 = out_guard.load_ps256(bi + 2, _token);
                let a3 = out_guard.load_ps256(bi + 3, _token);

                out_guard.store_ps256(bi, _mm256_fmadd_ps(s0, w, a0), _token);
                out_guard.store_ps256(bi + 1, _mm256_fmadd_ps(s1, w, a1), _token);
                out_guard.store_ps256(bi + 2, _mm256_fmadd_ps(s2, w, a2), _token);
                out_guard.store_ps256(bi + 3, _mm256_fmadd_ps(s3, w, a3), _token);
            }

            // Remaining 8-float chunks
            let rem_start = blocks4 * 4;
            for i in 0..block_rem {
                let ci = rem_start + i;
                let src = row_guard.load_ps256(ci, _token);
                let acc = out_guard.load_ps256(ci, _token);
                out_guard.store_ps256(ci, _mm256_fmadd_ps(src, w, acc), _token);
            }
        }

        // Scalar tail (out_guard dropped, output available)
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
    let len = input.len();

    // Shuffle mask to broadcast alpha within each pixel pair:
    // Input i16: [R0, G0, B0, A0, R1, G1, B1, A1]
    // Want:      [A0, A0, A0, A0, A1, A1, A1, A1]
    let alpha_bcast = _mm_set_epi8(
        15, 14, 15, 14, 15, 14, 15, 14, // A1 broadcast
        7, 6, 7, 6, 7, 6, 7, 6, // A0 broadcast
    );
    // Mask for blending: keep original alpha, use premultiplied RGB
    // Positions 3,7 (alpha channels in pixel 0 and 1) from original
    let alpha_blend = _mm_set_epi16(
        -1, 0, 0, 0, // pixel 1: A from orig, RGB from premul
        -1, 0, 0, 0, // pixel 0: A from orig, RGB from premul
    );
    let bias = _mm_set1_epi16(127);

    let chunks2 = len / 8; // 2 pixels at a time
    // Guard: reads 8 bytes at i*8 for i in 0..chunks2
    let in_guard = GuardedSlice::<u8, _, 8>::new(input, |i| i * 8, 0..chunks2);
    // Guard: writes 8 bytes at i*8 for i in 0..chunks2
    let mut out_guard = GuardedSliceMut::<u8, _, 8>::new(output, |i| i * 8, 0..chunks2);

    for i in 0..chunks2 {
        let bytes = in_guard.loadl_epi64(i, _token);
        let ext = _mm_cvtepu8_epi16(bytes); // [R0,G0,B0,A0,R1,G1,B1,A1] as i16

        // Broadcast alpha
        let alpha = _mm_shuffle_epi8(ext, alpha_bcast);

        // C * A
        let product = _mm_mullo_epi16(ext, alpha);

        // (C * A + 127) / 255 ≈ (product + 127 + ((product + 127) >> 8)) >> 8
        let biased = _mm_add_epi16(product, bias);
        let approx = _mm_srli_epi16::<8>(_mm_add_epi16(biased, _mm_srli_epi16::<8>(biased)));

        // Restore original alpha (blend: alpha positions from ext, rest from approx)
        let result = _mm_blendv_epi8(approx, ext, alpha_blend);

        // Pack i16 → u8
        let packed = _mm_packus_epi16(result, result);

        // Store low 8 bytes
        out_guard.storel_epi64(i, packed, _token);
    }

    // Scalar tail
    let tail = chunks2 * 8;
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

    // Process 1 pixel (4 bytes) at a time
    let pixels = row.len() / 4;
    let mut guard = GuardedSliceMut::<u8, _, 4>::new(row, |i| i * 4, 0..pixels);

    for i in 0..pixels {
        let a = guard.read(i)[3];
        if a == 0 {
            let arr = guard.write(i);
            arr[0] = 0;
            arr[1] = 0;
            arr[2] = 0;
            continue;
        }
        if a == 255 {
            continue;
        }

        // Load pixel as 4 × i32 → f32
        let bytes = guard.load_si32(i, _token);
        let ext = _mm_cvtepu8_epi32(bytes);
        let fpixel = _mm_cvtepi32_ps(ext);

        // Compute C * 255 / A using reciprocal
        let fa = _mm_set1_ps(a as f32);
        let inv_a = _mm_rcp_ps(fa); // approximate 1/A
        // Newton-Raphson refinement: inv_a = inv_a * (2 - a * inv_a)
        let refined = _mm_mul_ps(inv_a, _mm_sub_ps(_mm_set1_ps(2.0), _mm_mul_ps(fa, inv_a)));
        let result = _mm_add_ps(_mm_mul_ps(_mm_mul_ps(fpixel, scale), refined), half);
        let clamped = _mm_min_ps(_mm_max_ps(result, zero_f), max_val);

        // Convert back to u8
        let ints = _mm_cvttps_epi32(clamped);
        let packed16 = _mm_packs_epi32(ints, ints);
        let packed8 = _mm_packus_epi16(packed16, packed16);
        let val = _mm_cvtsi128_si32(packed8) as u32;

        // Restore original alpha
        let val_with_alpha = (val & 0x00FF_FFFF) | ((a as u32) << 24);
        guard.write_u32_ne(i, val_with_alpha);
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

    // 256-bit shuffle mask: within each 128-bit lane, rearrange
    // [R0,G0,B0,A0,R1,G1,B1,A1] → [R0,R1,G0,G1,B0,B1,A0,A1]
    let ymm_shuffle = _mm256_set_epi8(
        15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0,
        15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0,
    );

    let half = _mm_set1_epi32(1 << (I16_PRECISION - 1));
    let zero = _mm_setzero_si128();

    // Global guards: padding guarantees all accesses are in-bounds,
    // no clamping needed. One guard per buffer, constructed once.
    let in_guard = GuardedSlice::<u8, _, 16>::new(
        input,
        |px_start| px_start * 4,
        0..max_left + groups4 * 4,
    );

    let ew_all = weights.expanded_4ch_all();
    let ew_guard = GuardedSlice::<i16, _, 16>::new(
        ew_all,
        |group_idx| group_idx * 16,
        0..out_width * groups4,
    );

    let mut out_guard = GuardedSliceMut::<u8, _, 4>::new(output, |x| x * 4, 0..out_width);

    for out_x in 0..out_width {
        let left = weights.left[out_x] as usize;
        let ew_base = out_x * groups4;

        let mut acc = _mm256_setzero_si256();

        // groups4 is constant across all pixels — enables LLVM unrolling.
        for g in 0..groups4 {
            let pixels = in_guard.load_si128(left + g * 4, _token);
            let ext = _mm256_cvtepu8_epi16(pixels);
            let shuffled = _mm256_shuffle_epi8(ext, ymm_shuffle);
            let w = ew_guard.load_si256(ew_base + g, _token);
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
        out_guard.write_u32_ne(out_x, pixel_val);
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
        let mut out_guard = GuardedSliceMut::<u8, _, 4>::new(output, |x| x * 4, 0..safe_end);

        for out_x in 0..safe_end {
            let left = weights.left[out_x] as usize;

            let ew = weights.weights_expanded_4ch(out_x);
            let ew_guard = GuardedSlice::<i16, _, 16>::new(ew, |g| g * 16, 0..groups4);
            let in_guard = GuardedSlice::<u8, _, 16>::new(input, |g| left * 4 + g * 16, 0..groups4);

            let mut acc = _mm256_setzero_si256();

            for g in 0..groups4 {
                let pixels = in_guard.load_si128(g, _token);
                let ext = _mm256_cvtepu8_epi16(pixels);
                let shuffled = _mm256_shuffle_epi8(ext, ymm_shuffle);
                let w = ew_guard.load_si256(g, _token);
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
            out_guard.write_u32_ne(out_x, pixel_val);
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

    // Global guards: padding guarantees all accesses in-bounds, no clamping.
    let guard_range = 0..max_left + groups4 * 4;
    let ig0 = GuardedSlice::<u8, _, 16>::new(in0, |px| px * 4, guard_range.clone());
    let ig1 = GuardedSlice::<u8, _, 16>::new(in1, |px| px * 4, guard_range.clone());
    let ig2 = GuardedSlice::<u8, _, 16>::new(in2, |px| px * 4, guard_range.clone());
    let ig3 = GuardedSlice::<u8, _, 16>::new(in3, |px| px * 4, guard_range);

    let ew_all = weights.expanded_4ch_all();
    let ew_guard = GuardedSlice::<i16, _, 16>::new(
        ew_all,
        |group_idx| group_idx * 16,
        0..out_width * groups4,
    );

    let mut og0 = GuardedSliceMut::<u8, _, 4>::new(out0, |x| x * 4, 0..out_width);
    let mut og1 = GuardedSliceMut::<u8, _, 4>::new(out1, |x| x * 4, 0..out_width);
    let mut og2 = GuardedSliceMut::<u8, _, 4>::new(out2, |x| x * 4, 0..out_width);
    let mut og3 = GuardedSliceMut::<u8, _, 4>::new(out3, |x| x * 4, 0..out_width);

    for out_x in 0..out_width {
        let left = weights.left[out_x] as usize;
        let ew_base = out_x * groups4;

        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        let mut acc2 = _mm256_setzero_si256();
        let mut acc3 = _mm256_setzero_si256();

        for g in 0..groups4 {
            let w = ew_guard.load_si256(ew_base + g, _token);
            let px = left + g * 4;

            let p0 = _mm256_cvtepu8_epi16(ig0.load_si128(px, _token));
            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(_mm256_shuffle_epi8(p0, ymm_shuffle), w));

            let p1 = _mm256_cvtepu8_epi16(ig1.load_si128(px, _token));
            acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(_mm256_shuffle_epi8(p1, ymm_shuffle), w));

            let p2 = _mm256_cvtepu8_epi16(ig2.load_si128(px, _token));
            acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(_mm256_shuffle_epi8(p2, ymm_shuffle), w));

            let p3 = _mm256_cvtepu8_epi16(ig3.load_si128(px, _token));
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

        og0.write_u32_ne(out_x, _mm_cvtsi128_si32(result) as u32);
        og1.write_u32_ne(out_x, _mm_extract_epi32::<1>(result) as u32);
        og2.write_u32_ne(out_x, _mm_extract_epi32::<2>(result) as u32);
        og3.write_u32_ne(out_x, _mm_extract_epi32::<3>(result) as u32);
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
    let width = output.len();
    let num_rows = rows.len();
    debug_assert_eq!(num_rows, weights.len());

    let half = _mm256_set1_epi32(1 << (I16_PRECISION - 1));

    // Pre-compute all paired weight vectors before chunk loop.
    let pairs = num_rows / 2;
    let odd = num_rows % 2 != 0;

    assert!(num_rows <= 128, "V kernel: too many taps ({num_rows} > 128)");
    let mut paired_weights = [_mm256_setzero_si256(); 64];
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

    let chunks16 = width / 16;

    // Output guard: writes 16 bytes at chunk * 16
    let mut out_guard = GuardedSliceMut::<u8, _, 16>::new(output, |c| c * 16, 0..chunks16);

    for chunk in 0..chunks16 {
        let base = chunk * 16;
        let mut acc_lo = _mm256_setzero_si256();
        let mut acc_hi = _mm256_setzero_si256();

        for p in 0..pairs {
            let pw = paired_weights[p];
            // Per-load guards: each row is a separate slice, O(1) construction cost
            let g0 = GuardedSlice::<u8, _, 16>::new(rows[p * 2], |_| base, 0..1);
            let g1 = GuardedSlice::<u8, _, 16>::new(rows[p * 2 + 1], |_| base, 0..1);
            let src0 = g0.load_si128(0, _token);
            let src1 = g1.load_si128(0, _token);

            let interleaved_lo = _mm_unpacklo_epi8(src0, src1);
            let interleaved_hi = _mm_unpackhi_epi8(src0, src1);

            let ext_lo = _mm256_cvtepu8_epi16(interleaved_lo);
            let ext_hi = _mm256_cvtepu8_epi16(interleaved_hi);

            acc_lo = _mm256_add_epi32(acc_lo, _mm256_madd_epi16(ext_lo, pw));
            acc_hi = _mm256_add_epi32(acc_hi, _mm256_madd_epi16(ext_hi, pw));
        }

        if odd {
            let g = GuardedSlice::<u8, _, 16>::new(rows[num_rows - 1], |_| base, 0..1);
            let src = g.load_si128(0, _token);
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

        out_guard.store_si128(chunk, result, _token);
    }

    // Scalar tail (out_guard dropped, output available)
    let tail_start = chunks16 * 16;
    for x in tail_start..width {
        let mut acc: i32 = 0;
        for (row, &w) in rows.iter().zip(weights.iter()) {
            acc += row[x] as i32 * w as i32;
        }
        let rounded = (acc + (1 << (I16_PRECISION - 1))) >> I16_PRECISION;
        output[x] = rounded.clamp(0, 255) as u8;
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

    for out_y in 0..out_h {
        let left = weights.left[out_y];
        let tap_count = weights.tap_count(out_y);
        let w = weights.weights(out_y);
        let out_base = out_y * h_row_len;

        // Pre-compute paired weights and row offsets for this output row.
        let pairs = tap_count / 2;
        let odd = tap_count % 2 != 0;

        // Stack arrays: row byte offsets and paired weight vectors.
        // 128 taps max (Lanczos3 at 10× downscale).
        let mut row_offsets = [0usize; 128];
        let mut paired_wts = [_mm256_setzero_si256(); 64];

        for t in 0..tap_count {
            let in_y = (left + t as i32).clamp(0, in_h_i32 - 1) as usize;
            row_offsets[t] = in_y * h_row_len;
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

        // 2D guard on intermediate: (tap, chunk) → row_offsets[tap] + chunk * 16
        // row_offsets is monotonically non-decreasing (clamped row indices).
        let int_guard = GuardedSlice2D::<u8, _, 16>::new(
            intermediate,
            |tap, chunk| row_offsets[tap] + chunk * 16,
            0..tap_count,
            0..chunks16,
        );

        // Output guard scoped to this row's SIMD region; dropped before scalar tail.
        {
            let mut out_guard = GuardedSliceMut::<u8, _, 16>::new(
                &mut output[out_base..out_base + h_row_len],
                |c| c * 16,
                0..chunks16,
            );

            for chunk in 0..chunks16 {
                let mut acc_lo = _mm256_setzero_si256();
                let mut acc_hi = _mm256_setzero_si256();

                for p in 0..pairs {
                    let pw = paired_wts[p];
                    let src0 = int_guard.load_si128(p * 2, chunk, _token);
                    let src1 = int_guard.load_si128(p * 2 + 1, chunk, _token);

                    let il_lo = _mm_unpacklo_epi8(src0, src1);
                    let il_hi = _mm_unpackhi_epi8(src0, src1);

                    let ext_lo = _mm256_cvtepu8_epi16(il_lo);
                    let ext_hi = _mm256_cvtepu8_epi16(il_hi);

                    acc_lo = _mm256_add_epi32(acc_lo, _mm256_madd_epi16(ext_lo, pw));
                    acc_hi = _mm256_add_epi32(acc_hi, _mm256_madd_epi16(ext_hi, pw));
                }

                if odd {
                    let src = int_guard.load_si128(tap_count - 1, chunk, _token);
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

                out_guard.store_si128(chunk, result, _token);
            }
        }

        // Scalar tail (out_guard dropped, output available)
        let tail_start = chunks16 * 16;
        for x in tail_start..h_row_len {
            let mut acc: i32 = 0;
            for t in 0..tap_count {
                let in_y = (left + t as i32).clamp(0, in_h_i32 - 1) as usize;
                acc += intermediate[in_y * h_row_len + x] as i32 * w[t] as i32;
            }
            let rounded = (acc + (1 << (I16_PRECISION - 1))) >> I16_PRECISION;
            output[out_base + x] = rounded.clamp(0, 255) as u8;
        }
    }
}

