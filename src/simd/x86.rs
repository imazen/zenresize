//! x86-64 AVX2+FMA convolution and conversion kernels.
#![allow(unsafe_code)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::weights::{F32WeightTable, I16_PRECISION, I16WeightTable};
use archmage::X64V3Token;

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
    for i in 0..chunks8 {
        let base = i * 8;
        // Load 8 bytes into low 64 bits of __m128i
        let bytes = unsafe { _mm_loadl_epi64(input.as_ptr().add(base) as *const __m128i) };
        // Zero-extend 8×u8 → 8×i32
        let ints = _mm256_cvtepu8_epi32(bytes);
        // Convert 8×i32 → 8×f32
        let floats = _mm256_cvtepi32_ps(ints);
        // Multiply by 1/255
        let result = _mm256_mul_ps(floats, scale);
        unsafe { _mm256_storeu_ps(output.as_mut_ptr().add(base), result) };
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
    for i in 0..chunks8 {
        let base = i * 8;
        let floats = unsafe { _mm256_loadu_ps(input.as_ptr().add(base)) };
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
        unsafe {
            // Write 4 bytes from each lane
            let lo_val = _mm_cvtsi128_si32(lo) as u32;
            let hi_val = _mm_cvtsi128_si32(hi) as u32;
            core::ptr::write_unaligned(output.as_mut_ptr().add(base) as *mut u32, lo_val);
            core::ptr::write_unaligned(output.as_mut_ptr().add(base + 4) as *mut u32, hi_val);
        }
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
    for i in 0..chunks2 {
        let base = i * 8;
        let px = unsafe { _mm256_loadu_ps(row.as_ptr().add(base)) };
        // Extract alpha values: indices 3 and 7
        // Shuffle to broadcast alpha within each pixel
        // For pixel 0: [r,g,b,a] → need [a,a,a,a]
        // For pixel 1: [r,g,b,a] → need [a,a,a,a]
        let alpha = _mm256_permutevar8x32_ps(px, _mm256_set_epi32(7, 7, 7, 7, 3, 3, 3, 3));
        let result = _mm256_mul_ps(px, alpha);
        // Restore original alpha values (don't multiply alpha by itself)
        let mask = _mm256_blend_ps::<0b10001000>(result, px); // bits 3,7 from px
        unsafe { _mm256_storeu_ps(row.as_mut_ptr().add(base), mask) };
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

    for pixel in row.chunks_exact_mut(4) {
        let px = unsafe { _mm_loadu_ps(pixel.as_ptr()) };
        let alpha = _mm_shuffle_ps::<0xFF>(px, px); // broadcast alpha
        let mask = _mm_cmpgt_ps(alpha, threshold);
        let inv_alpha = _mm_div_ps(one, alpha);
        let inv_alpha_masked = _mm_and_ps(inv_alpha, mask); // zero if alpha <= threshold
        let unpremul = _mm_mul_ps(px, inv_alpha_masked);
        // Restore alpha channel
        let result = _mm_blend_ps::<0b1000>(unpremul, px);
        unsafe { _mm_storeu_ps(pixel.as_mut_ptr(), result) };
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
#[archmage::rite]
fn filter_h_4ch(_token: X64V3Token, input: &[f32], output: &mut [f32], weights: &F32WeightTable) {
    let out_width = weights.len();
    let max_taps = weights.max_taps;
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();

    for out_x in 0..out_width {
        let left = weights.left[out_x] as usize;
        let w_ptr = weights.weights_padded(out_x).as_ptr();
        let base_in = left * 4;

        let mut acc0 = _mm_setzero_ps();
        let mut acc1 = _mm_setzero_ps();
        let mut acc2 = _mm_setzero_ps();
        let mut acc3 = _mm_setzero_ps();

        let chunks4 = max_taps / 4;
        let remainder = max_taps % 4;

        for c in 0..chunks4 {
            let t = c * 4;
            unsafe {
                let w0 = _mm_set1_ps(*w_ptr.add(t));
                let w1 = _mm_set1_ps(*w_ptr.add(t + 1));
                let w2 = _mm_set1_ps(*w_ptr.add(t + 2));
                let w3 = _mm_set1_ps(*w_ptr.add(t + 3));

                let p0 = _mm_loadu_ps(in_ptr.add(base_in + t * 4));
                let p1 = _mm_loadu_ps(in_ptr.add(base_in + (t + 1) * 4));
                let p2 = _mm_loadu_ps(in_ptr.add(base_in + (t + 2) * 4));
                let p3 = _mm_loadu_ps(in_ptr.add(base_in + (t + 3) * 4));

                acc0 = _mm_fmadd_ps(p0, w0, acc0);
                acc1 = _mm_fmadd_ps(p1, w1, acc1);
                acc2 = _mm_fmadd_ps(p2, w2, acc2);
                acc3 = _mm_fmadd_ps(p3, w3, acc3);
            }
        }

        let t_start = chunks4 * 4;
        for t in 0..remainder {
            let tt = t_start + t;
            unsafe {
                let w_val = _mm_set1_ps(*w_ptr.add(tt));
                let pixel = _mm_loadu_ps(in_ptr.add(base_in + tt * 4));
                acc0 = _mm_fmadd_ps(pixel, w_val, acc0);
            }
        }

        let sum01 = _mm_add_ps(acc0, acc1);
        let sum23 = _mm_add_ps(acc2, acc3);
        let acc = _mm_add_ps(sum01, sum23);

        unsafe { _mm_storeu_ps(out_ptr.add(out_x * 4), acc) };
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

    let out_ptr = output.as_mut_ptr();
    let chunks8 = width / 8;
    let base8 = chunks8 * 8;

    // Zero the output buffer using AVX2
    let zero = _mm256_setzero_ps();
    for i in 0..chunks8 {
        unsafe { _mm256_storeu_ps(out_ptr.add(i * 8), zero) };
    }
    for v in &mut output[base8..width] {
        *v = 0.0;
    }

    // Row-major accumulation: broadcast weight once, sweep entire row
    for (row, &weight) in rows.iter().zip(weights.iter()) {
        let w = _mm256_set1_ps(weight);
        let row_ptr = row.as_ptr();

        // Process 32 floats (4×8) at a time for ILP
        let blocks4 = chunks8 / 4;
        let block_rem = chunks8 % 4;

        for b in 0..blocks4 {
            let base = b * 32;
            unsafe {
                let s0 = _mm256_loadu_ps(row_ptr.add(base));
                let s1 = _mm256_loadu_ps(row_ptr.add(base + 8));
                let s2 = _mm256_loadu_ps(row_ptr.add(base + 16));
                let s3 = _mm256_loadu_ps(row_ptr.add(base + 24));

                let a0 = _mm256_loadu_ps(out_ptr.add(base));
                let a1 = _mm256_loadu_ps(out_ptr.add(base + 8));
                let a2 = _mm256_loadu_ps(out_ptr.add(base + 16));
                let a3 = _mm256_loadu_ps(out_ptr.add(base + 24));

                _mm256_storeu_ps(out_ptr.add(base), _mm256_fmadd_ps(s0, w, a0));
                _mm256_storeu_ps(out_ptr.add(base + 8), _mm256_fmadd_ps(s1, w, a1));
                _mm256_storeu_ps(out_ptr.add(base + 16), _mm256_fmadd_ps(s2, w, a2));
                _mm256_storeu_ps(out_ptr.add(base + 24), _mm256_fmadd_ps(s3, w, a3));
            }
        }

        // Remaining 8-float chunks
        let rem_start = blocks4 * 32;
        for i in 0..block_rem {
            let base = rem_start + i * 8;
            unsafe {
                let src = _mm256_loadu_ps(row_ptr.add(base));
                let acc = _mm256_loadu_ps(out_ptr.add(base));
                _mm256_storeu_ps(out_ptr.add(base), _mm256_fmadd_ps(src, w, acc));
            }
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
    for i in 0..chunks2 {
        let base = i * 8;
        let bytes = unsafe { _mm_loadl_epi64(input.as_ptr().add(base) as *const __m128i) };
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

        // Store 8 bytes
        unsafe {
            core::ptr::copy_nonoverlapping(
                &packed as *const __m128i as *const u8,
                output.as_mut_ptr().add(base),
                8,
            );
        }
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
    for pixel in row.chunks_exact_mut(4) {
        let a = pixel[3];
        if a == 0 {
            pixel[0] = 0;
            pixel[1] = 0;
            pixel[2] = 0;
            continue;
        }
        if a == 255 {
            continue;
        }

        // Load pixel as 4 × i32 → f32
        let bytes =
            unsafe { _mm_cvtsi32_si128(core::ptr::read_unaligned(pixel.as_ptr() as *const i32)) };
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
        unsafe {
            core::ptr::write_unaligned(pixel.as_mut_ptr() as *mut u32, val_with_alpha);
        }
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
/// Port-5 pressure: 2 uops per 4 taps (cvtepu8 + shuffle) vs 6 uops in xmm path.
#[archmage::rite]
fn filter_h_u8_4ch(_token: X64V3Token, input: &[u8], output: &mut [u8], weights: &I16WeightTable) {
    let out_width = weights.len();
    let max_taps = weights.max_taps;
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();

    // Groups of 4 taps processed per ymm iteration
    let groups4 = max_taps / 4;
    let rem_taps = max_taps % 4;

    let ew_ptr = weights.expanded_4ch_ptr();
    let expanded_stride = weights.expanded_stride;

    // 256-bit shuffle mask: within each 128-bit lane, rearrange
    // [R0,G0,B0,A0,R1,G1,B1,A1] → [R0,R1,G0,G1,B0,B1,A0,A1]
    let ymm_shuffle = _mm256_set_epi8(
        15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0,
        15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0,
    );

    // XMM version for remainder handling (2 taps)
    let xmm_shuffle = _mm_set_epi8(
        15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0,
    );
    let half = _mm_set1_epi32(1 << (I16_PRECISION - 1));
    let zero = _mm_setzero_si128();

    for out_x in 0..out_width {
        let left = weights.left[out_x] as usize;
        let in_base = unsafe { in_ptr.add(left * 4) };
        let ew_base = unsafe { ew_ptr.add(out_x * expanded_stride) };

        let mut acc = _mm256_setzero_si256();

        // Process 4 taps per iteration using 256-bit registers
        for g in 0..groups4 {
            unsafe {
                // Load 4 RGBA pixels (16 bytes) → zero-extend to 16 × i16
                let pixels = _mm_loadu_si128(in_base.add(g * 16) as *const __m128i);
                let ext = _mm256_cvtepu8_epi16(pixels);
                let shuffled = _mm256_shuffle_epi8(ext, ymm_shuffle);

                // Load pre-expanded weights (32 bytes)
                let w = _mm256_loadu_si256(ew_base.add(g * 16) as *const __m256i);

                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(shuffled, w));
            }
        }

        // Combine upper and lower 128-bit lanes
        let lo = _mm256_castsi256_si128(acc);
        let hi = _mm256_extracti128_si256::<1>(acc);
        let mut final_acc = _mm_add_epi32(lo, hi);

        // Handle remaining 2 taps (if max_taps % 4 >= 2)
        if rem_taps >= 2 {
            let t = groups4 * 4;
            unsafe {
                let bytes = _mm_loadl_epi64(in_base.add(t * 4) as *const __m128i);
                let shuffled = _mm_shuffle_epi8(_mm_cvtepu8_epi16(bytes), xmm_shuffle);
                let w = _mm_broadcastd_epi32(
                    _mm_loadu_si32(weights.weights_padded(out_x).as_ptr().add(t) as *const _),
                );
                final_acc = _mm_add_epi32(final_acc, _mm_madd_epi16(shuffled, w));
            }
        }

        // Round, shift, pack to u8
        let rounded = _mm_add_epi32(final_acc, half);
        let shifted = _mm_srai_epi32::<{ I16_PRECISION }>(rounded);
        let packed16 = _mm_packs_epi32(shifted, zero);
        let packed8 = _mm_packus_epi16(packed16, zero);

        let pixel_val = _mm_cvtsi128_si32(packed8) as u32;
        unsafe {
            core::ptr::write_unaligned(out_ptr.add(out_x * 4) as *mut u32, pixel_val);
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
    let max_taps = weights.max_taps;

    let in_ptr0 = in0.as_ptr();
    let in_ptr1 = in1.as_ptr();
    let in_ptr2 = in2.as_ptr();
    let in_ptr3 = in3.as_ptr();
    let out_ptr0 = out0.as_mut_ptr();
    let out_ptr1 = out1.as_mut_ptr();
    let out_ptr2 = out2.as_mut_ptr();
    let out_ptr3 = out3.as_mut_ptr();

    let groups4 = max_taps / 4;
    let rem_taps = max_taps % 4;

    let ew_ptr = weights.expanded_4ch_ptr();
    let expanded_stride = weights.expanded_stride;

    let ymm_shuffle = _mm256_set_epi8(
        15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0,
        15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0,
    );

    let xmm_shuffle = _mm_set_epi8(
        15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0,
    );
    let half = _mm_set1_epi32(1 << (I16_PRECISION - 1));

    for out_x in 0..out_width {
        let left = weights.left[out_x] as usize;
        let byte_off = left * 4;
        let ew_base = unsafe { ew_ptr.add(out_x * expanded_stride) };

        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        let mut acc2 = _mm256_setzero_si256();
        let mut acc3 = _mm256_setzero_si256();

        for g in 0..groups4 {
            unsafe {
                let w = _mm256_loadu_si256(ew_base.add(g * 16) as *const __m256i);
                let g_off = g * 16;

                let p0 = _mm256_cvtepu8_epi16(
                    _mm_loadu_si128(in_ptr0.add(byte_off + g_off) as *const __m128i),
                );
                acc0 = _mm256_add_epi32(
                    acc0,
                    _mm256_madd_epi16(_mm256_shuffle_epi8(p0, ymm_shuffle), w),
                );

                let p1 = _mm256_cvtepu8_epi16(
                    _mm_loadu_si128(in_ptr1.add(byte_off + g_off) as *const __m128i),
                );
                acc1 = _mm256_add_epi32(
                    acc1,
                    _mm256_madd_epi16(_mm256_shuffle_epi8(p1, ymm_shuffle), w),
                );

                let p2 = _mm256_cvtepu8_epi16(
                    _mm_loadu_si128(in_ptr2.add(byte_off + g_off) as *const __m128i),
                );
                acc2 = _mm256_add_epi32(
                    acc2,
                    _mm256_madd_epi16(_mm256_shuffle_epi8(p2, ymm_shuffle), w),
                );

                let p3 = _mm256_cvtepu8_epi16(
                    _mm_loadu_si128(in_ptr3.add(byte_off + g_off) as *const __m128i),
                );
                acc3 = _mm256_add_epi32(
                    acc3,
                    _mm256_madd_epi16(_mm256_shuffle_epi8(p3, ymm_shuffle), w),
                );
            }
        }

        // Combine ymm lanes → xmm accumulators
        let mut f0 = _mm_add_epi32(_mm256_castsi256_si128(acc0), _mm256_extracti128_si256::<1>(acc0));
        let mut f1 = _mm_add_epi32(_mm256_castsi256_si128(acc1), _mm256_extracti128_si256::<1>(acc1));
        let mut f2 = _mm_add_epi32(_mm256_castsi256_si128(acc2), _mm256_extracti128_si256::<1>(acc2));
        let mut f3 = _mm_add_epi32(_mm256_castsi256_si128(acc3), _mm256_extracti128_si256::<1>(acc3));

        // Handle remaining 2 taps
        if rem_taps >= 2 {
            let t = groups4 * 4;
            let t_off = t * 4;
            unsafe {
                let w = _mm_broadcastd_epi32(
                    _mm_loadu_si32(weights.weights_padded(out_x).as_ptr().add(t) as *const _),
                );

                let s0 = _mm_shuffle_epi8(
                    _mm_cvtepu8_epi16(_mm_loadl_epi64(in_ptr0.add(byte_off + t_off) as *const __m128i)),
                    xmm_shuffle,
                );
                f0 = _mm_add_epi32(f0, _mm_madd_epi16(s0, w));

                let s1 = _mm_shuffle_epi8(
                    _mm_cvtepu8_epi16(_mm_loadl_epi64(in_ptr1.add(byte_off + t_off) as *const __m128i)),
                    xmm_shuffle,
                );
                f1 = _mm_add_epi32(f1, _mm_madd_epi16(s1, w));

                let s2 = _mm_shuffle_epi8(
                    _mm_cvtepu8_epi16(_mm_loadl_epi64(in_ptr2.add(byte_off + t_off) as *const __m128i)),
                    xmm_shuffle,
                );
                f2 = _mm_add_epi32(f2, _mm_madd_epi16(s2, w));

                let s3 = _mm_shuffle_epi8(
                    _mm_cvtepu8_epi16(_mm_loadl_epi64(in_ptr3.add(byte_off + t_off) as *const __m128i)),
                    xmm_shuffle,
                );
                f3 = _mm_add_epi32(f3, _mm_madd_epi16(s3, w));
            }
        }

        // Round and shift all 4 rows
        let s0 = _mm_srai_epi32::<{ I16_PRECISION }>(_mm_add_epi32(f0, half));
        let s1 = _mm_srai_epi32::<{ I16_PRECISION }>(_mm_add_epi32(f1, half));
        let s2 = _mm_srai_epi32::<{ I16_PRECISION }>(_mm_add_epi32(f2, half));
        let s3 = _mm_srai_epi32::<{ I16_PRECISION }>(_mm_add_epi32(f3, half));

        // Pack 4 pixels: i32×4 → i16×8 → u8×16
        let pack01 = _mm_packs_epi32(s0, s1);
        let pack23 = _mm_packs_epi32(s2, s3);
        let result = _mm_packus_epi16(pack01, pack23);

        // Store one pixel per row (4 bytes each)
        let out_off = out_x * 4;
        unsafe {
            core::ptr::write_unaligned(out_ptr0.add(out_off) as *mut u32, _mm_cvtsi128_si32(result) as u32);
            core::ptr::write_unaligned(out_ptr1.add(out_off) as *mut u32, _mm_extract_epi32::<1>(result) as u32);
            core::ptr::write_unaligned(out_ptr2.add(out_off) as *mut u32, _mm_extract_epi32::<2>(result) as u32);
            core::ptr::write_unaligned(out_ptr3.add(out_off) as *mut u32, _mm_extract_epi32::<3>(result) as u32);
        }
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
/// Pre-extracts raw row pointers to eliminate per-access bounds checking.
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
    let out_ptr = output.as_mut_ptr();

    // Pre-extract row pointers to avoid repeated bounds checking in inner loop.
    // 128 taps covers Lanczos3 at up to ~10x downscale.
    assert!(num_rows <= 128, "V kernel: too many taps ({num_rows} > 128)");
    let mut row_ptrs = [core::ptr::null::<u8>(); 128];
    for i in 0..num_rows {
        row_ptrs[i] = rows[i].as_ptr();
    }
    let row_ptrs = &row_ptrs[..num_rows];

    // Pre-compute all paired weight vectors before chunk loop.
    // For N taps, we have N/2 pairs + possibly 1 odd row.
    let pairs = num_rows / 2;
    let odd = num_rows % 2 != 0;

    // Max 64 pairs (128 rows). Stack array avoids allocation.
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

    for chunk in 0..chunks16 {
        let base = chunk * 16;
        let mut acc_lo = _mm256_setzero_si256();
        let mut acc_hi = _mm256_setzero_si256();

        for p in 0..pairs {
            let pw = paired_weights[p];
            unsafe {
                let src0 = _mm_loadu_si128(row_ptrs[p * 2].add(base) as *const __m128i);
                let src1 = _mm_loadu_si128(row_ptrs[p * 2 + 1].add(base) as *const __m128i);

                let interleaved_lo = _mm_unpacklo_epi8(src0, src1);
                let interleaved_hi = _mm_unpackhi_epi8(src0, src1);

                let ext_lo = _mm256_cvtepu8_epi16(interleaved_lo);
                let ext_hi = _mm256_cvtepu8_epi16(interleaved_hi);

                acc_lo = _mm256_add_epi32(acc_lo, _mm256_madd_epi16(ext_lo, pw));
                acc_hi = _mm256_add_epi32(acc_hi, _mm256_madd_epi16(ext_hi, pw));
            }
        }

        if odd {
            unsafe {
                let src = _mm_loadu_si128(row_ptrs[num_rows - 1].add(base) as *const __m128i);
                let zero_src = _mm_setzero_si128();

                let interleaved_lo = _mm_unpacklo_epi8(src, zero_src);
                let interleaved_hi = _mm_unpackhi_epi8(src, zero_src);

                let ext_lo = _mm256_cvtepu8_epi16(interleaved_lo);
                let ext_hi = _mm256_cvtepu8_epi16(interleaved_hi);

                acc_lo = _mm256_add_epi32(acc_lo, _mm256_madd_epi16(ext_lo, odd_weight));
                acc_hi = _mm256_add_epi32(acc_hi, _mm256_madd_epi16(ext_hi, odd_weight));
            }
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

        unsafe {
            _mm_storeu_si128(out_ptr.add(base) as *mut __m128i, result);
        }
    }

    // Scalar tail
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

// =============================================================================
// Pre-converted i16 path (for benchmarking alternative approaches)
// =============================================================================

/// Convert u8 → i16 (zero-extend) using AVX2.
#[archmage::arcane]
pub(crate) fn u8_to_i16_row_v3(_token: X64V3Token, input: &[u8], output: &mut [i16]) {
    debug_assert_eq!(input.len(), output.len());
    let len = input.len();

    let chunks16 = len / 16;
    for i in 0..chunks16 {
        let base = i * 16;
        unsafe {
            let bytes = _mm_loadu_si128(input.as_ptr().add(base) as *const __m128i);
            let extended = _mm256_cvtepu8_epi16(bytes);
            _mm256_storeu_si256(output.as_mut_ptr().add(base) as *mut __m256i, extended);
        }
    }

    for i in (chunks16 * 16)..len {
        output[i] = input[i] as i16;
    }
}

/// Integer horizontal convolution from pre-converted i16 input → u8 output.
#[archmage::arcane]
pub(crate) fn filter_h_i16_to_u8_v3(
    _token: X64V3Token,
    input: &[i16],
    output: &mut [u8],
    weights: &I16WeightTable,
    channels: usize,
) {
    match channels {
        4 => filter_h_i16_4ch(_token, input, output, weights),
        _ => filter_h_i16_generic(_token, input, output, weights, channels),
    }
}

/// Integer horizontal filter for 4-channel i16 input → u8 output.
#[archmage::rite]
fn filter_h_i16_4ch(
    _token: X64V3Token,
    input: &[i16],
    output: &mut [u8],
    weights: &I16WeightTable,
) {
    let out_width = weights.len();
    let max_taps = weights.max_taps;
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();

    let shuffle_mask = _mm_set_epi8(
        15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0,
    );

    let half = _mm_set1_epi32(1 << (I16_PRECISION - 1));
    let zero = _mm_setzero_si128();

    let pairs = max_taps / 2;
    let pairs2 = pairs / 2;
    let pairs_rem = pairs % 2;

    for out_x in 0..out_width {
        let left = weights.left[out_x] as usize;
        let w_ptr = weights.weights_padded(out_x).as_ptr();
        let in_base = unsafe { in_ptr.add(left * 4) };

        let mut acc0 = _mm_setzero_si128();
        let mut acc1 = _mm_setzero_si128();

        for p in 0..pairs2 {
            let t = p * 4;
            unsafe {
                let pixels0 = _mm_loadu_si128(in_base.add(t * 4) as *const __m128i);
                let shuffled0 = _mm_shuffle_epi8(pixels0, shuffle_mask);
                let w0 = _mm_broadcastd_epi32(_mm_loadu_si32(w_ptr.add(t) as *const _));
                acc0 = _mm_add_epi32(acc0, _mm_madd_epi16(shuffled0, w0));

                let pixels1 = _mm_loadu_si128(in_base.add((t + 2) * 4) as *const __m128i);
                let shuffled1 = _mm_shuffle_epi8(pixels1, shuffle_mask);
                let w1 = _mm_broadcastd_epi32(_mm_loadu_si32(w_ptr.add(t + 2) as *const _));
                acc1 = _mm_add_epi32(acc1, _mm_madd_epi16(shuffled1, w1));
            }
        }

        if pairs_rem > 0 {
            let t = pairs2 * 4;
            unsafe {
                let pixels = _mm_loadu_si128(in_base.add(t * 4) as *const __m128i);
                let shuffled = _mm_shuffle_epi8(pixels, shuffle_mask);
                let w = _mm_broadcastd_epi32(_mm_loadu_si32(w_ptr.add(t) as *const _));
                acc0 = _mm_add_epi32(acc0, _mm_madd_epi16(shuffled, w));
            }
        }

        let acc = _mm_add_epi32(acc0, acc1);
        let rounded = _mm_add_epi32(acc, half);
        let shifted = _mm_srai_epi32::<{ I16_PRECISION }>(rounded);
        let packed16 = _mm_packs_epi32(shifted, zero);
        let packed8 = _mm_packus_epi16(packed16, zero);

        let pixel_val = _mm_cvtsi128_si32(packed8) as u32;
        unsafe {
            core::ptr::write_unaligned(out_ptr.add(out_x * 4) as *mut u32, pixel_val);
        }
    }
}

/// Scalar fallback for i16 horizontal filter (arbitrary channel count).
#[archmage::rite]
fn filter_h_i16_generic(
    _token: X64V3Token,
    input: &[i16],
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
            let rounded = (acc + (1 << (I16_PRECISION - 1))) >> I16_PRECISION;
            output[out_base + c] = rounded.clamp(0, 255) as u8;
        }
    }
}
