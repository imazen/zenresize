//! x86-64 AVX2+FMA convolution and conversion kernels.
#![allow(unsafe_code)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::weights::{F32WeightTable, I16WeightTable, I16_PRECISION};
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
        let alpha = _mm256_permutevar8x32_ps(
            px,
            _mm256_set_epi32(7, 7, 7, 7, 3, 3, 3, 3),
        );
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
    for x in base8..width {
        output[x] = 0.0;
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
/// Processes 2 taps per madd_epi16 instruction. Shuffle mask rearranges
/// zero-extended pixel pairs into channel-interleaved format for paired
/// multiply-accumulate.
#[archmage::rite]
fn filter_h_u8_4ch(
    _token: X64V3Token,
    input: &[u8],
    output: &mut [u8],
    weights: &I16WeightTable,
) {
    let out_width = weights.len();
    let max_taps = weights.max_taps;

    // Shuffle mask to rearrange [R0,G0,B0,A0,R1,G1,B1,A1] (i16)
    // into [R0,R1,G0,G1,B0,B1,A0,A1] (i16) for madd_epi16
    let shuffle_mask = _mm_set_epi8(
        15, 14, 7, 6,   // A1, A0
        13, 12, 5, 4,   // B1, B0
        11, 10, 3, 2,   // G1, G0
        9, 8, 1, 0,     // R1, R0
    );

    let half = _mm_set1_epi32(1 << (I16_PRECISION as i32 - 1)); // rounding bias
    let zero = _mm_setzero_si128();

    for out_x in 0..out_width {
        let left = weights.left[out_x] as usize;
        let w_ptr = weights.weights_padded(out_x);
        let in_base = left * 4;

        let mut acc = _mm_setzero_si128(); // i32x4: [R, G, B, A]

        // Process 2 taps at a time via madd_epi16
        let pairs = max_taps / 2;
        for p in 0..pairs {
            let t = p * 2;
            let pixel_offset = in_base + t * 4;

            // Load 2 RGBA pixels (8 bytes) → zero-extend to 8×i16
            let bytes = unsafe {
                _mm_loadl_epi64(input.as_ptr().add(pixel_offset) as *const __m128i)
            };
            let pixels_i16 = _mm_cvtepu8_epi16(bytes);

            // Shuffle to channel-pair format: [R0,R1,G0,G1,B0,B1,A0,A1]
            let shuffled = _mm_shuffle_epi8(pixels_i16, shuffle_mask);

            // Prepare paired weights: [w0, w1, w0, w1, w0, w1, w0, w1]
            let w0 = w_ptr[t] as i32;
            let w1 = w_ptr[t + 1] as i32;
            let paired_w = _mm_set_epi16(
                w1 as i16, w0 as i16,
                w1 as i16, w0 as i16,
                w1 as i16, w0 as i16,
                w1 as i16, w0 as i16,
            );

            // madd_epi16: [R0*w0+R1*w1, G0*w0+G1*w1, B0*w0+B1*w1, A0*w0+A1*w1]
            let product = _mm_madd_epi16(shuffled, paired_w);
            acc = _mm_add_epi32(acc, product);
        }

        // Shift right by I16_PRECISION with rounding, clamp to u8
        let rounded = _mm_add_epi32(acc, half);
        let shifted = _mm_srai_epi32::<{ I16_PRECISION as i32 }>(rounded);
        // Pack i32 → i16 → u8 with saturation
        let packed16 = _mm_packs_epi32(shifted, zero); // i32→i16 (saturating)
        let packed8 = _mm_packus_epi16(packed16, zero); // i16→u8 (saturating)

        // Store 4 bytes (one RGBA pixel)
        let pixel_val = _mm_cvtsi128_si32(packed8) as u32;
        let out_offset = out_x * 4;
        unsafe {
            core::ptr::write_unaligned(
                output.as_mut_ptr().add(out_offset) as *mut u32,
                pixel_val,
            );
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
/// Loads u8 from input rows, zero-extends to i16, multiplies by i16 weights,
/// accumulates in i32, then packs back to u8.
#[archmage::arcane]
pub(crate) fn filter_v_u8_i16_v3(
    _token: X64V3Token,
    rows: &[&[u8]],
    output: &mut [u8],
    weights: &[i16],
) {
    let width = output.len();
    debug_assert_eq!(rows.len(), weights.len());

    let half = _mm256_set1_epi32(1 << (I16_PRECISION as i32 - 1));

    // Process 16 bytes (16 u8 values = 4 RGBA pixels) at a time
    let chunks16 = width / 16;

    for chunk in 0..chunks16 {
        let base = chunk * 16;
        let mut acc_lo = _mm256_setzero_si256(); // i32x8 for first 8 channels
        let mut acc_hi = _mm256_setzero_si256(); // i32x8 for next 8 channels

        // Process rows in pairs for madd_epi16
        let pairs = rows.len() / 2;
        for p in 0..pairs {
            let r0 = p * 2;
            let r1 = r0 + 1;
            let w0 = weights[r0] as i32;
            let w1 = weights[r1] as i32;
            let paired_w = _mm256_set1_epi32((w1 << 16) | (w0 & 0xFFFF));

            // Load 16 bytes from each row
            let src0 = unsafe {
                _mm_loadu_si128(rows[r0].as_ptr().add(base) as *const __m128i)
            };
            let src1 = unsafe {
                _mm_loadu_si128(rows[r1].as_ptr().add(base) as *const __m128i)
            };

            // Interleave bytes from row0 and row1:
            // unpacklo: [r0[0],r1[0],r0[1],r1[1],...,r0[7],r1[7]] → first 8 channels
            let interleaved_lo = _mm_unpacklo_epi8(src0, src1);
            let interleaved_hi = _mm_unpackhi_epi8(src0, src1);

            // Zero-extend i8 pairs to i16 pairs, then madd
            // Low 8 channels: extend to 256-bit
            let ext_lo = _mm256_cvtepu8_epi16(interleaved_lo);
            let ext_hi = _mm256_cvtepu8_epi16(interleaved_hi);

            // madd: multiply each i16 by paired weight, accumulate pairs → i32
            acc_lo = _mm256_add_epi32(acc_lo, _mm256_madd_epi16(ext_lo, paired_w));
            acc_hi = _mm256_add_epi32(acc_hi, _mm256_madd_epi16(ext_hi, paired_w));
        }

        // Handle odd last row (if any)
        if rows.len() % 2 != 0 {
            let r = rows.len() - 1;
            let w = _mm256_set1_epi16(weights[r]);

            let src = unsafe {
                _mm_loadu_si128(rows[r].as_ptr().add(base) as *const __m128i)
            };

            // Zero-extend to i16
            let ext = _mm256_cvtepu8_epi16(src);
            // Split into low and high halves for separate accumulation
            let lo = _mm256_unpacklo_epi16(ext, _mm256_setzero_si256());
            let hi = _mm256_unpackhi_epi16(ext, _mm256_setzero_si256());

            // Multiply and accumulate
            let w32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(w));
            acc_lo = _mm256_add_epi32(acc_lo, _mm256_mullo_epi32(lo, w32));
            acc_hi = _mm256_add_epi32(acc_hi, _mm256_mullo_epi32(hi, w32));
        }

        // Round, shift, pack to u8
        let rounded_lo = _mm256_add_epi32(acc_lo, half);
        let rounded_hi = _mm256_add_epi32(acc_hi, half);
        let shifted_lo = _mm256_srai_epi32::<{ I16_PRECISION as i32 }>(rounded_lo);
        let shifted_hi = _mm256_srai_epi32::<{ I16_PRECISION as i32 }>(rounded_hi);

        // Pack i32 → i16 (signed saturating)
        let packed16 = _mm256_packs_epi32(shifted_lo, shifted_hi);
        // Pack i16 → u8 (unsigned saturating)
        let packed8 = _mm256_packus_epi16(packed16, packed16);

        // AVX2 packing crosses lanes. Permute to get bytes in order.
        let result = _mm256_permute4x64_epi64::<0b11011000>(packed8);

        // Store 16 bytes
        unsafe {
            _mm_storeu_si128(
                output.as_mut_ptr().add(base) as *mut __m128i,
                _mm256_castsi256_si128(result),
            );
        }
    }

    // Scalar tail for remaining bytes
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
