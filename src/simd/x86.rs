//! x86-64 AVX2+FMA convolution and conversion kernels.
#![allow(unsafe_code)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::weights::F32WeightTable;
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

/// Fused vertical convolution + f32→u8 using AVX2+FMA.
///
/// Performs vertical filter and converts to u8 directly, eliminating the
/// temporary f32 output buffer for non-alpha sRGB path.
#[archmage::arcane]
pub(crate) fn filter_v_row_f32_to_u8_v3(
    _token: X64V3Token,
    rows: &[&[f32]],
    output: &mut [u8],
    weights: &[f32],
) {
    let width = output.len();
    debug_assert_eq!(rows.len(), weights.len());

    let scale = _mm256_set1_ps(255.0);
    let half = _mm256_set1_ps(0.5);
    let zero_ps = _mm256_setzero_ps();
    let max_val = _mm256_set1_ps(255.0);

    let chunks8 = width / 8;
    let base8 = chunks8 * 8;

    for chunk_idx in 0..chunks8 {
        let base = chunk_idx * 8;
        let mut acc = _mm256_setzero_ps();

        for (row, &weight) in rows.iter().zip(weights.iter()) {
            let w = _mm256_set1_ps(weight);
            let src = unsafe { _mm256_loadu_ps(row.as_ptr().add(base)) };
            acc = _mm256_fmadd_ps(src, w, acc);
        }

        // Convert f32 → u8 inline: val * 255 + 0.5, clamp, pack
        let scaled = _mm256_fmadd_ps(acc, scale, half);
        let clamped = _mm256_min_ps(_mm256_max_ps(scaled, zero_ps), max_val);
        let ints = _mm256_cvttps_epi32(clamped);
        let packed16 = _mm256_packus_epi32(ints, ints);
        let packed8 = _mm256_packus_epi16(packed16, packed16);
        let lo = _mm256_extracti128_si256::<0>(packed8);
        let hi = _mm256_extracti128_si256::<1>(packed8);
        unsafe {
            let lo_val = _mm_cvtsi128_si32(lo) as u32;
            let hi_val = _mm_cvtsi128_si32(hi) as u32;
            core::ptr::write_unaligned(output.as_mut_ptr().add(base) as *mut u32, lo_val);
            core::ptr::write_unaligned(output.as_mut_ptr().add(base + 4) as *mut u32, hi_val);
        }
    }

    // Scalar tail
    for x in base8..width {
        let mut sum = 0.0f32;
        for (row, &weight) in rows.iter().zip(weights.iter()) {
            sum += row[x] * weight;
        }
        output[x] = (sum * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
    }
}

/// Fused u8→f32 + horizontal convolution using AVX2+FMA.
///
/// Converts each input u8 pixel to f32 inline during convolution, eliminating
/// the separate u8→f32 pass. For sRGB-space resize (no gamma correction).
#[archmage::arcane]
pub(crate) fn filter_h_row_u8_srgb_v3(
    _token: X64V3Token,
    input: &[u8],
    output: &mut [f32],
    weights: &F32WeightTable,
    channels: usize,
) {
    match channels {
        4 => filter_h_u8_4ch(_token, input, output, weights),
        _ => {
            // Fallback: separate conversion + filter for non-4ch
            let len = input.len();
            // Stack-allocate for small rows, heap for large
            let mut temp = vec![0.0f32; len];
            let scale = 1.0 / 255.0;
            for (i, &b) in input.iter().enumerate() {
                temp[i] = b as f32 * scale;
            }
            match channels {
                3 => filter_h_3ch(_token, &temp, output, weights),
                _ => filter_h_generic(_token, &temp, output, weights, channels),
            }
        }
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

/// Fused u8→f32 horizontal filter for 4-channel data.
///
/// Loads u8 RGBA pixels, converts to f32 inline via cvtepu8_epi32 + cvtepi32_ps,
/// then multiplies by weight × (1/255) combined constant.
#[archmage::rite]
fn filter_h_u8_4ch(_token: X64V3Token, input: &[u8], output: &mut [f32], weights: &F32WeightTable) {
    let out_width = weights.len();
    let scale = 1.0f32 / 255.0;
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();

    for out_x in 0..out_width {
        let left = weights.left[out_x] as usize;
        let w_ptr = weights.weights_padded(out_x).as_ptr();
        let max_taps = weights.max_taps;

        let mut acc = _mm_setzero_ps();

        let chunks4 = max_taps / 4;
        let remainder = max_taps % 4;

        for c in 0..chunks4 {
            let t = c * 4;
            unsafe {
                // Pre-combine weight × (1/255) into a single scale factor
                let sw0 = _mm_set1_ps(*w_ptr.add(t) * scale);
                let sw1 = _mm_set1_ps(*w_ptr.add(t + 1) * scale);
                let sw2 = _mm_set1_ps(*w_ptr.add(t + 2) * scale);
                let sw3 = _mm_set1_ps(*w_ptr.add(t + 3) * scale);

                // Load 4 bytes (1 RGBA pixel), zero-extend to i32, convert to f32
                let b0 = _mm_cvtsi32_si128(core::ptr::read_unaligned(
                    in_ptr.add((left + t) * 4) as *const i32,
                ));
                let b1 = _mm_cvtsi32_si128(core::ptr::read_unaligned(
                    in_ptr.add((left + t + 1) * 4) as *const i32,
                ));
                let b2 = _mm_cvtsi32_si128(core::ptr::read_unaligned(
                    in_ptr.add((left + t + 2) * 4) as *const i32,
                ));
                let b3 = _mm_cvtsi32_si128(core::ptr::read_unaligned(
                    in_ptr.add((left + t + 3) * 4) as *const i32,
                ));

                let p0 = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(b0));
                let p1 = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(b1));
                let p2 = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(b2));
                let p3 = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(b3));

                acc = _mm_fmadd_ps(p0, sw0, acc);
                acc = _mm_fmadd_ps(p1, sw1, acc);
                acc = _mm_fmadd_ps(p2, sw2, acc);
                acc = _mm_fmadd_ps(p3, sw3, acc);
            }
        }

        let t_start = chunks4 * 4;
        for t in 0..remainder {
            let tt = t_start + t;
            unsafe {
                let sw = _mm_set1_ps(*w_ptr.add(tt) * scale);
                let bytes = _mm_cvtsi32_si128(core::ptr::read_unaligned(
                    in_ptr.add((left + tt) * 4) as *const i32,
                ));
                let pixel = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(bytes));
                acc = _mm_fmadd_ps(pixel, sw, acc);
            }
        }

        unsafe { _mm_storeu_ps(out_ptr.add(out_x * 4), acc) };
    }
}

/// Horizontal filter for 4-channel (RGBA) data.
///
/// Uses padded weights (fixed iteration count across all output pixels)
/// with manual 4-tap unrolling for instruction-level parallelism.
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

        let mut acc = _mm_setzero_ps();

        // Process 4 taps at a time for better ILP
        let chunks4 = max_taps / 4;
        let remainder = max_taps % 4;

        for c in 0..chunks4 {
            let t = c * 4;
            // SAFETY: weight table guarantees in-bounds access
            unsafe {
                let w0 = _mm_set1_ps(*w_ptr.add(t));
                let w1 = _mm_set1_ps(*w_ptr.add(t + 1));
                let w2 = _mm_set1_ps(*w_ptr.add(t + 2));
                let w3 = _mm_set1_ps(*w_ptr.add(t + 3));

                let p0 = _mm_loadu_ps(in_ptr.add(base_in + t * 4));
                let p1 = _mm_loadu_ps(in_ptr.add(base_in + (t + 1) * 4));
                let p2 = _mm_loadu_ps(in_ptr.add(base_in + (t + 2) * 4));
                let p3 = _mm_loadu_ps(in_ptr.add(base_in + (t + 3) * 4));

                acc = _mm_fmadd_ps(p0, w0, acc);
                acc = _mm_fmadd_ps(p1, w1, acc);
                acc = _mm_fmadd_ps(p2, w2, acc);
                acc = _mm_fmadd_ps(p3, w3, acc);
            }
        }

        // Handle remaining taps (0-3)
        let t_start = chunks4 * 4;
        for t in 0..remainder {
            let tt = t_start + t;
            unsafe {
                let w_val = _mm_set1_ps(*w_ptr.add(tt));
                let pixel = _mm_loadu_ps(in_ptr.add(base_in + tt * 4));
                acc = _mm_fmadd_ps(pixel, w_val, acc);
            }
        }

        // SAFETY: out_offset + 4 <= output.len() guaranteed by caller
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

    // Zero the output buffer using AVX2
    let chunks8 = width / 8;
    let base8 = chunks8 * 8;
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
