//! x86-64 AVX2+FMA convolution kernels.
#![allow(unsafe_code)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::weights::F32WeightTable;
use archmage::X64V3Token;

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
/// Each output pixel is accumulated using __m128 FMA.
#[archmage::rite]
fn filter_h_4ch(_token: X64V3Token, input: &[f32], output: &mut [f32], weights: &F32WeightTable) {
    let out_width = weights.len();

    for out_x in 0..out_width {
        let left = weights.left[out_x] as usize;
        let w = weights.weights(out_x);
        let out_offset = out_x * 4;

        let mut acc = _mm_setzero_ps();

        for (t, &weight) in w.iter().enumerate() {
            let in_offset = (left + t) * 4;
            // SAFETY: in_offset + 4 <= input.len() guaranteed by weight table construction
            let pixel = unsafe { _mm_loadu_ps(input.as_ptr().add(in_offset)) };
            let w_vec = _mm_set1_ps(weight);
            acc = _mm_fmadd_ps(pixel, w_vec, acc);
        }

        // SAFETY: out_offset + 4 <= output.len() guaranteed by caller
        unsafe { _mm_storeu_ps(output.as_mut_ptr().add(out_offset), acc) };
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
/// This is the hot path: processes 8 floats at a time across the row width
/// using FMA for each row × weight accumulation.
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

    // Process 8 floats at a time using AVX2 FMA
    for chunk_idx in 0..chunks8 {
        let base = chunk_idx * 8;
        let mut acc = _mm256_setzero_ps();

        for (row, &weight) in rows.iter().zip(weights.iter()) {
            // SAFETY: base + 8 <= row.len() since row.len() >= width
            let src = unsafe { _mm256_loadu_ps(row.as_ptr().add(base)) };
            let w = _mm256_set1_ps(weight);
            acc = _mm256_fmadd_ps(src, w, acc);
        }

        // SAFETY: base + 8 <= output.len()
        unsafe { _mm256_storeu_ps(output.as_mut_ptr().add(base), acc) };
    }

    // Remainder: 4-wide SSE pass if possible
    let remaining = width - base8;
    if remaining >= 4 {
        let mut acc = _mm_setzero_ps();
        for (row, &weight) in rows.iter().zip(weights.iter()) {
            let src = unsafe { _mm_loadu_ps(row.as_ptr().add(base8)) };
            let w = _mm_set1_ps(weight);
            acc = _mm_fmadd_ps(src, w, acc);
        }
        unsafe { _mm_storeu_ps(output.as_mut_ptr().add(base8), acc) };

        // Scalar tail for last 1-3 elements
        for x in (base8 + 4)..width {
            let mut sum = 0.0f32;
            for (row, &weight) in rows.iter().zip(weights.iter()) {
                sum += row[x] * weight;
            }
            output[x] = sum;
        }
    } else {
        // Scalar for < 4 remaining
        for x in base8..width {
            let mut sum = 0.0f32;
            for (row, &weight) in rows.iter().zip(weights.iter()) {
                sum += row[x] * weight;
            }
            output[x] = sum;
        }
    }
}
