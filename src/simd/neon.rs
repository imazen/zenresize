//! AArch64 NEON convolution kernels.
#![allow(unsafe_code)]

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

use crate::weights::F32WeightTable;
use archmage::NeonToken;

/// Horizontal convolution using NEON.
///
/// For each output pixel, accumulate weighted input pixels.
/// Dispatches to specialized channel-count implementations.
#[archmage::arcane]
pub(crate) fn filter_h_row_f32_neon(
    _token: NeonToken,
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

/// Horizontal filter for 4-channel (RGBA) data using NEON.
/// Each output pixel is accumulated using float32x4_t FMA.
#[archmage::rite]
fn filter_h_4ch(_token: NeonToken, input: &[f32], output: &mut [f32], weights: &F32WeightTable) {
    let out_width = weights.len();

    for out_x in 0..out_width {
        let left = weights.left[out_x] as usize;
        let w = weights.weights(out_x);
        let out_offset = out_x * 4;

        // SAFETY: vdupq_n_f32 is a value-based intrinsic
        let mut acc = unsafe { vdupq_n_f32(0.0) };

        for (t, &weight) in w.iter().enumerate() {
            let in_offset = (left + t) * 4;
            // SAFETY: in_offset + 4 <= input.len() guaranteed by weight table construction
            let pixel = unsafe { vld1q_f32(input.as_ptr().add(in_offset)) };
            let w_vec = unsafe { vdupq_n_f32(weight) };
            acc = unsafe { vfmaq_f32(acc, pixel, w_vec) };
        }

        // SAFETY: out_offset + 4 <= output.len() guaranteed by caller
        unsafe { vst1q_f32(output.as_mut_ptr().add(out_offset), acc) };
    }
}

/// Horizontal filter for 3-channel (RGB) data.
/// Scalar with compiler auto-vectorization.
#[archmage::rite]
fn filter_h_3ch(_token: NeonToken, input: &[f32], output: &mut [f32], weights: &F32WeightTable) {
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
    _token: NeonToken,
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

/// Vertical convolution using NEON.
///
/// Processes 4 floats at a time across the row width using FMA.
#[archmage::arcane]
pub(crate) fn filter_v_row_f32_neon(
    _token: NeonToken,
    rows: &[&[f32]],
    output: &mut [f32],
    weights: &[f32],
) {
    let width = output.len();
    debug_assert_eq!(rows.len(), weights.len());

    let chunks4 = width / 4;
    let base4 = chunks4 * 4;

    // Process 4 floats at a time using NEON FMA
    for chunk_idx in 0..chunks4 {
        let base = chunk_idx * 4;
        let mut acc = unsafe { vdupq_n_f32(0.0) };

        for (row, &weight) in rows.iter().zip(weights.iter()) {
            // SAFETY: base + 4 <= row.len() since row.len() >= width
            let src = unsafe { vld1q_f32(row.as_ptr().add(base)) };
            let w = unsafe { vdupq_n_f32(weight) };
            acc = unsafe { vfmaq_f32(acc, src, w) };
        }

        // SAFETY: base + 4 <= output.len()
        unsafe { vst1q_f32(output.as_mut_ptr().add(base), acc) };
    }

    // Scalar tail for remaining 1-3 elements
    for x in base4..width {
        let mut sum = 0.0f32;
        for (row, &weight) in rows.iter().zip(weights.iter()) {
            sum += row[x] * weight;
        }
        output[x] = sum;
    }
}
