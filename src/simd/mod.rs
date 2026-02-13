//! SIMD-accelerated convolution kernels.
//!
//! Uses archmage incant! dispatch to select the best available implementation:
//! - x86_64: AVX2+FMA (X64V3Token)
//! - AArch64: NEON (NeonToken)
//! - Fallback: Scalar

#![allow(unsafe_code)]

mod scalar;
#[allow(unused_imports)]
use scalar::*;

#[cfg(target_arch = "x86_64")]
mod x86;
#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
use x86::*;

#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "aarch64")]
#[allow(unused_imports)]
use neon::*;

use crate::weights::{F32WeightTable, I16WeightTable};

/// Horizontally filter one row of f32 pixels.
pub(crate) fn filter_h_row_f32(
    input: &[f32],
    output: &mut [f32],
    weights: &F32WeightTable,
    channels: usize,
) {
    archmage::incant!(filter_h_row_f32(input, output, weights, channels))
}

/// Vertically filter accumulated rows into one output row.
pub(crate) fn filter_v_row_f32(rows: &[&[f32]], output: &mut [f32], weights: &[f32]) {
    archmage::incant!(filter_v_row_f32(rows, output, weights))
}

/// Convert a row of u8 pixels to f32 (divide by 255).
pub(crate) fn u8_to_f32_row(input: &[u8], output: &mut [f32]) {
    archmage::incant!(u8_to_f32_row(input, output))
}

/// Convert a row of f32 pixels to u8 (multiply by 255, round, clamp).
pub(crate) fn f32_to_u8_row(input: &[f32], output: &mut [u8]) {
    archmage::incant!(f32_to_u8_row(input, output))
}

/// Premultiply alpha in-place on RGBA f32 row.
pub(crate) fn premultiply_alpha_row(row: &mut [f32]) {
    archmage::incant!(premultiply_alpha_row(row))
}

/// Unpremultiply alpha in-place on RGBA f32 row.
pub(crate) fn unpremultiply_alpha_row(row: &mut [f32]) {
    archmage::incant!(unpremultiply_alpha_row(row))
}

/// Integer horizontal filter: u8 input → u8 output via i16 weights.
pub(crate) fn filter_h_u8_i16(
    input: &[u8],
    output: &mut [u8],
    weights: &I16WeightTable,
    channels: usize,
) {
    archmage::incant!(filter_h_u8_i16(input, output, weights, channels))
}

/// Integer horizontal filter: 4 rows at once, RGBA only.
/// Shares weight computation across rows for better throughput.
pub(crate) fn filter_h_u8_i16_4rows(
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
    archmage::incant!(filter_h_u8_i16_4rows(
        in0, in1, in2, in3, out0, out1, out2, out3, weights
    ))
}

/// Integer vertical filter: u8 rows → u8 output via i16 weights.
pub(crate) fn filter_v_u8_i16(rows: &[&[u8]], output: &mut [u8], weights: &[i16]) {
    archmage::incant!(filter_v_u8_i16(rows, output, weights))
}

/// Premultiply alpha on RGBA u8 row: input → output.
pub(crate) fn premultiply_u8_row(input: &[u8], output: &mut [u8]) {
    archmage::incant!(premultiply_u8_row(input, output))
}

/// Unpremultiply alpha in-place on RGBA u8 row.
pub(crate) fn unpremultiply_u8_row(row: &mut [u8]) {
    archmage::incant!(unpremultiply_u8_row(row))
}

/// Convert u8 row to i16 (zero-extend) for pre-conversion before horizontal filter.
pub(crate) fn u8_to_i16_row(input: &[u8], output: &mut [i16]) {
    archmage::incant!(u8_to_i16_row(input, output))
}

/// Integer horizontal filter from pre-converted i16 input → u8 output.
/// Input must be zero-extended u8 values (0..255 range as i16).
pub(crate) fn filter_h_i16_to_u8(
    input: &[i16],
    output: &mut [u8],
    weights: &I16WeightTable,
    channels: usize,
) {
    archmage::incant!(filter_h_i16_to_u8(input, output, weights, channels))
}
