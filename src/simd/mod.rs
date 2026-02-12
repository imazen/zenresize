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

use crate::weights::F32WeightTable;

/// Horizontally filter one row of u8 pixels (sRGB space, no gamma conversion).
///
/// Fuses u8→f32 conversion directly into the horizontal convolution: each input
/// pixel is converted on-the-fly as it's loaded, eliminating the separate u8→f32
/// pass and saving one full read+write over the input row.
pub(crate) fn filter_h_row_u8_srgb(
    input: &[u8],
    output: &mut [f32],
    weights: &F32WeightTable,
    channels: usize,
) {
    archmage::incant!(filter_h_row_u8_srgb(input, output, weights, channels))
}

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

/// Fused vertical filter + f32→u8 conversion.
///
/// Performs vertical convolution and converts the result directly to u8,
/// eliminating the temporary f32 output buffer for the non-alpha sRGB path.
pub(crate) fn filter_v_row_f32_to_u8(rows: &[&[f32]], output: &mut [u8], weights: &[f32]) {
    archmage::incant!(filter_v_row_f32_to_u8(rows, output, weights))
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
