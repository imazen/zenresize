//! SIMD-accelerated convolution kernels.
//!
//! Uses archmage incant! dispatch to select the best available implementation:
//! - x86_64: AVX2+FMA (X64V3Token)
//! - AArch64: NEON via wide (NeonToken)
//! - WASM32: SIMD128 via wide (Wasm128Token)
//! - Fallback: Scalar
#![allow(clippy::too_many_arguments)]

mod scalar;
#[allow(unused_imports)]
use scalar::*;

#[cfg(target_arch = "x86_64")]
mod x86;
#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
use x86::*;

// Portable wide SIMD kernels (shared by NEON and WASM128)
#[cfg(any(target_arch = "aarch64", target_arch = "wasm32"))]
mod wide_kernels;

#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "aarch64")]
#[allow(unused_imports)]
use neon::*;

#[cfg(target_arch = "wasm32")]
mod wasm128;
#[cfg(target_arch = "wasm32")]
#[allow(unused_imports)]
use wasm128::*;

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

/// Batch vertical filter: process all output rows from the intermediate buffer.
///
/// Avoids per-row dispatch overhead and row pointer construction.
pub(crate) fn filter_v_all_u8_i16(
    intermediate: &[u8],
    output: &mut [u8],
    h_row_len: usize,
    in_h: usize,
    out_h: usize,
    weights: &crate::weights::I16WeightTable,
) {
    archmage::incant!(filter_v_all_u8_i16(
        intermediate,
        output,
        h_row_len,
        in_h,
        out_h,
        weights
    ))
}

/// Premultiply alpha on RGBA u8 row: input → output.
pub(crate) fn premultiply_u8_row(input: &[u8], output: &mut [u8]) {
    archmage::incant!(premultiply_u8_row(input, output))
}

/// Unpremultiply alpha in-place on RGBA u8 row.
pub(crate) fn unpremultiply_u8_row(row: &mut [u8]) {
    archmage::incant!(unpremultiply_u8_row(row))
}

/// Convert sRGB u8 → linear f32 (LUT-based, dispatched via token).
pub(crate) fn srgb_u8_to_linear_f32(
    input: &[u8],
    output: &mut [f32],
    channels: usize,
    has_alpha: bool,
) {
    archmage::incant!(srgb_u8_to_linear_f32(input, output, channels, has_alpha))
}

/// Convert linear f32 → sRGB u8 (LUT-based, dispatched via token).
pub(crate) fn linear_f32_to_srgb_u8(
    input: &[f32],
    output: &mut [u8],
    channels: usize,
    has_alpha: bool,
) {
    archmage::incant!(linear_f32_to_srgb_u8(input, output, channels, has_alpha))
}

/// Integer horizontal filter: i16 input → i16 output via i16 weights.
/// For linear-light i12 path (values 0-4095).
pub(crate) fn filter_h_i16_i16(
    input: &[i16],
    output: &mut [i16],
    weights: &I16WeightTable,
    channels: usize,
) {
    archmage::incant!(filter_h_i16_i16(input, output, weights, channels))
}

/// Streaming V-filter: u8 rows → u8 output via i16 weights.
/// For sRGB gamma i16 streaming path.
pub(crate) fn filter_v_row_u8_i16(rows: &[&[u8]], output: &mut [u8], weights: &[i16]) {
    archmage::incant!(filter_v_row_u8_i16(rows, output, weights))
}

/// Streaming V-filter: i16 rows → i16 output via i16 weights.
/// For linear i12 streaming path.
pub(crate) fn filter_v_row_i16(rows: &[&[i16]], output: &mut [i16], weights: &[i16]) {
    archmage::incant!(filter_v_row_i16(rows, output, weights))
}

/// Batch vertical filter: i16 intermediate → i16 output, all rows at once.
/// For linear-light i12 path (values 0-4095).
pub(crate) fn filter_v_all_i16_i16(
    intermediate: &[i16],
    output: &mut [i16],
    h_row_len: usize,
    in_h: usize,
    out_h: usize,
    weights: &crate::weights::I16WeightTable,
) {
    archmage::incant!(filter_v_all_i16_i16(
        intermediate,
        output,
        h_row_len,
        in_h,
        out_h,
        weights
    ))
}
