//! WASM SIMD128 convolution kernels via portable `wide` SIMD.
//!
//! All implementations delegate to `wide_kernels` which uses `wide` crate types
//! (f32x4, i16x8, etc.) that compile to WASM SIMD128 instructions when built
//! with `-C target-feature=+simd128`.

use crate::weights::{F32WeightTable, I16WeightTable};
use archmage::Wasm128Token;

#[archmage::arcane]
pub(crate) fn filter_h_row_f32_wasm128(
    _token: Wasm128Token,
    input: &[f32],
    output: &mut [f32],
    weights: &F32WeightTable,
    channels: usize,
) {
    super::wide_kernels::filter_h_row_f32(input, output, weights, channels)
}

#[archmage::arcane]
pub(crate) fn filter_v_row_f32_wasm128(
    _token: Wasm128Token,
    rows: &[&[f32]],
    output: &mut [f32],
    weights: &[f32],
) {
    super::wide_kernels::filter_v_row_f32(rows, output, weights)
}

#[archmage::arcane]
pub(crate) fn u8_to_f32_row_wasm128(_token: Wasm128Token, input: &[u8], output: &mut [f32]) {
    super::wide_kernels::u8_to_f32_row(input, output)
}

#[archmage::arcane]
pub(crate) fn f32_to_u8_row_wasm128(_token: Wasm128Token, input: &[f32], output: &mut [u8]) {
    super::wide_kernels::f32_to_u8_row(input, output)
}

#[archmage::arcane]
pub(crate) fn premultiply_alpha_row_wasm128(_token: Wasm128Token, row: &mut [f32]) {
    super::wide_kernels::premultiply_alpha_row(row)
}

#[archmage::arcane]
pub(crate) fn unpremultiply_alpha_row_wasm128(_token: Wasm128Token, row: &mut [f32]) {
    super::wide_kernels::unpremultiply_alpha_row(row)
}

#[archmage::arcane]
pub(crate) fn filter_h_u8_i16_wasm128(
    _token: Wasm128Token,
    input: &[u8],
    output: &mut [u8],
    weights: &I16WeightTable,
    channels: usize,
) {
    super::wide_kernels::filter_h_u8_i16(input, output, weights, channels)
}

#[archmage::arcane]
pub(crate) fn filter_h_u8_i16_4rows_wasm128(
    _token: Wasm128Token,
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
    super::wide_kernels::filter_h_u8_i16_4rows(in0, in1, in2, in3, out0, out1, out2, out3, weights)
}

#[archmage::arcane]
pub(crate) fn filter_v_all_u8_i16_wasm128(
    _token: Wasm128Token,
    intermediate: &[u8],
    output: &mut [u8],
    h_row_len: usize,
    in_h: usize,
    out_h: usize,
    weights: &crate::weights::I16WeightTable,
) {
    super::wide_kernels::filter_v_all_u8_i16(intermediate, output, h_row_len, in_h, out_h, weights)
}

#[archmage::arcane]
pub(crate) fn premultiply_u8_row_wasm128(_token: Wasm128Token, input: &[u8], output: &mut [u8]) {
    super::wide_kernels::premultiply_u8_row(input, output)
}

#[archmage::arcane]
pub(crate) fn unpremultiply_u8_row_wasm128(_token: Wasm128Token, row: &mut [u8]) {
    super::wide_kernels::unpremultiply_u8_row(row)
}

#[archmage::arcane]
pub(crate) fn srgb_u8_to_linear_f32_wasm128(
    _token: Wasm128Token,
    input: &[u8],
    output: &mut [f32],
    channels: usize,
    has_alpha: bool,
) {
    crate::color::srgb_u8_to_linear_f32_impl(input, output, channels, has_alpha);
}

#[archmage::arcane]
pub(crate) fn linear_f32_to_srgb_u8_wasm128(
    _token: Wasm128Token,
    input: &[f32],
    output: &mut [u8],
    channels: usize,
    has_alpha: bool,
) {
    crate::color::linear_f32_to_srgb_u8_impl(input, output, channels, has_alpha);
}
