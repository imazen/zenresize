//! AArch64 NEON convolution kernels via portable `wide` SIMD.
//!
//! All implementations delegate to `wide_kernels` which uses `wide` crate types
//! (f32x4, i16x8, etc.) that compile to NEON instructions on AArch64.

use crate::weights::{F32WeightTable, I16WeightTable};
use archmage::NeonToken;

#[archmage::arcane]
pub(crate) fn filter_h_row_f32_neon(
    _token: NeonToken,
    input: &[f32],
    output: &mut [f32],
    weights: &F32WeightTable,
    channels: usize,
) {
    super::wide_kernels::filter_h_row_f32(input, output, weights, channels)
}

#[archmage::arcane]
pub(crate) fn filter_v_row_f32_neon(
    _token: NeonToken,
    rows: &[&[f32]],
    output: &mut [f32],
    weights: &[f32],
) {
    super::wide_kernels::filter_v_row_f32(rows, output, weights)
}

#[archmage::arcane]
pub(crate) fn u8_to_f32_row_neon(_token: NeonToken, input: &[u8], output: &mut [f32]) {
    super::wide_kernels::u8_to_f32_row(input, output)
}

#[archmage::arcane]
pub(crate) fn f32_to_u8_row_neon(_token: NeonToken, input: &[f32], output: &mut [u8]) {
    super::wide_kernels::f32_to_u8_row(input, output)
}

#[archmage::arcane]
pub(crate) fn premultiply_alpha_row_neon(_token: NeonToken, row: &mut [f32]) {
    super::wide_kernels::premultiply_alpha_row(row)
}

#[archmage::arcane]
pub(crate) fn unpremultiply_alpha_row_neon(_token: NeonToken, row: &mut [f32]) {
    super::wide_kernels::unpremultiply_alpha_row(row)
}

#[archmage::arcane]
pub(crate) fn filter_h_u8_i16_neon(
    _token: NeonToken,
    input: &[u8],
    output: &mut [u8],
    weights: &I16WeightTable,
    channels: usize,
) {
    super::wide_kernels::filter_h_u8_i16(input, output, weights, channels)
}

#[archmage::arcane]
pub(crate) fn filter_h_u8_to_i16_neon(
    _token: NeonToken,
    input: &[u8],
    output: &mut [i16],
    weights: &I16WeightTable,
    channels: usize,
) {
    super::wide_kernels::filter_h_u8_to_i16(input, output, weights, channels)
}

#[archmage::arcane]
pub(crate) fn filter_h_u8_to_i16_4rows_neon(
    _token: NeonToken,
    in0: &[u8],
    in1: &[u8],
    in2: &[u8],
    in3: &[u8],
    out0: &mut [i16],
    out1: &mut [i16],
    out2: &mut [i16],
    out3: &mut [i16],
    weights: &I16WeightTable,
) {
    super::wide_kernels::filter_h_u8_to_i16_4rows(in0, in1, in2, in3, out0, out1, out2, out3, weights)
}

#[archmage::arcane]
pub(crate) fn filter_h_u8_i16_4rows_neon(
    _token: NeonToken,
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
pub(crate) fn filter_v_all_u8_i16_neon(
    _token: NeonToken,
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
pub(crate) fn premultiply_u8_row_neon(_token: NeonToken, input: &[u8], output: &mut [u8]) {
    super::wide_kernels::premultiply_u8_row(input, output)
}

#[archmage::arcane]
pub(crate) fn unpremultiply_u8_row_neon(_token: NeonToken, row: &mut [u8]) {
    super::wide_kernels::unpremultiply_u8_row(row)
}

#[archmage::arcane]
pub(crate) fn filter_h_i16_i16_neon(
    _token: NeonToken,
    input: &[i16],
    output: &mut [i16],
    weights: &I16WeightTable,
    channels: usize,
) {
    super::wide_kernels::filter_h_i16_i16(input, output, weights, channels)
}

#[archmage::arcane]
pub(crate) fn filter_v_all_i16_i16_neon(
    _token: NeonToken,
    intermediate: &[i16],
    output: &mut [i16],
    h_row_len: usize,
    in_h: usize,
    out_h: usize,
    weights: &crate::weights::I16WeightTable,
) {
    super::wide_kernels::filter_v_all_i16_i16(intermediate, output, h_row_len, in_h, out_h, weights)
}

#[archmage::arcane]
pub(crate) fn filter_v_row_u8_i16_neon(
    _token: NeonToken,
    rows: &[&[u8]],
    output: &mut [u8],
    weights: &[i16],
) {
    super::wide_kernels::filter_v_row_u8_i16(rows, output, weights)
}

#[archmage::arcane]
pub(crate) fn filter_v_row_i16_neon(
    _token: NeonToken,
    rows: &[&[i16]],
    output: &mut [i16],
    weights: &[i16],
) {
    super::wide_kernels::filter_v_row_i16(rows, output, weights)
}

// f16 kernels — delegate to wide_kernels (scalar-style)

#[archmage::arcane]
pub(crate) fn f32_to_f16_row_neon(_token: NeonToken, input: &[f32], output: &mut [u16]) {
    super::wide_kernels::f32_to_f16_row(input, output)
}

#[archmage::arcane]
pub(crate) fn f16_to_f32_row_neon(_token: NeonToken, input: &[u16], output: &mut [f32]) {
    super::wide_kernels::f16_to_f32_row(input, output)
}

#[archmage::arcane]
pub(crate) fn filter_h_row_f32_to_f16_neon(
    _token: NeonToken,
    input: &[f32],
    output: &mut [u16],
    weights: &F32WeightTable,
    channels: usize,
) {
    super::wide_kernels::filter_h_row_f32_to_f16(input, output, weights, channels)
}

#[archmage::arcane]
pub(crate) fn filter_v_row_f16_neon(
    _token: NeonToken,
    rows: &[&[u16]],
    output: &mut [f32],
    weights: &[f32],
) {
    super::wide_kernels::filter_v_row_f16(rows, output, weights)
}

#[archmage::arcane]
pub(crate) fn filter_v_all_f16_neon(
    _token: NeonToken,
    intermediate: &[u16],
    output: &mut [f32],
    h_row_len: usize,
    in_h: usize,
    out_h: usize,
    weights: &F32WeightTable,
) {
    super::wide_kernels::filter_v_all_f16(intermediate, output, h_row_len, in_h, out_h, weights)
}

// Transfer function batch processors — wrap linear-srgb rites via closures.

use magetypes::simd::generic::f32x4;

macro_rules! tf_neon {
    ($name_neon:ident, $rite_fn:path, $scalar_fn:path) => {
        #[archmage::arcane]
        pub(crate) fn $name_neon(
            _token: NeonToken,
            row: &mut [f32],
            channels: usize,
            has_alpha: bool,
        ) {
            super::tf_portable::tf_row_inplace(
                _token,
                row,
                channels,
                has_alpha,
                |t, v: f32x4<NeonToken>| f32x4::from_array(t, $rite_fn(t, v.to_array())),
                $scalar_fn,
            );
        }
    };
}

tf_neon!(
    srgb_to_linear_row_neon,
    linear_srgb::tokens::x4::tf_srgb_to_linear_neon,
    linear_srgb::tf::srgb_to_linear
);
tf_neon!(
    srgb_from_linear_row_neon,
    linear_srgb::tokens::x4::tf_linear_to_srgb_neon,
    linear_srgb::tf::linear_to_srgb
);
tf_neon!(
    bt709_to_linear_row_neon,
    linear_srgb::tokens::x4::bt709_to_linear_neon,
    linear_srgb::tf::bt709_to_linear
);
tf_neon!(
    bt709_from_linear_row_neon,
    linear_srgb::tokens::x4::linear_to_bt709_neon,
    linear_srgb::tf::linear_to_bt709
);
tf_neon!(
    pq_to_linear_row_neon,
    linear_srgb::tokens::x4::pq_to_linear_neon,
    linear_srgb::tf::pq_to_linear
);
tf_neon!(
    pq_from_linear_row_neon,
    linear_srgb::tokens::x4::linear_to_pq_neon,
    linear_srgb::tf::linear_to_pq
);
tf_neon!(
    hlg_to_linear_row_neon,
    linear_srgb::tokens::x4::hlg_to_linear_neon,
    linear_srgb::tf::hlg_to_linear
);
tf_neon!(
    hlg_from_linear_row_neon,
    linear_srgb::tokens::x4::linear_to_hlg_neon,
    linear_srgb::tf::linear_to_hlg
);

#[archmage::arcane]
pub(crate) fn srgb_u8_to_linear_f32_neon(
    _token: NeonToken,
    input: &[u8],
    output: &mut [f32],
    channels: usize,
    has_alpha: bool,
) {
    crate::color::srgb_u8_to_linear_f32_impl(input, output, channels, has_alpha);
}

#[archmage::arcane]
pub(crate) fn linear_f32_to_srgb_u8_neon(
    _token: NeonToken,
    input: &[f32],
    output: &mut [u8],
    channels: usize,
    has_alpha: bool,
) {
    crate::color::linear_f32_to_srgb_u8_impl(input, output, channels, has_alpha);
}
