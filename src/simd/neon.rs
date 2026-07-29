//! AArch64 NEON convolution kernels via portable magetypes SIMD.
//!
//! All implementations delegate to `wide_kernels` which uses magetypes generic
//! types that compile to NEON instructions on AArch64. The `#[magetypes(neon, wasm128)]`
//! macro generates `_neon` suffixed variants that these wrappers call.

use crate::weights::{F32WeightTable, I16WeightTable};
use archmage::NeonToken;
use archmage::prelude::*;

#[archmage::arcane]
pub(crate) fn filter_h_row_f32_neon(
    _token: NeonToken,
    input: &[f32],
    output: &mut [f32],
    weights: &F32WeightTable,
    channels: usize,
) {
    super::wide_kernels::filter_h_row_f32_impl_neon(_token, input, output, weights, channels)
}

#[archmage::arcane]
pub(crate) fn filter_v_row_f32_neon(
    _token: NeonToken,
    rows: &[&[f32]],
    output: &mut [f32],
    weights: &[f32],
) {
    super::wide_kernels::filter_v_row_f32_impl_neon(_token, rows, output, weights)
}

#[archmage::arcane]
pub(crate) fn u8_to_f32_row_neon(_token: NeonToken, input: &[u8], output: &mut [f32]) {
    super::wide_kernels::u8_to_f32_row_impl_neon(_token, input, output)
}

#[archmage::arcane]
pub(crate) fn f32_to_u8_row_neon(_token: NeonToken, input: &[f32], output: &mut [u8]) {
    super::wide_kernels::f32_to_u8_row_impl_neon(_token, input, output)
}

/// Portable body, deliberately.
///
/// A hand-written `vld4q_f32` version was tried and MEASURED NO GAIN: 1.00x
/// against the forced-scalar tier both before and after (CI straddling zero,
/// reproduced across runs). NEON is baseline on aarch64, so LLVM already
/// vectorizes this loop about as well as the intrinsics do — see
/// `unpremultiply_alpha_row_neon` below for the case where hand-writing DOES
/// pay, and why the two differ.
#[archmage::arcane]
pub(crate) fn premultiply_alpha_row_neon(_token: NeonToken, row: &mut [f32]) {
    super::wide_kernels::premultiply_alpha_row_impl_neon(_token, row)
}

/// Hand-written NEON unpremultiply: 4 f32 pixels per `vld4q_f32`.
///
/// Same shape as [`premultiply_alpha_row_neon`] — the portable body is scalar
/// in every tier (measured 1.00x, 5.41 GiB/s).
///
/// Bit-exact: the scalar body computes `inv_a = 1.0 / a` and then `c * inv_a`,
/// two roundings in that order, and so does this — `vdivq_f32` is a correctly
/// rounded IEEE divide, matching `1.0 / a`, and `vmulq_f32` matches the
/// multiply. The `a > 1/1024` guard becomes a select, leaving the pixel
/// untouched below the threshold exactly as the branch did.
#[archmage::arcane]
pub(crate) fn unpremultiply_alpha_row_neon(token: NeonToken, row: &mut [f32]) {
    const PX: usize = 4;
    let full = row.len() / (PX * 4) * (PX * 4);
    let (body, tail) = row.split_at_mut(full);
    let thresh = vdupq_n_f32(1.0 / 1024.0);
    let one = vdupq_n_f32(1.0);
    for chunk in body.chunks_exact_mut(PX * 4) {
        let block: &mut [f32; PX * 4] = chunk.try_into().unwrap();
        let p = vld4q_f32(block);
        // Select on the RESULT, not on the multiplier: below the threshold the
        // scalar branch leaves the channel completely untouched, and `c * 1.0`
        // — while equal for every finite value — is not guaranteed to preserve
        // a NaN payload. Selecting the original lane is unconditionally exact.
        let live = vcgtq_f32(p.3, thresh);
        let inv = vdivq_f32(one, p.3);
        vst4q_f32(
            block,
            float32x4x4_t(
                vbslq_f32(live, vmulq_f32(p.0, inv), p.0),
                vbslq_f32(live, vmulq_f32(p.1, inv), p.1),
                vbslq_f32(live, vmulq_f32(p.2, inv), p.2),
                p.3,
            ),
        );
    }
    if !tail.is_empty() {
        super::wide_kernels::unpremultiply_alpha_row_impl_neon(token, tail);
    }
}

#[archmage::arcane]
pub(crate) fn filter_h_u8_i16_neon(
    _token: NeonToken,
    input: &[u8],
    output: &mut [u8],
    weights: &I16WeightTable,
    channels: usize,
) {
    super::wide_kernels::filter_h_u8_i16_impl_neon(_token, input, output, weights, channels)
}

#[archmage::arcane]
pub(crate) fn filter_h_u8_to_i16_neon(
    _token: NeonToken,
    input: &[u8],
    output: &mut [i16],
    weights: &I16WeightTable,
    channels: usize,
) {
    super::wide_kernels::filter_h_u8_to_i16_impl_neon(_token, input, output, weights, channels)
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
    super::wide_kernels::filter_h_u8_to_i16_4rows_impl_neon(
        _token, in0, in1, in2, in3, out0, out1, out2, out3, weights,
    )
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
    super::wide_kernels::filter_h_u8_i16_4rows_impl_neon(
        _token, in0, in1, in2, in3, out0, out1, out2, out3, weights,
    )
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
    super::wide_kernels::filter_v_all_u8_i16_impl_neon(
        _token,
        intermediate,
        output,
        h_row_len,
        in_h,
        out_h,
        weights,
    )
}

#[archmage::arcane]
pub(crate) fn filter_v_all_u8_i16_tiled_neon(
    _token: NeonToken,
    intermediate: &[u8],
    output: &mut [u8],
    h_row_len: usize,
    in_h: usize,
    out_h: usize,
    weights: &crate::weights::I16WeightTable,
    tile_chunks: usize,
) {
    super::wide_kernels::filter_v_all_u8_i16_tiled_impl_neon(
        _token,
        intermediate,
        output,
        h_row_len,
        in_h,
        out_h,
        weights,
        tile_chunks,
    )
}

#[archmage::arcane]
pub(crate) fn premultiply_u8_row_neon(_token: NeonToken, input: &[u8], output: &mut [u8]) {
    super::wide_kernels::premultiply_u8_row_impl_neon(_token, input, output)
}

/// Widen a `u8x16` into four `u32x4` groups (lossless).
#[archmage::rite]
fn widen_u8x16(_t: NeonToken, v: uint8x16_t) -> [uint32x4_t; 4] {
    let lo16 = vmovl_u8(vget_low_u8(v));
    let hi16 = vmovl_u8(vget_high_u8(v));
    [
        vmovl_u16(vget_low_u16(lo16)),
        vmovl_u16(vget_high_u16(lo16)),
        vmovl_u16(vget_low_u16(hi16)),
        vmovl_u16(vget_high_u16(hi16)),
    ]
}

/// Narrow four `u32x4` groups (all values <= 255) back into a `u8x16`.
#[archmage::rite]
fn narrow_u32x4x4(_t: NeonToken, g: [uint32x4_t; 4]) -> uint8x16_t {
    let n0 = vcombine_u16(vmovn_u32(g[0]), vmovn_u32(g[1]));
    let n1 = vcombine_u16(vmovn_u32(g[2]), vmovn_u32(g[3]));
    vcombine_u8(vmovn_u16(n0), vmovn_u16(n1))
}

/// One 4-lane group of the unpremultiply: `min(255, (c*255 + a/2) / a)`, and 0
/// where `a == 0`.
///
/// `num` is built in integer (exact), then converted — `num <= 65152 < 2^24`
/// and `a <= 255` are both exact in f32, so `vdivq_f32` returns the correctly
/// rounded quotient and truncating it equals integer floor. See
/// `tests/unpremul_u8_exhaustive.rs`, which enumerates the whole domain.
///
/// `a == 255` needs no special case: `(c*255 + 127) / 255 == c` exactly.
/// `a == 0` divides by zero, giving inf or NaN, so it is selected away.
#[archmage::rite]
fn unpremul_group(_t: NeonToken, c: uint32x4_t, a: uint32x4_t) -> uint32x4_t {
    let num = vaddq_u32(vmulq_u32(c, vdupq_n_u32(255)), vshrq_n_u32::<1>(a));
    let q = vdivq_f32(vcvtq_f32_u32(num), vcvtq_f32_u32(a));
    let r = vminq_u32(vcvtq_u32_f32(q), vdupq_n_u32(255));
    vbslq_u32(vceqq_u32(a, vdupq_n_u32(0)), vdupq_n_u32(0), r)
}

/// Hand-written NEON unpremultiply: 16 pixels per iteration.
///
/// The divisor is the pixel's own alpha, so there is nothing a portable
/// integer kernel can vectorize — the generic body measured 3.0us / 2.37 GiB/s
/// with NEON and forced-scalar identical (1.00x). A portable f32-divide
/// rewrite was tried first and was WORSE (4.1us): marshalling one pixel at a
/// time through a vector cost more than the three integer divides it removed.
///
/// `vld4q_u8` is what makes it work — it deinterleaves RGBA into four planes
/// in one instruction, so the alpha for a whole group is already a vector and
/// no per-pixel shuffling is needed.
#[archmage::arcane]
pub(crate) fn unpremultiply_u8_row_neon(token: NeonToken, row: &mut [u8]) {
    const PX: usize = 16;
    let full = row.len() / (PX * 4) * (PX * 4);
    let (body, tail) = row.split_at_mut(full);

    for chunk in body.chunks_exact_mut(PX * 4) {
        let block: &mut [u8; PX * 4] = chunk.try_into().unwrap();
        let p = vld4q_u8(block);
        let a_groups = widen_u8x16(token, p.3);

        let mut planes = [p.0, p.1, p.2];
        for plane in planes.iter_mut() {
            let c = widen_u8x16(token, *plane);
            *plane = narrow_u32x4x4(
                token,
                [
                    unpremul_group(token, c[0], a_groups[0]),
                    unpremul_group(token, c[1], a_groups[1]),
                    unpremul_group(token, c[2], a_groups[2]),
                    unpremul_group(token, c[3], a_groups[3]),
                ],
            );
        }
        vst4q_u8(block, uint8x16x4_t(planes[0], planes[1], planes[2], p.3));
    }

    if !tail.is_empty() {
        super::wide_kernels::unpremultiply_u8_row_impl_neon(token, tail);
    }
}

#[archmage::arcane]
pub(crate) fn filter_h_i16_i16_neon(
    _token: NeonToken,
    input: &[i16],
    output: &mut [i16],
    weights: &I16WeightTable,
    channels: usize,
) {
    super::wide_kernels::filter_h_i16_i16_impl_neon(_token, input, output, weights, channels)
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
    super::wide_kernels::filter_v_all_i16_i16_impl_neon(
        _token,
        intermediate,
        output,
        h_row_len,
        in_h,
        out_h,
        weights,
    )
}

#[archmage::arcane]
pub(crate) fn filter_v_row_u8_i16_neon(
    _token: NeonToken,
    rows: &[&[u8]],
    output: &mut [u8],
    weights: &[i16],
) {
    super::wide_kernels::filter_v_row_u8_i16_impl_neon(_token, rows, output, weights)
}

#[archmage::arcane]
pub(crate) fn filter_v_row_i16_neon(
    _token: NeonToken,
    rows: &[&[i16]],
    output: &mut [i16],
    weights: &[i16],
) {
    super::wide_kernels::filter_v_row_i16_impl_neon(_token, rows, output, weights)
}

// f16 kernels — delegate to wide_kernels

#[archmage::arcane]
pub(crate) fn f32_to_f16_row_neon(_token: NeonToken, input: &[f32], output: &mut [u16]) {
    super::wide_kernels::f32_to_f16_row_impl_neon(_token, input, output)
}

#[archmage::arcane]
pub(crate) fn f16_to_f32_row_neon(_token: NeonToken, input: &[u16], output: &mut [f32]) {
    super::wide_kernels::f16_to_f32_row_impl_neon(_token, input, output)
}

#[archmage::arcane]
pub(crate) fn filter_h_row_f32_to_f16_neon(
    _token: NeonToken,
    input: &[f32],
    output: &mut [u16],
    weights: &F32WeightTable,
    channels: usize,
) {
    super::wide_kernels::filter_h_row_f32_to_f16_impl_neon(_token, input, output, weights, channels)
}

#[archmage::arcane]
pub(crate) fn filter_v_row_f16_neon(
    _token: NeonToken,
    rows: &[&[u16]],
    output: &mut [f32],
    weights: &[f32],
) {
    super::wide_kernels::filter_v_row_f16_impl_neon(_token, rows, output, weights)
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
    super::wide_kernels::filter_v_all_f16_impl_neon(
        _token,
        intermediate,
        output,
        h_row_len,
        in_h,
        out_h,
        weights,
    )
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
