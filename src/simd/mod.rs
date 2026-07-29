//! SIMD-accelerated convolution kernels.
//!
//! Uses archmage incant! dispatch to select the best available implementation:
//! - x86_64: AVX2+FMA (X64V3Token)
//! - AArch64: NEON via magetypes (NeonToken)
//! - WASM32: SIMD128 via magetypes (Wasm128Token)
//! - Fallback: Scalar
#![allow(clippy::too_many_arguments)]

mod scalar;
#[allow(unused_imports)]
use scalar::*;

#[cfg(target_arch = "x86_64")]
#[allow(clippy::excessive_precision)]
mod x86;
#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
use x86::*;

// Portable SIMD kernels via magetypes (shared by NEON and WASM128)
// allow(dead_code): #[magetypes] generates _scalar variants that are unused
// because scalar.rs provides the scalar fallbacks for incant!
#[cfg(any(target_arch = "aarch64", target_arch = "wasm32"))]
#[allow(dead_code)]
mod wide_kernels;

// Portable transfer function SIMD kernels via magetypes f32x4
#[cfg(any(target_arch = "aarch64", target_arch = "wasm32"))]
#[allow(clippy::excessive_precision)]
mod tf_portable;

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
    // aarch64: the NEON kernel is SLOWER than the scalar one here. NEON is
    // baseline on AArch64, so LLVM autovectorises the scalar body, and this
    // kernel's per-4-pixel `i32x4::from_array([a as i32, b as i32, ...])`
    // widening is scalar lane-assembly that the autovectoriser does better.
    // Measured on a 1920-px RGBA row (benches/kernel_tiers.rs):
    //   neon 2.00us vs scalar 1.20us  (0.60x — 1.67x slower)
    // Bit-identical: verified 0 differing lanes at n = 7680/1023/17/4/3/1.
    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        return crate::simd::scalar::u8_to_f32_row_scalar(
            archmage::ScalarToken::summon().expect("scalar token is infallible"),
            input,
            output,
        );
    }
    #[cfg(not(target_arch = "aarch64"))]
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

/// Integer horizontal filter: u8 input → i16 output (unclamped) via i16 weights.
/// Preserves Lanczos ringing in the intermediate without [0,255] clamping.
pub(crate) fn filter_h_u8_to_i16(
    input: &[u8],
    output: &mut [i16],
    weights: &I16WeightTable,
    channels: usize,
) {
    archmage::incant!(filter_h_u8_to_i16(input, output, weights, channels))
}

/// Integer horizontal filter: 4 rows at once, u8 input → i16 output (unclamped), RGBA only.
#[allow(dead_code)]
pub(crate) fn filter_h_u8_to_i16_4rows(
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
    archmage::incant!(filter_h_u8_to_i16_4rows(
        in0, in1, in2, in3, out0, out1, out2, out3, weights
    ))
}

/// Batch vertical filter: process all output rows from the intermediate buffer.
///
/// Avoids per-row dispatch overhead and row pointer construction.
pub fn filter_v_all_u8_i16(
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

/// Tiled batch V-filter: u8 intermediate → u8 output with column tiling.
///
/// `tile_chunks` is the number of 16-byte chunks per tile.
#[allow(dead_code)]
pub fn filter_v_all_u8_i16_tiled(
    intermediate: &[u8],
    output: &mut [u8],
    h_row_len: usize,
    in_h: usize,
    out_h: usize,
    weights: &crate::weights::I16WeightTable,
    tile_chunks: usize,
) {
    archmage::incant!(filter_v_all_u8_i16_tiled(
        intermediate,
        output,
        h_row_len,
        in_h,
        out_h,
        weights,
        tile_chunks
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

// =========================================================================
// f16 (half-precision) pipeline kernels
// =========================================================================

/// Bulk convert f32 → f16 (stored as u16).
pub(crate) fn f32_to_f16_row(input: &[f32], output: &mut [u16]) {
    archmage::incant!(f32_to_f16_row(input, output))
}

/// Bulk convert f16 (stored as u16) → f32.
#[allow(dead_code)]
pub(crate) fn f16_to_f32_row(input: &[u16], output: &mut [f32]) {
    archmage::incant!(f16_to_f32_row(input, output))
}

/// Horizontal filter: f32 input → f16 (u16) output.
/// Accumulates in f32, converts to f16 on store.
pub(crate) fn filter_h_row_f32_to_f16(
    input: &[f32],
    output: &mut [u16],
    weights: &F32WeightTable,
    channels: usize,
) {
    archmage::incant!(filter_h_row_f32_to_f16(input, output, weights, channels))
}

/// Streaming V-filter: f16 rows → f32 output via f32 weights.
pub(crate) fn filter_v_row_f16(rows: &[&[u16]], output: &mut [f32], weights: &[f32]) {
    archmage::incant!(filter_v_row_f16(rows, output, weights))
}

/// Batch V-filter for fullframe: f16 intermediate → f32 output.
#[allow(dead_code)]
pub(crate) fn filter_v_all_f16(
    intermediate: &[u16],
    output: &mut [f32],
    h_row_len: usize,
    in_h: usize,
    out_h: usize,
    weights: &F32WeightTable,
) {
    archmage::incant!(filter_v_all_f16(
        intermediate,
        output,
        h_row_len,
        in_h,
        out_h,
        weights
    ))
}

// =========================================================================
// Transfer function batch processors (SIMD-dispatched)
// =========================================================================

macro_rules! tf_dispatch {
    ($name:ident) => {
        pub(crate) fn $name(row: &mut [f32], channels: usize, has_alpha: bool) {
            archmage::incant!($name(row, channels, has_alpha))
        }
    };
}

tf_dispatch!(srgb_to_linear_row);
tf_dispatch!(srgb_from_linear_row);
tf_dispatch!(bt709_to_linear_row);
tf_dispatch!(bt709_from_linear_row);
tf_dispatch!(pq_to_linear_row);
tf_dispatch!(pq_from_linear_row);
tf_dispatch!(hlg_to_linear_row);
tf_dispatch!(hlg_from_linear_row);

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

/// Dev-only per-kernel access for `benches/kernel_tiers.rs`.
///
/// NOT public API and NOT semver-covered — `simd` is already `#[doc(hidden)]`.
/// These exist because the crate was only ever measured end-to-end (resize of a
/// whole image), which cannot show a single kernel being SLOWER than its own
/// scalar fallback. That failure mode was found in garb, zensim, zentone and
/// zenpng during the 2026-07-28 aarch64 sweep, so it is worth checking here.
#[doc(hidden)]
pub mod __bench_kernels {
    // Thin forwarders rather than `pub use`: the kernels are `pub(crate)`,
    // and re-exporting them directly would widen their visibility.
    pub fn u8_to_f32_row(i: &[u8], o: &mut [f32]) { super::u8_to_f32_row(i, o) }
    pub fn f32_to_u8_row(i: &[f32], o: &mut [u8]) { super::f32_to_u8_row(i, o) }
    pub fn premultiply_alpha_row(r: &mut [f32]) { super::premultiply_alpha_row(r) }
    pub fn unpremultiply_alpha_row(r: &mut [f32]) { super::unpremultiply_alpha_row(r) }
    pub fn premultiply_u8_row(i: &[u8], o: &mut [u8]) { super::premultiply_u8_row(i, o) }
    pub fn unpremultiply_u8_row(r: &mut [u8]) { super::unpremultiply_u8_row(r) }

    // ── The H/V filter kernels: the resizer's actual hot path. The six above
    // are per-pixel conversions; these are the convolutions every resize runs
    // once per output row, so a loss here costs far more.
    use crate::weights::{F32WeightTable, I16WeightTable};

    pub fn filter_h_row_f32(i: &[f32], o: &mut [f32], w: &F32WeightTable, c: usize) {
        super::filter_h_row_f32(i, o, w, c)
    }
    pub fn filter_v_row_f32(rows: &[&[f32]], o: &mut [f32], w: &[f32]) {
        super::filter_v_row_f32(rows, o, w)
    }
    pub fn filter_h_u8_i16(i: &[u8], o: &mut [u8], w: &I16WeightTable, c: usize) {
        super::filter_h_u8_i16(i, o, w, c)
    }
    pub fn filter_h_u8_to_i16(i: &[u8], o: &mut [i16], w: &I16WeightTable, c: usize) {
        super::filter_h_u8_to_i16(i, o, w, c)
    }
    pub fn filter_h_i16_i16(i: &[i16], o: &mut [i16], w: &I16WeightTable, c: usize) {
        super::filter_h_i16_i16(i, o, w, c)
    }
    pub fn filter_v_row_u8_i16(rows: &[&[u8]], o: &mut [u8], w: &[i16]) {
        super::filter_v_row_u8_i16(rows, o, w)
    }
    pub fn filter_v_row_i16(rows: &[&[i16]], o: &mut [i16], w: &[i16]) {
        super::filter_v_row_i16(rows, o, w)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn filter_h_u8_i16_4rows(
        i0: &[u8], i1: &[u8], i2: &[u8], i3: &[u8],
        o0: &mut [u8], o1: &mut [u8], o2: &mut [u8], o3: &mut [u8],
        w: &I16WeightTable,
    ) {
        super::filter_h_u8_i16_4rows(i0, i1, i2, i3, o0, o1, o2, o3, w)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn filter_h_u8_to_i16_4rows(
        i0: &[u8], i1: &[u8], i2: &[u8], i3: &[u8],
        o0: &mut [i16], o1: &mut [i16], o2: &mut [i16], o3: &mut [i16],
        w: &I16WeightTable,
    ) {
        super::filter_h_u8_to_i16_4rows(i0, i1, i2, i3, o0, o1, o2, o3, w)
    }
    pub fn filter_v_all_i16_i16(
        inter: &[i16], o: &mut [i16], h_row_len: usize, in_h: usize, out_h: usize,
        w: &I16WeightTable,
    ) {
        super::filter_v_all_i16_i16(inter, o, h_row_len, in_h, out_h, w)
    }

    // ── f16 pipeline
    pub fn f32_to_f16_row(i: &[f32], o: &mut [u16]) { super::f32_to_f16_row(i, o) }
    pub fn f16_to_f32_row(i: &[u16], o: &mut [f32]) { super::f16_to_f32_row(i, o) }
    pub fn filter_h_row_f32_to_f16(i: &[f32], o: &mut [u16], w: &F32WeightTable, c: usize) {
        super::filter_h_row_f32_to_f16(i, o, w, c)
    }
    pub fn filter_v_row_f16(rows: &[&[u16]], o: &mut [f32], w: &[f32]) {
        super::filter_v_row_f16(rows, o, w)
    }
    pub fn filter_v_all_f16(
        inter: &[u16], o: &mut [f32], h_row_len: usize, in_h: usize, out_h: usize,
        w: &F32WeightTable,
    ) {
        super::filter_v_all_f16(inter, o, h_row_len, in_h, out_h, w)
    }

    // ── colour / transfer
    pub fn srgb_u8_to_linear_f32(i: &[u8], o: &mut [f32], c: usize, a: bool) {
        super::srgb_u8_to_linear_f32(i, o, c, a)
    }
    pub fn linear_f32_to_srgb_u8(i: &[f32], o: &mut [u8], c: usize, a: bool) {
        super::linear_f32_to_srgb_u8(i, o, c, a)
    }
    pub fn srgb_to_linear_row(r: &mut [f32], c: usize, a: bool) { super::srgb_to_linear_row(r, c, a) }
    pub fn srgb_from_linear_row(r: &mut [f32], c: usize, a: bool) { super::srgb_from_linear_row(r, c, a) }
    pub fn pq_to_linear_row(r: &mut [f32], c: usize, a: bool) { super::pq_to_linear_row(r, c, a) }
    pub fn hlg_to_linear_row(r: &mut [f32], c: usize, a: bool) { super::hlg_to_linear_row(r, c, a) }
}
