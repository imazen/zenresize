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
