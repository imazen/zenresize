//! Portable transfer function SIMD kernels using magetypes f32x4.
//!
//! Generic over any token implementing `F32x4Convert` (NEON, WASM128, etc.).
//! 4-wide f32 SIMD for all transfer functions — same algorithms as x86.rs
//! but using magetypes portable abstractions instead of raw intrinsics.

#![allow(clippy::excessive_precision)]

use crate::fastmath;
use magetypes::simd::backends::F32x4Convert;
use magetypes::simd::generic::{f32x4, i32x4};

// =============================================================================
// Core fast math primitives (4-wide)
// =============================================================================

/// Evaluate degree-4 rational polynomial P(x)/Q(x) on 4 f32 values via FMA.
#[inline(always)]
fn eval_rational_poly_x4<T: F32x4Convert>(
    t: T,
    x: f32x4<T>,
    p: [f32; 5],
    q: [f32; 5],
) -> f32x4<T> {
    let mut yp = f32x4::splat(t, p[4]);
    yp = yp.mul_add(x, f32x4::splat(t, p[3]));
    yp = yp.mul_add(x, f32x4::splat(t, p[2]));
    yp = yp.mul_add(x, f32x4::splat(t, p[1]));
    yp = yp.mul_add(x, f32x4::splat(t, p[0]));

    let mut yq = f32x4::splat(t, q[4]);
    yq = yq.mul_add(x, f32x4::splat(t, q[3]));
    yq = yq.mul_add(x, f32x4::splat(t, q[2]));
    yq = yq.mul_add(x, f32x4::splat(t, q[1]));
    yq = yq.mul_add(x, f32x4::splat(t, q[0]));

    yp / yq
}

/// fast_log2f on 4 f32 values — bit manipulation + rational polynomial.
#[inline(always)]
fn fast_log2f_x4<T: F32x4Convert>(t: T, x: f32x4<T>) -> f32x4<T> {
    const LOG2_P: [f32; 3] = [-1.8503833400518310e-6, 1.4287160470083755, 7.4245873327820566e-1];
    const LOG2_Q: [f32; 3] = [9.9032814277590719e-1, 1.0096718572241148, 1.7409343003366853e-1];

    let x_bits = x.bitcast_to_i32();
    let magic = i32x4::splat(t, 0x3f2aaaab_u32 as i32);
    let exp_bits = x_bits - magic;
    let exp_shifted = exp_bits.shr_arithmetic_const::<23>();

    // Reconstruct mantissa: x_bits - (exp_shifted << 23)
    let shifted_back = exp_shifted.shl_const::<23>();
    let mantissa_bits = x_bits - shifted_back;
    let mantissa = f32x4::from_i32_bitcast(t, mantissa_bits);

    let exp_f = f32x4::from_i32(t, exp_shifted);
    let one = f32x4::splat(t, 1.0);
    let m = mantissa - one;

    // Degree-2 rational polynomial on (mantissa - 1.0)
    let mut yp = f32x4::splat(t, LOG2_P[2]);
    yp = yp.mul_add(m, f32x4::splat(t, LOG2_P[1]));
    yp = yp.mul_add(m, f32x4::splat(t, LOG2_P[0]));

    let mut yq = f32x4::splat(t, LOG2_Q[2]);
    yq = yq.mul_add(m, f32x4::splat(t, LOG2_Q[1]));
    yq = yq.mul_add(m, f32x4::splat(t, LOG2_Q[0]));

    let poly = yp / yq;
    poly + exp_f
}

/// fast_pow2f on 4 f32 values — integer bit manipulation + polynomial.
#[inline(always)]
fn fast_pow2f_x4<T: F32x4Convert>(t: T, x: f32x4<T>) -> f32x4<T> {
    const NUM: [f32; 3] = [1.01749063e1, 4.88687798e1, 9.85506591e1];
    const DEN: [f32; 4] = [2.10242958e-1, -2.22328856e-2, -1.94414990e1, 9.85506633e1];

    let x_floor = x.floor();
    let frac = x - x_floor;

    // exp = 2^floor(x) via integer bit manipulation
    let x_floor_i = x_floor.to_i32();
    let bias = i32x4::splat(t, 127);
    let exp_bits = (x_floor_i + bias).shl_const::<23>();
    let exp = f32x4::from_i32_bitcast(t, exp_bits);

    // Numerator: ((frac + NUM[0]) * frac + NUM[1]) * frac + NUM[2]) * exp
    let mut num = frac + f32x4::splat(t, NUM[0]);
    num = num.mul_add(frac, f32x4::splat(t, NUM[1]));
    num = num.mul_add(frac, f32x4::splat(t, NUM[2]));
    num = num * exp;

    // Denominator: ((DEN[0]*frac + DEN[1]) * frac + DEN[2]) * frac + DEN[3]
    let mut den = f32x4::splat(t, DEN[0]).mul_add(frac, f32x4::splat(t, DEN[1]));
    den = den.mul_add(frac, f32x4::splat(t, DEN[2]));
    den = den.mul_add(frac, f32x4::splat(t, DEN[3]));

    num / den
}

/// fast_powf(base, exp) on 4 f32 values: pow2f(exp * log2f(base)).
#[inline(always)]
fn fast_powf_x4<T: F32x4Convert>(t: T, base: f32x4<T>, exponent: f32) -> f32x4<T> {
    let log2 = fast_log2f_x4(t, base);
    let scaled = log2 * f32x4::splat(t, exponent);
    fast_pow2f_x4(t, scaled)
}

// =============================================================================
// sRGB SIMD kernels
// =============================================================================

const SRGB_TO_LINEAR_P: [f32; 5] = [
    2.200248328e-4, 1.043637593e-2, 1.624820318e-1, 7.961564959e-1, 8.210152774e-1,
];
const SRGB_TO_LINEAR_Q: [f32; 5] = [
    2.631846970e-1, 1.076976492, 4.987528350e-1, -5.512498495e-2, 6.521209011e-3,
];
const SRGB_FROM_LINEAR_P: [f32; 5] = [
    -5.135152395e-4, 5.287254571e-3, 3.903842876e-1, 1.474205315, 7.352629620e-1,
];
const SRGB_FROM_LINEAR_Q: [f32; 5] = [
    1.004519624e-2, 3.036675394e-1, 1.340816930, 9.258482155e-1, 2.424867759e-2,
];

#[inline(always)]
fn srgb_to_linear_x4<T: F32x4Convert>(t: T, v: f32x4<T>) -> f32x4<T> {
    let threshold = f32x4::splat(t, 0.04045);
    let inv_12_92 = f32x4::splat(t, 1.0 / 12.92);

    let linear = v * inv_12_92;
    let poly = eval_rational_poly_x4(t, v, SRGB_TO_LINEAR_P, SRGB_TO_LINEAR_Q);

    // v <= threshold ? linear : poly
    let mask = v.simd_le(threshold);
    f32x4::blend(mask, linear, poly)
}

#[inline(always)]
fn srgb_from_linear_x4<T: F32x4Convert>(t: T, v: f32x4<T>) -> f32x4<T> {
    let threshold = f32x4::splat(t, 0.0031308);
    let scale = f32x4::splat(t, 12.92);

    let linear = v * scale;
    let s = v.sqrt();
    let poly = eval_rational_poly_x4(t, s, SRGB_FROM_LINEAR_P, SRGB_FROM_LINEAR_Q);

    let mask = v.simd_le(threshold);
    f32x4::blend(mask, linear, poly)
}

// =============================================================================
// PQ SIMD kernels
// =============================================================================

const PQ_EOTF_P: [f32; 5] = [
    2.6297566e-4, -6.235531e-3, 7.386023e-1, 2.6455317, 5.500349e-1,
];
const PQ_EOTF_Q: [f32; 5] = [4.213501e2, -4.2873682e2, 1.7436467e2, -3.3907887e1, 2.6771877];

const PQ_INV_P_LARGE: [f32; 5] = [1.351392e-2, -1.095778, 5.522776e1, 1.492516e2, 4.838434e1];
const PQ_INV_Q_LARGE: [f32; 5] = [1.012416, 2.016708e1, 9.26371e1, 1.120607e2, 2.590418e1];

const PQ_INV_P_SMALL: [f32; 5] = [9.863406e-6, 3.881234e-1, 1.352821e2, 6.889862e4, -2.864824e5];
const PQ_INV_Q_SMALL: [f32; 5] = [3.371868e1, 1.477719e3, 1.608477e4, -4.389884e4, -2.072546e5];

#[inline(always)]
fn pq_to_linear_x4<T: F32x4Convert>(t: T, v: f32x4<T>) -> f32x4<T> {
    let zero = f32x4::zero(t);
    let a = v.max(zero);
    // x = a + a*a
    let x = a.mul_add(a, a);
    let result = eval_rational_poly_x4(t, x, PQ_EOTF_P, PQ_EOTF_Q);
    // Clamp negative inputs to 0
    let mask = v.simd_gt(zero);
    result & mask
}

#[inline(always)]
fn pq_from_linear_x4<T: F32x4Convert>(t: T, v: f32x4<T>) -> f32x4<T> {
    let zero = f32x4::zero(t);
    let a = v.max(zero);
    // Fourth root: sqrt(sqrt(x))
    let a4 = a.sqrt().sqrt();

    // Two-range approximation
    let threshold = f32x4::splat(t, 0.1);
    let large = eval_rational_poly_x4(t, a4, PQ_INV_P_LARGE, PQ_INV_Q_LARGE);
    let small = eval_rational_poly_x4(t, a4, PQ_INV_P_SMALL, PQ_INV_Q_SMALL);

    let mask = a4.simd_lt(threshold);
    let result = f32x4::blend(mask, small, large);

    // Clamp negative inputs to 0
    let pos_mask = v.simd_gt(zero);
    result & pos_mask
}

// =============================================================================
// BT.709 SIMD kernels
// =============================================================================

const BT709_ALPHA_F: f32 = 0.09929682680944;
const BT709_BETA_F: f32 = 0.018053968510807;

#[inline(always)]
fn bt709_to_linear_x4<T: F32x4Convert>(t: T, v: f32x4<T>) -> f32x4<T> {
    let threshold = f32x4::splat(t, 4.5 * BT709_BETA_F);
    let inv_4_5 = f32x4::splat(t, 1.0 / 4.5);
    let alpha = f32x4::splat(t, BT709_ALPHA_F);
    let one_plus_alpha = f32x4::splat(t, 1.0 + BT709_ALPHA_F);

    let linear = v * inv_4_5;

    // Power region: powf((v + alpha) / (1 + alpha), 1/0.45)
    let normalized = (v + alpha) / one_plus_alpha;
    let safe = normalized.max(f32x4::splat(t, f32::MIN_POSITIVE));
    let power = fast_powf_x4(t, safe, 1.0 / 0.45);

    let mask = v.simd_lt(threshold);
    f32x4::blend(mask, linear, power)
}

#[inline(always)]
fn bt709_from_linear_x4<T: F32x4Convert>(t: T, v: f32x4<T>) -> f32x4<T> {
    let threshold = f32x4::splat(t, BT709_BETA_F);
    let scale_4_5 = f32x4::splat(t, 4.5);
    let one_plus_alpha = f32x4::splat(t, 1.0 + BT709_ALPHA_F);
    let alpha = f32x4::splat(t, BT709_ALPHA_F);

    let linear = v * scale_4_5;

    // Power region: (1+alpha) * powf(v, 0.45) - alpha
    let safe = v.max(f32x4::splat(t, f32::MIN_POSITIVE));
    let power = fast_powf_x4(t, safe, 0.45);
    let power = one_plus_alpha.mul_add(power, -alpha);

    let mask = v.simd_lt(threshold);
    f32x4::blend(mask, linear, power)
}

// =============================================================================
// HLG SIMD kernels
// =============================================================================

#[inline(always)]
fn hlg_to_linear_x4<T: F32x4Convert>(t: T, v: f32x4<T>) -> f32x4<T> {
    let zero = f32x4::zero(t);
    let half = f32x4::splat(t, 0.5);
    let third = f32x4::splat(t, 1.0 / 3.0);
    let inv_12 = f32x4::splat(t, 1.0 / 12.0);
    let hlg_b = f32x4::splat(t, 0.28466892);
    let hlg_c = f32x4::splat(t, 0.55991073);
    let hlg_inv_a_log2e = f32x4::splat(t, core::f32::consts::LOG2_E / 0.17883277);

    let a = v.max(zero);

    // Low region: v^2 / 3
    let low = (a * a) * third;

    // High region: (exp((v - C) / A) + B) / 12 = (pow2((v-C) * log2e/A) + B) / 12
    let exp_arg = (a - hlg_c) * hlg_inv_a_log2e;
    let exp_val = fast_pow2f_x4(t, exp_arg);
    let high = (exp_val + hlg_b) * inv_12;

    // v <= 0.5 ? low : high
    let mask = a.simd_le(half);
    let result = f32x4::blend(mask, low, high);

    // v <= 0 ? 0 : result
    let pos_mask = v.simd_gt(zero);
    result & pos_mask
}

#[inline(always)]
fn hlg_from_linear_x4<T: F32x4Convert>(t: T, v: f32x4<T>) -> f32x4<T> {
    let zero = f32x4::zero(t);
    let threshold = f32x4::splat(t, 1.0 / 12.0);
    let three = f32x4::splat(t, 3.0);
    let twelve = f32x4::splat(t, 12.0);
    let hlg_a_ln2 = f32x4::splat(t, 0.17883277 * core::f32::consts::LN_2);
    let hlg_b = f32x4::splat(t, 0.28466892);
    let hlg_c = f32x4::splat(t, 0.55991073);

    let a = v.max(zero);

    // Low region: sqrt(3 * v)
    let low = (three * a).sqrt();

    // High region: A * ln(12*v - B) + C = A*ln2 * log2(12*v - B) + C
    let arg = twelve * a - hlg_b;
    let safe_arg = arg.max(f32x4::splat(t, f32::MIN_POSITIVE));
    let log2_val = fast_log2f_x4(t, safe_arg);
    let high = hlg_a_ln2.mul_add(log2_val, hlg_c);

    // v <= 1/12 ? low : high
    let mask = a.simd_le(threshold);
    let result = f32x4::blend(mask, low, high);

    let pos_mask = v.simd_gt(zero);
    result & pos_mask
}

// =============================================================================
// Row-level batch processors
// =============================================================================

/// Apply a 4-wide transfer function to a row of f32 values in-place.
///
/// For RGBA (4ch, has_alpha): processes 4 floats (1 pixel) at a time,
/// restoring alpha channel. For no-alpha: processes 4 floats flat.
#[inline(always)]
pub(super) fn tf_row_inplace<T: F32x4Convert>(
    t: T,
    row: &mut [f32],
    channels: usize,
    has_alpha: bool,
    tf_x4: fn(T, f32x4<T>) -> f32x4<T>,
    tf_scalar: fn(f32) -> f32,
) {
    if has_alpha && channels == 4 {
        // RGBA: process 4 floats (1 pixel) at a time, restore alpha
        let (chunks, _tail) = f32x4::<T>::partition_slice_mut(t, row);
        for chunk in chunks.iter_mut() {
            let alpha = chunk[3];
            let v = f32x4::load(t, chunk);
            let converted = tf_x4(t, v);
            converted.store(chunk);
            chunk[3] = alpha;
        }
        // No tail — RGBA row length is always a multiple of 4
    } else if has_alpha && channels >= 2 {
        // Non-4ch with alpha: per-pixel, skip last channel
        for pixel in row.chunks_exact_mut(channels) {
            for v in &mut pixel[..channels - 1] {
                *v = tf_scalar(*v);
            }
        }
    } else {
        // No alpha: process all values flat, 4 at a time
        let (chunks, tail) = f32x4::<T>::partition_slice_mut(t, row);
        for chunk in chunks.iter_mut() {
            let v = f32x4::load(t, chunk);
            let converted = tf_x4(t, v);
            converted.store(chunk);
        }
        for v in tail.iter_mut() {
            *v = tf_scalar(*v);
        }
    }
}

// Public batch functions — called from neon.rs and wasm128.rs

#[inline(always)]
pub(super) fn srgb_to_linear_row<T: F32x4Convert>(
    t: T, row: &mut [f32], channels: usize, has_alpha: bool,
) {
    tf_row_inplace(t, row, channels, has_alpha, srgb_to_linear_x4, fastmath::srgb_to_linear);
}

#[inline(always)]
pub(super) fn srgb_from_linear_row<T: F32x4Convert>(
    t: T, row: &mut [f32], channels: usize, has_alpha: bool,
) {
    tf_row_inplace(t, row, channels, has_alpha, srgb_from_linear_x4, fastmath::srgb_from_linear);
}

#[inline(always)]
pub(super) fn bt709_to_linear_row<T: F32x4Convert>(
    t: T, row: &mut [f32], channels: usize, has_alpha: bool,
) {
    tf_row_inplace(t, row, channels, has_alpha, bt709_to_linear_x4, fastmath::bt709_to_linear);
}

#[inline(always)]
pub(super) fn bt709_from_linear_row<T: F32x4Convert>(
    t: T, row: &mut [f32], channels: usize, has_alpha: bool,
) {
    tf_row_inplace(t, row, channels, has_alpha, bt709_from_linear_x4, fastmath::bt709_from_linear);
}

#[inline(always)]
pub(super) fn pq_to_linear_row<T: F32x4Convert>(
    t: T, row: &mut [f32], channels: usize, has_alpha: bool,
) {
    tf_row_inplace(t, row, channels, has_alpha, pq_to_linear_x4, fastmath::pq_to_linear);
}

#[inline(always)]
pub(super) fn pq_from_linear_row<T: F32x4Convert>(
    t: T, row: &mut [f32], channels: usize, has_alpha: bool,
) {
    tf_row_inplace(t, row, channels, has_alpha, pq_from_linear_x4, fastmath::pq_from_linear);
}

#[inline(always)]
pub(super) fn hlg_to_linear_row<T: F32x4Convert>(
    t: T, row: &mut [f32], channels: usize, has_alpha: bool,
) {
    tf_row_inplace(t, row, channels, has_alpha, hlg_to_linear_x4, fastmath::hlg_to_linear);
}

#[inline(always)]
pub(super) fn hlg_from_linear_row<T: F32x4Convert>(
    t: T, row: &mut [f32], channels: usize, has_alpha: bool,
) {
    tf_row_inplace(t, row, channels, has_alpha, hlg_from_linear_x4, fastmath::hlg_from_linear);
}
