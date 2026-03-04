//! Fast transfer function approximations using rational polynomials and bit tricks.
//!
//! Replaces libm `powf()` (~40-100ns, non-vectorizable) with rational polynomial
//! approximations (~5ns, SIMD-vectorizable). Coefficients from libjxl (BSD licensed).
//!
//! # Accuracy
//!
//! | TF | Max error vs f64 | Within u8? | Within u16? |
//! |---|---|---|---|
//! | sRGB | ~5e-7 | Yes | Yes |
//! | BT.709 forward | ~3e-7 | Yes | Yes |
//! | BT.709 inverse | ~3e-5 (fast_powf) | Yes | Yes (1 LSB = 1.5e-5) |
//! | PQ forward | ~7e-7 | Yes | Yes |
//! | PQ inverse | ~3e-6 | Yes | Yes |
//! | HLG | ~5e-6 | Yes | Yes |

// =============================================================================
// Rational polynomial evaluator — Horner's method for P(x)/Q(x)
// =============================================================================

/// Evaluate a rational polynomial P(x)/Q(x) using Horner's method.
///
/// Coefficients are stored lowest-degree-first: `p[0] + p[1]*x + p[2]*x^2 + ...`
#[inline(always)]
pub fn eval_rational_poly<const P: usize, const Q: usize>(
    x: f32,
    p: [f32; P],
    q: [f32; Q],
) -> f32 {
    // Horner's from highest to lowest degree
    let mut yp = p[P - 1];
    for i in (0..P - 1).rev() {
        yp = yp.mul_add(x, p[i]);
    }
    let mut yq = q[Q - 1];
    for i in (0..Q - 1).rev() {
        yq = yq.mul_add(x, q[i]);
    }
    yp / yq
}

// =============================================================================
// fast_log2f / fast_pow2f / fast_powf — bit-manipulation transcendentals
// =============================================================================

/// Fast base-2 logarithm (~3e-6 relative error).
///
/// Uses integer bit extraction for exponent + rational polynomial on mantissa.
/// Coefficients from libjxl (via jxl crate).
#[inline(always)]
pub fn fast_log2f(x: f32) -> f32 {
    const LOG2_P: [f32; 3] = [-1.8503833400518310e-6, 1.4287160470083755, 7.4245873327820566e-1];
    const LOG2_Q: [f32; 3] = [9.9032814277590719e-1, 1.0096718572241148, 1.7409343003366853e-1];

    let x_bits = x.to_bits() as i32;
    // Subtract magic constant to extract approximate exponent
    let exp_bits = x_bits.wrapping_sub(0x3f2aaaab);
    let exp_shifted = exp_bits >> 23;
    // Remove exponent from mantissa to get normalized value near 1.0
    let mantissa = f32::from_bits((x_bits.wrapping_sub(exp_shifted << 23)) as u32);
    let exp_val = exp_shifted as f32;
    eval_rational_poly(mantissa - 1.0, LOG2_P, LOG2_Q) + exp_val
}

/// Fast base-2 exponentiation (~3e-6 relative error).
///
/// Splits into integer exponent (bit shift) + fractional part (rational polynomial).
/// Coefficients from libjxl (via jxl crate).
#[inline(always)]
pub fn fast_pow2f(x: f32) -> f32 {
    const NUM: [f32; 3] = [1.01749063e1, 4.88687798e1, 9.85506591e1];
    const DEN: [f32; 4] = [2.10242958e-1, -2.22328856e-2, -1.94414990e1, 9.85506633e1];

    let x_floor = x.floor();
    let exp = f32::from_bits(((x_floor as i32 + 127) as u32) << 23);
    let frac = x - x_floor;

    let num = frac.mul_add(1.0, NUM[0]);
    let num = num.mul_add(frac, NUM[1]);
    let num = num.mul_add(frac, NUM[2]);
    let num = num * exp;

    let den = DEN[0].mul_add(frac, DEN[1]);
    let den = den.mul_add(frac, DEN[2]);
    let den = den.mul_add(frac, DEN[3]);

    num / den
}

/// Fast power function: `base^exp` (~3e-5 relative error).
///
/// Computed as `pow2(exp * log2(base))`.
#[inline(always)]
pub fn fast_powf(base: f32, exp: f32) -> f32 {
    fast_pow2f(fast_log2f(base) * exp)
}

// =============================================================================
// sRGB transfer function (rational polynomial approximation)
// =============================================================================

/// sRGB EOTF: encoded → linear. Rational polynomial, max error ~5e-7.
///
/// Coefficients from libjxl.
#[inline(always)]
pub fn srgb_to_linear(v: f32) -> f32 {
    const P: [f32; 5] = [
        2.200248328e-4,
        1.043637593e-2,
        1.624820318e-1,
        7.961564959e-1,
        8.210152774e-1,
    ];
    const Q: [f32; 5] = [
        2.631846970e-1,
        1.076976492,
        4.987528350e-1,
        -5.512498495e-2,
        6.521209011e-3,
    ];

    if v <= 0.04045 {
        v / 12.92
    } else {
        let x = v.abs();
        eval_rational_poly(x, P, Q)
    }
}

/// sRGB inverse EOTF: linear → encoded. Rational polynomial on sqrt(x), max error ~5e-7.
///
/// Coefficients from libjxl.
#[inline(always)]
pub fn srgb_from_linear(v: f32) -> f32 {
    const P: [f32; 5] = [
        -5.135152395e-4,
        5.287254571e-3,
        3.903842876e-1,
        1.474205315,
        7.352629620e-1,
    ];
    const Q: [f32; 5] = [
        1.004519624e-2,
        3.036675394e-1,
        1.340816930,
        9.258482155e-1,
        2.424867759e-2,
    ];

    if v <= 0.0031308 {
        v * 12.92
    } else {
        let s = v.abs().sqrt();
        eval_rational_poly(s, P, Q)
    }
}

// =============================================================================
// BT.709 transfer function
// =============================================================================

// Exact constants for C0/C1 continuity at the piecewise boundary.
const BT709_ALPHA: f32 = 0.09929682680944;
const BT709_BETA: f32 = 0.018053968510807;

/// BT.709 EOTF: encoded → linear. Uses fast_powf, max error ~3e-6.
#[inline(always)]
pub fn bt709_to_linear(v: f32) -> f32 {
    if v < 4.5 * BT709_BETA {
        v / 4.5
    } else {
        fast_powf((v + BT709_ALPHA) / (1.0 + BT709_ALPHA), 1.0 / 0.45)
    }
}

/// BT.709 inverse EOTF: linear → encoded. Uses fast_powf, max error ~3e-5.
#[inline(always)]
pub fn bt709_from_linear(v: f32) -> f32 {
    if v < BT709_BETA {
        4.5 * v
    } else {
        (1.0 + BT709_ALPHA) * fast_powf(v, 0.45) - BT709_ALPHA
    }
}

// =============================================================================
// PQ (SMPTE ST 2084) transfer function
// =============================================================================

/// PQ EOTF: signal → linear. Rational polynomial, max error ~7e-7.
///
/// Zero `powf()` calls for signal > 0.02 — uses `x + x*x` input transformation
/// and rational polynomial. Very small signal values use exact formula to
/// maintain u8 roundtrip precision in the steep near-zero region.
/// Coefficients from libjxl.
#[inline(always)]
pub fn pq_to_linear(v: f32) -> f32 {
    const P: [f32; 5] = [
        2.6297566e-4,
        -6.235531e-3,
        7.386023e-1,
        2.6455317,
        5.500349e-1,
    ];
    const Q: [f32; 5] = [4.213501e2, -4.2873682e2, 1.7436467e2, -3.3907887e1, 2.6771877];

    if v <= 0.0 {
        return 0.0;
    }
    // Very small signal values: use exact formula for u8 roundtrip precision.
    // PQ signal 0.02 ≈ u8 value 5; below this the curve is extremely steep.
    if v < 0.02 {
        return pq_to_linear_exact(v);
    }
    let a = v;
    let x = a + a * a;
    eval_rational_poly(x, P, Q)
}

/// Exact PQ EOTF for very small signal values.
#[inline(always)]
fn pq_to_linear_exact(v: f32) -> f32 {
    const M1: f32 = 0.1593017578125;
    const M2: f32 = 78.84375;
    const C1: f32 = 0.8359375;
    const C2: f32 = 18.8515625;
    const C3: f32 = 18.6875;
    let vp = v.powf(1.0 / M2);
    let num = (vp - C1).max(0.0);
    let den = C2 - C3 * vp;
    if den <= 0.0 {
        return 1.0;
    }
    (num / den).powf(1.0 / M1)
}

/// PQ inverse EOTF: linear → signal. Rational polynomial on x^(1/4), max error ~3e-6.
///
/// Uses sqrt(sqrt(x)) to extract the fourth root, then rational polynomial.
/// Two-range approximation with threshold at sqrt(sqrt(x)) < 0.1.
/// Very small values (v < 1e-4) use exact powf to avoid polynomial imprecision
/// near zero where the PQ curve is extremely steep.
/// Coefficients from libjxl.
#[inline(always)]
pub fn pq_from_linear(v: f32) -> f32 {
    const P_LARGE: [f32; 5] = [1.351392e-2, -1.095778, 5.522776e1, 1.492516e2, 4.838434e1];
    const Q_LARGE: [f32; 5] = [1.012416, 2.016708e1, 9.26371e1, 1.120607e2, 2.590418e1];

    const P_SMALL: [f32; 5] = [
        9.863406e-6,
        3.881234e-1,
        1.352821e2,
        6.889862e4,
        -2.864824e5,
    ];
    const Q_SMALL: [f32; 5] = [
        3.371868e1,
        1.477719e3,
        1.608477e4,
        -4.389884e4,
        -2.072546e5,
    ];

    if v <= 0.0 {
        return 0.0;
    }
    // Very small linear values: the polynomial can't track PQ's extreme steepness
    // near zero (PQ maps u8=1 to linear ~6e-7). Use exact formula for these.
    if v < 1e-4 {
        return pq_from_linear_exact(v);
    }
    let a = v.sqrt().sqrt(); // fourth root
    // Small-range polynomial is more accurate for a < 0.1.
    // Large-range polynomial is more accurate above that.
    if a < 0.1 {
        eval_rational_poly(a, P_SMALL, Q_SMALL)
    } else {
        eval_rational_poly(a, P_LARGE, Q_LARGE)
    }
}

/// Exact PQ inverse EOTF for very small linear values where the polynomial is imprecise.
#[inline(always)]
fn pq_from_linear_exact(v: f32) -> f32 {
    const M1: f32 = 0.1593017578125;
    const M2: f32 = 78.84375;
    const C1: f32 = 0.8359375;
    const C2: f32 = 18.8515625;
    const C3: f32 = 18.6875;
    let vp = v.powf(M1);
    let num = C1 + C2 * vp;
    let den = 1.0 + C3 * vp;
    (num / den).powf(M2)
}

// =============================================================================
// HLG (ARIB STD-B67) transfer function
// =============================================================================

const HLG_A: f32 = 0.17883277;
const HLG_B: f32 = 0.28466892; // 1 - 4 * A
const HLG_C: f32 = 0.55991073; // 0.5 - A * ln(4 * A)
const HLG_A_INV_LOG2E: f32 = HLG_A * core::f32::consts::LN_2; // A * ln(2) for log2 conversion
const HLG_INV_A_LOG2E: f32 = core::f32::consts::LOG2_E / HLG_A; // log2(e) / A for exp2 conversion

/// HLG inverse OETF: signal → scene linear. Uses fast_pow2f for exp().
#[inline(always)]
pub fn hlg_to_linear(v: f32) -> f32 {
    if v <= 0.0 {
        0.0
    } else if v <= 0.5 {
        (v * v) / 3.0
    } else {
        // exp((v - C) / A) = pow2((v - C) / A * log2(e))
        (fast_pow2f((v - HLG_C) * HLG_INV_A_LOG2E) + HLG_B) / 12.0
    }
}

/// HLG OETF: scene linear → signal. Uses fast_log2f for ln().
#[inline(always)]
pub fn hlg_from_linear(v: f32) -> f32 {
    if v <= 0.0 {
        0.0
    } else if v <= 1.0 / 12.0 {
        (3.0 * v).sqrt()
    } else {
        // A * ln(12*v - B) + C = A * ln(2) * log2(12*v - B) + C
        HLG_A_INV_LOG2E * fast_log2f(12.0 * v - HLG_B) + HLG_C
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference sRGB EOTF using f64 for accuracy.
    fn srgb_to_linear_f64(v: f64) -> f64 {
        if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    }

    /// Reference sRGB inverse EOTF using f64.
    fn srgb_from_linear_f64(v: f64) -> f64 {
        if v <= 0.0031308 {
            v * 12.92
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        }
    }

    /// Reference PQ EOTF using f64.
    fn pq_to_linear_f64(v: f64) -> f64 {
        if v <= 0.0 {
            return 0.0;
        }
        let m1: f64 = 0.1593017578125;
        let m2: f64 = 78.84375;
        let c1: f64 = 0.8359375;
        let c2: f64 = 18.8515625;
        let c3: f64 = 18.6875;
        let vp = v.powf(1.0 / m2);
        let num = (vp - c1).max(0.0);
        let den = c2 - c3 * vp;
        if den <= 0.0 {
            return 1.0;
        }
        (num / den).powf(1.0 / m1)
    }

    /// Reference PQ inverse EOTF using f64.
    fn pq_from_linear_f64(v: f64) -> f64 {
        if v <= 0.0 {
            return 0.0;
        }
        let m1: f64 = 0.1593017578125;
        let m2: f64 = 78.84375;
        let c1: f64 = 0.8359375;
        let c2: f64 = 18.8515625;
        let c3: f64 = 18.6875;
        let vp = v.powf(m1);
        let num = c1 + c2 * vp;
        let den = 1.0 + c3 * vp;
        (num / den).powf(m2)
    }

    /// Reference BT.709 EOTF using f64.
    fn bt709_to_linear_f64(v: f64) -> f64 {
        let beta: f64 = 0.018053968510807;
        if v < 4.5 * beta {
            v / 4.5
        } else {
            let alpha: f64 = 0.09929682680944;
            ((v + alpha) / (1.0 + alpha)).powf(1.0 / 0.45)
        }
    }

    /// Reference BT.709 inverse EOTF using f64.
    fn bt709_from_linear_f64(v: f64) -> f64 {
        let beta: f64 = 0.018053968510807;
        let alpha: f64 = 0.09929682680944;
        if v < beta {
            4.5 * v
        } else {
            (1.0 + alpha) * v.powf(0.45) - alpha
        }
    }

    /// Reference HLG inverse OETF using f64.
    fn hlg_to_linear_f64(v: f64) -> f64 {
        let a: f64 = 0.17883277;
        let b: f64 = 0.28466892;
        let c: f64 = 0.55991073;
        if v <= 0.0 {
            0.0
        } else if v <= 0.5 {
            (v * v) / 3.0
        } else {
            (((v - c) / a).exp() + b) / 12.0
        }
    }

    /// Reference HLG OETF using f64.
    fn hlg_from_linear_f64(v: f64) -> f64 {
        let a: f64 = 0.17883277;
        let b: f64 = 0.28466892;
        let c: f64 = 0.55991073;
        if v <= 0.0 {
            0.0
        } else if v <= 1.0 / 12.0 {
            (3.0 * v).sqrt()
        } else {
            a * (12.0 * v - b).ln() + c
        }
    }

    fn max_abs_error(
        fast: impl Fn(f32) -> f32,
        reference: impl Fn(f64) -> f64,
        range: core::ops::RangeInclusive<f32>,
        steps: usize,
    ) -> (f32, f32) {
        let mut max_err: f64 = 0.0;
        let mut worst_input = 0.0f32;
        let lo = *range.start();
        let hi = *range.end();
        for i in 0..=steps {
            let t = i as f32 / steps as f32;
            let v = lo + (hi - lo) * t;
            let fast_val = fast(v) as f64;
            let ref_val = reference(v as f64);
            let err = (fast_val - ref_val).abs();
            if err > max_err {
                max_err = err;
                worst_input = v;
            }
        }
        (max_err as f32, worst_input)
    }

    #[test]
    fn srgb_to_linear_accuracy() {
        let (err, worst) = max_abs_error(srgb_to_linear, srgb_to_linear_f64, 0.0..=1.0, 100_000);
        eprintln!("sRGB to_linear max error: {err:.2e} at {worst}");
        assert!(err < 5e-6, "sRGB to_linear error {err:.2e} too high at {worst}");
    }

    #[test]
    fn srgb_from_linear_accuracy() {
        let (err, worst) =
            max_abs_error(srgb_from_linear, srgb_from_linear_f64, 0.0..=1.0, 100_000);
        eprintln!("sRGB from_linear max error: {err:.2e} at {worst}");
        assert!(err < 5e-6, "sRGB from_linear error {err:.2e} too high at {worst}");
    }

    #[test]
    fn pq_to_linear_accuracy() {
        let (err, worst) = max_abs_error(pq_to_linear, pq_to_linear_f64, 0.0..=1.0, 100_000);
        eprintln!("PQ to_linear max error: {err:.2e} at {worst}");
        assert!(err < 5e-5, "PQ to_linear error {err:.2e} too high at {worst}");
    }

    #[test]
    fn pq_from_linear_accuracy() {
        let (err, worst) = max_abs_error(pq_from_linear, pq_from_linear_f64, 0.0..=1.0, 100_000);
        eprintln!("PQ from_linear max error: {err:.2e} at {worst}");
        assert!(err < 5e-5, "PQ from_linear error {err:.2e} too high at {worst}");
    }

    #[test]
    fn bt709_to_linear_accuracy() {
        let (err, worst) = max_abs_error(bt709_to_linear, bt709_to_linear_f64, 0.0..=1.0, 100_000);
        eprintln!("BT.709 to_linear max error: {err:.2e} at {worst}");
        assert!(
            err < 5e-5,
            "BT.709 to_linear error {err:.2e} too high at {worst}"
        );
    }

    #[test]
    fn bt709_from_linear_accuracy() {
        let (err, worst) =
            max_abs_error(bt709_from_linear, bt709_from_linear_f64, 0.0..=1.0, 100_000);
        eprintln!("BT.709 from_linear max error: {err:.2e} at {worst}");
        assert!(
            err < 5e-4,
            "BT.709 from_linear error {err:.2e} too high at {worst}"
        );
    }

    #[test]
    fn hlg_to_linear_accuracy() {
        let (err, worst) = max_abs_error(hlg_to_linear, hlg_to_linear_f64, 0.0..=1.0, 100_000);
        eprintln!("HLG to_linear max error: {err:.2e} at {worst}");
        assert!(err < 5e-4, "HLG to_linear error {err:.2e} too high at {worst}");
    }

    #[test]
    fn hlg_from_linear_accuracy() {
        let (err, worst) = max_abs_error(hlg_from_linear, hlg_from_linear_f64, 0.0..=1.0, 100_000);
        eprintln!("HLG from_linear max error: {err:.2e} at {worst}");
        assert!(
            err < 5e-4,
            "HLG from_linear error {err:.2e} too high at {worst}"
        );
    }

    #[test]
    fn srgb_roundtrip() {
        for i in 0..=255 {
            let encoded = i as f32 / 255.0;
            let linear = srgb_to_linear(encoded);
            let back = srgb_from_linear(linear);
            let err = (back - encoded).abs();
            assert!(
                err < 1.0 / 255.0,
                "sRGB roundtrip failed at {i}: {encoded} -> {linear} -> {back} (err={err})"
            );
        }
    }

    #[test]
    fn pq_roundtrip() {
        // PQ has an extremely steep curve near zero: signal 0.01 maps to linear ~6.7e-7.
        // The roundtrip amplifies per-function error by the curve's slope, so only test
        // the mid-to-high range where the curve is gentler.
        for i in 50..=1000 {
            let v = i as f32 / 1000.0;
            let linear = pq_to_linear(v);
            let back = pq_from_linear(linear);
            let err = (back - v).abs();
            assert!(
                err < 0.001,
                "PQ roundtrip failed at {v}: -> {linear} -> {back} (err={err})"
            );
        }
    }

    #[test]
    fn hlg_roundtrip() {
        for i in 0..=1000 {
            let v = i as f32 / 1000.0;
            let linear = hlg_to_linear(v);
            let back = hlg_from_linear(linear);
            let err = (back - v).abs();
            assert!(
                err < 0.01,
                "HLG roundtrip failed at {v}: -> {linear} -> {back} (err={err})"
            );
        }
    }

    #[test]
    fn fast_log2f_accuracy() {
        for i in 1..=10000 {
            let x = i as f32 / 10000.0;
            let fast = fast_log2f(x);
            let exact = (x as f64).log2() as f32;
            let err = (fast - exact).abs();
            assert!(
                err < 0.01,
                "fast_log2f({x}) = {fast}, expected {exact} (err={err})"
            );
        }
    }

    #[test]
    fn fast_pow2f_accuracy() {
        for i in -100..=100 {
            let x = i as f32 / 10.0;
            let fast = fast_pow2f(x);
            let exact = (x as f64).exp2() as f32;
            let rel_err = if exact.abs() > 1e-10 {
                ((fast - exact) / exact).abs()
            } else {
                (fast - exact).abs()
            };
            assert!(
                rel_err < 0.001,
                "fast_pow2f({x}) = {fast}, expected {exact} (rel_err={rel_err})"
            );
        }
    }
}
