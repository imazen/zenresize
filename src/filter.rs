//! Interpolation filters ported from imageflow.
//!
//! This module provides all 31 resampling filters from imageflow, enabling
//! high-quality image resizing with various tradeoffs between sharpness,
//! ringing artifacts, and performance.
//!
//! ## Filter Categories
//!
//! - **Lanczos family**: Sharp, some ringing — best for photos
//! - **Robidoux family**: Balanced sharpness/smoothness — good default
//! - **Cubic family**: Smooth, minimal ringing
//! - **Ginseng**: Jinc-windowed sinc — excellent for upscaling
//! - **Box/Triangle/Linear**: Simple, fast filters

use core::f64::consts::PI;

/// Zero-cost sharpness adjustment applied during weight computation.
///
/// All variants modify the interpolation weights at weight-table construction
/// time. There is no extra convolution pass and no runtime cost beyond the
/// (negligible) difference in weight values.
///
/// # Kernel scaling vs. lobe amplification
///
/// `KernelScale` changes the kernel's effective width. A wider kernel
/// averages more input pixels (softer), a narrower one samples fewer
/// (sharper, more aliasing risk). This is equivalent to imageflow's `blur`
/// parameter on `InterpolationDetails`.
///
/// `SharpenPercent` leaves the kernel width unchanged but amplifies the
/// negative lobes, increasing perceived edge contrast. This is equivalent
/// to imageflow's `f.sharpen` parameter. It only adds sharpness — if the
/// filter already exceeds the target, no change is made.
///
/// # Kernel scaling vs. post-resize Gaussian blur
///
/// `KernelScale(f)` with `f > 1.0` and `post_blur(sigma)` both soften the
/// output, but they are *not* equivalent:
///
/// - **Kernel scaling** multiplies the resampling kernel's coordinate axis
///   by `f`, uniformly stretching it. In frequency domain this compresses
///   the response by `1/f` — a hard shift of the whole MTF curve.
///
/// - **Post-resize Gaussian blur** convolves the output with
///   `exp(-x² / 2σ²)`, which applies a Gaussian roll-off in frequency
///   domain: `exp(-2π²σ²f²)`. This attenuates high frequencies more
///   aggressively than low ones.
///
/// For small adjustments (within ~20% of the default) the visual difference
/// is negligible and `KernelScale` should be preferred since it's free.
/// Post-resize blur is only useful when you need a Gaussian-shaped frequency
/// response (e.g., matched to a specific noise model) or a fixed sigma
/// independent of the resize ratio.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FilterSharpness {
    /// Use the filter preset's default kernel shape. No adjustment.
    Preset,

    /// Scale the kernel width by a factor.
    ///
    /// `> 1.0`: wider kernel → softer output (e.g., `1.1` = 10% wider).
    /// `< 1.0`: narrower kernel → sharper output (e.g., `0.9` = 10% narrower).
    ///
    /// Multiplied with the filter preset's built-in blur value. For example,
    /// `LanczosSharp` (built-in blur ≈ 0.98) with `KernelScale(0.95)` yields
    /// effective blur ≈ 0.93.
    KernelScale(f64),

    /// Amplify the kernel's negative lobes to a target percentage.
    ///
    /// Range: `0.0` (no extra sharpening) to ~`50.0` (aggressive).
    /// Only increases sharpness beyond what the filter naturally provides.
    /// Matches imageflow's `f.sharpen` parameter.
    ///
    /// Unlike `KernelScale`, this does not change the kernel width — it
    /// separately scales positive and negative weights so the negative-weight
    /// area reaches the target ratio, increasing perceived edge contrast.
    SharpenPercent(f32),
}

impl Default for FilterSharpness {
    fn default() -> Self {
        FilterSharpness::Preset
    }
}

impl FilterSharpness {
    /// Convert to the raw blur factor multiplier.
    /// Returns `None` for `SharpenPercent` (handled in weight tables).
    pub(crate) fn blur_factor(&self) -> f64 {
        match self {
            FilterSharpness::Preset => 1.0,
            FilterSharpness::KernelScale(f) => *f,
            FilterSharpness::SharpenPercent(_) => 1.0,
        }
    }

    /// Get the sharpen-percent goal, if any.
    pub(crate) fn sharpen_percent(&self) -> f32 {
        match self {
            FilterSharpness::SharpenPercent(p) => *p,
            _ => 0.0,
        }
    }
}

/// Named interpolation filter presets.
///
/// These match imageflow's filter names exactly for compatibility.
#[repr(u8)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash, Default)]
pub enum Filter {
    /// Robidoux with smaller window — faster, slight quality loss
    RobidouxFast = 1,
    /// Balanced cubic filter — good default for downscaling
    #[default]
    Robidoux = 2,
    /// Robidoux with sharpening — more detail, slight ringing
    RobidouxSharp = 3,
    /// Jinc-windowed sinc — excellent for upscaling
    Ginseng = 4,
    /// Ginseng with sharpening
    GinsengSharp = 5,
    /// Lanczos-3 windowed sinc — sharp, some ringing
    Lanczos = 6,
    /// Lanczos-3 with sharpening
    LanczosSharp = 7,
    /// Lanczos-2 windowed sinc — less ringing than Lanczos-3
    Lanczos2 = 8,
    /// Lanczos-2 with sharpening
    Lanczos2Sharp = 9,
    /// Fast bicubic approximation
    CubicFast = 10,
    /// Standard cubic (B=0, C=1)
    Cubic = 11,
    /// Cubic with sharpening
    CubicSharp = 12,
    /// Catmull-Rom spline (B=0, C=0.5)
    CatmullRom = 13,
    /// Mitchell-Netravali (B=1/3, C=1/3) — balanced blur/ringing
    Mitchell = 14,
    /// Cubic B-spline (B=1, C=0) — very smooth, blurs
    CubicBSpline = 15,
    /// Hermite (B=0, C=0) — smooth interpolation
    Hermite = 16,
    /// Jinc function — circular sinc
    Jinc = 17,
    /// Raw Lanczos-3 without windowing
    RawLanczos3 = 18,
    /// Raw Lanczos-3 with sharpening
    RawLanczos3Sharp = 19,
    /// Raw Lanczos-2 without windowing
    RawLanczos2 = 20,
    /// Raw Lanczos-2 with sharpening
    RawLanczos2Sharp = 21,
    /// Triangle/tent filter — linear interpolation
    Triangle = 22,
    /// Linear interpolation (same as Triangle)
    Linear = 23,
    /// Box/nearest neighbor — fastest, blocky
    Box = 24,
    /// Fast Catmull-Rom with smaller window
    CatmullRomFast = 25,
    /// Fast Catmull-Rom with sharpening
    CatmullRomFastSharp = 26,
    /// Fastest filter — minimal quality
    Fastest = 27,
    /// Fast Mitchell with smaller window
    MitchellFast = 28,
    /// NCubic — optimized cubic
    NCubic = 29,
    /// NCubic with sharpening
    NCubicSharp = 30,
    /// Legacy IDCT filter
    LegacyIDCTFilter = 31,
}

impl Filter {
    /// Get all filter variants for iteration.
    pub fn all() -> &'static [Filter] {
        use Filter::*;
        &[
            RobidouxFast,
            Robidoux,
            RobidouxSharp,
            Ginseng,
            GinsengSharp,
            Lanczos,
            LanczosSharp,
            Lanczos2,
            Lanczos2Sharp,
            CubicFast,
            Cubic,
            CubicSharp,
            CatmullRom,
            Mitchell,
            CubicBSpline,
            Hermite,
            Jinc,
            RawLanczos3,
            RawLanczos3Sharp,
            RawLanczos2,
            RawLanczos2Sharp,
            Triangle,
            Linear,
            Box,
            CatmullRomFast,
            CatmullRomFastSharp,
            Fastest,
            MitchellFast,
            NCubic,
            NCubicSharp,
            LegacyIDCTFilter,
        ]
    }

    /// Get the filter name as a string.
    pub fn name(&self) -> &'static str {
        match self {
            Filter::RobidouxFast => "robidoux_fast",
            Filter::Robidoux => "robidoux",
            Filter::RobidouxSharp => "robidoux_sharp",
            Filter::Ginseng => "ginseng",
            Filter::GinsengSharp => "ginseng_sharp",
            Filter::Lanczos => "lanczos",
            Filter::LanczosSharp => "lanczos_sharp",
            Filter::Lanczos2 => "lanczos2",
            Filter::Lanczos2Sharp => "lanczos2_sharp",
            Filter::CubicFast => "cubic_fast",
            Filter::Cubic => "cubic",
            Filter::CubicSharp => "cubic_sharp",
            Filter::CatmullRom => "catmull_rom",
            Filter::Mitchell => "mitchell",
            Filter::CubicBSpline => "cubic_b_spline",
            Filter::Hermite => "hermite",
            Filter::Jinc => "jinc",
            Filter::RawLanczos3 => "raw_lanczos3",
            Filter::RawLanczos3Sharp => "raw_lanczos3_sharp",
            Filter::RawLanczos2 => "raw_lanczos2",
            Filter::RawLanczos2Sharp => "raw_lanczos2_sharp",
            Filter::Triangle => "triangle",
            Filter::Linear => "linear",
            Filter::Box => "box",
            Filter::CatmullRomFast => "catmull_rom_fast",
            Filter::CatmullRomFastSharp => "catmull_rom_fast_sharp",
            Filter::Fastest => "fastest",
            Filter::MitchellFast => "mitchell_fast",
            Filter::NCubic => "n_cubic",
            Filter::NCubicSharp => "n_cubic_sharp",
            Filter::LegacyIDCTFilter => "legacy_idct",
        }
    }
}

/// Interpolation filter details and configuration.
///
/// Contains the filter function and parameters needed to compute weights.
#[derive(Clone)]
pub struct InterpolationDetails {
    /// Filter support window (half-width in input pixels)
    pub window: f64,
    /// Blur factor (< 1.0 sharpens, > 1.0 blurs)
    pub blur: f64,
    /// Bicubic coefficient p1
    pub p1: f64,
    /// Bicubic coefficient p2
    pub p2: f64,
    /// Bicubic coefficient p3
    pub p3: f64,
    /// Bicubic coefficient q1
    pub q1: f64,
    /// Bicubic coefficient q2
    pub q2: f64,
    /// Bicubic coefficient q3
    pub q3: f64,
    /// Bicubic coefficient q4
    pub q4: f64,
    /// The filter function to use
    filter_fn: FilterFn,
    /// Sharpening goal (0.0 = none, positive = sharpen).
    ///
    /// Set via [`with_sharpen_percent`](Self::with_sharpen_percent).
    /// Read by [`F32WeightTable::new`](crate::weights::F32WeightTable::new)
    /// to amplify negative lobes during weight computation.
    pub sharpen_percent_goal: f32,
}

/// Filter function type.
type FilterFn = fn(&InterpolationDetails, f64) -> f64;

impl Default for InterpolationDetails {
    fn default() -> Self {
        Self {
            window: 2.0,
            blur: 1.0,
            p1: 0.0,
            p2: 1.0,
            p3: 1.0,
            q1: 0.0,
            q2: 1.0,
            q3: 1.0,
            q4: 1.0,
            filter_fn: filter_box,
            sharpen_percent_goal: 0.0,
        }
    }
}

impl InterpolationDetails {
    /// Create interpolation details for a named filter.
    pub fn create(filter: Filter) -> Self {
        match filter {
            Filter::Triangle | Filter::Linear => Self {
                window: 1.0,
                blur: 1.0,
                filter_fn: filter_triangle,
                ..Default::default()
            },
            Filter::RawLanczos2 => Self {
                window: 2.0,
                blur: 1.0,
                filter_fn: filter_sinc,
                ..Default::default()
            },
            Filter::RawLanczos3 => Self {
                window: 3.0,
                blur: 1.0,
                filter_fn: filter_sinc,
                ..Default::default()
            },
            Filter::RawLanczos2Sharp => Self {
                window: 2.0,
                blur: 0.9549963639785485,
                filter_fn: filter_sinc,
                ..Default::default()
            },
            Filter::RawLanczos3Sharp => Self {
                window: 3.0,
                blur: 0.9812505644269356,
                filter_fn: filter_sinc,
                ..Default::default()
            },
            Filter::Lanczos2 => Self {
                window: 2.0,
                blur: 1.0,
                filter_fn: filter_sinc_windowed,
                ..Default::default()
            },
            Filter::Lanczos => Self {
                window: 3.0,
                blur: 1.0,
                filter_fn: filter_sinc_windowed,
                ..Default::default()
            },
            Filter::Lanczos2Sharp => Self {
                window: 2.0,
                blur: 0.9549963639785485,
                filter_fn: filter_sinc_windowed,
                ..Default::default()
            },
            Filter::LanczosSharp => Self {
                window: 3.0,
                blur: 0.9812505644269356,
                filter_fn: filter_sinc_windowed,
                ..Default::default()
            },
            Filter::CubicFast => Self {
                window: 2.0,
                blur: 1.0,
                filter_fn: filter_bicubic_fast,
                ..Default::default()
            },
            Filter::Box => Self {
                window: 0.5,
                blur: 1.0,
                filter_fn: filter_box,
                ..Default::default()
            },
            Filter::Ginseng => Self {
                window: 3.0,
                blur: 1.0,
                filter_fn: filter_ginseng,
                ..Default::default()
            },
            Filter::GinsengSharp => Self {
                window: 3.0,
                blur: 0.9812505644269356,
                filter_fn: filter_ginseng,
                ..Default::default()
            },
            Filter::Jinc => Self {
                window: 6.0,
                blur: 1.0,
                filter_fn: filter_jinc,
                ..Default::default()
            },
            Filter::CubicBSpline => Self::bicubic(2.0, 1.0, 1.0, 0.0),
            Filter::Cubic => Self::bicubic(2.0, 1.0, 0.0, 1.0),
            Filter::CubicSharp => Self::bicubic(2.0, 0.9549963639785485, 0.0, 1.0),
            Filter::CatmullRom => Self::bicubic(2.0, 1.0, 0.0, 0.5),
            Filter::CatmullRomFast => Self::bicubic(1.0, 1.0, 0.0, 0.5),
            Filter::CatmullRomFastSharp => Self::bicubic(1.0, 13.0 / 16.0, 0.0, 0.5),
            Filter::Mitchell => Self::bicubic(2.0, 1.0, 1.0 / 3.0, 1.0 / 3.0),
            Filter::MitchellFast => Self::bicubic(1.0, 1.0, 1.0 / 3.0, 1.0 / 3.0),
            Filter::NCubic => Self::bicubic(
                2.5,
                1.0 / 1.1685777620836933,
                0.3782157550939987,
                0.3108921224530007,
            ),
            Filter::NCubicSharp => Self::bicubic(
                2.5,
                1.0 / 1.105822933719019,
                0.2620145123990142,
                0.3689927438004929,
            ),
            Filter::Robidoux => Self::bicubic(2.0, 1.0, 0.3782157550939987, 0.3108921224530007),
            Filter::LegacyIDCTFilter => Self::bicubic(
                2.0,
                1.0 / 1.1685777620836932,
                0.3782157550939987,
                0.3108921224530007,
            ),
            Filter::Fastest => Self::bicubic(0.74, 0.74, 0.3782157550939987, 0.3108921224530007),
            Filter::RobidouxFast => {
                Self::bicubic(1.05, 1.0, 0.3782157550939987, 0.3108921224530007)
            }
            Filter::RobidouxSharp => {
                Self::bicubic(2.0, 1.0, 0.2620145123990142, 0.3689927438004929)
            }
            Filter::Hermite => Self::bicubic(1.0, 1.0, 0.0, 0.0),
        }
    }

    /// Create bicubic interpolation with B and C parameters.
    fn bicubic(window: f64, blur: f64, b: f64, c: f64) -> Self {
        let bx2 = b + b;
        Self {
            window,
            blur,
            filter_fn: filter_flex_cubic,
            p1: 1.0 - (1.0 / 3.0) * b,
            p2: -3.0 + bx2 + c,
            p3: 2.0 - 1.5 * b - c,
            q1: (4.0 / 3.0) * b + 4.0 * c,
            q2: -8.0 * c - bx2,
            q3: b + 5.0 * c,
            q4: (-1.0 / 6.0) * b - c,
            sharpen_percent_goal: 0.0,
        }
    }

    /// Multiply the blur factor.
    pub fn with_blur(mut self, factor: f64) -> Self {
        self.blur *= factor;
        self
    }

    /// Set the sharpen-percent goal.
    ///
    /// Amplifies negative lobes during weight computation to reach the
    /// target negative-weight ratio. Range: 0–100. Only increases sharpness
    /// beyond the filter's natural ratio. Matches imageflow's `f.sharpen`.
    pub fn with_sharpen_percent(mut self, goal: f32) -> Self {
        self.sharpen_percent_goal = goal;
        self
    }

    /// Evaluate the filter at position x.
    #[inline]
    pub fn filter(&self, x: f64) -> f64 {
        (self.filter_fn)(self, x)
    }

    /// Get the effective window size accounting for blur.
    #[cfg(test)]
    #[inline]
    pub fn effective_window(&self) -> f64 {
        self.window * self.blur
    }

    /// Calculate the ratio of negative weight area to positive weight area.
    ///
    /// Used by [`FilterSharpness::SharpenPercent`] to determine how much
    /// additional negative-lobe amplification is needed.
    pub fn calculate_percent_negative_weight(&self) -> f64 {
        let samples = 50i32;
        let step = self.window / samples as f64;
        let mut last_height = self.filter(-step);
        let mut positive_area = 0.0;
        let mut negative_area = 0.0;

        for i in 0..(samples + 3) {
            let height = self.filter(i as f64 * step);
            let area = (height + last_height) / 2.0 * step;
            last_height = height;
            if area > 0.0 {
                positive_area += area;
            } else {
                negative_area -= area;
            }
        }
        negative_area / positive_area
    }
}

// =============================================================================
// Filter Functions
// =============================================================================

/// Flexible bicubic filter with configurable B and C parameters.
fn filter_flex_cubic(d: &InterpolationDetails, x: f64) -> f64 {
    let t = f64_abs(x) / d.blur;
    if t < 1.0 {
        return d.p1 + t * (t * (d.p2 + t * d.p3));
    }
    if t < 2.0 {
        return d.q1 + t * (d.q2 + t * (d.q3 + t * d.q4));
    }
    0.0
}

/// Fast bicubic approximation.
fn filter_bicubic_fast(d: &InterpolationDetails, t: f64) -> f64 {
    let abs_t = f64_abs(t) / d.blur;
    let abs_t_sq = abs_t * abs_t;
    if abs_t < 1.0 {
        1.0 - 2.0 * abs_t_sq + abs_t_sq * abs_t
    } else if abs_t < 2.0 {
        4.0 - 8.0 * abs_t + 5.0 * abs_t_sq - abs_t_sq * abs_t
    } else {
        0.0
    }
}

/// Sinc function (without windowing).
fn filter_sinc(d: &InterpolationDetails, t: f64) -> f64 {
    let abs_t = f64_abs(t) / d.blur;
    if abs_t == 0.0 {
        1.0
    } else if abs_t > d.window {
        0.0
    } else {
        let a = abs_t * PI;
        f64_sin(a) / a
    }
}

/// Box/nearest neighbor filter.
fn filter_box(d: &InterpolationDetails, t: f64) -> f64 {
    let x = t / d.blur;
    if x >= -d.window && x < d.window {
        1.0
    } else {
        0.0
    }
}

/// Triangle/tent filter (linear interpolation).
fn filter_triangle(d: &InterpolationDetails, t: f64) -> f64 {
    let x = f64_abs(t) / d.blur;
    if x < 1.0 { 1.0 - x } else { 0.0 }
}

/// Lanczos-windowed sinc filter.
fn filter_sinc_windowed(d: &InterpolationDetails, t: f64) -> f64 {
    let x = t / d.blur;
    let abs_t = f64_abs(x);
    if abs_t == 0.0 {
        1.0
    } else if abs_t > d.window {
        0.0
    } else {
        d.window * f64_sin(PI * x / d.window) * f64_sin(x * PI) / (PI * PI * x * x)
    }
}

/// Jinc filter (circular sinc using Bessel function).
fn filter_jinc(d: &InterpolationDetails, t: f64) -> f64 {
    let x = f64_abs(t) / d.blur;
    if x == 0.0 {
        0.5 * PI
    } else {
        bessj1(PI * x) / x
    }
}

/// Ginseng filter (sinc windowed by jinc).
fn filter_ginseng(d: &InterpolationDetails, t: f64) -> f64 {
    let abs_t = f64_abs(t) / d.blur;
    let t_pi = abs_t * PI;
    if abs_t == 0.0 {
        1.0
    } else if abs_t > 3.0 {
        0.0
    } else {
        let jinc_input = 1.2196698912665046 * t_pi / d.window;
        let jinc_output = bessj1(jinc_input) / (jinc_input * 0.5);
        jinc_output * f64_sin(t_pi) / t_pi
    }
}

/// Bessel function of the first kind, order 1.
fn bessj1(x: f64) -> f64 {
    let ax = f64_abs(x);
    let ans = if ax < 8.0 {
        let y = x * x;
        let ans1 = x
            * (72362614232.0
                + y * (-7895059235.0
                    + y * (242396853.1
                        + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606))))));
        let ans2 = 144725228442.0
            + y * (2300535178.0
                + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y * 1.0))));
        ans1 / ans2
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 2.356194491;
        let ans1 = 1.0
            + y * (0.183105e-2
                + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * (-0.240337019e-6))));
        let ans2 = 0.04687499995
            + y * (-0.2002690873e-3
                + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)));
        f64_sqrt(core::f64::consts::FRAC_2_PI / ax) * (f64_cos(xx) * ans1 - z * f64_sin(xx) * ans2)
    };
    if x < 0.0 { -ans } else { ans }
}

// =============================================================================
// no_std math helpers
// =============================================================================

#[inline]
fn f64_abs(x: f64) -> f64 {
    if x < 0.0 { -x } else { x }
}

#[cfg(feature = "std")]
#[inline]
fn f64_sin(x: f64) -> f64 {
    x.sin()
}

#[cfg(not(feature = "std"))]
#[inline]
fn f64_sin(x: f64) -> f64 {
    libm::sin(x)
}

#[cfg(feature = "std")]
#[inline]
fn f64_cos(x: f64) -> f64 {
    x.cos()
}

#[cfg(not(feature = "std"))]
#[inline]
fn f64_cos(x: f64) -> f64 {
    libm::cos(x)
}

#[cfg(feature = "std")]
#[inline]
fn f64_sqrt(x: f64) -> f64 {
    x.sqrt()
}

#[cfg(not(feature = "std"))]
#[inline]
fn f64_sqrt(x: f64) -> f64 {
    libm::sqrt(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_filters_create() {
        for filter in Filter::all() {
            let details = InterpolationDetails::create(*filter);
            assert!(
                details.window > 0.0,
                "Filter {:?} has invalid window",
                filter
            );
            assert!(details.blur > 0.0, "Filter {:?} has invalid blur", filter);
        }
    }

    #[test]
    fn test_filter_at_zero() {
        for filter in Filter::all() {
            let details = InterpolationDetails::create(*filter);
            let value = details.filter(0.0);
            assert!(
                (0.5..=PI).contains(&value),
                "Filter {:?} at 0 = {}, expected near 1.0",
                filter,
                value
            );
        }
    }

    #[test]
    fn test_filter_symmetry() {
        for filter in Filter::all() {
            if *filter == Filter::Box {
                continue;
            }
            let details = InterpolationDetails::create(*filter);
            for x in [0.25, 0.75, 1.0, 1.5, 2.0] {
                let pos = details.filter(x);
                let neg = details.filter(-x);
                assert!(
                    (pos - neg).abs() < 1e-10,
                    "Filter {:?} not symmetric at x={}: {} vs {}",
                    filter,
                    x,
                    pos,
                    neg
                );
            }
        }
    }

    #[test]
    fn test_filter_outside_window() {
        for filter in Filter::all() {
            if *filter == Filter::Jinc {
                continue;
            }
            let details = InterpolationDetails::create(*filter);
            let x = details.effective_window() + 1.0;
            let value = details.filter(x);
            assert!(
                value.abs() < 0.01,
                "Filter {:?} at {} (outside window {}) = {}, expected ~0",
                filter,
                x,
                details.effective_window(),
                value
            );
        }
    }

    #[test]
    fn test_lanczos_known_values() {
        let details = InterpolationDetails::create(Filter::Lanczos);
        let v0 = details.filter(0.0);
        assert!((v0 - 1.0).abs() < 1e-10);

        let v1 = details.filter(1.0);
        assert!(v1.abs() < 0.1, "Lanczos at 1.0 = {}", v1);
    }

    #[test]
    fn test_box_filter() {
        let details = InterpolationDetails::create(Filter::Box);
        assert_eq!(details.filter(0.0), 1.0);
        assert_eq!(details.filter(0.4), 1.0);
        assert_eq!(details.filter(0.5), 0.0);
        assert_eq!(details.filter(1.0), 0.0);
    }

    #[test]
    fn test_triangle_filter() {
        let details = InterpolationDetails::create(Filter::Triangle);
        assert!((details.filter(0.0) - 1.0).abs() < 1e-10);
        assert!((details.filter(0.5) - 0.5).abs() < 1e-10);
        assert!((details.filter(1.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_negative_weight_percentage() {
        let lanczos = InterpolationDetails::create(Filter::Lanczos);
        let neg_pct = lanczos.calculate_percent_negative_weight();
        assert!(
            neg_pct > 0.0 && neg_pct < 0.5,
            "Lanczos negative weight % = {}",
            neg_pct
        );

        let box_filter = InterpolationDetails::create(Filter::Box);
        let box_neg = box_filter.calculate_percent_negative_weight();
        assert!(box_neg < 1e-10, "Box should have no negative weights");
    }
}
