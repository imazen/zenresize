//! Aspect-ratio constraint solver — compute output dimensions from a target box.
//!
//! Small self-contained helper so callers don't have to reimplement the
//! fit/within/cover arithmetic that every image-processing pipeline needs.
//! Used via [`ResizeConfigBuilder::fit`](crate::ResizeConfigBuilder::fit)
//! or the free [`fit_dims`] function.
//!
//! The algorithm is a port of `zenlayout`'s `fit_inside` / `crop_to_aspect` —
//! bit-identical output across a brute-force sweep (`tests/vs_zenlayout.rs`)
//! so callers that migrate from zenlayout don't see pixel-level differences.

/// How to fit an input size into a target bounding box.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FitMode {
    /// Aspect-preserving. Scale so the image fits entirely inside the bounds.
    /// May up- or down-scale. Output is `≤` bounds on both axes, `==` on one.
    /// Classic letterbox behavior when the source aspect differs from bounds.
    #[default]
    Fit,

    /// Like [`Fit`](Self::Fit), but never upscale past the input size.
    /// Output is `≤ min(bounds, input)` on both axes.
    Within,

    /// Aspect-preserving, fill bounds. The source is center-cropped to the
    /// target aspect ratio, then scaled to exact bounds — imageflow-style
    /// "fit=crop". For this mode [`fit_dims`] returns `(max_w, max_h)`;
    /// use [`fit_cover_source_crop`] to get the source-side rectangle that
    /// produces that result. [`ResizeConfigBuilder::fit`](crate::ResizeConfigBuilder::fit)
    /// wires both together automatically.
    Cover,

    /// Stretch to exactly the bounds, ignoring aspect ratio.
    Stretch,
}

/// Compute output dimensions from an input size, target box, and fit mode.
///
/// Returns `(0, 0)` if any input is zero. For [`FitMode::Cover`] returns
/// `(max_w, max_h)` — the final post-resize dimensions. Minimum result is
/// `1` on each axis when inputs are non-zero (so a 1-pixel source into a
/// 1000-pixel box doesn't collapse to 0).
///
/// Algorithm matches `zenlayout::fit_inside`, including the snap-to-target
/// rounding that prevents off-by-one drift when the computed dimension is
/// within rounding-loss distance of the target.
///
/// # Example
///
/// ```
/// use zenresize::{FitMode, fit_dims};
///
/// // A 1600×900 source into a 800×600 box:
/// assert_eq!(fit_dims(1600, 900, 800, 600, FitMode::Fit),     (800, 450));
/// assert_eq!(fit_dims(1600, 900, 800, 600, FitMode::Cover),   (800, 600));
/// assert_eq!(fit_dims(1600, 900, 800, 600, FitMode::Stretch), (800, 600));
///
/// // Within never upscales:
/// assert_eq!(fit_dims(400, 300, 800, 600, FitMode::Within),   (400, 300));
/// assert_eq!(fit_dims(400, 300, 800, 600, FitMode::Fit),      (800, 600));
/// ```
pub fn fit_dims(in_w: u32, in_h: u32, max_w: u32, max_h: u32, mode: FitMode) -> (u32, u32) {
    if in_w == 0 || in_h == 0 || max_w == 0 || max_h == 0 {
        return (0, 0);
    }

    match mode {
        FitMode::Stretch => (max_w, max_h),
        FitMode::Cover => (max_w, max_h),
        FitMode::Fit => fit_inside(in_w, in_h, max_w, max_h),
        FitMode::Within => {
            if in_w <= max_w && in_h <= max_h {
                (in_w, in_h)
            } else {
                fit_inside(in_w, in_h, max_w, max_h)
            }
        }
    }
}

/// Compute a center-crop of the source that, when resized to exactly
/// `(max_w, max_h)`, produces [`FitMode::Cover`] behavior — fills the target
/// box without overflow or stretching.
///
/// Returns `(crop_x, crop_y, crop_w, crop_h)`. Center-anchored. Pair with
/// `ResizeConfigBuilder::out_width/out_height = (max_w, max_h)` and
/// `.crop(x, y, w, h)` to get imageflow-style "fit=crop" behavior.
///
/// Returns `(0, 0, 0, 0)` if any input is zero. When the source and target
/// aspect ratios match exactly, returns `(0, 0, in_w, in_h)` (no crop).
pub fn fit_cover_source_crop(in_w: u32, in_h: u32, max_w: u32, max_h: u32) -> (u32, u32, u32, u32) {
    if in_w == 0 || in_h == 0 || max_w == 0 || max_h == 0 {
        return (0, 0, 0, 0);
    }

    // Exact aspect match via cross-multiplication: sw * th == sh * tw.
    // Avoids floating-point drift for ratios like 2:1 vs 200:100.
    let cross_s = in_w as u64 * max_h as u64;
    let cross_t = in_h as u64 * max_w as u64;
    if cross_s == cross_t {
        return (0, 0, in_w, in_h);
    }

    if cross_s > cross_t {
        // Source is wider than target — crop horizontally, keep full height.
        // new_w = in_h * (max_w / max_h), via proportional() to match zenlayout's snap.
        let new_w = proportional(max_w, max_h, in_h, false, in_w, in_h);
        if new_w >= in_w {
            return (0, 0, in_w, in_h);
        }
        let x = (in_w - new_w) / 2;
        (x, 0, new_w, in_h)
    } else {
        // Source is taller than target — crop vertically, keep full width.
        let new_h = proportional(max_w, max_h, in_w, true, in_w, in_h);
        if new_h >= in_h {
            return (0, 0, in_w, in_h);
        }
        let y = (in_h - new_h) / 2;
        (0, y, in_w, new_h)
    }
}

// ── Internal helpers (ported from zenlayout::constraint) ─────────────────────

/// Compute dimensions that fit inside a target box, preserving aspect ratio.
/// One dimension matches the target; the other is `≤` target.
///
/// Port of `zenlayout::fit_inside` — identical output across the brute-force
/// test sweep in `tests/vs_zenlayout.rs`.
fn fit_inside(sw: u32, sh: u32, tw: u32, th: u32) -> (u32, u32) {
    let ratio_w = tw as f64 / sw as f64;
    let ratio_h = th as f64 / sh as f64;
    if ratio_w <= ratio_h {
        // Width constrains — derive height.
        let h = proportional(sw, sh, tw, true, tw, th);
        (tw, h)
    } else {
        // Height constrains — derive width.
        let w = proportional(sw, sh, th, false, tw, th);
        (w, th)
    }
}

/// Compute a proportional value with snap-to-target rounding.
///
/// `ratio_w:ratio_h` defines the aspect. `basis` is the known value on one
/// axis (width if `basis_is_width`, else height); the function returns the
/// other axis. The snap logic prefers rounding to the source free axis or to
/// the target free axis whenever the exact computation is within
/// rounding-loss distance — kills off-by-one drift when aspects are nearly
/// (but not exactly) preserved.
///
/// Port of `zenlayout::proportional`. Zero result is clamped to 1.
fn proportional(
    ratio_w: u32,
    ratio_h: u32,
    basis: u32,
    basis_is_width: bool,
    target_w: u32,
    target_h: u32,
) -> u32 {
    let ratio = ratio_w as f64 / ratio_h as f64;

    let snap_amount = if basis_is_width {
        rounding_loss_height(ratio_w, ratio_h, target_h)
    } else {
        rounding_loss_width(ratio_w, ratio_h, target_w)
    };

    let snap_a = if basis_is_width { ratio_h } else { ratio_w };
    let snap_b = if basis_is_width { target_h } else { target_w };

    let float = if basis_is_width {
        basis as f64 / ratio
    } else {
        ratio * basis as f64
    };

    let delta_a = (float - snap_a as f64).abs();
    let delta_b = (float - snap_b as f64).abs();

    let v = if delta_a <= snap_amount && delta_a <= delta_b {
        snap_a
    } else if delta_b <= snap_amount {
        snap_b
    } else {
        float.round() as u32
    };

    if v == 0 { 1 } else { v }
}

/// Rounding loss when target width is used as basis. Port of
/// `zenlayout::rounding_loss_width`.
fn rounding_loss_width(ratio_w: u32, ratio_h: u32, target_width: u32) -> f64 {
    let ratio = ratio_w as f64 / ratio_h as f64;
    let target_x_to_self_x = target_width as f64 / ratio_w as f64;
    let recreate_y = ratio_h as f64 * target_x_to_self_x;
    let rounded_y = recreate_y.round();
    let recreate_x_from_rounded_y = rounded_y * ratio;
    (target_width as f64 - recreate_x_from_rounded_y).abs()
}

/// Rounding loss when target height is used as basis. Port of
/// `zenlayout::rounding_loss_height`.
fn rounding_loss_height(ratio_w: u32, ratio_h: u32, target_height: u32) -> f64 {
    let ratio = ratio_w as f64 / ratio_h as f64;
    let target_y_to_self_y = target_height as f64 / ratio_h as f64;
    let recreate_x = ratio_w as f64 * target_y_to_self_y;
    let rounded_x = recreate_x.round();
    let recreate_y_from_rounded_x = rounded_x / ratio;
    (target_height as f64 - recreate_y_from_rounded_x).abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_inputs() {
        assert_eq!(fit_dims(0, 100, 800, 600, FitMode::Fit), (0, 0));
        assert_eq!(fit_dims(100, 0, 800, 600, FitMode::Fit), (0, 0));
        assert_eq!(fit_dims(100, 100, 0, 600, FitMode::Fit), (0, 0));
        assert_eq!(fit_dims(100, 100, 800, 0, FitMode::Fit), (0, 0));
    }

    #[test]
    fn stretch_ignores_aspect() {
        assert_eq!(fit_dims(100, 100, 800, 600, FitMode::Stretch), (800, 600));
        assert_eq!(fit_dims(1, 10000, 500, 500, FitMode::Stretch), (500, 500));
    }

    #[test]
    fn fit_wider_source() {
        assert_eq!(fit_dims(1600, 900, 800, 600, FitMode::Fit), (800, 450));
    }

    #[test]
    fn fit_taller_source() {
        // 900x1600 → height constrains. zenlayout parity: 338x600.
        assert_eq!(fit_dims(900, 1600, 800, 600, FitMode::Fit), (338, 600));
    }

    #[test]
    fn fit_upscales() {
        assert_eq!(fit_dims(400, 300, 800, 600, FitMode::Fit), (800, 600));
    }

    #[test]
    fn within_refuses_upscale() {
        assert_eq!(fit_dims(400, 300, 800, 600, FitMode::Within), (400, 300));
        assert_eq!(fit_dims(1600, 900, 800, 600, FitMode::Within), (800, 450));
    }

    #[test]
    fn cover_returns_target_dims() {
        assert_eq!(fit_dims(1600, 900, 800, 600, FitMode::Cover), (800, 600));
        assert_eq!(fit_dims(900, 1600, 800, 600, FitMode::Cover), (800, 600));
    }

    #[test]
    fn never_collapses_to_zero() {
        let (w, h) = fit_dims(10_000, 1, 10, 10, FitMode::Fit);
        assert!(w >= 1 && h >= 1, "got {w}x{h}");
    }

    #[test]
    fn cover_source_crop_wider_source() {
        let (_x, y, w, h) = fit_cover_source_crop(1600, 900, 800, 600);
        assert_eq!((w, h), (1200, 900));
        assert_eq!(y, 0);
    }

    #[test]
    fn cover_source_crop_taller_source() {
        let (x, _y, w, h) = fit_cover_source_crop(900, 1600, 800, 600);
        assert_eq!((w, h), (900, 675));
        assert_eq!(x, 0);
    }

    #[test]
    fn cover_source_crop_exact_aspect() {
        assert_eq!(
            fit_cover_source_crop(1000, 500, 800, 400),
            (0, 0, 1000, 500)
        );
    }

    #[test]
    fn cover_source_crop_zero_inputs() {
        assert_eq!(fit_cover_source_crop(0, 100, 800, 600), (0, 0, 0, 0));
        assert_eq!(fit_cover_source_crop(100, 100, 0, 600), (0, 0, 0, 0));
    }

    #[test]
    fn defaults() {
        assert_eq!(FitMode::default(), FitMode::Fit);
    }

    #[test]
    fn fit_snap_to_target() {
        // 1000x500 (2:1) into 400x200 (2:1, exact aspect) → should match target.
        assert_eq!(fit_dims(1000, 500, 400, 200, FitMode::Fit), (400, 200));
        // 1001x500 (near 2:1) into 400x200 → would naively compute 400x199.8 → snap to 200.
        assert_eq!(fit_dims(1001, 500, 400, 200, FitMode::Fit), (400, 200));
    }
}
