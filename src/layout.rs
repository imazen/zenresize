//! Layout constraint computation for resize operations.
//!
//! Computes dimensions, crop regions, and padding from a constraint mode,
//! source dimensions, and target dimensions. Pure geometry — no pixel
//! operations, no allocations, `no_std` compatible.
//!
//! # Example
//!
//! ```
//! use zenresize::layout::{Constraint, ConstraintMode};
//!
//! let layout = Constraint::new(ConstraintMode::FitCrop, 400, 300)
//!     .compute(1000, 500)
//!     .unwrap();
//!
//! // Source cropped to 4:3 aspect ratio, then resized to 400×300
//! assert_eq!(layout.resize_to, (400, 300));
//! assert!(layout.source_crop.is_some());
//! ```

/// How to fit a source image into target dimensions.
#[non_exhaustive]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ConstraintMode {
    /// Scale to exact target dimensions, distorting aspect ratio.
    Distort,

    /// Scale to fit within target, preserving aspect ratio.
    /// Upscales or downscales as needed.
    /// Output may be smaller than target on one axis.
    Fit,

    /// Like [`Fit`](Self::Fit), but never upscales.
    /// Images already smaller than target stay their original size.
    Within,

    /// Scale to fill target, crop overflow to exact target dimensions.
    /// Preserves aspect ratio. Upscales or downscales as needed.
    FitCrop,

    /// Like [`FitCrop`](Self::FitCrop), but never upscales.
    WithinCrop,

    /// Scale to fit within target, pad to exact target dimensions.
    /// Preserves aspect ratio. Upscales or downscales as needed.
    FitPad,

    /// Like [`FitPad`](Self::FitPad), but never upscales.
    /// Images smaller than target are centered on the canvas without scaling.
    WithinPad,

    /// Crop to target aspect ratio without any scaling.
    AspectCrop,
}

/// Where to position the image when cropping or padding.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub enum Gravity {
    /// Center on both axes.
    #[default]
    Center,
    /// Position by percentage. `(0.0, 0.0)` = top-left, `(1.0, 1.0)` = bottom-right.
    Percentage(f32, f32),
}

/// Canvas background color for pad modes.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum CanvasColor {
    /// Transparent black `[0, 0, 0, 0]`.
    #[default]
    Transparent,
    /// sRGB color with alpha.
    Srgb { r: u8, g: u8, b: u8, a: u8 },
}

impl CanvasColor {
    /// White, fully opaque.
    pub const fn white() -> Self {
        Self::Srgb {
            r: 255,
            g: 255,
            b: 255,
            a: 255,
        }
    }

    /// Black, fully opaque.
    pub const fn black() -> Self {
        Self::Srgb {
            r: 0,
            g: 0,
            b: 0,
            a: 255,
        }
    }
}

/// Region of source image to use before applying the constraint.
///
/// Either absolute pixel coordinates or percentages of source dimensions.
/// The caller resolves this — for JPEG, the decoder can skip IDCT outside
/// the crop region; for raw pixels, it's sub-buffer addressing.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum SourceCrop {
    /// Absolute pixel coordinates.
    Pixels(Rect),
    /// Percentage of source dimensions. All values in `0.0..=1.0`.
    ///
    /// `x=0.1, y=0.1, width=0.8, height=0.8` crops 10% from each edge.
    Percent {
        x: f32,
        y: f32,
        width: f32,
        height: f32,
    },
}

impl SourceCrop {
    /// Create a pixel-based crop region.
    pub fn pixels(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self::Pixels(Rect {
            x,
            y,
            width,
            height,
        })
    }

    /// Create a percentage-based crop region.
    ///
    /// `x` and `y` are the top-left origin (0.0–1.0), `width` and `height`
    /// are the region size as a fraction of source dimensions.
    pub fn percent(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self::Percent {
            x,
            y,
            width,
            height,
        }
    }

    /// Crop equal margins from all edges.
    ///
    /// `margin` is the fraction to remove from each side (0.0–0.5).
    /// `margin_percent(0.1)` removes 10% from each edge, keeping the center 80%.
    pub fn margin_percent(margin: f32) -> Self {
        Self::Percent {
            x: margin,
            y: margin,
            width: (1.0 - 2.0 * margin).max(0.0),
            height: (1.0 - 2.0 * margin).max(0.0),
        }
    }

    /// Crop specific margins from each edge (CSS order: top, right, bottom, left).
    ///
    /// All values are fractions of source dimensions (0.0–1.0).
    pub fn margins_percent(top: f32, right: f32, bottom: f32, left: f32) -> Self {
        Self::Percent {
            x: left,
            y: top,
            width: (1.0 - left - right).max(0.0),
            height: (1.0 - top - bottom).max(0.0),
        }
    }

    /// Resolve to pixel coordinates for a given source size.
    pub fn resolve(&self, source_w: u32, source_h: u32) -> Rect {
        match *self {
            Self::Pixels(r) => r.clamp_to(source_w, source_h),
            Self::Percent {
                x,
                y,
                width,
                height,
            } => {
                let px = (source_w as f64 * x.clamp(0.0, 1.0) as f64).round() as u32;
                let py = (source_h as f64 * y.clamp(0.0, 1.0) as f64).round() as u32;
                let pw = (source_w as f64 * width.clamp(0.0, 1.0) as f64).round() as u32;
                let ph = (source_h as f64 * height.clamp(0.0, 1.0) as f64).round() as u32;
                Rect {
                    x: px,
                    y: py,
                    width: pw,
                    height: ph,
                }
                .clamp_to(source_w, source_h)
            }
        }
    }
}

/// Axis-aligned rectangle in pixel coordinates.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl Rect {
    /// Create a new rect.
    pub const fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Clamp this rect to fit within `(0, 0, max_w, max_h)`.
    /// Width and height are clamped to at least 1.
    pub fn clamp_to(self, max_w: u32, max_h: u32) -> Self {
        let x = self.x.min(max_w.saturating_sub(1));
        let y = self.y.min(max_h.saturating_sub(1));
        let w = self.width.min(max_w.saturating_sub(x)).max(1);
        let h = self.height.min(max_h.saturating_sub(y)).max(1);
        Self {
            x,
            y,
            width: w,
            height: h,
        }
    }

    /// Whether this rect covers the full source (no actual crop).
    pub fn is_full(&self, source_w: u32, source_h: u32) -> bool {
        self.x == 0 && self.y == 0 && self.width == source_w && self.height == source_h
    }
}

/// Layout constraint specification.
///
/// Describes how to fit a source image into target dimensions,
/// with optional explicit cropping and canvas padding.
///
/// # Example
///
/// ```
/// use zenresize::layout::{Constraint, ConstraintMode, CanvasColor, Gravity};
///
/// let layout = Constraint::new(ConstraintMode::FitPad, 400, 300)
///     .gravity(Gravity::Center)
///     .canvas_color(CanvasColor::white())
///     .compute(1000, 500)
///     .unwrap();
///
/// assert_eq!(layout.canvas, (400, 300));
/// assert!(layout.needs_padding());
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Constraint {
    pub mode: ConstraintMode,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub gravity: Gravity,
    pub canvas_color: CanvasColor,
    pub source_crop: Option<SourceCrop>,
}

impl Constraint {
    /// Create a constraint with both target dimensions.
    pub fn new(mode: ConstraintMode, width: u32, height: u32) -> Self {
        Self {
            mode,
            width: Some(width),
            height: Some(height),
            gravity: Gravity::Center,
            canvas_color: CanvasColor::Transparent,
            source_crop: None,
        }
    }

    /// Constrain only width (height derived from source aspect ratio).
    pub fn width_only(mode: ConstraintMode, width: u32) -> Self {
        Self {
            mode,
            width: Some(width),
            height: None,
            gravity: Gravity::Center,
            canvas_color: CanvasColor::Transparent,
            source_crop: None,
        }
    }

    /// Constrain only height (width derived from source aspect ratio).
    pub fn height_only(mode: ConstraintMode, height: u32) -> Self {
        Self {
            mode,
            width: None,
            height: Some(height),
            gravity: Gravity::Center,
            canvas_color: CanvasColor::Transparent,
            source_crop: None,
        }
    }

    /// Set gravity for crop/pad positioning.
    pub fn gravity(mut self, gravity: Gravity) -> Self {
        self.gravity = gravity;
        self
    }

    /// Set canvas background color (for pad modes).
    pub fn canvas_color(mut self, color: CanvasColor) -> Self {
        self.canvas_color = color;
        self
    }

    /// Set explicit source crop (pixel or percentage).
    ///
    /// Applied before the constraint mode. For JPEG decode pipelines,
    /// the caller can pass the resolved crop to the decoder to skip
    /// IDCT outside the region.
    pub fn source_crop(mut self, crop: SourceCrop) -> Self {
        self.source_crop = Some(crop);
        self
    }

    /// Compute the layout for a source image of the given dimensions.
    pub fn compute(&self, source_w: u32, source_h: u32) -> Result<Layout, LayoutError> {
        if source_w == 0 || source_h == 0 {
            return Err(LayoutError::ZeroSourceDimension);
        }

        // Step 1: Apply explicit source crop.
        let (user_crop, sw, sh) = match &self.source_crop {
            Some(crop) => {
                let r = crop.resolve(source_w, source_h);
                (Some(r), r.width, r.height)
            }
            None => (None, source_w, source_h),
        };

        // Step 2: Resolve target dimensions (fill in missing axis from aspect ratio).
        let (tw, th) = self.resolve_target(sw, sh)?;

        // Step 2b: Single-axis shortcut.
        // When only one dimension is specified, the derived dimension already
        // preserves aspect ratio. Calling fit_inside or crop_to_aspect with
        // the rounded derived dimension can cause cascading rounding errors
        // (the wrong axis constrains). All modes degenerate: no crop/pad
        // is needed because source and target aspect ratios match (within
        // rounding).
        use ConstraintMode::*;
        let single_axis = self.width.is_none() || self.height.is_none();
        if single_axis {
            let no_upscale = matches!(self.mode, Within | WithinCrop | WithinPad);
            let (rw, rh) = if self.mode == AspectCrop {
                // AspectCrop = crop only, no scaling. Single-axis means
                // source and target aspect ratios match → no crop → use source.
                (sw, sh)
            } else if no_upscale && sw <= tw && sh <= th {
                (sw, sh)
            } else {
                (tw, th)
            };
            let (canvas, placement) = match self.mode {
                FitPad | WithinPad => {
                    let (px, py) = gravity_offset(tw, th, rw, rh, &self.gravity);
                    ((tw, th), (px, py))
                }
                _ => ((rw, rh), (0, 0)),
            };
            return Ok(Layout {
                source: (source_w, source_h),
                source_crop: user_crop,
                resize_to: (rw, rh),
                canvas,
                placement,
                canvas_color: self.canvas_color,
            }
            .normalize());
        }

        // Step 3: Compute layout based on mode.
        let layout = match self.mode {
            Distort => Layout {
                source: (source_w, source_h),
                source_crop: user_crop,
                resize_to: (tw, th),
                canvas: (tw, th),
                placement: (0, 0),
                canvas_color: self.canvas_color,
            },

            Fit => {
                let (rw, rh) = fit_inside(sw, sh, tw, th);
                Layout {
                    source: (source_w, source_h),
                    source_crop: user_crop,
                    resize_to: (rw, rh),
                    canvas: (rw, rh),
                    placement: (0, 0),
                    canvas_color: self.canvas_color,
                }
            }

            Within => {
                let (rw, rh) = if sw <= tw && sh <= th {
                    (sw, sh)
                } else {
                    fit_inside(sw, sh, tw, th)
                };
                Layout {
                    source: (source_w, source_h),
                    source_crop: user_crop,
                    resize_to: (rw, rh),
                    canvas: (rw, rh),
                    placement: (0, 0),
                    canvas_color: self.canvas_color,
                }
            }

            FitCrop => {
                let aspect_crop = crop_to_aspect(sw, sh, tw, th, &self.gravity);
                let combined = combine_crops(user_crop, aspect_crop);
                Layout {
                    source: (source_w, source_h),
                    source_crop: Some(combined),
                    resize_to: (tw, th),
                    canvas: (tw, th),
                    placement: (0, 0),
                    canvas_color: self.canvas_color,
                }
            }

            WithinCrop => {
                let aspect_crop = crop_to_aspect(sw, sh, tw, th, &self.gravity);
                let combined = combine_crops(user_crop, aspect_crop);
                let (rw, rh) = if combined.width <= tw && combined.height <= th {
                    (combined.width, combined.height)
                } else {
                    (tw, th)
                };
                Layout {
                    source: (source_w, source_h),
                    source_crop: Some(combined),
                    resize_to: (rw, rh),
                    canvas: (rw, rh),
                    placement: (0, 0),
                    canvas_color: self.canvas_color,
                }
            }

            FitPad => {
                let (rw, rh) = fit_inside(sw, sh, tw, th);
                let (px, py) = gravity_offset(tw, th, rw, rh, &self.gravity);
                Layout {
                    source: (source_w, source_h),
                    source_crop: user_crop,
                    resize_to: (rw, rh),
                    canvas: (tw, th),
                    placement: (px, py),
                    canvas_color: self.canvas_color,
                }
            }

            WithinPad => {
                let (rw, rh) = if sw <= tw && sh <= th {
                    (sw, sh)
                } else {
                    fit_inside(sw, sh, tw, th)
                };
                let (px, py) = gravity_offset(tw, th, rw, rh, &self.gravity);
                Layout {
                    source: (source_w, source_h),
                    source_crop: user_crop,
                    resize_to: (rw, rh),
                    canvas: (tw, th),
                    placement: (px, py),
                    canvas_color: self.canvas_color,
                }
            }

            AspectCrop => {
                let aspect_crop = crop_to_aspect(sw, sh, tw, th, &self.gravity);
                let combined = combine_crops(user_crop, aspect_crop);
                Layout {
                    source: (source_w, source_h),
                    source_crop: Some(combined),
                    resize_to: (combined.width, combined.height),
                    canvas: (combined.width, combined.height),
                    placement: (0, 0),
                    canvas_color: self.canvas_color,
                }
            }
        };

        // Normalize: if source_crop covers the full source, set to None.
        Ok(layout.normalize())
    }

    /// Resolve target dimensions, filling in the missing axis from source aspect ratio.
    ///
    /// When only one dimension is specified, crop/pad modes become equivalent
    /// to fit — the derived dimension matches the source aspect ratio exactly.
    fn resolve_target(&self, sw: u32, sh: u32) -> Result<(u32, u32), LayoutError> {
        match (self.width, self.height) {
            (Some(w), Some(h)) if w == 0 || h == 0 => Err(LayoutError::ZeroTargetDimension),
            (Some(w), Some(h)) => Ok((w, h)),
            (Some(0), None) => Err(LayoutError::ZeroTargetDimension),
            (Some(w), None) => {
                let h = (sh as f64 * w as f64 / sw as f64).round().max(1.0) as u32;
                Ok((w, h))
            }
            (None, Some(0)) => Err(LayoutError::ZeroTargetDimension),
            (None, Some(h)) => {
                let w = (sw as f64 * h as f64 / sh as f64).round().max(1.0) as u32;
                Ok((w, h))
            }
            (None, None) => Ok((sw, sh)),
        }
    }
}

/// Computed layout from applying a [`Constraint`] to source dimensions.
///
/// Contains everything needed to execute the resize:
/// - Which region of the source to read
/// - What dimensions to resize to
/// - Final canvas size and image placement (for padding)
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Layout {
    /// Original source dimensions.
    pub source: (u32, u32),
    /// Region of source to use. `None` = full source.
    pub source_crop: Option<Rect>,
    /// Dimensions to resize the (cropped) source to.
    pub resize_to: (u32, u32),
    /// Final output canvas dimensions (≥ `resize_to`).
    pub canvas: (u32, u32),
    /// Top-left offset where the resized image sits on the canvas.
    pub placement: (u32, u32),
    /// Canvas background color (for padding areas).
    pub canvas_color: CanvasColor,
}

impl Layout {
    /// Whether resampling is needed (dimensions change).
    pub fn needs_resize(&self) -> bool {
        let (sw, sh) = self.effective_source();
        self.resize_to != (sw, sh)
    }

    /// Whether padding is needed (canvas larger than resized image).
    pub fn needs_padding(&self) -> bool {
        self.canvas != self.resize_to
    }

    /// Whether a source crop is applied (excludes full-source no-ops).
    pub fn needs_crop(&self) -> bool {
        self.source_crop.is_some()
    }

    /// Effective source dimensions after crop.
    pub fn effective_source(&self) -> (u32, u32) {
        match &self.source_crop {
            Some(r) => (r.width, r.height),
            None => self.source,
        }
    }

    /// Normalize: clear source_crop if it covers the full source.
    fn normalize(mut self) -> Self {
        if let Some(r) = &self.source_crop
            && r.is_full(self.source.0, self.source.1)
        {
            self.source_crop = None;
        }
        self
    }
}

/// Layout computation error.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LayoutError {
    /// Source image has zero width or height.
    ZeroSourceDimension,
    /// Target width or height is zero.
    ZeroTargetDimension,
}

// ============================================================================
// Internal geometry
// ============================================================================

/// Compute dimensions that fit inside the target box, preserving aspect ratio.
/// One dimension matches the target; the other is ≤ target.
fn fit_inside(sw: u32, sh: u32, tw: u32, th: u32) -> (u32, u32) {
    let ratio_w = tw as f64 / sw as f64;
    let ratio_h = th as f64 / sh as f64;
    if ratio_w <= ratio_h {
        // Width constrains — compute height.
        // Snap to target height if rounding would land within 1 pixel,
        // preventing cascading rounding errors (e.g., 1200×400 → 100×33).
        let h = round_snap(sh as f64 * ratio_w, th);
        (tw, h)
    } else {
        // Height constrains — compute width.
        let w = round_snap(sw as f64 * ratio_h, tw);
        (w, th)
    }
}

/// Crop source to match target aspect ratio.
fn crop_to_aspect(sw: u32, sh: u32, tw: u32, th: u32, gravity: &Gravity) -> Rect {
    // Use cross-multiplication to avoid floating-point comparison for exact matches.
    let cross_s = sw as u64 * th as u64;
    let cross_t = sh as u64 * tw as u64;
    if cross_s == cross_t {
        return Rect {
            x: 0,
            y: 0,
            width: sw,
            height: sh,
        };
    }

    let target_ratio = tw as f64 / th as f64;
    let source_ratio = sw as f64 / sh as f64;

    if source_ratio > target_ratio {
        // Source is wider — crop width.
        let new_w = round_snap(sh as f64 * target_ratio, tw);
        if new_w >= sw {
            return Rect {
                x: 0,
                y: 0,
                width: sw,
                height: sh,
            };
        }
        let x = gravity_offset_1d(sw - new_w, gravity, true);
        Rect {
            x,
            y: 0,
            width: new_w,
            height: sh,
        }
    } else {
        // Source is taller — crop height.
        let new_h = round_snap(sw as f64 / target_ratio, th);
        if new_h >= sh {
            return Rect {
                x: 0,
                y: 0,
                width: sw,
                height: sh,
            };
        }
        let y = gravity_offset_1d(sh - new_h, gravity, false);
        Rect {
            x: 0,
            y,
            width: sw,
            height: new_h,
        }
    }
}

/// Combine an explicit user crop with a constraint-computed crop.
/// The constraint crop is in post-user-crop coordinates.
fn combine_crops(user_crop: Option<Rect>, constraint_crop: Rect) -> Rect {
    match user_crop {
        None => constraint_crop,
        Some(uc) => Rect {
            x: uc.x + constraint_crop.x,
            y: uc.y + constraint_crop.y,
            width: constraint_crop
                .width
                .min(uc.width.saturating_sub(constraint_crop.x)),
            height: constraint_crop
                .height
                .min(uc.height.saturating_sub(constraint_crop.y)),
        },
    }
}

/// Compute placement offset for a resized image within a canvas.
fn gravity_offset(cw: u32, ch: u32, iw: u32, ih: u32, gravity: &Gravity) -> (u32, u32) {
    let x = gravity_offset_1d(cw.saturating_sub(iw), gravity, true);
    let y = gravity_offset_1d(ch.saturating_sub(ih), gravity, false);
    (x, y)
}

fn gravity_offset_1d(space: u32, gravity: &Gravity, horizontal: bool) -> u32 {
    if space == 0 {
        return 0;
    }
    match gravity {
        Gravity::Center => space / 2,
        Gravity::Percentage(x, y) => {
            let pct = if horizontal { *x } else { *y };
            (space as f64 * pct.clamp(0.0, 1.0) as f64).round() as u32
        }
    }
}

/// Round a computed dimension, snapping to `snap_target` if within 1 pixel.
///
/// This prevents cascading rounding errors when computing one dimension
/// from the other via aspect ratio. For example, source 1200×400 at
/// target 100×33: `width_for(33) = 33 * 3.0 = 99`, but should be 100.
/// Snapping to the target value (100) when within 1 pixel fixes this.
fn round_snap(value: f64, snap_target: u32) -> u32 {
    if (value - snap_target as f64).abs() <= 1.0 {
        snap_target
    } else {
        (value.round() as u32).max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── fit_inside ──────────────────────────────────────────────────────

    #[test]
    fn fit_inside_landscape_into_landscape() {
        // 1000×500 (2:1) into 400×300 (4:3) → width constrains → 400×200
        assert_eq!(fit_inside(1000, 500, 400, 300), (400, 200));
    }

    #[test]
    fn fit_inside_portrait_into_landscape() {
        // 500×1000 (1:2) into 400×300 → height constrains → 150×300
        assert_eq!(fit_inside(500, 1000, 400, 300), (150, 300));
    }

    #[test]
    fn fit_inside_same_aspect() {
        // 1000×500 (2:1) into 400×200 (2:1) → exact fit
        assert_eq!(fit_inside(1000, 500, 400, 200), (400, 200));
    }

    #[test]
    fn fit_inside_snap_rounding() {
        // 1200×400 (3:1) into 100×33 — without snap, width_for(33) = 99.
        // fit_inside should produce (100, 33) because height constrains and
        // width snaps to target.
        assert_eq!(fit_inside(1200, 400, 100, 33), (100, 33));
    }

    #[test]
    fn fit_inside_square() {
        assert_eq!(fit_inside(1000, 500, 200, 200), (200, 100));
    }

    // ── crop_to_aspect ──────────────────────────────────────────────────

    #[test]
    fn crop_aspect_wider_source() {
        // 1000×500 (2:1) to 4:3 → crop width
        let r = crop_to_aspect(1000, 500, 400, 300, &Gravity::Center);
        // Expected width: 500 * 4/3 = 666.67 → 667
        assert_eq!(r.width, 667);
        assert_eq!(r.height, 500);
        // Centered: (1000-667)/2 = 166.5 → 167 (round)
        assert_eq!(r.x, 166);
        assert_eq!(r.y, 0);
    }

    #[test]
    fn crop_aspect_taller_source() {
        // 500×1000 (1:2) to 4:3 → crop height
        let r = crop_to_aspect(500, 1000, 400, 300, &Gravity::Center);
        assert_eq!(r.width, 500);
        // Expected height: 500 / (4/3) = 375
        assert_eq!(r.height, 375);
    }

    #[test]
    fn crop_aspect_same_ratio() {
        let r = crop_to_aspect(800, 600, 400, 300, &Gravity::Center);
        assert_eq!(
            r,
            Rect {
                x: 0,
                y: 0,
                width: 800,
                height: 600
            }
        );
    }

    #[test]
    fn crop_aspect_gravity_top_left() {
        let r = crop_to_aspect(1000, 500, 400, 300, &Gravity::Percentage(0.0, 0.0));
        assert_eq!(r.x, 0);
        assert_eq!(r.y, 0);
    }

    #[test]
    fn crop_aspect_gravity_bottom_right() {
        let r = crop_to_aspect(1000, 500, 400, 300, &Gravity::Percentage(1.0, 1.0));
        assert_eq!(r.x, 1000 - r.width);
        assert_eq!(r.y, 0); // Only width was cropped
    }

    // ── ConstraintMode::Distort ─────────────────────────────────────────

    #[test]
    fn distort_ignores_aspect() {
        let l = Constraint::new(ConstraintMode::Distort, 400, 300)
            .compute(1000, 500)
            .unwrap();
        assert_eq!(l.resize_to, (400, 300));
        assert_eq!(l.canvas, (400, 300));
        assert!(l.source_crop.is_none());
    }

    // ── ConstraintMode::Fit ─────────────────────────────────────────────

    #[test]
    fn fit_downscale() {
        let l = Constraint::new(ConstraintMode::Fit, 400, 300)
            .compute(1000, 500)
            .unwrap();
        assert_eq!(l.resize_to, (400, 200));
        assert_eq!(l.canvas, (400, 200));
        assert!(!l.needs_padding());
    }

    #[test]
    fn fit_upscale() {
        let l = Constraint::new(ConstraintMode::Fit, 400, 300)
            .compute(200, 100)
            .unwrap();
        assert_eq!(l.resize_to, (400, 200));
    }

    // ── ConstraintMode::Within ──────────────────────────────────────────

    #[test]
    fn within_no_upscale() {
        let l = Constraint::new(ConstraintMode::Within, 400, 300)
            .compute(200, 100)
            .unwrap();
        // Source fits within target → no resize.
        assert_eq!(l.resize_to, (200, 100));
        assert!(!l.needs_resize());
    }

    #[test]
    fn within_downscale() {
        let l = Constraint::new(ConstraintMode::Within, 400, 300)
            .compute(1000, 500)
            .unwrap();
        assert_eq!(l.resize_to, (400, 200));
    }

    // ── ConstraintMode::FitCrop ─────────────────────────────────────────

    #[test]
    fn fit_crop_exact_dimensions() {
        let l = Constraint::new(ConstraintMode::FitCrop, 400, 300)
            .compute(1000, 500)
            .unwrap();
        assert_eq!(l.resize_to, (400, 300));
        assert_eq!(l.canvas, (400, 300));
        assert!(l.source_crop.is_some());
        let crop = l.source_crop.unwrap();
        // Source cropped to 4:3 aspect ratio.
        assert_eq!(crop.height, 500);
        // crop.width ≈ 667
        assert!(crop.width > 650 && crop.width < 680);
    }

    #[test]
    fn fit_crop_same_aspect() {
        let l = Constraint::new(ConstraintMode::FitCrop, 400, 200)
            .compute(1000, 500)
            .unwrap();
        assert_eq!(l.resize_to, (400, 200));
        // Same aspect ratio → no crop needed (normalized to None).
        assert!(!l.needs_crop());
        assert!(l.source_crop.is_none());
    }

    // ── ConstraintMode::WithinCrop ──────────────────────────────────────

    #[test]
    fn within_crop_no_upscale() {
        let l = Constraint::new(ConstraintMode::WithinCrop, 400, 300)
            .compute(200, 100)
            .unwrap();
        // Source fits, aspect crop applies (200×100 is 2:1, target is 4:3).
        let crop = l.source_crop.unwrap();
        // Cropped to 4:3: width = 100 * 4/3 = 133
        assert_eq!(crop.height, 100);
        assert!(crop.width < 200);
        // No upscale → resize_to = crop dimensions.
        assert_eq!(l.resize_to, (crop.width, crop.height));
    }

    // ── ConstraintMode::FitPad ──────────────────────────────────────────

    #[test]
    fn fit_pad_adds_padding() {
        let l = Constraint::new(ConstraintMode::FitPad, 400, 300)
            .canvas_color(CanvasColor::white())
            .compute(1000, 500)
            .unwrap();
        assert_eq!(l.resize_to, (400, 200));
        assert_eq!(l.canvas, (400, 300));
        assert_eq!(l.placement, (0, 50)); // 50px top padding
        assert!(l.needs_padding());
    }

    #[test]
    fn fit_pad_no_padding_when_aspect_matches() {
        let l = Constraint::new(ConstraintMode::FitPad, 400, 200)
            .compute(1000, 500)
            .unwrap();
        assert_eq!(l.resize_to, (400, 200));
        assert_eq!(l.canvas, (400, 200));
        assert!(!l.needs_padding());
    }

    // ── ConstraintMode::WithinPad ───────────────────────────────────────

    #[test]
    fn within_pad_canvas_expand() {
        // Source smaller than target → no resize, just pad.
        let l = Constraint::new(ConstraintMode::WithinPad, 400, 300)
            .canvas_color(CanvasColor::white())
            .compute(200, 100)
            .unwrap();
        assert_eq!(l.resize_to, (200, 100));
        assert_eq!(l.canvas, (400, 300));
        assert_eq!(l.placement, (100, 100)); // Centered
        assert!(!l.needs_resize());
        assert!(l.needs_padding());
    }

    #[test]
    fn within_pad_downscale_and_pad() {
        let l = Constraint::new(ConstraintMode::WithinPad, 400, 300)
            .compute(1000, 500)
            .unwrap();
        assert_eq!(l.resize_to, (400, 200));
        assert_eq!(l.canvas, (400, 300));
        assert_eq!(l.placement, (0, 50));
    }

    // ── ConstraintMode::AspectCrop ──────────────────────────────────────

    #[test]
    fn aspect_crop_no_scaling() {
        let l = Constraint::new(ConstraintMode::AspectCrop, 400, 300)
            .compute(1000, 500)
            .unwrap();
        let crop = l.source_crop.unwrap();
        // Crop to 4:3 from a 2:1 source, no scaling.
        assert_eq!(l.resize_to, (crop.width, crop.height));
        assert!(!l.needs_resize());
    }

    // ── Source crop ─────────────────────────────────────────────────────

    #[test]
    fn source_crop_pixels() {
        let l = Constraint::new(ConstraintMode::Fit, 200, 200)
            .source_crop(SourceCrop::pixels(100, 100, 500, 500))
            .compute(1000, 1000)
            .unwrap();
        assert_eq!(
            l.source_crop,
            Some(Rect {
                x: 100,
                y: 100,
                width: 500,
                height: 500
            })
        );
        assert_eq!(l.resize_to, (200, 200));
    }

    #[test]
    fn source_crop_percent() {
        let l = Constraint::new(ConstraintMode::Fit, 200, 200)
            .source_crop(SourceCrop::percent(0.25, 0.25, 0.5, 0.5))
            .compute(1000, 1000)
            .unwrap();
        assert_eq!(
            l.source_crop,
            Some(Rect {
                x: 250,
                y: 250,
                width: 500,
                height: 500
            })
        );
    }

    #[test]
    fn source_crop_combined_with_fit_crop() {
        // User crops to center 50%, then FitCrop to 4:3.
        let l = Constraint::new(ConstraintMode::FitCrop, 400, 300)
            .source_crop(SourceCrop::percent(0.25, 0.25, 0.5, 0.5))
            .compute(1000, 1000)
            .unwrap();
        let crop = l.source_crop.unwrap();
        // User crop is (250, 250, 500, 500).
        // 500×500 → FitCrop to 4:3 → crop height to 375.
        assert_eq!(crop.width, 500);
        assert_eq!(crop.height, 375);
        // Origin offset: user's 250 + aspect crop offset.
        assert_eq!(crop.x, 250);
        assert!(crop.y > 250);
    }

    // ── Source crop margins ───────────────────────────────────────────

    #[test]
    fn margin_percent_symmetric() {
        let crop = SourceCrop::margin_percent(0.1);
        let r = crop.resolve(1000, 500);
        assert_eq!(
            r,
            Rect {
                x: 100,
                y: 50,
                width: 800,
                height: 400
            }
        );
    }

    #[test]
    fn margins_percent_asymmetric() {
        // CSS order: top, right, bottom, left
        let crop = SourceCrop::margins_percent(0.1, 0.2, 0.1, 0.2);
        let r = crop.resolve(1000, 500);
        assert_eq!(
            r,
            Rect {
                x: 200,
                y: 50,
                width: 600,
                height: 400
            }
        );
    }

    #[test]
    fn rect_is_full() {
        assert!(Rect::new(0, 0, 100, 100).is_full(100, 100));
        assert!(!Rect::new(1, 0, 99, 100).is_full(100, 100));
        assert!(!Rect::new(0, 0, 99, 100).is_full(100, 100));
    }

    // ── Width-only / height-only ────────────────────────────────────────

    #[test]
    fn width_only_computes_height() {
        let l = Constraint::width_only(ConstraintMode::Fit, 500)
            .compute(1000, 600)
            .unwrap();
        assert_eq!(l.resize_to, (500, 300));
    }

    #[test]
    fn height_only_computes_width() {
        let l = Constraint::height_only(ConstraintMode::Fit, 300)
            .compute(1000, 600)
            .unwrap();
        assert_eq!(l.resize_to, (500, 300));
    }

    #[test]
    fn width_only_fit_crop_no_crop() {
        // With only width specified, aspect ratios match → no crop.
        let l = Constraint::width_only(ConstraintMode::FitCrop, 500)
            .compute(1000, 600)
            .unwrap();
        assert!(!l.needs_crop());
        assert!(l.source_crop.is_none());
    }

    // ── Gravity ─────────────────────────────────────────────────────────

    #[test]
    fn gravity_top_left_pad() {
        let l = Constraint::new(ConstraintMode::FitPad, 400, 400)
            .gravity(Gravity::Percentage(0.0, 0.0))
            .compute(1000, 500)
            .unwrap();
        assert_eq!(l.placement, (0, 0));
    }

    #[test]
    fn gravity_bottom_right_pad() {
        let l = Constraint::new(ConstraintMode::FitPad, 400, 400)
            .gravity(Gravity::Percentage(1.0, 1.0))
            .compute(1000, 500)
            .unwrap();
        assert_eq!(l.resize_to, (400, 200));
        assert_eq!(l.placement, (0, 200));
    }

    // ── Error cases ─────────────────────────────────────────────────────

    #[test]
    fn zero_source_errors() {
        assert_eq!(
            Constraint::new(ConstraintMode::Fit, 100, 100).compute(0, 100),
            Err(LayoutError::ZeroSourceDimension)
        );
    }

    #[test]
    fn zero_target_errors() {
        assert_eq!(
            Constraint::new(ConstraintMode::Fit, 0, 100).compute(100, 100),
            Err(LayoutError::ZeroTargetDimension)
        );
    }

    // ── Rect clamping ───────────────────────────────────────────────────

    #[test]
    fn rect_clamp_oversized() {
        let r = Rect {
            x: 900,
            y: 900,
            width: 500,
            height: 500,
        };
        let c = r.clamp_to(1000, 1000);
        assert_eq!(
            c,
            Rect {
                x: 900,
                y: 900,
                width: 100,
                height: 100
            }
        );
    }

    #[test]
    fn rect_clamp_zero_width() {
        let r = Rect {
            x: 1000,
            y: 0,
            width: 0,
            height: 100,
        };
        let c = r.clamp_to(1000, 1000);
        assert!(c.width >= 1);
        assert!(c.x < 1000);
    }

    // ── Layout helpers ──────────────────────────────────────────────────

    #[test]
    fn needs_resize_false_for_identity() {
        let l = Constraint::new(ConstraintMode::Within, 1000, 1000)
            .compute(500, 300)
            .unwrap();
        assert!(!l.needs_resize());
    }

    #[test]
    fn needs_padding_true_for_pad() {
        let l = Constraint::new(ConstraintMode::FitPad, 400, 400)
            .compute(1000, 500)
            .unwrap();
        assert!(l.needs_padding());
    }

    // ════════════════════════════════════════════════════════════════════
    // Step 1: Rounding regression (1,185 cases from imageflow RIAPI)
    // ════════════════════════════════════════════════════════════════════

    #[rustfmt::skip]
    static SHRINK_WITHIN_TESTS: [(i32, i32, i32, i32); 1185] = [(1399,697, 280, -1),(1399,689, 200, -1),(1399,685, 193, -1),(1399,683, 212, -1),(1399,673, 396, -1),(1399,671, 270, -1),(1399,665, 365, -1),(1399,659, 190, -1),(1399,656, 193, -1),(1399,652, 162, -1),(1399,643, 260, -1),(1399,643, 260, -1),(1399,637, 291, -1),(1399,628, 362, -1),(1399,628, 362, -1),(1399,622, 343, -1),(1399,614, 270, -1),(1399,614, 270, -1),(1399,607, 363, -1),(1399,600, 232, -1),(1399,600, 232, -1),(1399,594, 305, -1),(1399,587, 342, -1),(1399,585, 391, -1),(1399,582, 256, -1),(1399,577, 217, -1),(1399,569, 193, -1),(1399,568, 383, -1),(1399,564, 222, -1),(1399,560, 346, -1),(1399,556, 39, -1),(1399,554, 125, -1),(1399,551, 179, -1),(1399,545, 163, -1),(1399,540, 307, -1),(1399,537, 353, -1),(1399,534, 93, -1),(1399,530, 260, -1),(1399,526, 254, -1),(1399,526, 254, -1),(1399,520, 265, -1),(1399,516, 61, -1),(1399,512, 291, -1),(1399,512, 97, -1),(1399,508, 263, -1),(1399,500, 270, -1),(1399,497, 190, -1),(1399,497, 114, -1),(1399,493, 271, -1),(1399,489, 216, -1),(1399,481, 397, -1),(1399,480, 290, -1),(1399,480, 290, -1),(1399,474, 152, -1),(1399,468, 139, -1),(1399,464, 300, -1),(1399,459, 32, -1),(1399,456, 158, -1),(1399,450, 300, -1),(1399,449, 148, -1),(1399,445, 11, -1),(1399,440, 310, -1),(1399,438, 107, -1),(1399,435, 320, -1),(1399,431, 297, -1),(1399,427, 172, -1),(1399,424, 325, -1),(1399,419, 202, -1),(1399,417, 52, -1),(1399,413, 188, -1),(1399,408, 108, -1),(1399,406, 143, -1),(1399,401, 232, -1),(1399,397, 259, -1),(1399,394, 158, -1),(1399,392, 298, -1),(1399,389, 196, -1),(1399,387, 338, -1),(1399,384, 388, -1),(1399,380, 289, -1),(1399,377, 154, -1),(1399,372, 220, -1),(1399,370, 259, -1),(1399,367, 223, -1),(1399,364, 98, -1),(1399,362, 114, -1),(1399,360, 68, -1),(1399,359, 189, -1),(1399,355, 333, -1),(1399,351, 277, -1),(1399,348, 203, -1),(1399,346, 374, -1),(1399,345, 221, -1),(1399,341, 240, -1),(1399,338, 387, -1),(1399,335, 332, -1),(1399,333, 355, -1),(1399,330, 248, -1),(1399,328, 386, -1),(1399,326, 324, -1),(1399,324, 326, -1),(1399,321, 146, -1),(1399,319, 182, -1),(1399,317, 267, -1),(1399,314, 274, -1),(1399,313, 257, -1),(1399,310, 264, -1),(1399,309, 206, -1),(1399,307, 180, -1),(1399,305, 383, -1),(1399,304, 237, -1),(1399,301, 244, -1),(1399,299, 386, -1),(1399,299, 255, -1),(1399,295, 377, -1),(1399,295, 230, -1),(1399,293, 74, -1),(1399,291, 387, -1),(1399,290, 41, -1),(1399,288, 85, -1),(1399,286, 203, -1),(1399,283, 393, -1),(1399,279, 183, -1),(1399,279, 178, -1),(1399,278, 234, -1),(1399,277, 250, -1),(1399,275, 379, -1),(1399,272, 162, -1),(1399,272, 54, -1),(1399,269, 169, -1),(1399,269, 91, -1),(1399,269, 13, -1),(1399,267, 317, -1),(1399,264, 310, -1),(1399,263, 125, -1),(1399,261, 335, -1),(1399,259, 397, -1),(1399,258, 122, -1),(1399,257, 313, -1),(1399,256, 194, -1),(1399,255, 299, -1),(1399,253, 235, -1),(1399,252, 297, -1),(1399,250, 277, -1),(1399,248, 330, -1),(1399,247, 337, -1),(1399,246, 327, -1),(1399,245, 197, -1),(1399,244, 43, -1),(1399,242, 211, -1),(1399,241, 357, -1),(1399,240, 341, -1),(1399,238, 385, -1),(1399,237, 304, -1),(1399,236, 329, -1),(1399,234, 278, -1),(1399,233, 9, -1),(1399,231, 324, -1),(1399,230, 295, -1),(1399,229, 168, -1),(1399,228, 316, -1),(1399,225, 115, -1),(1399,224, 153, -1),(1399,223, 367, -1),(1399,221, 345, -1),(1399,220, 124, -1),(1399,219, 214, -1),(1399,218, 369, -1),(1399,216, 204, -1),(1399,216, 68, -1),(1399,215, 244, -1),(1399,214, 219, -1),(1399,213, 266, -1),(1399,211, 242, -1),(1399,209, 251, -1),(1399,208, 380, -1),(1399,206, 309, -1),(1399,205, 58, -1),(1399,204, 120, -1),(1399,203, 286, -1),(1399,201, 261, -1),(1399,199, 355, -1),(1399,198, 378, -1),(1399,197, 316, -1),(1399,197, 245, -1),(1399,196, 389, -1),(1399,195, 391, -1),(1399,194, 256, -1),(1399,191, 260, -1),(1399,190, 335, -1),(1399,189, 396, -1),(1399,189, 359, -1),(1399,187, 288, -1),(1399,186, 267, -1),(1399,185, 397, -1),(1399,184, 19, -1),(1399,183, 172, -1),(1399,182, 319, -1),(1399,181, 228, -1),(1399,180, 307, -1),(1399,179, 254, -1),(1399,178, 279, -1),(1399,177, 328, -1),(1399,176, 155, -1),(1399,174, 205, -1),(1399,173, 376, -1),(1399,172, 305, -1),(1399,172, 61, -1),(1399,170, 144, -1),(1399,168, 229, -1),(1399,167, 289, -1),(1399,165, 284, -1),(1399,165, 89, -1),(1399,164, 354, -1),(1399,163, 339, -1),(1399,162, 367, -1),(1399,161, 265, -1),(1399,160, 153, -1),(1399,158, 394, -1),(1399,157, 147, -1),(1399,157, 49, -1),(1399,156, 139, -1),(1399,155, 176, -1),(1399,154, 377, -1),(1399,153, 224, -1),(1399,153, 96, -1),(1399,152, 69, -1),(1399,152, 23, -1),(1399,151, 88, -1),(1399,149, 399, -1),(1399,149, 61, -1),(1399,148, 345, -1),(1399,147, 157, -1),(1399,146, 321, -1),(1399,145, 82, -1),(1399,144, 306, -1),(1399,144, 170, -1),(1399,144, 34, -1),(1399,143, 44, -1),(1399,142, 399, -1),(1399,141, 253, -1),(1399,139, 317, -1),(1399,139, 156, -1),(1399,138, 370, -1),(1399,137, 291, -1),(1399,137, 97, -1),(1399,136, 324, -1),(1399,136, 180, -1),(1399,136, 36, -1),(1399,134, 214, -1),(1399,133, 163, -1),(1399,133, 142, -1),(1399,131, 315, -1),(1399,131, 283, -1),(1399,130, 382, -1),(1399,129, 244, -1),(1399,128, 388, -1),(1399,127, 369, -1),(1399,127, 358, -1),(1399,126, 272, -1),(1399,125, 263, -1),(1399,124, 220, -1),(1399,123, 381, -1),(1399,122, 258, -1),(1399,121, 237, -1),(1399,120, 204, -1),(1399,119, 335, -1),(1399,119, 288, -1),(1399,119, 241, -1),(1399,118, 326, -1),(1399,117, 269, -1),(1399,116, 211, -1),(1399,115, 371, -1),(1399,114, 362, -1),(1399,113, 229, -1),(1399,112, 331, -1),(1399,111, 397, -1),(1399,110, 337, -1),(1399,110, 248, -1),(1399,108, 395, -1),(1399,108, 136, -1),(1399,107, 268, -1),(1399,105, 393, -1),(1399,104, 343, -1),(1399,103, 292, -1),(1399,102, 336, -1),(1399,102, 144, -1),(1399,101, 367, -1),(1399,101, 90, -1),(1399,99, 332, -1),(1399,99, 219, -1),(1399,98, 364, -1),(1399,97, 310, -1),(1399,97, 137, -1),(1399,96, 357, -1),(1399,96, 255, -1),(1399,96, 153, -1),(1399,96, 51, -1),(1399,95, 346, -1),(1399,94, 305, -1),(1399,94, 186, -1),(1399,93, 203, -1),(1399,93, 188, -1),(1399,92, 190, -1),(1399,92, 114, -1),(1399,92, 38, -1),(1399,91, 392, -1),(1399,90, 272, -1),(1399,89, 275, -1),(1399,89, 165, -1),(1399,89, 55, -1),(1399,88, 310, -1),(1399,87, 217, -1),(1399,87, 201, -1),(1399,86, 366, -1),(1399,86, 122, -1),(1399,85, 288, -1),(1399,84, 358, -1),(1399,83, 396, -1),(1399,82, 179, -1),(1399,82, 162, -1),(1399,82, 145, -1),(1399,81, 354, -1),(1399,80, 341, -1),(1399,79, 363, -1),(1399,78, 278, -1),(1399,77, 227, -1),(1399,77, 118, -1),(1399,76, 322, -1),(1399,76, 230, -1),(1399,76, 138, -1),(1399,76, 46, -1),(1399,75, 345, -1),(1399,74, 293, -1),(1399,73, 297, -1),(1399,72, 340, -1),(1399,72, 204, -1),(1399,72, 68, -1),(1399,71, 325, -1),(1399,71, 266, -1),(1399,69, 375, -1),(1399,69, 152, -1),(1399,68, 360, -1),(1399,68, 216, -1),(1399,68, 72, -1),(1399,67, 261, -1),(1399,66, 392, -1),(1399,65, 398, -1),(1399,65, 355, -1),(1399,65, 312, -1),(1399,65, 269, -1),(1399,64, 295, -1),(1399,64, 142, -1),(1399,63, 344, -1),(1399,63, 233, -1),(1399,63, 122, -1),(1399,62, 327, -1),(1399,62, 282, -1),(1399,61, 172, -1),(1399,60, 338, -1),(1399,59, 391, -1),(1399,59, 320, -1),(1399,58, 253, -1),(1399,58, 229, -1),(1399,58, 205, -1),(1399,57, 233, -1),(1399,57, 184, -1),(1399,56, 387, -1),(1399,55, 394, -1),(1399,55, 267, -1),(1399,55, 89, -1),(1399,54, 272, -1),(1399,53, 277, -1),(1399,52, 390, -1),(1399,52, 121, -1),(1399,51, 288, -1),(1399,51, 96, -1),(1399,49, 385, -1),(1399,49, 328, -1),(1399,49, 271, -1),(1399,49, 214, -1),(1399,49, 157, -1),(1399,48, 335, -1),(1399,48, 306, -1),(1399,48, 102, -1),(1399,47, 372, -1),(1399,47, 253, -1),(1399,46, 380, -1),(1399,46, 228, -1),(1399,46, 76, -1),(1399,45, 295, -1),(1399,45, 264, -1),(1399,45, 233, -1),(1399,45, 202, -1),(1399,44, 302, -1),(1399,43, 374, -1),(1399,43, 309, -1),(1399,43, 244, -1),(1399,42, 383, -1),(1399,41, 392, -1),(1399,41, 358, -1),(1399,41, 324, -1),(1399,41, 290, -1),(1399,40, 367, -1),(1399,39, 376, -1),(1399,39, 269, -1),(1399,38, 276, -1),(1399,38, 92, -1),(1399,37, 397, -1),(1399,36, 369, -1),(1399,36, 136, -1),(1399,34, 390, -1),(1399,34, 349, -1),(1399,34, 308, -1),(1399,34, 267, -1),(1399,34, 226, -1),(1399,34, 185, -1),(1399,34, 144, -1),(1399,33, 360, -1),(1399,33, 233, -1),(1399,32, 371, -1),(1399,32, 284, -1),(1399,32, 153, -1),(1399,31, 383, -1),(1399,31, 338, -1),(1399,31, 293, -1),(1399,31, 248, -1),(1399,31, 203, -1),(1399,30, 396, -1),(1399,30, 303, -1),(1399,29, 361, -1),(1399,29, 313, -1),(1399,29, 265, -1),(1399,29, 217, -1),(1399,28, 374, -1),(1399,27, 388, -1),(1399,27, 233, -1),(1399,26, 349, -1),(1399,26, 242, -1),(1399,25, 363, -1),(1399,24, 378, -1),(1399,24, 320, -1),(1399,24, 262, -1),(1399,24, 204, -1),(1399,23, 395, -1),(1399,23, 152, -1),(1399,22, 349, -1),(1399,22, 286, -1),(1399,21, 366, -1),(1399,21, 233, -1),(1399,20, 384, -1),(1399,19, 331, -1),(1399,19, 184, -1),(1399,18, 349, -1),(1399,18, 272, -1),(1399,17, 370, -1),(1399,17, 288, -1),(1399,16, 393, -1),(1399,16, 306, -1),(1399,15, 326, -1),(1399,15, 233, -1),(1399,14, 349, -1),(1399,13, 376, -1),(1399,13, 269, -1),(1399,12, 291, -1),(1399,11, 317, -1),(1399,11, 190, -1),(1399,10, 349, -1),(1399,9, 388, -1),(1399,9, 233, -1),(1399,8, 262, -1),(1399,7, 299, -1),(1399,6, 349, -1),(1399,5, 399, -1),(1398,5, 399, -1),(1397,5, 399, -1),(1396,5, 399, -1),(1395,5, 399, -1),(1394,5, 399, -1),(1393,5, 399, -1),(1392,5, 399, -1),(1391,5, 399, -1),(1390,5, 399, -1),(1389,5, 399, -1),(1388,5, 399, -1),(1387,5, 399, -1),(1386,5, 399, -1),(1385,5, 399, -1),(1384,5, 399, -1),(1383,5, 399, -1),(1382,5, 399, -1),(1381,5, 399, -1),(1380,5, 399, -1),(1379,5, 399, -1),(1378,5, 399, -1),(1377,5, 399, -1),(1376,5, 399, -1),(1375,5, 399, -1),(1374,5, 399, -1),(1373,5, 399, -1),(1372,5, 399, -1),(1371,5, 399, -1),(1370,5, 399, -1),(1369,5, 399, -1),(1368,5, 399, -1),(1367,5, 399, -1),(1366,5, 399, -1),(1365,5, 399, -1),(1364,5, 399, -1),(1363,5, 399, -1),(1362,5, 399, -1),(1361,5, 399, -1),(1360,5, 399, -1),(1359,5, 399, -1),(1358,5, 399, -1),(1357,5, 399, -1),(1356,5, 399, -1),(1355,5, 399, -1),(1354,5, 399, -1),(1353,5, 399, -1),(1352,5, 399, -1),(1351,5, 399, -1),(1350,5, 399, -1),(1349,5, 399, -1),(1348,5, 399, -1),(1347,5, 399, -1),(1346,5, 399, -1),(1345,5, 399, -1),(1344,5, 399, -1),(1343,5, 399, -1),(1342,5, 399, -1),(1341,5, 399, -1),(1340,5, 399, -1),(1339,5, 399, -1),(1338,5, 399, -1),(1337,5, 399, -1),(1336,5, 399, -1),(1335,5, 399, -1),(1334,5, 399, -1),(1333,5, 399, -1),(1332,5, 399, -1),(1331,5, 399, -1),(697,1399, -1, 280),(689,1399, -1, 200),(683,1399, -1, 212),(674,1398, -1, 28),(667,1398, -1, 284),(660,1399, -1, 124),(654,1399, -1, 123),(647,1399, -1, 40),(641,1399, -1, 287),(635,1398, -1, 142),(629,1398, -1, 10),(623,1398, -1, 46),(618,1399, -1, 103),(612,1399, -1, 8),(606,1399, -1, 202),(600,1399, -1, 232),(594,1399, -1, 305),(588,1397, -1, 177),(582,1399, -1, 256),(577,1399, -1, 217),(571,1396, -1, 11),(567,1399, -1, 359),(563,1399, -1, 41),(557,1399, -1, 162),(552,1399, -1, 313),(547,1399, -1, 211),(541,1399, -1, 128),(536,1399, -1, 338),(532,1398, -1, 293),(527,1399, -1, 73),(521,1398, -1, 377),(517,1399, -1, 115),(513,1397, -1, 241),(509,1399, -1, 224),(505,1396, -1, 217),(502,1398, -1, 110),(498,1397, -1, 108),(495,1399, -1, 366),(490,1398, -1, 398),(485,1397, -1, 301),(482,1398, -1, 364),(479,1399, -1, 92),(474,1399, -1, 152),(470,1398, -1, 58),(467,1399, -1, 349),(463,1399, -1, 210),(459,1399, -1, 32),(456,1399, -1, 158),(452,1398, -1, 283),(449,1399, -1, 148),(445,1399, -1, 11),(442,1398, -1, 68),(439,1398, -1, 164),(436,1398, -1, 101),(433,1399, -1, 63),(430,1399, -1, 353),(426,1399, -1, 133),(423,1399, -1, 339),(419,1399, -1, 202),(417,1399, -1, 52),(413,1399, -1, 188),(409,1396, -1, 285),(406,1399, -1, 143),(402,1397, -1, 384),(400,1399, -1, 348),(397,1399, -1, 111),(393,1399, -1, 283),(391,1399, -1, 195),(388,1399, -1, 384),(385,1398, -1, 187),(383,1399, -1, 305),(380,1399, -1, 289),(377,1399, -1, 154),(374,1398, -1, 271),(372,1399, -1, 220),(370,1399, -1, 259),(368,1398, -1, 340),(366,1399, -1, 86),(364,1397, -1, 71),(361,1398, -1, 91),(359,1399, -1, 189),(356,1397, -1, 155),(355,1399, -1, 333),(351,1399, -1, 277),(349,1398, -1, 6),(347,1398, -1, 280),(345,1399, -1, 221),(341,1399, -1, 240),(338,1399, -1, 387),(335,1399, -1, 332),(333,1399, -1, 355),(330,1399, -1, 248),(328,1399, -1, 386),(326,1399, -1, 324),(324,1399, -1, 326),(322,1398, -1, 89),(320,1398, -1, 391),(318,1397, -1, 380),(317,1399, -1, 267),(315,1397, -1, 51),(313,1399, -1, 257),(311,1398, -1, 227),(310,1399, -1, 264),(309,1399, -1, 206),(307,1399, -1, 180),(305,1399, -1, 383),(304,1399, -1, 237),(301,1399, -1, 244),(299,1399, -1, 386),(299,1399, -1, 255),(296,1398, -1, 196),(294,1397, -1, 354),(292,1399, -1, 103),(291,1397, -1, 12),(289,1399, -1, 380),(288,1399, -1, 17),(286,1399, -1, 203),(284,1397, -1, 273),(283,1399, -1, 393),(281,1397, -1, 261),(280,1398, -1, 347),(278,1399, -1, 390),(277,1399, -1, 250),(275,1399, -1, 379),(273,1397, -1, 284),(272,1399, -1, 54),(270,1398, -1, 277),(269,1399, -1, 65),(267,1399, -1, 317),(265,1398, -1, 182),(263,1399, -1, 125),(262,1398, -1, 8),(261,1399, -1, 201),(259,1399, -1, 397),(258,1399, -1, 122),(257,1399, -1, 313),(256,1399, -1, 194),(255,1399, -1, 299),(253,1399, -1, 235),(252,1399, -1, 297),(251,1398, -1, 220),(250,1399, -1, 277),(248,1399, -1, 330),(247,1399, -1, 337),(246,1399, -1, 327),(245,1399, -1, 197),(244,1399, -1, 43),(242,1399, -1, 211),(241,1399, -1, 357),(240,1399, -1, 341),(238,1399, -1, 385),(238,1398, -1, 326),(237,1399, -1, 304),(236,1399, -1, 329),(234,1399, -1, 278),(233,1399, -1, 9),(232,1397, -1, 280),(230,1399, -1, 295),(229,1399, -1, 168),(228,1399, -1, 316),(226,1398, -1, 300),(225,1398, -1, 146),(224,1399, -1, 153),(223,1399, -1, 367),(221,1399, -1, 345),(220,1399, -1, 124),(219,1399, -1, 214),(218,1399, -1, 369),(216,1399, -1, 204),(216,1399, -1, 68),(215,1399, -1, 244),(214,1399, -1, 219),(213,1399, -1, 266),(212,1397, -1, 313),(211,1399, -1, 242),(209,1399, -1, 251),(208,1399, -1, 380),(207,1398, -1, 260),(206,1399, -1, 309),(205,1399, -1, 58),(204,1399, -1, 120),(203,1399, -1, 286),(202,1398, -1, 218),(201,1399, -1, 261),(200,1398, -1, 346),(199,1396, -1, 235),(198,1399, -1, 378),(197,1399, -1, 316),(197,1399, -1, 245),(196,1399, -1, 389),(195,1399, -1, 391),(194,1399, -1, 256),(193,1398, -1, 134),(191,1399, -1, 260),(191,1398, -1, 172),(189,1399, -1, 396),(189,1399, -1, 359),(188,1398, -1, 145),(187,1398, -1, 385),(186,1399, -1, 267),(185,1399, -1, 397),(184,1399, -1, 19),(183,1399, -1, 172),(182,1399, -1, 319),(181,1399, -1, 228),(180,1399, -1, 307),(179,1399, -1, 254),(178,1399, -1, 279),(177,1399, -1, 328),(176,1399, -1, 155),(175,1396, -1, 347),(174,1399, -1, 205),(173,1399, -1, 376),(173,1398, -1, 101),(172,1399, -1, 305),(172,1399, -1, 61),(171,1397, -1, 241),(170,1399, -1, 144),(169,1398, -1, 335),(168,1399, -1, 229),(167,1399, -1, 289),(166,1398, -1, 240),(166,1398, -1, 80),(165,1398, -1, 72),(164,1399, -1, 354),(163,1399, -1, 339),(162,1399, -1, 367),(162,1397, -1, 332),(161,1398, -1, 178),(160,1399, -1, 153),(159,1396, -1, 259),(158,1399, -1, 394),(157,1399, -1, 147),(157,1399, -1, 49),(156,1399, -1, 139),(155,1399, -1, 176),(154,1399, -1, 377),(153,1399, -1, 224),(153,1399, -1, 96),(152,1399, -1, 69),(152,1399, -1, 23),(151,1399, -1, 88),(150,1398, -1, 219),(149,1399, -1, 399),(149,1399, -1, 61),(149,1396, -1, 89),(148,1399, -1, 345),(148,1398, -1, 392),(147,1399, -1, 157),(146,1399, -1, 321),(145,1399, -1, 82),(144,1399, -1, 306),(144,1399, -1, 170),(144,1399, -1, 34),(143,1399, -1, 44),(142,1399, -1, 399),(141,1399, -1, 253),(140,1396, -1, 344),(139,1399, -1, 317),(139,1399, -1, 156),(138,1399, -1, 370),(138,1397, -1, 329),(137,1399, -1, 291),(137,1399, -1, 97),(136,1399, -1, 324),(136,1399, -1, 180),(136,1399, -1, 36),(135,1398, -1, 88),(135,1397, -1, 119),(134,1399, -1, 214),(134,1398, -1, 193),(133,1399, -1, 142),(132,1398, -1, 323),(131,1399, -1, 315),(131,1399, -1, 283),(130,1399, -1, 382),(130,1398, -1, 371),(129,1399, -1, 244),(128,1399, -1, 388),(127,1399, -1, 369),(127,1399, -1, 358),(126,1399, -1, 272),(125,1399, -1, 263),(124,1399, -1, 220),(123,1399, -1, 381),(122,1399, -1, 258),(121,1399, -1, 237),(120,1399, -1, 204),(119,1399, -1, 335),(119,1399, -1, 288),(119,1399, -1, 241),(118,1399, -1, 326),(118,1398, -1, 77),(117,1399, -1, 269),(117,1397, -1, 197),(116,1399, -1, 211),(116,1398, -1, 235),(115,1399, -1, 371),(115,1398, -1, 79),(114,1399, -1, 362),(113,1399, -1, 229),(113,1398, -1, 167),(112,1399, -1, 331),(111,1399, -1, 397),(110,1399, -1, 337),(110,1399, -1, 248),(109,1398, -1, 109),(108,1399, -1, 136),(107,1399, -1, 268),(106,1398, -1, 389),(106,1398, -1, 178),(106,1397, -1, 112),(105,1399, -1, 393),(105,1397, -1, 153),(104,1399, -1, 343),(103,1399, -1, 292),(102,1399, -1, 336),(102,1399, -1, 144),(101,1399, -1, 367),(101,1399, -1, 90),(100,1396, -1, 342),(99,1399, -1, 332),(99,1399, -1, 219),(98,1399, -1, 364),(97,1399, -1, 310),(97,1399, -1, 137),(96,1399, -1, 357),(96,1399, -1, 255),(96,1399, -1, 153),(96,1399, -1, 51),(95,1399, -1, 346),(95,1397, -1, 272),(95,1396, -1, 360),(94,1399, -1, 305),(94,1399, -1, 186),(94,1398, -1, 290),(93,1399, -1, 203),(93,1399, -1, 188),(93,1397, -1, 353),(92,1399, -1, 190),(92,1399, -1, 114),(92,1399, -1, 38),(91,1399, -1, 392),(91,1397, -1, 284),(90,1399, -1, 272),(89,1399, -1, 275),(89,1399, -1, 165),(89,1399, -1, 55),(88,1399, -1, 310),(87,1399, -1, 217),(87,1399, -1, 201),(86,1399, -1, 366),(86,1399, -1, 122),(85,1399, -1, 288),(85,1398, -1, 74),(84,1399, -1, 358),(84,1398, -1, 208),(83,1399, -1, 396),(83,1398, -1, 261),(83,1398, -1, 160),(82,1399, -1, 162),(82,1399, -1, 145),(81,1399, -1, 354),(81,1398, -1, 302),(80,1399, -1, 341),(79,1399, -1, 363),(78,1399, -1, 278),(77,1399, -1, 227),(77,1399, -1, 118),(77,1398, -1, 354),(77,1398, -1, 118),(76,1399, -1, 230),(76,1399, -1, 138),(76,1399, -1, 46),(75,1399, -1, 345),(75,1396, -1, 214),(75,1394, -1, 381),(75,1393, -1, 65),(74,1399, -1, 293),(74,1398, -1, 85),(73,1399, -1, 297),(73,1398, -1, 67),(72,1399, -1, 340),(72,1399, -1, 204),(72,1399, -1, 68),(71,1399, -1, 325),(71,1399, -1, 266),(70,1396, -1, 329),(70,1395, -1, 269),(70,1394, -1, 229),(69,1399, -1, 375),(69,1399, -1, 152),(69,1398, -1, 314),(68,1399, -1, 360),(68,1399, -1, 216),(68,1399, -1, 72),(67,1399, -1, 261),(66,1399, -1, 392),(66,1398, -1, 180),(65,1399, -1, 398),(65,1399, -1, 355),(65,1399, -1, 312),(65,1399, -1, 269),(64,1399, -1, 295),(64,1399, -1, 142),(64,1398, -1, 273),(64,1397, -1, 251),(63,1399, -1, 344),(63,1399, -1, 233),(63,1399, -1, 122),(63,1398, -1, 122),(63,1397, -1, 255),(62,1399, -1, 327),(62,1399, -1, 282),(62,1398, -1, 124),(61,1399, -1, 172),(60,1399, -1, 338),(60,1398, -1, 198),(59,1399, -1, 391),(59,1399, -1, 320),(59,1398, -1, 154),(58,1399, -1, 229),(58,1399, -1, 205),(57,1399, -1, 233),(57,1399, -1, 184),(57,1398, -1, 282),(56,1399, -1, 387),(56,1398, -1, 337),(55,1399, -1, 267),(55,1399, -1, 89),(54,1399, -1, 272),(53,1399, -1, 277),(53,1398, -1, 356),(53,1398, -1, 145),(53,1397, -1, 224),(53,1395, -1, 329),(52,1399, -1, 390),(52,1399, -1, 121),(52,1397, -1, 94),(51,1399, -1, 288),(51,1399, -1, 96),(50,1397, -1, 377),(50,1396, -1, 321),(50,1395, -1, 265),(49,1399, -1, 328),(49,1399, -1, 271),(49,1399, -1, 214),(49,1399, -1, 157),(48,1399, -1, 335),(48,1399, -1, 306),(48,1399, -1, 102),(47,1399, -1, 372),(47,1399, -1, 253),(46,1399, -1, 380),(46,1399, -1, 228),(46,1399, -1, 76),(45,1399, -1, 295),(45,1399, -1, 264),(45,1399, -1, 233),(45,1399, -1, 202),(45,1397, -1, 357),(44,1399, -1, 302),(43,1399, -1, 374),(43,1399, -1, 309),(43,1399, -1, 244),(42,1399, -1, 383),(41,1399, -1, 392),(41,1399, -1, 358),(41,1399, -1, 324),(41,1399, -1, 290),(40,1399, -1, 367),(40,1398, -1, 332),(39,1399, -1, 269),(38,1399, -1, 276),(38,1399, -1, 92),(37,1399, -1, 397),(36,1399, -1, 369),(36,1399, -1, 136),(35,1398, -1, 379),(35,1397, -1, 379),(35,1396, -1, 339),(34,1399, -1, 308),(34,1399, -1, 267),(34,1399, -1, 226),(34,1399, -1, 185),(34,1399, -1, 144),(33,1399, -1, 360),(33,1399, -1, 233),(33,1398, -1, 360),(32,1399, -1, 371),(32,1399, -1, 284),(32,1399, -1, 153),(31,1399, -1, 383),(31,1399, -1, 338),(31,1399, -1, 293),(31,1399, -1, 248),(31,1399, -1, 203),(31,1398, -1, 248),(30,1399, -1, 396),(30,1399, -1, 303),(30,1396, -1, 349),(29,1399, -1, 361),(29,1399, -1, 313),(29,1399, -1, 265),(29,1399, -1, 217),(28,1399, -1, 374),(28,1398, -1, 374),(28,1397, -1, 374),(28,1396, -1, 324),(28,1395, -1, 274),(27,1399, -1, 388),(27,1399, -1, 233),(27,1397, -1, 388),(26,1399, -1, 349),(26,1399, -1, 242),(26,1397, -1, 188),(25,1399, -1, 363),(25,1398, -1, 363),(25,1397, -1, 363),(25,1396, -1, 307),(25,1393, -1, 195),(24,1399, -1, 378),(24,1399, -1, 320),(24,1399, -1, 262),(24,1399, -1, 204),(23,1399, -1, 395),(23,1399, -1, 152),(22,1399, -1, 349),(22,1399, -1, 286),(21,1399, -1, 366),(21,1399, -1, 233),(20,1399, -1, 384),(20,1398, -1, 384),(20,1397, -1, 384),(20,1396, -1, 314),(19,1399, -1, 331),(19,1399, -1, 184),(18,1399, -1, 349),(18,1399, -1, 272),(17,1399, -1, 370),(17,1399, -1, 288),(16,1399, -1, 393),(16,1399, -1, 306),(15,1399, -1, 326),(15,1399, -1, 233),(14,1399, -1, 349),(14,1398, -1, 349),(14,1397, -1, 349),(14,1395, -1, 249),(13,1399, -1, 376),(13,1399, -1, 269),(12,1399, -1, 291),(12,1398, -1, 291),(12,1397, -1, 291),(11,1399, -1, 317),(11,1399, -1, 190),(11,1398, -1, 190),(11,1397, -1, 317),(11,1396, -1, 317),(11,1395, -1, 317),(10,1399, -1, 349),(10,1398, -1, 349),(10,1397, -1, 349),(9,1399, -1, 388),(9,1399, -1, 233),(8,1399, -1, 262),(8,1398, -1, 262),(7,1399, -1, 299),(7,1398, -1, 299),(7,1397, -1, 299),(7,1396, -1, 299),(6,1399, -1, 349),(6,1398, -1, 349),(6,1397, -1, 349),(5,1399, -1, 399),(5,1398, -1, 399),(5,1397, -1, 399),(5,1396, -1, 399),(5,1395, -1, 399),(5,1394, -1, 399),(5,1393, -1, 399),(5,1392, -1, 399),(5,1391, -1, 399),(5,1390, -1, 399),(5,1389, -1, 399),(5,1388, -1, 399),(5,1387, -1, 399),(5,1386, -1, 399),(5,1385, -1, 399),(5,1384, -1, 399),(5,1383, -1, 399),(5,1382, -1, 399),(5,1381, -1, 399),(5,1380, -1, 399),(5,1379, -1, 399),(5,1378, -1, 399),(5,1377, -1, 399),(5,1376, -1, 399),(5,1375, -1, 399),(5,1374, -1, 399),(5,1373, -1, 399),(5,1372, -1, 399),(5,1371, -1, 399),(5,1370, -1, 399),(5,1369, -1, 399),(5,1368, -1, 399),(5,1367, -1, 399),(5,1366, -1, 399),(5,1365, -1, 399),(5,1364, -1, 399),(5,1363, -1, 399),(5,1362, -1, 399),(5,1361, -1, 399),(5,1360, -1, 399),(5,1359, -1, 399),(5,1358, -1, 399),(5,1357, -1, 399),(5,1356, -1, 399),(5,1355, -1, 399),(5,1354, -1, 399),(5,1353, -1, 399),(5,1352, -1, 399),(5,1351, -1, 399),(5,1350, -1, 399),(5,1349, -1, 399),(5,1348, -1, 399),(5,1347, -1, 399),(5,1346, -1, 399),(5,1345, -1, 399),(5,1344, -1, 399),(5,1343, -1, 399),(5,1342, -1, 399),(5,1341, -1, 399),(5,1340, -1, 399),(5,1339, -1, 399),(5,1338, -1, 399),(5,1337, -1, 399),(5,1336, -1, 399),(5,1335, -1, 399),(5,1334, -1, 399),(5,1333, -1, 399),(5,1332, -1, 399),(5,1331, -1, 399)];

    #[test]
    fn rounding_regression_shrink_within() {
        let mut failures = Vec::new();
        for (i, &(ow, oh, tw, th)) in SHRINK_WITHIN_TESTS.iter().enumerate() {
            let ow = ow as u32;
            let oh = oh as u32;
            if tw > 0 {
                let tw = tw as u32;
                let layout = Constraint::width_only(ConstraintMode::Fit, tw)
                    .compute(ow, oh)
                    .unwrap();
                if layout.resize_to.0 != tw {
                    failures.push(format!(
                        "case {i}: ({ow}x{oh}, w={tw}) -> resize_to.0={}, expected {tw}",
                        layout.resize_to.0
                    ));
                }
                if layout.source_crop.is_some() {
                    failures.push(format!(
                        "case {i}: ({ow}x{oh}, w={tw}) -> unexpected source_crop"
                    ));
                }
            } else if th > 0 {
                let th = th as u32;
                let layout = Constraint::height_only(ConstraintMode::Fit, th)
                    .compute(ow, oh)
                    .unwrap();
                if layout.resize_to.1 != th {
                    failures.push(format!(
                        "case {i}: ({ow}x{oh}, h={th}) -> resize_to.1={}, expected {th}",
                        layout.resize_to.1
                    ));
                }
                if layout.source_crop.is_some() {
                    failures.push(format!(
                        "case {i}: ({ow}x{oh}, h={th}) -> unexpected source_crop"
                    ));
                }
            }
        }
        assert!(
            failures.is_empty(),
            "Rounding regression failures ({} of {}):\n{}",
            failures.len(),
            SHRINK_WITHIN_TESTS.len(),
            failures.join("\n")
        );
    }

    // ════════════════════════════════════════════════════════════════════
    // Step 2: Parametric invariant tests
    // ════════════════════════════════════════════════════════════════════

    const TARGETS: [(u32, u32); 11] = [
        (1, 1),
        (1, 3),
        (3, 1),
        (7, 3),
        (90, 45),
        (10, 10),
        (100, 33),
        (1621, 883),
        (971, 967),
        (17, 1871),
        (512, 512),
    ];

    fn gen_source_sizes(tw: u32, th: u32) -> Vec<(u32, u32)> {
        fn vary(v: u32) -> Vec<u32> {
            let mut vals = vec![v, v.saturating_add(1), v.saturating_sub(1).max(1)];
            vals.extend([v * 2, v * 3, v * 10]);
            vals.extend([(v / 2).max(1), (v / 3).max(1), (v / 10).max(1)]);
            vals.push(v.next_power_of_two());
            vals.extend([1, 2, 3, 5, 7, 16, 100, 1000]);
            vals.sort_unstable();
            vals.dedup();
            vals.retain(|&x| x > 0);
            vals
        }

        let w_vals = vary(tw);
        let h_vals = vary(th);
        let mut sizes: Vec<(u32, u32)> = Vec::new();
        for &w in &w_vals {
            for &h in &h_vals {
                sizes.push((w, h));
            }
        }

        // Aspect ratio inner/outer boxes
        let ratios: [(u64, u64); 6] = [(1, 1), (1, 3), (3, 1), (4, 3), (16, 9), (1200, 400)];
        for (rw, rh) in ratios {
            let inner_w = (tw as u64).min(th as u64 * rw / rh.max(1)).max(1) as u32;
            let inner_h = (th as u64).min(tw as u64 * rh / rw.max(1)).max(1) as u32;
            sizes.push((inner_w, inner_h));
            let outer_w = (tw as u64).max(th as u64 * rw / rh.max(1)).max(1) as u32;
            let outer_h = (th as u64).max(tw as u64 * rh / rw.max(1)).max(1) as u32;
            sizes.push((outer_w, outer_h));
        }

        sizes.sort_unstable();
        sizes.dedup();
        sizes
    }

    #[test]
    fn parametric_invariants() {
        let mut failures = Vec::new();
        let mut checked = 0u64;

        for &(tw, th) in &TARGETS {
            let sources = gen_source_sizes(tw, th);
            for &(sw, sh) in &sources {
                use ConstraintMode::*;
                let modes = [
                    Distort, Fit, Within, FitCrop, WithinCrop, FitPad, WithinPad, AspectCrop,
                ];
                for mode in modes {
                    let c = Constraint::new(mode, tw, th);
                    let layout = match c.compute(sw, sh) {
                        Ok(l) => l,
                        Err(e) => {
                            failures
                                .push(format!("{mode:?} ({sw}x{sh} -> {tw}x{th}): error {e:?}"));
                            continue;
                        }
                    };
                    let (rw, rh) = layout.resize_to;
                    let (cw, ch) = layout.canvas;
                    let (px, py) = layout.placement;
                    let tag = format!("{mode:?} ({sw}x{sh} -> {tw}x{th})");

                    match mode {
                        Distort => {
                            if (cw, ch) != (tw, th) {
                                failures.push(format!(
                                    "{tag}: canvas ({cw},{ch}) != target ({tw},{th})"
                                ));
                            }
                            if (rw, rh) != (tw, th) {
                                failures.push(format!(
                                    "{tag}: resize_to ({rw},{rh}) != target ({tw},{th})"
                                ));
                            }
                            if layout.source_crop.is_some() {
                                failures.push(format!("{tag}: unexpected source_crop"));
                            }
                        }
                        Fit => {
                            if rw > tw || rh > th {
                                failures.push(format!(
                                    "{tag}: resize_to ({rw},{rh}) exceeds target ({tw},{th})"
                                ));
                            }
                            if rw != tw && rh != th {
                                failures.push(format!(
                                    "{tag}: doesn't touch either edge: ({rw},{rh}) vs ({tw},{th})"
                                ));
                            }
                            if layout.source_crop.is_some() {
                                failures.push(format!("{tag}: unexpected source_crop"));
                            }
                            if (cw, ch) != (rw, rh) {
                                failures.push(format!(
                                    "{tag}: canvas ({cw},{ch}) != resize_to ({rw},{rh})"
                                ));
                            }
                        }
                        Within => {
                            if rw > tw || rh > th {
                                failures.push(format!(
                                    "{tag}: resize_to ({rw},{rh}) exceeds target ({tw},{th})"
                                ));
                            }
                            if sw <= tw && sh <= th {
                                if (rw, rh) != (sw, sh) {
                                    failures.push(format!(
                                        "{tag}: no-upscale: ({rw},{rh}) != source ({sw},{sh})"
                                    ));
                                }
                            } else if rw != tw && rh != th {
                                failures.push(format!(
                                    "{tag}: doesn't touch either edge: ({rw},{rh}) vs ({tw},{th})"
                                ));
                            }
                            if layout.source_crop.is_some() {
                                failures.push(format!("{tag}: unexpected source_crop"));
                            }
                            if (cw, ch) != (rw, rh) {
                                failures.push(format!(
                                    "{tag}: canvas ({cw},{ch}) != resize_to ({rw},{rh})"
                                ));
                            }
                        }
                        FitCrop => {
                            if (cw, ch) != (tw, th) {
                                failures.push(format!(
                                    "{tag}: canvas ({cw},{ch}) != target ({tw},{th})"
                                ));
                            }
                            if (rw, rh) != (tw, th) {
                                failures.push(format!(
                                    "{tag}: resize_to ({rw},{rh}) != target ({tw},{th})"
                                ));
                            }
                            if let Some(crop) = &layout.source_crop {
                                if crop.x > 0
                                    && crop.y > 0
                                    && crop.x + crop.width < sw
                                    && crop.y + crop.height < sh
                                {
                                    failures.push(format!("{tag}: crop on all 4 sides: {crop:?}"));
                                }
                            }
                        }
                        WithinCrop => {
                            if rw > tw || rh > th {
                                failures.push(format!(
                                    "{tag}: resize_to ({rw},{rh}) exceeds target ({tw},{th})"
                                ));
                            }
                            if let Some(crop) = &layout.source_crop {
                                if rw > crop.width || rh > crop.height {
                                    failures.push(format!(
                                        "{tag}: upscale: resize ({rw},{rh}) > crop ({},{})",
                                        crop.width, crop.height
                                    ));
                                }
                                if crop.x > 0
                                    && crop.y > 0
                                    && crop.x + crop.width < sw
                                    && crop.y + crop.height < sh
                                {
                                    failures.push(format!("{tag}: crop on all 4 sides: {crop:?}"));
                                }
                            }
                        }
                        FitPad => {
                            if (cw, ch) != (tw, th) {
                                failures.push(format!(
                                    "{tag}: canvas ({cw},{ch}) != target ({tw},{th})"
                                ));
                            }
                            if rw > tw || rh > th {
                                failures.push(format!(
                                    "{tag}: resize_to ({rw},{rh}) exceeds target ({tw},{th})"
                                ));
                            }
                            if layout.source_crop.is_some() {
                                failures.push(format!("{tag}: unexpected source_crop"));
                            }
                            if px > 0 && py > 0 && px + rw < tw && py + rh < th {
                                failures.push(format!("{tag}: padding on all 4 sides"));
                            }
                            if px + rw > tw || py + rh > th {
                                failures.push(format!(
                                    "{tag}: placement overflow: ({px},{py})+({rw},{rh})>({tw},{th})"
                                ));
                            }
                        }
                        WithinPad => {
                            if (cw, ch) != (tw, th) {
                                failures.push(format!(
                                    "{tag}: canvas ({cw},{ch}) != target ({tw},{th})"
                                ));
                            }
                            if sw <= tw && sh <= th {
                                if (rw, rh) != (sw, sh) {
                                    failures.push(format!(
                                        "{tag}: no-upscale: ({rw},{rh}) != source ({sw},{sh})"
                                    ));
                                }
                            }
                            if layout.source_crop.is_some() {
                                failures.push(format!("{tag}: unexpected source_crop"));
                            }
                            if px + rw > tw || py + rh > th {
                                failures.push(format!(
                                    "{tag}: placement overflow: ({px},{py})+({rw},{rh})>({tw},{th})"
                                ));
                            }
                        }
                        AspectCrop => {
                            if let Some(crop) = &layout.source_crop {
                                if (rw, rh) != (crop.width, crop.height) {
                                    failures.push(format!(
                                        "{tag}: resize_to ({rw},{rh}) != crop ({},{})",
                                        crop.width, crop.height
                                    ));
                                }
                            } else if (rw, rh) != (sw, sh) {
                                failures.push(format!(
                                    "{tag}: no crop but resize ({rw},{rh}) != source ({sw},{sh})"
                                ));
                            }
                            if (cw, ch) != (rw, rh) {
                                failures.push(format!(
                                    "{tag}: canvas ({cw},{ch}) != resize_to ({rw},{rh})"
                                ));
                            }
                        }
                    }
                    checked += 1;
                }
            }
        }

        assert!(
            failures.is_empty(),
            "Parametric invariant failures ({} of {checked} checked):\n{}",
            failures.len(),
            failures.join("\n")
        );
        // Ensure we actually checked a reasonable number of combinations
        assert!(
            checked > 15_000,
            "Only checked {checked} combinations, expected >15,000"
        );
    }

    // ════════════════════════════════════════════════════════════════════
    // Step 3: Specific rounding edge cases
    // ════════════════════════════════════════════════════════════════════

    #[test]
    fn rounding_1200x400_to_100x33_fit() {
        let l = Constraint::new(ConstraintMode::Fit, 100, 33)
            .compute(1200, 400)
            .unwrap();
        assert_eq!(l.resize_to, (100, 33));
        assert!(l.source_crop.is_none());
    }

    #[test]
    fn rounding_1200x400_to_100x33_fit_crop() {
        let l = Constraint::new(ConstraintMode::FitCrop, 100, 33)
            .compute(1200, 400)
            .unwrap();
        assert_eq!(l.resize_to, (100, 33));
        assert_eq!(l.canvas, (100, 33));
    }

    #[test]
    fn crop_aspect_638x423_to_200x133() {
        let l = Constraint::new(ConstraintMode::FitCrop, 200, 133)
            .compute(638, 423)
            .unwrap();
        assert_eq!(l.resize_to, (200, 133));
        assert_eq!(l.canvas, (200, 133));
    }

    #[test]
    fn fit_2x4_to_1x3() {
        let l = Constraint::new(ConstraintMode::Fit, 1, 3)
            .compute(2, 4)
            .unwrap();
        assert!(l.resize_to.0 <= 1);
        assert!(l.resize_to.1 <= 3);
        assert!(l.resize_to.0 == 1 || l.resize_to.1 == 3);
    }

    #[test]
    fn fit_crop_2x4_to_1x3() {
        let l = Constraint::new(ConstraintMode::FitCrop, 1, 3)
            .compute(2, 4)
            .unwrap();
        assert_eq!(l.resize_to, (1, 3));
        assert_eq!(l.canvas, (1, 3));
    }

    #[test]
    fn fit_1399x5_to_width_399() {
        let l = Constraint::width_only(ConstraintMode::Fit, 399)
            .compute(1399, 5)
            .unwrap();
        assert_eq!(l.resize_to.0, 399);
        assert!(l.source_crop.is_none());
    }

    #[test]
    fn fit_5x1399_to_height_399() {
        let l = Constraint::height_only(ConstraintMode::Fit, 399)
            .compute(5, 1399)
            .unwrap();
        assert_eq!(l.resize_to.1, 399);
        assert!(l.source_crop.is_none());
    }

    #[test]
    fn fit_1621x883_to_100x33() {
        let l = Constraint::new(ConstraintMode::Fit, 100, 33)
            .compute(1621, 883)
            .unwrap();
        assert!(l.resize_to.0 <= 100 && l.resize_to.1 <= 33);
        assert!(l.resize_to.0 == 100 || l.resize_to.1 == 33);
    }

    #[test]
    fn fit_971x967_to_512x512() {
        let l = Constraint::new(ConstraintMode::Fit, 512, 512)
            .compute(971, 967)
            .unwrap();
        assert!(l.resize_to.0 <= 512 && l.resize_to.1 <= 512);
        assert!(l.resize_to.0 == 512 || l.resize_to.1 == 512);
    }

    #[test]
    fn fit_1000x500_to_1x1() {
        let l = Constraint::new(ConstraintMode::Fit, 1, 1)
            .compute(1000, 500)
            .unwrap();
        assert_eq!(l.resize_to, (1, 1));
    }

    #[test]
    fn fit_crop_1000x500_to_1x1() {
        let l = Constraint::new(ConstraintMode::FitCrop, 1, 1)
            .compute(1000, 500)
            .unwrap();
        assert_eq!(l.resize_to, (1, 1));
        assert_eq!(l.canvas, (1, 1));
    }

    #[test]
    fn fit_pad_1000x500_to_1x1() {
        let l = Constraint::new(ConstraintMode::FitPad, 1, 1)
            .compute(1000, 500)
            .unwrap();
        assert_eq!(l.canvas, (1, 1));
        assert!(l.resize_to.0 <= 1 && l.resize_to.1 <= 1);
    }

    #[test]
    fn fit_100x100_to_100x100() {
        let l = Constraint::new(ConstraintMode::Fit, 100, 100)
            .compute(100, 100)
            .unwrap();
        assert_eq!(l.resize_to, (100, 100));
        assert!(!l.needs_resize());
    }

    #[test]
    fn fit_crop_100x100_to_100x100() {
        let l = Constraint::new(ConstraintMode::FitCrop, 100, 100)
            .compute(100, 100)
            .unwrap();
        assert_eq!(l.resize_to, (100, 100));
        assert!(l.source_crop.is_none());
    }

    #[test]
    fn within_modes_no_upscale_small_source() {
        let modes = [
            ConstraintMode::Within,
            ConstraintMode::WithinCrop,
            ConstraintMode::WithinPad,
        ];
        for mode in modes {
            let l = Constraint::new(mode, 400, 300).compute(50, 30).unwrap();
            let tag = format!("{mode:?}");
            assert!(
                l.resize_to.0 <= 50 && l.resize_to.1 <= 30,
                "{tag}: upscaled to {:?}",
                l.resize_to
            );
        }
    }

    #[test]
    fn width_only_1399x697_to_280() {
        let l = Constraint::width_only(ConstraintMode::Fit, 280)
            .compute(1399, 697)
            .unwrap();
        assert_eq!(l.resize_to.0, 280);
        assert!(l.source_crop.is_none());
    }

    #[test]
    fn height_only_697x1399_to_280() {
        let l = Constraint::height_only(ConstraintMode::Fit, 280)
            .compute(697, 1399)
            .unwrap();
        assert_eq!(l.resize_to.1, 280);
        assert!(l.source_crop.is_none());
    }

    // ════════════════════════════════════════════════════════════════════
    // Step 4: Source crop + mode composition edge cases
    // ════════════════════════════════════════════════════════════════════

    #[test]
    fn percent_crop_99_percent_rounds_to_full() {
        let l = Constraint::new(ConstraintMode::Fit, 50, 50)
            .source_crop(SourceCrop::percent(0.0, 0.0, 0.99, 0.99))
            .compute(100, 100)
            .unwrap();
        // 0.99 * 100 = 99, not full source
        let crop = l.source_crop.unwrap();
        assert_eq!(crop.width, 99);
        assert_eq!(crop.height, 99);
    }

    #[test]
    fn percent_crop_plus_fit_crop_extreme_aspect() {
        let l = Constraint::new(ConstraintMode::FitCrop, 100, 10)
            .source_crop(SourceCrop::percent(0.0, 0.0, 0.5, 0.5))
            .compute(1000, 1000)
            .unwrap();
        assert_eq!(l.resize_to, (100, 10));
        assert_eq!(l.canvas, (100, 10));
        let crop = l.source_crop.unwrap();
        assert!(crop.width <= 500);
        assert!(crop.height <= 500);
    }

    #[test]
    fn pixel_crop_to_1x1_with_fit() {
        let l = Constraint::new(ConstraintMode::Fit, 200, 200)
            .source_crop(SourceCrop::pixels(500, 500, 1, 1))
            .compute(1000, 1000)
            .unwrap();
        assert_eq!(l.resize_to, (200, 200));
    }

    #[test]
    fn pixel_crop_exceeds_source_clamped() {
        let l = Constraint::new(ConstraintMode::Fit, 50, 50)
            .source_crop(SourceCrop::pixels(900, 900, 500, 500))
            .compute(1000, 1000)
            .unwrap();
        let crop = l.source_crop.unwrap();
        assert!(crop.x + crop.width <= 1000);
        assert!(crop.y + crop.height <= 1000);
        assert!(crop.width >= 1 && crop.height >= 1);
    }

    #[test]
    fn percent_crop_exceeds_100_clamped() {
        let l = Constraint::new(ConstraintMode::Fit, 50, 50)
            .source_crop(SourceCrop::percent(0.0, 0.0, 1.5, 1.5))
            .compute(100, 100)
            .unwrap();
        // Percent clamped to 1.0 -> full source -> normalized to None
        assert!(l.source_crop.is_none());
    }

    #[test]
    fn percent_crop_zero_area() {
        let l = Constraint::new(ConstraintMode::Fit, 50, 50)
            .source_crop(SourceCrop::percent(0.5, 0.5, 0.0, 0.0))
            .compute(100, 100)
            .unwrap();
        // Zero-area clamps to min 1x1 via Rect::clamp_to
        let crop = l.source_crop.unwrap();
        assert!(crop.width >= 1 && crop.height >= 1);
    }

    // ════════════════════════════════════════════════════════════════════
    // Step 5: Gravity edge cases
    // ════════════════════════════════════════════════════════════════════

    #[test]
    fn gravity_center_odd_padding() {
        // WithinPad: no upscale, so 100×100 stays at 100×100 in a 103×103 canvas
        let l = Constraint::new(ConstraintMode::WithinPad, 103, 103)
            .compute(100, 100)
            .unwrap();
        assert_eq!(l.canvas, (103, 103));
        assert_eq!(l.resize_to, (100, 100));
        // Center: 3/2 = 1 (integer division)
        assert_eq!(l.placement, (1, 1));
    }

    #[test]
    fn gravity_percentage_clamp_negative() {
        let l = Constraint::new(ConstraintMode::FitPad, 400, 400)
            .gravity(Gravity::Percentage(-1.0, -1.0))
            .compute(1000, 500)
            .unwrap();
        assert_eq!(l.placement, (0, 0));
    }

    #[test]
    fn gravity_percentage_clamp_over_1() {
        let l = Constraint::new(ConstraintMode::FitPad, 400, 400)
            .gravity(Gravity::Percentage(2.0, 2.0))
            .compute(1000, 500)
            .unwrap();
        let max_x = 400 - l.resize_to.0;
        let max_y = 400 - l.resize_to.1;
        assert_eq!(l.placement, (max_x, max_y));
    }

    #[test]
    fn gravity_50_50_equals_center() {
        let l_pct = Constraint::new(ConstraintMode::FitPad, 400, 400)
            .gravity(Gravity::Percentage(0.5, 0.5))
            .compute(1000, 500)
            .unwrap();
        let l_center = Constraint::new(ConstraintMode::FitPad, 400, 400)
            .gravity(Gravity::Center)
            .compute(1000, 500)
            .unwrap();
        assert_eq!(l_pct.placement, l_center.placement);
    }
}
