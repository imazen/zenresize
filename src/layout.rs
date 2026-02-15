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
        Self::Srgb { r: 255, g: 255, b: 255, a: 255 }
    }

    /// Black, fully opaque.
    pub const fn black() -> Self {
        Self::Srgb { r: 0, g: 0, b: 0, a: 255 }
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
    Percent { x: f32, y: f32, width: f32, height: f32 },
}

impl SourceCrop {
    /// Create a pixel-based crop region.
    pub fn pixels(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self::Pixels(Rect { x, y, width, height })
    }

    /// Create a percentage-based crop region.
    ///
    /// `x` and `y` are the top-left origin (0.0–1.0), `width` and `height`
    /// are the region size as a fraction of source dimensions.
    pub fn percent(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self::Percent { x, y, width, height }
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
            Self::Percent { x, y, width, height } => {
                let px = (source_w as f64 * x.clamp(0.0, 1.0) as f64).round() as u32;
                let py = (source_h as f64 * y.clamp(0.0, 1.0) as f64).round() as u32;
                let pw = (source_w as f64 * width.clamp(0.0, 1.0) as f64).round() as u32;
                let ph = (source_h as f64 * height.clamp(0.0, 1.0) as f64).round() as u32;
                Rect { x: px, y: py, width: pw, height: ph }.clamp_to(source_w, source_h)
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
        Self { x, y, width, height }
    }

    /// Clamp this rect to fit within `(0, 0, max_w, max_h)`.
    /// Width and height are clamped to at least 1.
    pub fn clamp_to(self, max_w: u32, max_h: u32) -> Self {
        let x = self.x.min(max_w.saturating_sub(1));
        let y = self.y.min(max_h.saturating_sub(1));
        let w = self.width.min(max_w.saturating_sub(x)).max(1);
        let h = self.height.min(max_h.saturating_sub(y)).max(1);
        Self { x, y, width: w, height: h }
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

        // Step 3: Compute layout based on mode.
        use ConstraintMode::*;
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
        return Rect { x: 0, y: 0, width: sw, height: sh };
    }

    let target_ratio = tw as f64 / th as f64;
    let source_ratio = sw as f64 / sh as f64;

    if source_ratio > target_ratio {
        // Source is wider — crop width.
        let new_w = round_snap(sh as f64 * target_ratio, tw);
        if new_w >= sw {
            return Rect { x: 0, y: 0, width: sw, height: sh };
        }
        let x = gravity_offset_1d(sw - new_w, gravity, true);
        Rect { x, y: 0, width: new_w, height: sh }
    } else {
        // Source is taller — crop height.
        let new_h = round_snap(sw as f64 / target_ratio, th);
        if new_h >= sh {
            return Rect { x: 0, y: 0, width: sw, height: sh };
        }
        let y = gravity_offset_1d(sh - new_h, gravity, false);
        Rect { x: 0, y, width: sw, height: new_h }
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
            width: constraint_crop.width.min(uc.width.saturating_sub(constraint_crop.x)),
            height: constraint_crop.height.min(uc.height.saturating_sub(constraint_crop.y)),
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
        assert_eq!(r, Rect { x: 0, y: 0, width: 800, height: 600 });
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
        assert_eq!(l.source_crop, Some(Rect { x: 100, y: 100, width: 500, height: 500 }));
        assert_eq!(l.resize_to, (200, 200));
    }

    #[test]
    fn source_crop_percent() {
        let l = Constraint::new(ConstraintMode::Fit, 200, 200)
            .source_crop(SourceCrop::percent(0.25, 0.25, 0.5, 0.5))
            .compute(1000, 1000)
            .unwrap();
        assert_eq!(l.source_crop, Some(Rect { x: 250, y: 250, width: 500, height: 500 }));
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
        assert_eq!(r, Rect { x: 100, y: 50, width: 800, height: 400 });
    }

    #[test]
    fn margins_percent_asymmetric() {
        // CSS order: top, right, bottom, left
        let crop = SourceCrop::margins_percent(0.1, 0.2, 0.1, 0.2);
        let r = crop.resolve(1000, 500);
        assert_eq!(r, Rect { x: 200, y: 50, width: 600, height: 400 });
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
        let r = Rect { x: 900, y: 900, width: 500, height: 500 };
        let c = r.clamp_to(1000, 1000);
        assert_eq!(c, Rect { x: 900, y: 900, width: 100, height: 100 });
    }

    #[test]
    fn rect_clamp_zero_width() {
        let r = Rect { x: 1000, y: 0, width: 0, height: 100 };
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
}
