//! Porter-Duff source-over compositing for blending resized images onto backgrounds.
//!
//! Compositing happens in premultiplied linear f32 space, between the vertical
//! filter output and unpremultiply. The [`Background`] trait is user-extensible.
//!
//! The formula for premultiplied source-over:
//! ```text
//! out[i] = fg[i] + bg[i] * (1.0 - fg_alpha)
//! ```

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use zenpixels::PixelDescriptor;

/// Error type for compositing configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompositeError {
    /// `RgbaPremul` input with compositing is mathematically incorrect:
    /// the pipeline would skip unpremultiply, so the composite formula
    /// would be applied to already-premultiplied data interpreted as straight.
    PremultipliedInput,
}

impl core::fmt::Display for CompositeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::PremultipliedInput => write!(
                f,
                "compositing with RgbaPremul input is not supported \
                 (premultiply + composite = mathematically incorrect)"
            ),
        }
    }
}

/// Background for source-over compositing.
///
/// Implementations describe what lies "behind" the resized foreground image.
/// The trait is not sealed — you can implement it for custom background sources.
///
/// All pixel data is premultiplied linear f32.
pub trait Background {
    /// Write premultiplied linear f32 background for output row `y` into `dst`.
    ///
    /// Only called when [`solid_pixel()`](Self::solid_pixel) returns `None`
    /// (non-solid backgrounds). `dst` has length `width * channels`.
    fn fill_row(&mut self, dst: &mut [f32], y: u32, channels: u8);

    /// If true, compositing is a no-op (skip entirely).
    ///
    /// Const-foldable for [`NoBackground`], allowing the compiler to eliminate
    /// all composite code when no background is set.
    fn is_transparent(&self) -> bool {
        false
    }

    /// If true, background alpha is always 1.0 — output alpha = 1.0 after blend.
    fn is_opaque(&self) -> bool {
        false
    }

    /// If `Some`, background is a single repeating pixel.
    ///
    /// Enables a fused solid-color fast path: one pass, no row buffer,
    /// pixel values live in registers.
    fn solid_pixel(&self) -> Option<&[f32; 4]> {
        None
    }
}

/// No background — compositing is a no-op.
///
/// This is the default. When used as a generic parameter, the compiler
/// eliminates all composite code paths (zero overhead).
pub struct NoBackground;

impl Background for NoBackground {
    fn fill_row(&mut self, _dst: &mut [f32], _y: u32, _channels: u8) {
        // Never called — is_transparent() returns true.
    }

    #[inline(always)]
    fn is_transparent(&self) -> bool {
        true
    }
}

/// Solid-color background (single premultiplied linear f32 pixel).
///
/// The fast path: no row buffer needed, pixel values fit in registers.
pub struct SolidBackground {
    pixel: [f32; 4],
    opaque: bool,
}

impl SolidBackground {
    /// Create from sRGB u8 color values.
    ///
    /// The color is converted to premultiplied linear f32.
    /// For non-alpha descriptors (Gray, Rgb), alpha is set to 1.0.
    pub fn from_srgb_u8(r: u8, g: u8, b: u8, a: u8, desc: PixelDescriptor) -> Self {
        let lr = linear_srgb::precise::srgb_to_linear(r as f32 / 255.0);
        let lg = linear_srgb::precise::srgb_to_linear(g as f32 / 255.0);
        let lb = linear_srgb::precise::srgb_to_linear(b as f32 / 255.0);
        let fa = if desc.has_alpha() {
            a as f32 / 255.0
        } else {
            1.0
        };
        // Premultiply
        Self {
            pixel: [lr * fa, lg * fa, lb * fa, fa],
            opaque: fa >= 1.0,
        }
    }

    /// Create from linear f32 color values (already in linear light).
    ///
    /// Values should be straight (non-premultiplied). They will be premultiplied internally.
    /// For non-alpha descriptors, alpha is forced to 1.0.
    pub fn from_linear(r: f32, g: f32, b: f32, a: f32, desc: PixelDescriptor) -> Self {
        let fa = if desc.has_alpha() { a } else { 1.0 };
        Self {
            pixel: [r * fa, g * fa, b * fa, fa],
            opaque: fa >= 1.0,
        }
    }

    /// Create from a [`CanvasColor`](crate::layout::CanvasColor).
    #[cfg(feature = "layout")]
    pub fn from_canvas_color(color: &crate::layout::CanvasColor, desc: PixelDescriptor) -> Self {
        match color {
            crate::layout::CanvasColor::Transparent => Self::transparent(desc),
            crate::layout::CanvasColor::Srgb { r, g, b, a } => {
                Self::from_srgb_u8(*r, *g, *b, *a, desc)
            }
            crate::layout::CanvasColor::Linear { r, g, b, a } => {
                Self::from_linear(*r, *g, *b, *a, desc)
            }
            // non_exhaustive fallback
            _ => Self::transparent(desc),
        }
    }

    /// Fully transparent background (equivalent to [`NoBackground`] but as a concrete type).
    pub fn transparent(_desc: PixelDescriptor) -> Self {
        Self {
            pixel: [0.0; 4],
            opaque: false,
        }
    }

    /// Opaque white background.
    pub fn white(desc: PixelDescriptor) -> Self {
        Self::from_srgb_u8(255, 255, 255, 255, desc)
    }

    /// Opaque black background.
    pub fn black(desc: PixelDescriptor) -> Self {
        Self::from_srgb_u8(0, 0, 0, 255, desc)
    }
}

impl Background for SolidBackground {
    fn fill_row(&mut self, _dst: &mut [f32], _y: u32, _channels: u8) {
        // Never called — solid_pixel() returns Some.
    }

    #[inline(always)]
    fn is_transparent(&self) -> bool {
        self.pixel[3] == 0.0
    }

    #[inline(always)]
    fn is_opaque(&self) -> bool {
        self.opaque
    }

    #[inline(always)]
    fn solid_pixel(&self) -> Option<&[f32; 4]> {
        Some(&self.pixel)
    }
}

/// Background from a borrowed premultiplied linear f32 buffer (row-major).
///
/// The buffer must contain `width * height * channels` elements.
/// `fill_row` copies the appropriate row into the destination.
pub struct SliceBackground<'a> {
    data: &'a [f32],
    row_len: usize,
}

impl<'a> SliceBackground<'a> {
    /// Create from a premultiplied linear f32 buffer.
    ///
    /// `data` must be row-major with `row_len` f32 elements per row
    /// (i.e., `width * channels`).
    pub fn new(data: &'a [f32], row_len: usize) -> Self {
        Self { data, row_len }
    }
}

impl Background for SliceBackground<'_> {
    fn fill_row(&mut self, dst: &mut [f32], y: u32, _channels: u8) {
        let start = y as usize * self.row_len;
        let end = start + self.row_len;
        let src = &self.data[start..end];
        let copy_len = dst.len().min(src.len());
        dst[..copy_len].copy_from_slice(&src[..copy_len]);
    }
}

/// Push-based streamed background with a ring buffer.
///
/// Caller pushes rows via [`push_row()`](Self::push_row) as they become available.
/// The ring buffer holds `capacity` rows; older rows are overwritten.
pub struct StreamedBackground {
    ring: Vec<Vec<f32>>,
    capacity: usize,
    write_idx: usize,
    rows_pushed: u32,
}

impl StreamedBackground {
    /// Create with the given ring buffer capacity (number of rows).
    pub fn new(capacity: usize, row_len: usize) -> Self {
        Self {
            ring: (0..capacity).map(|_| vec![0.0f32; row_len]).collect(),
            capacity,
            write_idx: 0,
            rows_pushed: 0,
        }
    }

    /// Push one row of premultiplied linear f32 background data.
    pub fn push_row(&mut self, row: &[f32]) {
        let slot = self.write_idx % self.capacity;
        let dest = &mut self.ring[slot];
        let copy_len = dest.len().min(row.len());
        dest[..copy_len].copy_from_slice(&row[..copy_len]);
        self.write_idx += 1;
        self.rows_pushed += 1;
    }

    /// Number of rows pushed so far.
    pub fn rows_pushed(&self) -> u32 {
        self.rows_pushed
    }
}

impl Background for StreamedBackground {
    fn fill_row(&mut self, dst: &mut [f32], y: u32, _channels: u8) {
        let slot = y as usize % self.capacity;
        let src = &self.ring[slot];
        let copy_len = dst.len().min(src.len());
        dst[..copy_len].copy_from_slice(&src[..copy_len]);
    }
}

// =============================================================================
// Composite kernels (scalar, autovectorizes reasonably)
// =============================================================================

/// Source-over composite: premultiplied foreground over premultiplied background.
///
/// For non-4-channel data (Gray, Rgb), this is a no-op since those formats
/// have no alpha and are therefore fully opaque.
#[inline]
pub(crate) fn composite_over_premul(src: &mut [f32], bg: &[f32], channels: u8) {
    if channels != 4 {
        return; // Gray/Rgb are opaque — no blending needed
    }
    for (s, b) in src.chunks_exact_mut(4).zip(bg.chunks_exact(4)) {
        let inv_a = 1.0 - s[3];
        s[0] += b[0] * inv_a;
        s[1] += b[1] * inv_a;
        s[2] += b[2] * inv_a;
        s[3] += b[3] * inv_a;
    }
}

/// Source-over composite with a solid premultiplied background pixel.
///
/// No row buffer needed — pixel values live in registers.
#[inline]
pub(crate) fn composite_over_solid_premul(src: &mut [f32], pixel: &[f32; 4]) {
    for s in src.chunks_exact_mut(4) {
        let inv_a = 1.0 - s[3];
        s[0] += pixel[0] * inv_a;
        s[1] += pixel[1] * inv_a;
        s[2] += pixel[2] * inv_a;
        s[3] += pixel[3] * inv_a;
    }
}

/// Source-over composite with a solid opaque premultiplied background pixel.
///
/// Output alpha is always 1.0 (opaque bg + any fg → opaque output).
#[inline]
pub(crate) fn composite_over_solid_opaque_premul(src: &mut [f32], pixel: &[f32; 4]) {
    for s in src.chunks_exact_mut(4) {
        let inv_a = 1.0 - s[3];
        s[0] += pixel[0] * inv_a;
        s[1] += pixel[1] * inv_a;
        s[2] += pixel[2] * inv_a;
        s[3] = 1.0; // bg opaque + any fg → opaque output
    }
}

/// Three-tier composite dispatch used by both [`Resizer`] and [`StreamingResize`].
///
/// Applies source-over compositing in-place on `src` (premultiplied linear f32).
/// `bg_row_buf` is only used for non-solid backgrounds.
#[inline]
pub(crate) fn composite_dispatch<B: Background>(
    src: &mut [f32],
    bg: &mut B,
    bg_row_buf: &mut [f32],
    out_y: u32,
    channels: u8,
) {
    if bg.is_transparent() {
        return;
    }
    if let Some(pixel) = bg.solid_pixel() {
        if bg.is_opaque() {
            composite_over_solid_opaque_premul(src, pixel);
        } else {
            composite_over_solid_premul(src, pixel);
        }
    } else {
        let row_len = src.len();
        bg.fill_row(&mut bg_row_buf[..row_len], out_y, channels);
        composite_over_premul(src, &bg_row_buf[..row_len], channels);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opaque_fg_ignores_bg() {
        // Fully opaque red fg over green bg → pure red
        let mut src = [1.0, 0.0, 0.0, 1.0];
        let bg = [0.0, 1.0, 0.0, 1.0];
        composite_over_premul(&mut src, &bg, 4);
        assert_eq!(src, [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn transparent_fg_passes_bg() {
        // Fully transparent fg → output = bg
        let mut src = [0.0, 0.0, 0.0, 0.0];
        let bg = [0.0, 0.5, 0.0, 1.0];
        composite_over_premul(&mut src, &bg, 4);
        assert_eq!(src, [0.0, 0.5, 0.0, 1.0]);
    }

    #[test]
    fn semi_transparent_blend() {
        // 50% red over opaque green
        // fg premul: [0.5, 0.0, 0.0, 0.5]
        // bg premul: [0.0, 1.0, 0.0, 1.0]
        // inv_a = 0.5
        // out = [0.5 + 0.0*0.5, 0.0 + 1.0*0.5, 0.0 + 0.0*0.5, 0.5 + 1.0*0.5]
        //      = [0.5, 0.5, 0.0, 1.0]
        let mut src = [0.5, 0.0, 0.0, 0.5];
        let bg = [0.0, 1.0, 0.0, 1.0];
        composite_over_premul(&mut src, &bg, 4);
        assert!((src[0] - 0.5).abs() < 1e-6);
        assert!((src[1] - 0.5).abs() < 1e-6);
        assert!((src[2] - 0.0).abs() < 1e-6);
        assert!((src[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn non_4ch_is_noop() {
        let mut src = [0.5, 0.3, 0.1, 0.5, 0.3, 0.1];
        let bg = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let original = src;
        composite_over_premul(&mut src, &bg, 3);
        assert_eq!(src, original);
    }

    #[test]
    fn solid_opaque_fast_path() {
        // Semi-transparent fg over opaque white → output alpha = 1.0
        let mut src = [0.3, 0.0, 0.0, 0.3];
        let pixel = [1.0, 1.0, 1.0, 1.0];
        composite_over_solid_opaque_premul(&mut src, &pixel);
        // inv_a = 0.7, out_r = 0.3 + 1.0*0.7 = 1.0
        assert!((src[0] - 1.0).abs() < 1e-6);
        assert!((src[1] - 0.7).abs() < 1e-6);
        assert!((src[2] - 0.7).abs() < 1e-6);
        assert_eq!(src[3], 1.0);
    }

    #[test]
    fn solid_non_opaque_fast_path() {
        let mut src = [0.0, 0.0, 0.0, 0.0];
        let pixel = [0.0, 0.25, 0.0, 0.5];
        composite_over_solid_premul(&mut src, &pixel);
        // Transparent fg → output = bg pixel
        assert_eq!(src, [0.0, 0.25, 0.0, 0.5]);
    }

    #[test]
    fn solid_background_white() {
        let bg = SolidBackground::white(PixelDescriptor::RGBA8_SRGB);
        assert!(bg.is_opaque());
        assert!(!bg.is_transparent());
        let pixel = bg.solid_pixel().unwrap();
        // White in linear: srgb_to_linear(255) = 1.0, premul by a=1.0 → 1.0
        assert!((pixel[0] - 1.0).abs() < 1e-6);
        assert!((pixel[1] - 1.0).abs() < 1e-6);
        assert!((pixel[2] - 1.0).abs() < 1e-6);
        assert_eq!(pixel[3], 1.0);
    }

    #[test]
    fn solid_background_transparent() {
        let bg = SolidBackground::transparent(PixelDescriptor::RGBA8_SRGB);
        assert!(bg.is_transparent());
        assert!(!bg.is_opaque());
        let pixel = bg.solid_pixel().unwrap();
        assert_eq!(*pixel, [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn solid_background_from_srgb() {
        let bg = SolidBackground::from_srgb_u8(128, 0, 0, 128, PixelDescriptor::RGBA8_SRGB);
        assert!(!bg.is_opaque());
        assert!(!bg.is_transparent());
        let pixel = bg.solid_pixel().unwrap();
        // alpha = 128/255 ≈ 0.502
        let expected_a = 128.0 / 255.0;
        assert!((pixel[3] - expected_a).abs() < 1e-3);
        // R = srgb_to_linear(128/255) * alpha
        let lr = linear_srgb::precise::srgb_to_linear(128.0 / 255.0);
        assert!((pixel[0] - lr * expected_a).abs() < 1e-3);
    }

    #[test]
    fn no_background_is_transparent() {
        let bg = NoBackground;
        assert!(bg.is_transparent());
    }

    #[test]
    fn slice_background_fill_row() {
        // 3 rows of 2 RGBA pixels each
        let row_len = 2 * 4;
        let data: Vec<f32> = (0..3 * row_len).map(|i| i as f32 * 0.01).collect();
        let mut bg = SliceBackground::new(&data, row_len);

        let mut dst = vec![0.0f32; row_len];
        bg.fill_row(&mut dst, 1, 4);
        // Row 1 starts at index 8
        assert_eq!(dst, &data[row_len..2 * row_len]);
    }

    #[test]
    fn streamed_background_push_and_fill() {
        let row_len = 8; // 2 RGBA pixels
        let mut bg = StreamedBackground::new(3, row_len);

        let row0: Vec<f32> = (0..row_len).map(|i| i as f32).collect();
        let row1: Vec<f32> = (0..row_len).map(|i| (i + 10) as f32).collect();
        let row2: Vec<f32> = (0..row_len).map(|i| (i + 20) as f32).collect();

        bg.push_row(&row0);
        bg.push_row(&row1);
        bg.push_row(&row2);
        assert_eq!(bg.rows_pushed(), 3);

        let mut dst = vec![0.0f32; row_len];

        bg.fill_row(&mut dst, 0, 4);
        assert_eq!(dst, row0);

        bg.fill_row(&mut dst, 1, 4);
        assert_eq!(dst, row1);

        bg.fill_row(&mut dst, 2, 4);
        assert_eq!(dst, row2);
    }

    #[test]
    fn composite_dispatch_no_background() {
        let mut bg = NoBackground;
        let mut src = [0.5, 0.3, 0.1, 0.7];
        let original = src;
        let mut buf = vec![0.0f32; 4];
        composite_dispatch(&mut src, &mut bg, &mut buf, 0, 4);
        assert_eq!(src, original);
    }

    #[test]
    fn composite_dispatch_solid_opaque() {
        let mut bg = SolidBackground::white(PixelDescriptor::RGBA8_SRGB);
        let mut src = [0.0, 0.0, 0.0, 0.0]; // transparent fg
        let mut buf = Vec::new(); // not used for solid
        composite_dispatch(&mut src, &mut bg, &mut buf, 0, 4);
        // Transparent fg over white → white
        assert!((src[0] - 1.0).abs() < 1e-6);
        assert!((src[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn composite_dispatch_slice_background() {
        let row_len = 4;
        let data = vec![0.0f32, 0.5, 0.0, 1.0]; // one RGBA pixel, opaque green
        let mut bg = SliceBackground::new(&data, row_len);
        let mut src = [0.0, 0.0, 0.0, 0.0]; // transparent fg
        let mut buf = vec![0.0f32; row_len];
        composite_dispatch(&mut src, &mut bg, &mut buf, 0, 4);
        // Transparent fg → output = bg
        assert_eq!(src, [0.0, 0.5, 0.0, 1.0]);
    }

    #[test]
    fn multi_pixel_composite() {
        // 3 pixels: opaque red, 50% green, transparent
        let mut src = [
            1.0, 0.0, 0.0, 1.0, // opaque red
            0.0, 0.25, 0.0, 0.5, // 50% green (premul)
            0.0, 0.0, 0.0, 0.0, // transparent
        ];
        let bg = [
            0.0, 0.0, 1.0, 1.0, // opaque blue
            0.0, 0.0, 1.0, 1.0, // opaque blue
            0.0, 0.0, 1.0, 1.0, // opaque blue
        ];
        composite_over_premul(&mut src, &bg, 4);

        // Pixel 0: opaque red, inv_a=0 → stays [1,0,0,1]
        assert_eq!(&src[0..4], &[1.0, 0.0, 0.0, 1.0]);

        // Pixel 1: 50% green over blue, inv_a=0.5
        // [0+0*0.5, 0.25+0*0.5, 0+1*0.5, 0.5+1*0.5] = [0, 0.25, 0.5, 1.0]
        assert!((src[4] - 0.0).abs() < 1e-6);
        assert!((src[5] - 0.25).abs() < 1e-6);
        assert!((src[6] - 0.5).abs() < 1e-6);
        assert!((src[7] - 1.0).abs() < 1e-6);

        // Pixel 2: transparent over blue → blue
        assert_eq!(&src[8..12], &[0.0, 0.0, 1.0, 1.0]);
    }
}
