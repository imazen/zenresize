//! Row-at-a-time streaming resize state machine.
//!
//! The streaming resizer uses vertical-first architecture with lazy pull-based
//! output production and natural backpressure:
//!
//! 1. `push_row()`: Linearize + premultiply the input row, cache it in the ring buffer
//! 2. `next_output_row()`: Lazily V-filter → H-filter one output row when enough cached rows exist
//! 3. Caller MUST drain all available output between `push_row` calls (ring buffer contract)
//!
//! V-first runs the H-filter only on output rows (not input rows), reducing H-filter
//! calls from `in_height` to `out_height` for downscaling.
//!
//! # Backpressure Contract
//!
//! The ring buffer has `max_taps + 2` slots. The caller MUST drain all
//! available output rows between `push_row` calls.
//!
//! ```ignore
//! // Single-row API:
//! for row in input_rows {
//!     resizer.push_row(row)?;
//!     while let Some(output) = resizer.next_output_row() {
//!         encoder.write_row(output);
//!     }
//! }
//! let remaining = resizer.finish();
//! for _ in 0..remaining {
//!     encoder.write_row(resizer.next_output_row().unwrap());
//! }
//!
//! // Batch API (for multi-row decode buffers):
//! let available = resizer.push_rows(&buf, stride, rows_read)?;
//! for _ in 0..available {
//!     encoder.write_row(resizer.next_output_row().unwrap());
//! }
//! ```

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::color;
use crate::composite::{self, Background, CompositeError, NoBackground};
use crate::filter::InterpolationDetails;
use crate::pixel::ResizeConfig;
use crate::simd;
use crate::transfer::{Bt709, Hlg, Pq, Srgb, TransferCurve};
use crate::weights::{F32WeightTable, I16WeightTable};
use whereat::{At, ResultAtExt, at};
use zenpixels::{AlphaMode, ChannelType, TransferFunction};

/// Estimate the maximum filter tap count for a given dimension and filter.
///
/// Used to decide whether the i16 integer pipeline would accumulate
/// unacceptable rounding error (falling back to f32 when taps are too many).
fn estimate_max_taps(in_size: u32, out_size: u32, filter: &InterpolationDetails) -> usize {
    let scale = out_size as f64 / in_size as f64;
    let downscale_factor = scale.min(1.0);
    // filter.window already incorporates blur
    let effective_window = filter.window / downscale_factor;
    // +2 margin for rounding and edge trimming variation
    (2.0 * effective_window).ceil() as usize + 2
}

/// Post-resize orientation transform.
///
/// Applied after resize completes. For streaming, this means the resizer
/// buffers output rows internally and applies the transform before serving.
///
/// Variants match EXIF orientation values 1–8 and the `Orientation` enum
/// from `zenlayout`, but this type is always available (no feature gate).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OrientOutput {
    /// No transform (EXIF 1).
    #[default]
    Identity,
    /// Horizontal flip — mirror each row (EXIF 2).
    /// Applied per-row; no buffering needed.
    FlipH,
    /// 180° rotation (EXIF 3).
    Rotate180,
    /// Vertical flip — reverse row order (EXIF 4).
    FlipV,
    /// Transpose — reflect over main diagonal (EXIF 5).
    Transpose,
    /// 90° clockwise rotation (EXIF 6).
    Rotate90,
    /// Reflect over anti-diagonal (EXIF 7).
    Transverse,
    /// 270° clockwise (90° counter-clockwise) rotation (EXIF 8).
    Rotate270,
}

impl OrientOutput {
    /// Whether this orientation swaps width and height.
    #[inline]
    pub const fn swaps_axes(self) -> bool {
        matches!(
            self,
            Self::Transpose | Self::Rotate90 | Self::Transverse | Self::Rotate270
        )
    }

    /// Whether this is a no-op.
    #[inline]
    pub const fn is_identity(self) -> bool {
        matches!(self, Self::Identity)
    }

    /// Whether this can be applied per-row without buffering the full image.
    #[inline]
    pub const fn is_row_local(self) -> bool {
        matches!(self, Self::Identity | Self::FlipH)
    }

    /// Compute output dimensions after applying this orientation.
    #[inline]
    pub const fn output_dimensions(self, w: u32, h: u32) -> (u32, u32) {
        if self.swaps_axes() { (h, w) } else { (w, h) }
    }

    /// Forward-map source pixel `(sx, sy)` to destination pixel `(dx, dy)`.
    ///
    /// `(w, h)` are the source dimensions (before orientation).
    #[inline]
    fn forward_map(self, sx: u32, sy: u32, w: u32, h: u32) -> (u32, u32) {
        match self {
            Self::Identity => (sx, sy),
            Self::FlipH => (w - 1 - sx, sy),
            Self::Rotate90 => (h - 1 - sy, sx),
            Self::Transpose => (sy, sx),
            Self::Rotate180 => (w - 1 - sx, h - 1 - sy),
            Self::FlipV => (sx, h - 1 - sy),
            Self::Rotate270 => (sy, w - 1 - sx),
            Self::Transverse => (h - 1 - sy, w - 1 - sx),
        }
    }
}

#[cfg(feature = "layout")]
impl From<crate::layout::Orientation> for OrientOutput {
    fn from(o: crate::layout::Orientation) -> Self {
        match o {
            crate::layout::Orientation::Identity => Self::Identity,
            crate::layout::Orientation::FlipH => Self::FlipH,
            crate::layout::Orientation::Rotate180 => Self::Rotate180,
            crate::layout::Orientation::FlipV => Self::FlipV,
            crate::layout::Orientation::Transpose => Self::Transpose,
            crate::layout::Orientation::Rotate90 => Self::Rotate90,
            crate::layout::Orientation::Transverse => Self::Transverse,
            crate::layout::Orientation::Rotate270 => Self::Rotate270,
            _ => Self::Identity, // non_exhaustive fallback
        }
    }
}

/// Build V-filter row references from a ring buffer cache.
///
/// Uses a 128-slot stack array for the common case (up to ~21× downscale with
/// Lanczos3). Falls back to heap allocation for extreme downscale ratios.
fn with_v_rows<T: Copy, R>(
    cache: &[Vec<T>],
    cache_size: usize,
    left: i32,
    tap_count: usize,
    in_height: u32,
    in_row_len: usize,
    f: impl FnOnce(&[&[T]]) -> R,
) -> R {
    const STACK_LIMIT: usize = 128;
    let clamp_max = in_height as i32 - 1;
    let empty: &[T] = &[];

    if tap_count <= STACK_LIMIT {
        let mut rows = [empty; STACK_LIMIT];
        for (t, slot) in rows.iter_mut().enumerate().take(tap_count) {
            let input_y = (left + t as i32).clamp(0, clamp_max) as usize;
            *slot = &cache[input_y % cache_size][..in_row_len];
        }
        f(&rows[..tap_count])
    } else {
        let rows: Vec<&[T]> = (0..tap_count)
            .map(|t| {
                let input_y = (left + t as i32).clamp(0, clamp_max) as usize;
                &cache[input_y % cache_size][..in_row_len]
            })
            .collect();
        f(&rows)
    }
}

/// Internal path selection for streaming resize.
///
/// Matches the fullframe Resizer's path 0/1/2 selection, adapted for
/// the V-first streaming architecture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StreamingPath {
    /// f32 path: sRGB u8 → linear f32 → filter → sRGB u8 (or f32 I/O)
    F32,
    /// sRGB gamma i16 H-first path: u8 → H-filter to i16 → ring buffer → V-filter → u8
    /// No linearization. 4ch only. Ring buffer is out_width wide.
    I16Srgb,
    /// Linear i12 i16 H-first path: sRGB u8 → i12 → H-filter to i16 → ring buffer → V-filter → sRGB u8
    /// 4ch, no premul (Rgbx or RgbaPremul). Ring buffer is out_width wide.
    I16Linear,
}

/// The internal working format of the streaming resizer.
///
/// Callers can use [`StreamingResize::working_format()`] to query this and
/// push data in the optimal format, avoiding redundant conversions. For
/// example, a JPEG decoder could fuse YCbCr→linear i12 conversion and
/// push via [`push_row_i16()`](StreamingResize::push_row_i16).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkingFormat {
    /// f32 linear-light (or f32 gamma for `TransferFunction::Linear`).
    F32,
    /// i16 in sRGB gamma space (u8 values zero-extended to i16).
    /// Only for 4ch, no linearization, u8 I/O.
    I16Srgb,
    /// i16 linear-light i12 (0–4095 range).
    /// Only for 4ch, linearized, no straight-alpha premul.
    I16Linear,
}

/// Errors from streaming resize push operations.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamingError {
    /// `push_row` was called after `finish()`.
    AlreadyFinished,
    /// Input row is too short for the configured width and channel count.
    InputTooShort,
    /// Ring buffer overflow: output rows were not drained between pushes.
    RingBufferOverflow,
}

impl core::fmt::Display for StreamingError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::AlreadyFinished => write!(f, "push_row called after finish()"),
            Self::InputTooShort => write!(f, "input row too short for configured dimensions"),
            Self::RingBufferOverflow => {
                write!(
                    f,
                    "ring buffer overflow: drain output rows before pushing more input"
                )
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for StreamingError {}

/// Streaming resize state machine.
///
/// Push input rows one at a time, pull output rows as they become available.
/// Uses lazy pull-based production with natural backpressure via the push/drain contract.
///
/// The pipeline is channel-order-agnostic: RGBA, BGRA, ARGB, BGRX all work
/// identically. The sRGB transfer function is the same for R, G, and B,
/// and the convolution kernels just operate on N floats per pixel. Pass
/// BGRA data straight through — no swizzle needed.
///
/// The generic parameter `B` controls background compositing. The default
/// [`NoBackground`] eliminates all composite code at compile time (zero overhead).
/// Use [`with_background()`](Self::with_background) to enable source-over compositing.
pub struct StreamingResize<B: Background = NoBackground> {
    config: ResizeConfig,
    /// Which internal pipeline path to use.
    path: StreamingPath,

    // === f32 path fields (empty when i16 path is active) ===
    h_weights: Option<F32WeightTable>,
    /// V-weights always allocated (used for scheduling even in i16 mode).
    v_weights: F32WeightTable,
    channels: usize,
    needs_premul: bool,
    alpha_is_last: bool,

    /// Ring buffer of input rows (f16 stored as u16, linearized + premultiplied).
    /// Each row is `in_width * channels + h_padding` wide (zero-padded for V-filter SIMD).
    /// Empty when i16 path is active.
    v_cache: Vec<Vec<u16>>,
    /// Which ring buffer slot to write the next input row into.
    cache_write_idx: usize,
    /// Total number of cache slots.
    cache_size: usize,

    /// Temporary buffer for format conversion (u8/u16 → f32).
    temp_input_f32: Vec<f32>,
    /// Temporary buffer for V-filter output (in_width-wide, zero-padded for H-filter SIMD).
    temp_v_output: Vec<f32>,
    /// Temporary buffer for H-filter output (out_width-wide).
    temp_output_f32: Vec<f32>,

    /// Reusable u8 conversion buffer (allocated once, used by `next_output_row`).
    output_buf_u8: Vec<u8>,
    /// Reusable u16 conversion buffer (allocated once, used by `next_output_row_u16`).
    output_buf_u16: Vec<u16>,

    // === i16 path fields (empty when f32 path is active) ===
    i16_h_weights: Option<I16WeightTable>,
    i16_v_weights: Option<I16WeightTable>,
    /// Ring buffer of u8 input rows (I16Srgb path).
    u8_v_cache: Vec<Vec<u8>>,
    /// Ring buffer of i16 input rows (I16Linear path).
    i16_v_cache: Vec<Vec<i16>>,
    /// V-filter output buffer for I16Srgb path (u8, with H-filter SIMD padding).
    temp_v_output_u8: Vec<u8>,
    /// V-filter output buffer for I16Linear path (i16, with H-filter SIMD padding).
    temp_v_output_i16: Vec<i16>,
    /// H-filter output buffer for I16Linear path (i16).
    temp_h_output_i16: Vec<i16>,
    /// Linearized i16 row buffer for I16Linear push.
    linearized_row_i16: Vec<i16>,

    /// Number of input rows received.
    input_rows_received: u32,
    /// Number of output rows produced.
    output_rows_produced: u32,
    /// End-of-input marker. Returns error on future pushes.
    finished: bool,

    /// Paired-row buffer for i16 paths: when consecutive output rows share the
    /// same V-filter window, both are V-filtered in one pass (L1-hot input data).
    /// The second row is buffered here and served on the next next_output_row() call.
    paired_row_buf: Vec<u8>,
    /// Whether `paired_row_buf` contains a ready output row.
    paired_row_ready: bool,

    /// Background for compositing.
    background: B,
    /// Row buffer for non-solid backgrounds. Empty for NoBackground and SolidBackground.
    composite_bg_row: Vec<f32>,
    /// Blend mode for compositing (default: SrcOver).
    blend_mode: composite::BlendMode,

    // === Crop state (zero-cost when not cropping) ===
    /// Current source row index (0-based, tracks full source image rows pushed).
    source_row_index: u32,
    /// Byte offset into source rows for horizontal crop (crop_x * channels).
    crop_x_offset: usize,
    /// First source row to use (inclusive).
    crop_y_start: u32,
    /// Last source row to use (exclusive: crop_y + crop_h).
    crop_y_end: u32,
    /// Full source width for push_row validation (differs from resize_in_w when cropping).
    source_in_width: u32,
    /// Whether cropping is active.
    has_crop: bool,

    // === Post-resize orientation ===
    /// Orientation transform applied after resize.
    orient: OrientOutput,
    /// Buffer for the full oriented output image (u8).
    /// Empty when `orient.is_row_local()`.
    /// Layout: `oriented_w * oriented_h * channels` bytes.
    orient_buf: Vec<u8>,
    /// Number of oriented rows already emitted.
    orient_rows_emitted: u32,
    /// Width after orientation (may differ from resize output if axes swap).
    oriented_w: u32,
    /// Height after orientation.
    oriented_h: u32,
    /// Whether all resize output has been captured into orient_buf.
    orient_ready: bool,
    /// Resize output rows captured so far (for non-row-local orientations).
    orient_captured: u32,

    // === Padding state (zero-cost when not padding) ===
    /// Padding pixels above the content.
    pad_top: u32,
    /// Padding pixels below the content.
    pad_bottom: u32,
    /// Number of elements for left padding (pad_left * channels).
    pad_left_elements: usize,
    /// Pre-filled full-width padding row (u8). Empty when no padding.
    pad_full_row_u8: Vec<u8>,
    /// Padded output buffer for content rows (u8, pre-filled with pad color).
    padded_output_buf_u8: Vec<u8>,
    /// Pre-filled full-width padding row (f32). Empty when no padding or u8-only output.
    pad_full_row_f32: Vec<f32>,
    /// Padded output buffer for content rows (f32).
    padded_output_buf_f32: Vec<f32>,
    /// Pre-filled full-width padding row (u16). Empty when no padding or u8-only output.
    pad_full_row_u16: Vec<u16>,
    /// Padded output buffer for content rows (u16).
    padded_output_buf_u16: Vec<u16>,
    /// Top padding rows emitted so far.
    pad_top_emitted: u32,
    /// Bottom padding rows emitted so far.
    pad_bottom_emitted: u32,
    /// Whether padding is active.
    has_padding: bool,
}

impl StreamingResize<NoBackground> {
    /// Create a new streaming resizer (no background compositing).
    pub fn new(config: &ResizeConfig) -> Self {
        Self::new_inner(config, NoBackground, false, 0)
    }

    /// Create a streaming resizer with a batch hint for [`push_rows()`](Self::push_rows).
    ///
    /// The `batch_hint` controls the ring buffer size: `max_taps + batch_hint + 2`.
    /// Set this to the number of rows you'll push per batch (e.g., 8 for zenjpeg).
    /// Without a hint (or 0), the buffer is sized for one-at-a-time push/drain.
    pub fn with_batch_hint(config: &ResizeConfig, batch_hint: u32) -> Self {
        Self::new_inner(config, NoBackground, false, batch_hint)
    }
}

impl<B: Background> StreamingResize<B> {
    /// Set a post-resize orientation transform.
    ///
    /// After all input rows have been pushed and all resize output produced,
    /// the resizer applies this orientation to the output. For non-row-local
    /// orientations (anything other than `Identity` or `FlipH`), the full
    /// output is buffered internally — memory cost is `out_w * out_h * channels`
    /// bytes.
    ///
    /// For FlipH, each output row is reversed in-place — no extra buffering.
    ///
    /// Must be called before pushing any rows.
    ///
    /// # Panics
    ///
    /// Panics if rows have already been pushed.
    pub fn with_orientation(mut self, orient: OrientOutput) -> Self {
        assert_eq!(
            self.input_rows_received, 0,
            "with_orientation must be called before pushing rows"
        );
        let (ow, oh) = orient.output_dimensions(
            self.config.total_output_width(),
            self.config.total_output_height(),
        );
        self.oriented_w = ow;
        self.oriented_h = oh;
        if !orient.is_row_local() {
            self.orient_buf = vec![0u8; ow as usize * oh as usize * self.channels];
        }
        self.orient = orient;
        self
    }

    /// Set the blend mode for compositing.
    ///
    /// Default is [`BlendMode::SrcOver`] (Porter-Duff source-over).
    /// Only meaningful when a background is set via [`with_background`](Self::with_background).
    ///
    /// Panics if rows have already been pushed.
    pub fn with_blend_mode(mut self, mode: composite::BlendMode) -> Self {
        assert_eq!(
            self.input_rows_received, 0,
            "with_blend_mode must be called before pushing rows"
        );
        self.blend_mode = mode;
        self
    }

    /// Create a streaming resizer with background compositing.
    ///
    /// Performs source-over compositing between the resized foreground and the
    /// given background. The compositing happens in premultiplied linear f32
    /// space, between the vertical filter and unpremultiply.
    ///
    /// # Errors
    ///
    /// Returns [`CompositeError::PremultipliedInput`] if the input format
    /// is `RgbaPremul` (compositing premultiplied input is mathematically incorrect).
    pub fn with_background(
        config: &ResizeConfig,
        background: B,
    ) -> Result<Self, At<CompositeError>> {
        if config.input.alpha == Some(AlphaMode::Premultiplied) {
            return Err(at!(CompositeError::PremultipliedInput));
        }
        Ok(Self::new_inner(config, background, true, 0))
    }

    fn new_inner(
        config: &ResizeConfig,
        background: B,
        has_composite: bool,
        batch_hint: u32,
    ) -> Self {
        config.validate().expect("invalid resize config");

        let mut config = config.clone();
        // Compositing requires linear f32 path
        let active_composite = has_composite && !background.is_transparent();
        if active_composite {
            config.linear = true;
        }

        let mut filter = InterpolationDetails::create(config.filter);
        if let Some(scale) = config.kernel_width_scale {
            filter = filter.with_blur(scale);
        }
        match &config.lobe_ratio {
            crate::pixel::LobeRatio::Natural => {}
            crate::pixel::LobeRatio::Exact(r) => filter = filter.with_lobe_ratio(*r),
            crate::pixel::LobeRatio::SharpenPercent(p) => filter = filter.with_sharpen_percent(*p),
        }

        // Crop: use crop dimensions for weight tables
        let resize_in_w = config.resize_in_width();
        let resize_in_h = config.resize_in_height();
        let v_weights = F32WeightTable::new(resize_in_h, config.out_height, &filter);

        let channels = config.channels();
        let needs_premul = config.needs_premultiply();
        let alpha_is_last = config.input.has_alpha();

        // Crop state
        let has_crop = config.source_region.is_some();
        let (crop_x_offset, crop_y_start, crop_y_end, source_in_width) =
            if let Some(ref r) = config.source_region {
                (
                    r.x as usize * channels,
                    r.y,
                    r.y + r.height,
                    config.in_width,
                )
            } else {
                (0, 0, config.in_height, config.in_width)
            };

        // Path selection: mirrors fullframe Resizer paths 0/1.
        // I16Srgb: identity transfer (no linearization), u8 4ch.
        // I16Linear: sRGB→linear via compile-time LUTs, u8 4ch, no premul.
        // Both i16 paths hardcode their transfer functions, so they require
        // exact match — any other transfer (BT.709, PQ, HLG) goes to f32.
        let input_tf = config.effective_input_transfer();
        let output_tf = config.effective_output_transfer();
        let is_u8_io = config.input.channel_type() == ChannelType::U8
            && config.output.channel_type() == ChannelType::U8;
        let path = if !active_composite
            && is_u8_io
            && input_tf == TransferFunction::Linear
            && output_tf == TransferFunction::Linear
            && channels == 4
        {
            StreamingPath::I16Srgb
        } else if !active_composite
            && is_u8_io
            && input_tf == TransferFunction::Srgb
            && output_tf == TransferFunction::Srgb
            && channels == 4
            && !needs_premul
        {
            StreamingPath::I16Linear
        } else {
            StreamingPath::F32
        };

        // i16 precision guard: fall back to f32 when filter tap count is high
        // enough that integer rounding error accumulates beyond acceptable levels.
        //
        // The i16 pipeline rounds to i16 after the H-filter, then the V-filter
        // accumulates these rounded values. With many taps (heavy downscale),
        // the double-rounding error grows. For I16Linear, the sRGB delinearization
        // curve amplifies small linear errors into large u8 errors in darks.
        //
        // Thresholds chosen from measured max diff vs f32 reference:
        //   I16Srgb: ≤7 u8 at ≤2x, acceptable up to ~8x downscale
        //   I16Linear: up to 43 u8 at 2x, use only for ≤2x downscale
        let path = match path {
            StreamingPath::I16Linear => {
                let h_taps = estimate_max_taps(resize_in_w, config.out_width, &filter);
                let v_taps = estimate_max_taps(resize_in_h, config.out_height, &filter);
                // 14 taps ≈ Lanczos3 at 2.3x downscale
                if h_taps.max(v_taps) > 14 {
                    StreamingPath::F32
                } else {
                    StreamingPath::I16Linear
                }
            }
            StreamingPath::I16Srgb => {
                let h_taps = estimate_max_taps(resize_in_w, config.out_width, &filter);
                let v_taps = estimate_max_taps(resize_in_h, config.out_height, &filter);
                // 50 taps ≈ Lanczos3 at 8x downscale
                if h_taps.max(v_taps) > 50 {
                    StreamingPath::F32
                } else {
                    StreamingPath::I16Srgb
                }
            }
            StreamingPath::F32 => StreamingPath::F32,
        };

        let extra_slack = if batch_hint > 0 {
            batch_hint as usize + 2
        } else {
            2
        };
        let cache_size = v_weights.max_taps + extra_slack;
        let in_row_len = resize_in_w as usize * channels;
        let out_row_len = config.out_width as usize * channels;
        let cfg_total_w = config.total_output_width();
        let cfg_total_h = config.total_output_height();

        // Only allocate bg row buffer for non-solid, non-transparent backgrounds
        let needs_bg_row =
            has_composite && !background.is_transparent() && background.solid_pixel().is_none();
        let composite_bg_row = if needs_bg_row {
            vec![0.0f32; out_row_len]
        } else {
            Vec::new()
        };

        // Reusable output conversion buffers (always needed for u8 output)
        let output_buf_u8 = vec![0u8; out_row_len];

        // Padding buffers
        let has_padding = config.padding.as_ref().is_some_and(|p| !p.is_empty());
        let (
            pad_top,
            pad_bottom,
            pad_left_elements,
            pad_full_row_u8,
            padded_output_buf_u8,
            pad_full_row_f32,
            padded_output_buf_f32,
            pad_full_row_u16,
            padded_output_buf_u16,
        ) = if has_padding {
            let p = config.padding.as_ref().unwrap();
            let total_w = (p.left + config.out_width + p.right) as usize;
            let total_row_len = total_w * channels;
            let pad_left_el = p.left as usize * channels;

            // Convert color to u8 pixel
            let mut pad_pixel_u8 = vec![0u8; channels];
            for (i, b) in pad_pixel_u8.iter_mut().enumerate() {
                *b = (p.color[i.min(3)] * 255.0 + 0.5) as u8;
            }
            let mut full_u8 = vec![0u8; total_row_len];
            for pixel in full_u8.chunks_exact_mut(channels) {
                pixel.copy_from_slice(&pad_pixel_u8);
            }
            let padded_u8 = full_u8.clone();

            // f32 padding buffers
            let mut pad_pixel_f32 = vec![0.0f32; channels];
            for (i, v) in pad_pixel_f32.iter_mut().enumerate() {
                *v = p.color[i.min(3)];
            }
            let mut full_f32 = vec![0.0f32; total_row_len];
            for pixel in full_f32.chunks_exact_mut(channels) {
                pixel.copy_from_slice(&pad_pixel_f32);
            }
            let padded_f32 = full_f32.clone();

            // u16 padding buffers
            let mut pad_pixel_u16 = vec![0u16; channels];
            for (i, v) in pad_pixel_u16.iter_mut().enumerate() {
                *v = (p.color[i.min(3)] * 65535.0 + 0.5) as u16;
            }
            let mut full_u16 = vec![0u16; total_row_len];
            for pixel in full_u16.chunks_exact_mut(channels) {
                pixel.copy_from_slice(&pad_pixel_u16);
            }
            let padded_u16 = full_u16.clone();

            (
                p.top,
                p.bottom,
                pad_left_el,
                full_u8,
                padded_u8,
                full_f32,
                padded_f32,
                full_u16,
                padded_u16,
            )
        } else {
            (
                0,
                0,
                0,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            )
        };

        match path {
            StreamingPath::F32 => {
                let h_weights = F32WeightTable::new(resize_in_w, config.out_width, &filter);
                let h_padding = h_weights.max_taps * channels;
                let v_cache_row_len = in_row_len + h_padding;

                let v_cache = (0..cache_size)
                    .map(|_| vec![0u16; v_cache_row_len])
                    .collect();

                Self {
                    config,
                    path,
                    h_weights: Some(h_weights),
                    v_weights,
                    channels,
                    needs_premul,
                    alpha_is_last,
                    v_cache,
                    cache_write_idx: 0,
                    cache_size,
                    temp_input_f32: vec![0.0f32; v_cache_row_len],
                    temp_v_output: vec![0.0f32; v_cache_row_len],
                    temp_output_f32: vec![0.0f32; out_row_len],
                    output_buf_u8,
                    output_buf_u16: vec![0u16; out_row_len],
                    i16_h_weights: None,
                    i16_v_weights: None,
                    u8_v_cache: Vec::new(),
                    i16_v_cache: Vec::new(),
                    temp_v_output_u8: Vec::new(),
                    temp_v_output_i16: Vec::new(),
                    temp_h_output_i16: Vec::new(),
                    linearized_row_i16: Vec::new(),
                    input_rows_received: 0,
                    output_rows_produced: 0,
                    finished: false,
                    paired_row_buf: vec![0u8; out_row_len],
                    paired_row_ready: false,
                    background,
                    composite_bg_row,
                    blend_mode: composite::BlendMode::SrcOver,
                    source_row_index: 0,
                    crop_x_offset,
                    crop_y_start,
                    crop_y_end,
                    source_in_width,
                    has_crop,
                    pad_top,
                    pad_bottom,
                    pad_left_elements,
                    pad_full_row_u8,
                    padded_output_buf_u8,
                    pad_full_row_f32,
                    padded_output_buf_f32,
                    pad_full_row_u16,
                    padded_output_buf_u16,
                    pad_top_emitted: 0,
                    pad_bottom_emitted: 0,
                    has_padding,
                    orient: OrientOutput::Identity,
                    orient_buf: Vec::new(),
                    orient_rows_emitted: 0,
                    oriented_w: cfg_total_w,
                    oriented_h: cfg_total_h,
                    orient_ready: false,
                    orient_captured: 0,
                }
            }
            StreamingPath::I16Srgb => {
                let i16_h_weights = I16WeightTable::new(resize_in_w, config.out_width, &filter);
                let i16_v_weights = I16WeightTable::new(resize_in_h, config.out_height, &filter);

                // H-first: ring buffer stores H-filtered rows at out_width (not in_width)
                let h_padding_bytes = i16_h_weights.groups4 * 16;
                let i16_v_cache =
                    (0..cache_size).map(|_| vec![0i16; out_row_len]).collect();

                Self {
                    config,
                    path,
                    h_weights: None,
                    v_weights,
                    channels,
                    needs_premul,
                    alpha_is_last,
                    v_cache: Vec::new(),
                    cache_write_idx: 0,
                    cache_size,
                    temp_input_f32: Vec::new(),
                    temp_v_output: Vec::new(),
                    temp_output_f32: Vec::new(),
                    output_buf_u8,
                    output_buf_u16: Vec::new(),
                    i16_h_weights: Some(i16_h_weights),
                    i16_v_weights: Some(i16_v_weights),
                    u8_v_cache: Vec::new(),
                    i16_v_cache,
                    // H-first: premul scratch and H-filter input buffer
                    temp_v_output_u8: if needs_premul {
                        vec![0u8; in_row_len + h_padding_bytes]
                    } else {
                        Vec::new()
                    },
                    // V-filter output buffer (out_width wide, for produce_next)
                    temp_v_output_i16: vec![0i16; out_row_len],
                    temp_h_output_i16: Vec::new(), // not used in H-first
                    linearized_row_i16: Vec::new(),
                    input_rows_received: 0,
                    output_rows_produced: 0,
                    finished: false,
                    paired_row_buf: vec![0u8; out_row_len],
                    paired_row_ready: false,
                    background,
                    composite_bg_row,
                    blend_mode: composite::BlendMode::SrcOver,
                    source_row_index: 0,
                    crop_x_offset,
                    crop_y_start,
                    crop_y_end,
                    source_in_width,
                    has_crop,
                    pad_top,
                    pad_bottom,
                    pad_left_elements,
                    pad_full_row_u8,
                    padded_output_buf_u8,
                    pad_full_row_f32,
                    padded_output_buf_f32,
                    pad_full_row_u16,
                    padded_output_buf_u16,
                    pad_top_emitted: 0,
                    pad_bottom_emitted: 0,
                    has_padding,
                    orient: OrientOutput::Identity,
                    orient_buf: Vec::new(),
                    orient_rows_emitted: 0,
                    oriented_w: cfg_total_w,
                    oriented_h: cfg_total_h,
                    orient_ready: false,
                    orient_captured: 0,
                }
            }
            StreamingPath::I16Linear => {
                let i16_h_weights = I16WeightTable::new(resize_in_w, config.out_width, &filter);
                let i16_v_weights = I16WeightTable::new(resize_in_h, config.out_height, &filter);

                // H-first: ring buffer stores H-filtered rows at out_width
                let h_padding_i16 = i16_h_weights.groups4 * 16;
                let i16_v_cache =
                    (0..cache_size).map(|_| vec![0i16; out_row_len]).collect();

                Self {
                    config,
                    path,
                    h_weights: None,
                    v_weights,
                    channels,
                    needs_premul,
                    alpha_is_last,
                    v_cache: Vec::new(),
                    cache_write_idx: 0,
                    cache_size,
                    temp_input_f32: Vec::new(),
                    temp_v_output: Vec::new(),
                    temp_output_f32: Vec::new(),
                    output_buf_u8,
                    output_buf_u16: Vec::new(),
                    i16_h_weights: Some(i16_h_weights),
                    i16_v_weights: Some(i16_v_weights),
                    u8_v_cache: Vec::new(),
                    i16_v_cache,
                    temp_v_output_u8: Vec::new(),
                    temp_v_output_i16: vec![0i16; out_row_len],
                    temp_h_output_i16: Vec::new(), // not used in H-first
                    linearized_row_i16: vec![0i16; in_row_len + h_padding_i16],
                    input_rows_received: 0,
                    output_rows_produced: 0,
                    finished: false,
                    paired_row_buf: vec![0u8; out_row_len],
                    paired_row_ready: false,
                    background,
                    composite_bg_row,
                    blend_mode: composite::BlendMode::SrcOver,
                    source_row_index: 0,
                    crop_x_offset,
                    crop_y_start,
                    crop_y_end,
                    source_in_width,
                    has_crop,
                    pad_top,
                    pad_bottom,
                    pad_left_elements,
                    pad_full_row_u8,
                    padded_output_buf_u8,
                    pad_full_row_f32,
                    padded_output_buf_f32,
                    pad_full_row_u16,
                    padded_output_buf_u16,
                    pad_top_emitted: 0,
                    pad_bottom_emitted: 0,
                    has_padding,
                    orient: OrientOutput::Identity,
                    orient_buf: Vec::new(),
                    orient_rows_emitted: 0,
                    oriented_w: cfg_total_w,
                    oriented_h: cfg_total_h,
                    orient_ready: false,
                    orient_captured: 0,
                }
            }
        }
    }

    // =========================================================================
    // Info accessors
    // =========================================================================

    /// Number of elements per input row (for sizing decode buffers).
    ///
    /// When a source region is set, this still returns the full source width
    /// (`in_width * channels`), since `push_row` accepts full-width rows.
    pub fn input_row_len(&self) -> usize {
        self.source_in_width as usize * self.channels
    }

    /// Number of elements per output row (for sizing encode buffers).
    ///
    /// When padding is set, this returns the padded width
    /// (`(left + out_width + right) * channels`).
    /// When orientation swaps axes, this returns the oriented width.
    pub fn output_row_len(&self) -> usize {
        if self.orient.is_identity() {
            self.config.total_output_row_len()
        } else {
            self.oriented_w as usize * self.channels
        }
    }

    /// Mutable reference to the background (e.g., for pushing rows to [`crate::StreamedBackground`]).
    pub fn background_mut(&mut self) -> &mut B {
        &mut self.background
    }

    /// Consume the streaming resizer and return the background.
    pub fn into_background(self) -> B {
        self.background
    }

    /// Query the internal working format.
    ///
    /// Callers can use this to produce input data in the optimal format,
    /// avoiding redundant conversions. For example, if `working_format()`
    /// returns [`WorkingFormat::I16Linear`], a JPEG decoder could fuse
    /// YCbCr→linear i12 conversion and push via [`push_row_i16()`](Self::push_row_i16).
    pub fn working_format(&self) -> WorkingFormat {
        match self.path {
            StreamingPath::F32 => WorkingFormat::F32,
            StreamingPath::I16Srgb => WorkingFormat::I16Srgb,
            StreamingPath::I16Linear => WorkingFormat::I16Linear,
        }
    }

    /// Reset the streaming resizer for reuse with a new image of the same dimensions.
    ///
    /// Clears all internal state (ring buffer indices, output counters, orientation
    /// buffers) while preserving weight tables and allocated buffers. This avoids
    /// the cost of recomputing weight tables (~1ms for 4K Lanczos3).
    pub fn reset(&mut self) {
        self.cache_write_idx = 0;
        self.input_rows_received = 0;
        self.output_rows_produced = 0;
        self.finished = false;
        self.paired_row_ready = false;
        self.source_row_index = 0;
        self.orient_rows_emitted = 0;
        self.orient_ready = false;
        self.orient_captured = 0;
        self.pad_top_emitted = 0;
        self.pad_bottom_emitted = 0;
    }

    /// How many input rows must be pushed before the first output row.
    ///
    /// When padding is set and `pad_top > 0`, returns 0 since top padding
    /// rows are available immediately without any input.
    pub fn initial_input_rows_needed(&self) -> u32 {
        if self.pad_top > 0 {
            return 0;
        }
        let first_right = self.first_output_row_max_input();
        let resize_in_h = self.config.resize_in_height();
        let crop_offset = self.crop_y_start;
        // Account for crop: the caller needs to push crop_y_start + needed rows
        (first_right + 1).min(resize_in_h) + crop_offset
    }

    /// Total output rows produced so far.
    pub fn output_rows_produced(&self) -> u32 {
        self.output_rows_produced
    }

    /// Total output rows produced so far, including padding rows.
    pub fn total_rows_emitted(&self) -> u32 {
        if !self.orient.is_row_local() {
            return self.orient_rows_emitted;
        }
        self.pad_top_emitted + self.output_rows_produced + self.pad_bottom_emitted
    }

    /// Total output height including padding and orientation.
    pub fn total_output_height(&self) -> u32 {
        self.oriented_h
    }

    /// Check if all output rows have been produced (including padding and orientation).
    pub fn is_complete(&self) -> bool {
        self.total_rows_emitted() >= self.total_output_height()
    }

    /// Count how many output rows can be produced right now without more input.
    ///
    /// This is the number of consecutive `next_output_row()` calls that will
    /// return `Some` before returning `None`. Includes padding rows.
    ///
    /// For non-row-local orientations, returns 0 until all resize output
    /// has been captured, then returns the remaining oriented rows.
    pub fn output_rows_available(&self) -> u32 {
        if !self.orient.is_row_local() {
            return if self.orient_ready {
                self.oriented_h - self.orient_rows_emitted
            } else {
                0
            };
        }
        self.output_rows_available_inner()
    }

    /// Inner implementation of output_rows_available (resize + padding only).
    fn output_rows_available_inner(&self) -> u32 {
        let mut count = 0u32;

        // Remaining top padding
        count += self.pad_top.saturating_sub(self.pad_top_emitted);

        // Content rows available
        let mut probe_y = self.output_rows_produced;
        let resize_in_h = self.config.resize_in_height();
        while probe_y < self.config.out_height {
            let left = self.v_weights.left[probe_y as usize];
            let tap_count = self.v_weights.tap_count(probe_y as usize);
            let right = left + tap_count as i32 - 1;
            let needed_max = right.min(resize_in_h as i32 - 1).max(0) as u32;
            if needed_max >= self.input_rows_received {
                break;
            }
            count += 1;
            probe_y += 1;
        }

        // Bottom padding (only available after all content is produced)
        if probe_y >= self.config.out_height {
            count += self.pad_bottom.saturating_sub(self.pad_bottom_emitted);
        }

        count
    }

    // =========================================================================
    // Input methods
    // =========================================================================

    /// Push one row of u8 input pixels. Linearizes, premultiplies, and caches the row.
    ///
    /// When a source region is set, rows outside the vertical range are skipped
    /// automatically. The row must be full-width (`in_width * channels` elements).
    ///
    /// Caller MUST drain all available output rows (via `next_output_row` or
    /// `next_output_row_into`) before pushing the next input row.
    ///
    /// # Errors
    ///
    /// Returns [`StreamingError::AlreadyFinished`] if called after `finish()`.
    /// Returns [`StreamingError::InputTooShort`] if `row` is shorter than required.
    /// Returns [`StreamingError::RingBufferOverflow`] if output was not drained.
    pub fn push_row(&mut self, row: &[u8]) -> Result<(), At<StreamingError>> {
        if self.finished {
            return Err(at!(StreamingError::AlreadyFinished));
        }

        // Validate against full source width
        let source_row_len = self.source_in_width as usize * self.channels;
        let stride = self.config.effective_in_stride();
        if row.len() < source_row_len.min(stride) {
            return Err(at!(StreamingError::InputTooShort));
        }

        // Vertical crop: skip rows outside the crop region
        if self.has_crop {
            let src_y = self.source_row_index;
            self.source_row_index += 1;
            if src_y < self.crop_y_start || src_y >= self.crop_y_end {
                return Ok(());
            }
        }

        // Extract horizontal crop region (or use full row)
        let resize_row_len = self.config.resize_in_width() as usize * self.channels;
        let pixel_data = &row[self.crop_x_offset..self.crop_x_offset + resize_row_len];
        let pixel_len = resize_row_len;

        match self.path {
            StreamingPath::I16Srgb => {
                // H-first: u8 → optional premul → H-filter to i16 → ring buffer
                self.check_ring_buffer().at()?;
                let cache_slot = self.cache_write_idx % self.cache_size;
                let i16_h_weights = self.i16_h_weights.as_ref().unwrap();
                let out_row_len = self.config.out_width as usize * self.channels;

                let src = if self.needs_premul {
                    simd::premultiply_u8_row(
                        pixel_data,
                        &mut self.temp_v_output_u8[..pixel_len],
                    );
                    &self.temp_v_output_u8[..]
                } else {
                    pixel_data
                };

                // H-filter u8 → i16 (unclamped, preserves ringing)
                simd::filter_h_u8_to_i16(
                    src,
                    &mut self.i16_v_cache[cache_slot][..out_row_len],
                    i16_h_weights,
                    self.channels,
                );
                self.cache_write_idx += 1;
                self.input_rows_received += 1;
                self.eagerly_capture_orient_rows();
                return Ok(());
            }
            StreamingPath::I16Linear => {
                // H-first: sRGB u8 → linear i12 (LUT) → H-filter i16 → ring buffer
                color::srgb_u8_to_linear_i12_row(
                    pixel_data,
                    &mut self.linearized_row_i16[..pixel_len],
                );
                // Zero SIMD padding
                for v in &mut self.linearized_row_i16[pixel_len..] {
                    *v = 0;
                }
                self.check_ring_buffer().at()?;
                let cache_slot = self.cache_write_idx % self.cache_size;
                let i16_h_weights = self.i16_h_weights.as_ref().unwrap();
                let out_row_len = self.config.out_width as usize * self.channels;
                simd::filter_h_i16_i16(
                    &self.linearized_row_i16,
                    &mut self.i16_v_cache[cache_slot][..out_row_len],
                    i16_h_weights,
                    self.channels,
                );
                self.cache_write_idx += 1;
                self.input_rows_received += 1;
                self.eagerly_capture_orient_rows();
                return Ok(());
            }
            StreamingPath::F32 => {}
        }

        // F32 path: linearize + premultiply → cache f32
        match self.config.effective_input_transfer() {
            TransferFunction::Srgb => {
                color::srgb_u8_to_linear_f32(
                    pixel_data,
                    &mut self.temp_input_f32[..pixel_len],
                    self.channels,
                    self.alpha_is_last,
                );
            }
            TransferFunction::Linear => {
                simd::u8_to_f32_row(pixel_data, &mut self.temp_input_f32[..pixel_len]);
            }
            TransferFunction::Bt709 => {
                Bt709.u8_to_linear_f32(
                    pixel_data,
                    &mut self.temp_input_f32[..pixel_len],
                    &(),
                    self.channels,
                    self.alpha_is_last,
                    false,
                );
            }
            TransferFunction::Pq => {
                Pq.u8_to_linear_f32(
                    pixel_data,
                    &mut self.temp_input_f32[..pixel_len],
                    &(),
                    self.channels,
                    self.alpha_is_last,
                    false,
                );
            }
            TransferFunction::Hlg => {
                Hlg.u8_to_linear_f32(
                    pixel_data,
                    &mut self.temp_input_f32[..pixel_len],
                    &(),
                    self.channels,
                    self.alpha_is_last,
                    false,
                );
            }
            _ => {
                simd::u8_to_f32_row(pixel_data, &mut self.temp_input_f32[..pixel_len]);
            }
        }

        if self.needs_premul {
            simd::premultiply_alpha_row(&mut self.temp_input_f32[..pixel_len]);
        }

        self.push_row_internal().at()
    }

    /// Push multiple rows of u8 input pixels from a contiguous buffer.
    ///
    /// Each row is `stride` bytes apart in `buf`. Pushes `count` rows and
    /// returns the number of output rows available to drain via `next_output_row()`.
    ///
    /// Caller MUST drain all available output rows between `push_rows` calls.
    /// The batch size must be small enough to fit in the ring buffer without
    /// draining (typically `max_taps + 2` slots). For typical photo sizes and
    /// filters, batches of 8 rows work well.
    ///
    /// # Errors
    ///
    /// Returns [`StreamingError::AlreadyFinished`] if called after `finish()`.
    /// Returns [`StreamingError::InputTooShort`] if `buf` is too short for the
    /// given `stride` and `count`.
    /// Returns [`StreamingError::RingBufferOverflow`] if the batch is too large
    /// for the ring buffer. Use a smaller batch or drain between pushes.
    pub fn push_rows(
        &mut self,
        buf: &[u8],
        stride: usize,
        count: u32,
    ) -> Result<u32, At<StreamingError>> {
        let source_row_len = self.source_in_width as usize * self.channels;
        let expected_len = if count == 0 {
            0
        } else {
            stride * (count as usize - 1) + source_row_len
        };
        if buf.len() < expected_len {
            return Err(at!(StreamingError::InputTooShort));
        }
        for i in 0..count as usize {
            self.push_row(&buf[i * stride..]).at()?;
        }
        Ok(self.output_rows_available())
    }

    /// Push one row of f32 input pixels. Premultiplies (if needed) and caches the row.
    ///
    /// When a source region is set, rows outside the vertical range are skipped.
    /// The row must be full-width (`in_width * channels` elements).
    ///
    /// Caller MUST drain all available output rows before pushing the next input row.
    ///
    /// # Errors
    ///
    /// Returns [`StreamingError::AlreadyFinished`] if called after `finish()`.
    /// Returns [`StreamingError::InputTooShort`] if `row` is shorter than required.
    /// Returns [`StreamingError::RingBufferOverflow`] if output was not drained.
    pub fn push_row_f32(&mut self, row: &[f32]) -> Result<(), At<StreamingError>> {
        debug_assert_eq!(
            self.path,
            StreamingPath::F32,
            "push_row_f32 requires f32 path"
        );
        if self.finished {
            return Err(at!(StreamingError::AlreadyFinished));
        }
        let source_row_len = self.source_in_width as usize * self.channels;
        if row.len() < source_row_len {
            return Err(at!(StreamingError::InputTooShort));
        }

        // Vertical crop
        if self.has_crop {
            let src_y = self.source_row_index;
            self.source_row_index += 1;
            if src_y < self.crop_y_start || src_y >= self.crop_y_end {
                return Ok(());
            }
        }

        let pixel_len = self.config.resize_in_width() as usize * self.channels;
        let crop_offset_f32 = self.crop_x_offset; // same element offset for f32
        self.temp_input_f32[..pixel_len]
            .copy_from_slice(&row[crop_offset_f32..crop_offset_f32 + pixel_len]);

        if self.needs_premul {
            simd::premultiply_alpha_row(&mut self.temp_input_f32[..pixel_len]);
        }

        self.push_row_internal().at()
    }

    /// Push one row of f32 input by writing directly into the resizer's internal buffer.
    ///
    /// The closure receives `&mut [f32]` of length `resize_in_width * channels`
    /// (crop width if cropping, else full width). Write your f32 pixel data into
    /// this slice. After the closure returns, premultiply and cache run without
    /// a `copy_from_slice` (saves one memcpy vs `push_row_f32`).
    ///
    /// When a source region is set, the caller must handle vertical crop
    /// externally and only call this for rows within the crop region.
    ///
    /// Caller MUST drain all available output rows before pushing the next input row.
    ///
    /// # Errors
    ///
    /// Returns [`StreamingError::AlreadyFinished`] if called after `finish()`.
    /// Returns [`StreamingError::RingBufferOverflow`] if output was not drained.
    pub fn push_row_f32_with<F: FnOnce(&mut [f32])>(
        &mut self,
        f: F,
    ) -> Result<(), At<StreamingError>> {
        debug_assert_eq!(
            self.path,
            StreamingPath::F32,
            "push_row_f32_with requires f32 path"
        );
        if self.finished {
            return Err(at!(StreamingError::AlreadyFinished));
        }
        let pixel_len = self.config.resize_in_width() as usize * self.channels;
        f(&mut self.temp_input_f32[..pixel_len]);

        if self.needs_premul {
            simd::premultiply_alpha_row(&mut self.temp_input_f32[..pixel_len]);
        }

        self.push_row_internal().at()
    }

    /// Push one row of u16 input pixels. Linearizes, premultiplies, and caches the row.
    ///
    /// Uses the sRGB transfer function to linearize u16 values (0-65535)
    /// before filtering in f32 linear-light space.
    ///
    /// Caller MUST drain all available output rows before pushing the next input row.
    ///
    /// # Errors
    ///
    /// Returns [`StreamingError::AlreadyFinished`] if called after `finish()`.
    /// Returns [`StreamingError::InputTooShort`] if `row` is shorter than required.
    /// Returns [`StreamingError::RingBufferOverflow`] if output was not drained.
    pub fn push_row_u16(&mut self, row: &[u16]) -> Result<(), At<StreamingError>> {
        if self.finished {
            return Err(at!(StreamingError::AlreadyFinished));
        }
        let source_row_len = self.source_in_width as usize * self.channels;
        let stride = self.config.effective_in_stride();
        if row.len() < source_row_len.min(stride) {
            return Err(at!(StreamingError::InputTooShort));
        }

        // Vertical crop
        if self.has_crop {
            let src_y = self.source_row_index;
            self.source_row_index += 1;
            if src_y < self.crop_y_start || src_y >= self.crop_y_end {
                return Ok(());
            }
        }

        let pixel_len = self.config.resize_in_width() as usize * self.channels;
        let pixel_data = &row[self.crop_x_offset..self.crop_x_offset + pixel_len];

        match self.config.effective_input_transfer() {
            TransferFunction::Srgb => {
                Srgb.u16_to_linear_f32(
                    pixel_data,
                    &mut self.temp_input_f32[..pixel_len],
                    &(),
                    self.channels,
                    self.alpha_is_last,
                    self.needs_premul,
                );
            }
            TransferFunction::Linear => {
                crate::transfer::NoTransfer.u16_to_linear_f32(
                    pixel_data,
                    &mut self.temp_input_f32[..pixel_len],
                    &(),
                    self.channels,
                    self.alpha_is_last,
                    self.needs_premul,
                );
            }
            TransferFunction::Bt709 => {
                Bt709.u16_to_linear_f32(
                    pixel_data,
                    &mut self.temp_input_f32[..pixel_len],
                    &(),
                    self.channels,
                    self.alpha_is_last,
                    self.needs_premul,
                );
            }
            TransferFunction::Pq => {
                Pq.u16_to_linear_f32(
                    pixel_data,
                    &mut self.temp_input_f32[..pixel_len],
                    &(),
                    self.channels,
                    self.alpha_is_last,
                    self.needs_premul,
                );
            }
            TransferFunction::Hlg => {
                Hlg.u16_to_linear_f32(
                    pixel_data,
                    &mut self.temp_input_f32[..pixel_len],
                    &(),
                    self.channels,
                    self.alpha_is_last,
                    self.needs_premul,
                );
            }
            _ => {
                crate::transfer::NoTransfer.u16_to_linear_f32(
                    pixel_data,
                    &mut self.temp_input_f32[..pixel_len],
                    &(),
                    self.channels,
                    self.alpha_is_last,
                    self.needs_premul,
                );
            }
        }

        self.push_row_internal().at()
    }

    /// Push one row of i16 data directly into the i16 ring buffer.
    ///
    /// For [`WorkingFormat::I16Srgb`]: values are u8-range (0–255) zero-extended to i16.
    /// For [`WorkingFormat::I16Linear`]: values are linear i12 (0–4095).
    ///
    /// This skips the input transfer function entirely — the caller is responsible
    /// for providing correctly-formatted data. Use [`working_format()`](Self::working_format)
    /// to determine what format is expected.
    ///
    /// # Panics (debug)
    ///
    /// Panics if `working_format()` is [`WorkingFormat::F32`].
    pub fn push_row_i16(&mut self, row: &[i16]) -> Result<(), At<StreamingError>> {
        debug_assert!(
            matches!(self.path, StreamingPath::I16Srgb | StreamingPath::I16Linear),
            "push_row_i16 requires an i16 path"
        );
        if self.finished {
            return Err(at!(StreamingError::AlreadyFinished));
        }
        let source_row_len = self.source_in_width as usize * self.channels;
        if row.len() < source_row_len {
            return Err(at!(StreamingError::InputTooShort));
        }

        // Vertical crop
        if self.has_crop {
            let src_y = self.source_row_index;
            self.source_row_index += 1;
            if src_y < self.crop_y_start || src_y >= self.crop_y_end {
                return Ok(());
            }
        }

        let pixel_len = self.config.resize_in_width() as usize * self.channels;
        let row = &row[self.crop_x_offset..self.crop_x_offset + pixel_len];

        match self.path {
            StreamingPath::I16Srgb => {
                // H-first: H-filter i16 input → ring buffer (out_width wide)
                self.check_ring_buffer().at()?;
                let cache_slot = self.cache_write_idx % self.cache_size;
                let i16_h_weights = self.i16_h_weights.as_ref().unwrap();
                let out_row_len = self.config.out_width as usize * self.channels;
                // Copy to linearized_row for SIMD padding, then H-filter
                self.linearized_row_i16.resize(pixel_len + i16_h_weights.groups4 * 16, 0);
                self.linearized_row_i16[..pixel_len].copy_from_slice(&row[..pixel_len]);
                for v in &mut self.linearized_row_i16[pixel_len..] { *v = 0; }
                simd::filter_h_i16_i16(
                    &self.linearized_row_i16,
                    &mut self.i16_v_cache[cache_slot][..out_row_len],
                    i16_h_weights,
                    self.channels,
                );
                self.cache_write_idx += 1;
                self.input_rows_received += 1;
                Ok(())
            }
            StreamingPath::I16Linear => {
                // H-first: H-filter i16 input → ring buffer (out_width wide)
                self.linearized_row_i16[..pixel_len].copy_from_slice(&row[..pixel_len]);
                for v in &mut self.linearized_row_i16[pixel_len..] { *v = 0; }
                self.check_ring_buffer().at()?;
                let cache_slot = self.cache_write_idx % self.cache_size;
                let i16_h_weights = self.i16_h_weights.as_ref().unwrap();
                let out_row_len = self.config.out_width as usize * self.channels;
                simd::filter_h_i16_i16(
                    &self.linearized_row_i16,
                    &mut self.i16_v_cache[cache_slot][..out_row_len],
                    i16_h_weights,
                    self.channels,
                );
                self.cache_write_idx += 1;
                self.input_rows_received += 1;
                self.eagerly_capture_orient_rows();
                Ok(())
            }
            StreamingPath::F32 => unreachable!("guarded by debug_assert"),
        }
    }

    /// Push one row of linear f32 data, skipping the input transfer function.
    ///
    /// Data must already be in linear-light f32 space. The resizer will still
    /// premultiply if needed, but skips linearization.
    ///
    /// # Panics (debug)
    ///
    /// Panics if `working_format()` is not [`WorkingFormat::F32`].
    pub fn push_row_linear_f32(&mut self, row: &[f32]) -> Result<(), At<StreamingError>> {
        debug_assert_eq!(
            self.path,
            StreamingPath::F32,
            "push_row_linear_f32 requires f32 path"
        );
        if self.finished {
            return Err(at!(StreamingError::AlreadyFinished));
        }
        let source_row_len = self.source_in_width as usize * self.channels;
        if row.len() < source_row_len {
            return Err(at!(StreamingError::InputTooShort));
        }

        // Vertical crop
        if self.has_crop {
            let src_y = self.source_row_index;
            self.source_row_index += 1;
            if src_y < self.crop_y_start || src_y >= self.crop_y_end {
                return Ok(());
            }
        }

        let pixel_len = self.config.resize_in_width() as usize * self.channels;
        self.temp_input_f32[..pixel_len]
            .copy_from_slice(&row[self.crop_x_offset..self.crop_x_offset + pixel_len]);

        if self.needs_premul {
            simd::premultiply_alpha_row(&mut self.temp_input_f32[..pixel_len]);
        }

        self.push_row_internal().at()
    }

    /// Signal end of input. No more rows may be pushed after this call.
    ///
    /// Returns the number of output rows still available to drain.
    ///
    /// After calling `finish()`, drain remaining output rows:
    /// ```ignore
    /// let remaining = resizer.finish();
    /// for _ in 0..remaining {
    ///     if let Some(row) = resizer.next_output_row() {
    ///         encoder.write_row(row);
    ///     }
    /// }
    /// ```
    pub fn finish(&mut self) -> u32 {
        self.finished = true;
        if !self.orient.is_row_local() {
            // Capture any remaining resize output rows
            let row_len = self.config.total_output_row_len();
            loop {
                let dst_offset = self.orient_captured as usize * row_len;
                if dst_offset + row_len > self.orient_buf.len() {
                    break;
                }
                if !self.capture_one_orient_row(dst_offset, row_len) {
                    break;
                }
                self.orient_captured += 1;
            }
            self.apply_orient_transform();
        }
        self.output_rows_available()
    }

    // =========================================================================
    // Output methods — lazy pull-based production
    // =========================================================================

    /// Pull the next output row as u8. Returns `None` if more input is needed
    /// or all output rows have been produced.
    ///
    /// When padding is set, emits padding rows (top/bottom) and pads content
    /// rows (left/right) automatically. The returned slice length is
    /// `output_row_len()` (which includes padding width).
    ///
    /// When orientation is set, non-row-local transforms buffer internally
    /// and only return rows after `finish()` has been called. FlipH is
    /// applied per-row with no buffering.
    ///
    /// Lazily produces one output row: V-filter → H-filter → composite → unpremultiply → u8 convert.
    /// The returned slice borrows from the resizer's internal buffer and is valid
    /// until the next method call on this resizer.
    pub fn next_output_row(&mut self) -> Option<&[u8]> {
        // Non-row-local orientation: serve from orient_buf after transform is applied
        if !self.orient.is_row_local() {
            if !self.orient_ready {
                return None;
            }
            if self.orient_rows_emitted >= self.oriented_h {
                return None;
            }
            let row_len = self.oriented_w as usize * self.channels;
            let offset = self.orient_rows_emitted as usize * row_len;
            self.orient_rows_emitted += 1;
            return Some(&self.orient_buf[offset..offset + row_len]);
        }

        // Identity: produce normally (most common path)
        if self.orient.is_identity() {
            return self.next_output_row_unoriented();
        }

        // FlipH: produce into output_buf_u8, flip pixels in-place.
        // We produce directly and flip, avoiding borrow aliasing.
        if !self.has_padding {
            if !self.can_produce_next_output() {
                return None;
            }
            let row_len = self.config.out_width as usize * self.channels;
            let mut tmp = core::mem::take(&mut self.output_buf_u8);
            match self.path {
                StreamingPath::I16Srgb => self.produce_next_i16_srgb(&mut tmp[..row_len]),
                StreamingPath::I16Linear => self.produce_next_i16_linear(&mut tmp[..row_len]),
                StreamingPath::F32 => {
                    self.produce_next_f32();
                    Self::encode_output_u8(
                        &self.temp_output_f32[..row_len],
                        &mut tmp[..row_len],
                        self.config.effective_output_transfer(),
                        self.channels,
                        self.alpha_is_last,
                    );
                }
            }
            flip_h_row(&mut tmp[..row_len], self.channels);
            self.output_buf_u8 = tmp;
            Some(&self.output_buf_u8[..row_len])
        } else {
            // Padded FlipH: produce into padded buffer, flip entire row
            let total_row_len = self.config.total_output_row_len();

            // Phase 1: top padding
            if self.pad_top_emitted < self.pad_top {
                self.pad_top_emitted += 1;
                // Padding is uniform color — flipping is identity, but do it for correctness
                if self.output_buf_u8.len() < total_row_len {
                    self.output_buf_u8.resize(total_row_len, 0);
                }
                self.output_buf_u8[..total_row_len]
                    .copy_from_slice(&self.pad_full_row_u8[..total_row_len]);
                flip_h_row(&mut self.output_buf_u8[..total_row_len], self.channels);
                return Some(&self.output_buf_u8[..total_row_len]);
            }

            // Phase 2: content rows with left/right padding
            if self.output_rows_produced < self.config.out_height {
                if !self.can_produce_next_output() {
                    return None;
                }
                let content_len = self.config.out_width as usize * self.channels;
                let content_start = self.pad_left_elements;
                let mut tmp = core::mem::take(&mut self.padded_output_buf_u8);
                match self.path {
                    StreamingPath::I16Srgb => {
                        self.produce_next_i16_srgb(
                            &mut tmp[content_start..content_start + content_len],
                        );
                    }
                    StreamingPath::I16Linear => {
                        self.produce_next_i16_linear(
                            &mut tmp[content_start..content_start + content_len],
                        );
                    }
                    StreamingPath::F32 => {
                        self.produce_next_f32();
                        Self::encode_output_u8(
                            &self.temp_output_f32[..content_len],
                            &mut tmp[content_start..content_start + content_len],
                            self.config.effective_output_transfer(),
                            self.channels,
                            self.alpha_is_last,
                        );
                    }
                }
                flip_h_row(&mut tmp[..total_row_len], self.channels);
                self.padded_output_buf_u8 = tmp;
                return Some(&self.padded_output_buf_u8[..total_row_len]);
            }

            // Phase 3: bottom padding
            if self.pad_bottom_emitted < self.pad_bottom {
                self.pad_bottom_emitted += 1;
                if self.output_buf_u8.len() < total_row_len {
                    self.output_buf_u8.resize(total_row_len, 0);
                }
                self.output_buf_u8[..total_row_len]
                    .copy_from_slice(&self.pad_full_row_u8[..total_row_len]);
                flip_h_row(&mut self.output_buf_u8[..total_row_len], self.channels);
                return Some(&self.output_buf_u8[..total_row_len]);
            }

            None
        }
    }

    /// Pull the next output row directly into a caller-provided u8 buffer.
    /// Returns `true` if a row was produced, `false` if more input is needed.
    ///
    /// `dst` must be at least `output_row_len()` elements long.
    /// Skips the internal `output_buf_u8` — writes directly to the caller's buffer.
    pub fn next_output_row_into(&mut self, dst: &mut [u8]) -> bool {
        if !self.has_padding {
            return self.next_content_row_u8_into(dst);
        }
        let total_row_len = self.config.total_output_row_len();

        // Phase 1: top padding
        if self.pad_top_emitted < self.pad_top {
            self.pad_top_emitted += 1;
            dst[..total_row_len].copy_from_slice(&self.pad_full_row_u8[..total_row_len]);
            return true;
        }

        // Phase 2: content rows with left/right padding
        if self.output_rows_produced < self.config.out_height {
            if !self.can_produce_next_output() {
                return false;
            }
            let content_len = self.config.out_width as usize * self.channels;
            let content_start = self.pad_left_elements;
            // Fill left and right padding
            dst[..content_start].copy_from_slice(&self.pad_full_row_u8[..content_start]);
            let right_start = content_start + content_len;
            dst[right_start..total_row_len]
                .copy_from_slice(&self.pad_full_row_u8[right_start..total_row_len]);
            // Produce content
            match self.path {
                StreamingPath::I16Srgb => {
                    self.produce_next_i16_srgb(
                        &mut dst[content_start..content_start + content_len],
                    );
                }
                StreamingPath::I16Linear => {
                    self.produce_next_i16_linear(
                        &mut dst[content_start..content_start + content_len],
                    );
                }
                StreamingPath::F32 => {
                    self.produce_next_f32();
                    Self::encode_output_u8(
                        &self.temp_output_f32[..content_len],
                        &mut dst[content_start..content_start + content_len],
                        self.config.effective_output_transfer(),
                        self.channels,
                        self.alpha_is_last,
                    );
                }
            }
            return true;
        }

        // Phase 3: bottom padding
        if self.pad_bottom_emitted < self.pad_bottom {
            self.pad_bottom_emitted += 1;
            dst[..total_row_len].copy_from_slice(&self.pad_full_row_u8[..total_row_len]);
            return true;
        }

        false
    }

    /// Pull the next output row as f32. Returns `None` if more input is needed.
    ///
    /// When padding is set, includes padding pixels in the returned slice.
    /// The returned slice borrows from the resizer's internal buffer.
    pub fn next_output_row_f32(&mut self) -> Option<&[f32]> {
        debug_assert_eq!(
            self.path,
            StreamingPath::F32,
            "next_output_row_f32 requires f32 path"
        );

        if !self.has_padding {
            if !self.can_produce_next_output() {
                return None;
            }
            self.produce_next_f32();
            let row_len = self.config.out_width as usize * self.channels;
            return Some(&self.temp_output_f32[..row_len]);
        }

        let total_row_len = self.config.total_output_row_len();

        // Phase 1: top padding
        if self.pad_top_emitted < self.pad_top {
            self.pad_top_emitted += 1;
            return Some(&self.pad_full_row_f32[..total_row_len]);
        }

        // Phase 2: content rows
        if self.output_rows_produced < self.config.out_height {
            if !self.can_produce_next_output() {
                return None;
            }
            self.produce_next_f32();
            let content_len = self.config.out_width as usize * self.channels;
            let content_start = self.pad_left_elements;
            self.padded_output_buf_f32[content_start..content_start + content_len]
                .copy_from_slice(&self.temp_output_f32[..content_len]);
            return Some(&self.padded_output_buf_f32[..total_row_len]);
        }

        // Phase 3: bottom padding
        if self.pad_bottom_emitted < self.pad_bottom {
            self.pad_bottom_emitted += 1;
            return Some(&self.pad_full_row_f32[..total_row_len]);
        }

        None
    }

    /// Pull the next output row as u16. Returns `None` if more input is needed.
    ///
    /// Uses the configured transfer function to convert from linear f32 to encoded u16.
    pub fn next_output_row_u16(&mut self) -> Option<&[u16]> {
        debug_assert_eq!(
            self.path,
            StreamingPath::F32,
            "next_output_row_u16 requires f32 path"
        );

        if !self.has_padding {
            if !self.can_produce_next_output() {
                return None;
            }
            self.produce_next_f32();
            let row_len = self.config.out_width as usize * self.channels;
            Self::encode_output_u16(
                &self.temp_output_f32[..row_len],
                &mut self.output_buf_u16[..row_len],
                self.config.effective_output_transfer(),
                self.channels,
                self.alpha_is_last,
            );
            return Some(&self.output_buf_u16[..row_len]);
        }

        let total_row_len = self.config.total_output_row_len();

        // Phase 1: top padding
        if self.pad_top_emitted < self.pad_top {
            self.pad_top_emitted += 1;
            return Some(&self.pad_full_row_u16[..total_row_len]);
        }

        // Phase 2: content rows
        if self.output_rows_produced < self.config.out_height {
            if !self.can_produce_next_output() {
                return None;
            }
            self.produce_next_f32();
            let content_len = self.config.out_width as usize * self.channels;
            let content_start = self.pad_left_elements;
            Self::encode_output_u16(
                &self.temp_output_f32[..content_len],
                &mut self.padded_output_buf_u16[content_start..content_start + content_len],
                self.config.effective_output_transfer(),
                self.channels,
                self.alpha_is_last,
            );
            return Some(&self.padded_output_buf_u16[..total_row_len]);
        }

        // Phase 3: bottom padding
        if self.pad_bottom_emitted < self.pad_bottom {
            self.pad_bottom_emitted += 1;
            return Some(&self.pad_full_row_u16[..total_row_len]);
        }

        None
    }

    /// Pull the next output row directly into a caller-provided u16 buffer.
    /// Returns `true` if a row was produced, `false` if more input is needed.
    ///
    /// `dst` must be at least `output_row_len()` elements long.
    pub fn next_output_row_u16_into(&mut self, dst: &mut [u16]) -> bool {
        debug_assert_eq!(
            self.path,
            StreamingPath::F32,
            "next_output_row_u16_into requires f32 path"
        );

        if !self.has_padding {
            if !self.can_produce_next_output() {
                return false;
            }
            self.produce_next_f32();
            let row_len = self.config.out_width as usize * self.channels;
            Self::encode_output_u16(
                &self.temp_output_f32[..row_len],
                &mut dst[..row_len],
                self.config.effective_output_transfer(),
                self.channels,
                self.alpha_is_last,
            );
            return true;
        }

        let total_row_len = self.config.total_output_row_len();

        // Phase 1: top padding
        if self.pad_top_emitted < self.pad_top {
            self.pad_top_emitted += 1;
            dst[..total_row_len].copy_from_slice(&self.pad_full_row_u16[..total_row_len]);
            return true;
        }

        // Phase 2: content rows
        if self.output_rows_produced < self.config.out_height {
            if !self.can_produce_next_output() {
                return false;
            }
            let content_len = self.config.out_width as usize * self.channels;
            let content_start = self.pad_left_elements;
            // Fill padding
            dst[..content_start].copy_from_slice(&self.pad_full_row_u16[..content_start]);
            let right_start = content_start + content_len;
            dst[right_start..total_row_len]
                .copy_from_slice(&self.pad_full_row_u16[right_start..total_row_len]);
            // Produce content
            self.produce_next_f32();
            Self::encode_output_u16(
                &self.temp_output_f32[..content_len],
                &mut dst[content_start..content_start + content_len],
                self.config.effective_output_transfer(),
                self.channels,
                self.alpha_is_last,
            );
            return true;
        }

        // Phase 3: bottom padding
        if self.pad_bottom_emitted < self.pad_bottom {
            self.pad_bottom_emitted += 1;
            dst[..total_row_len].copy_from_slice(&self.pad_full_row_u16[..total_row_len]);
            return true;
        }

        false
    }

    /// Pull the next output row as premultiplied linear f32 — **before** compositing
    /// and unpremultiply. This is the building block for external two-stream compositing.
    ///
    /// Returns `None` if more input rows are needed. Does NOT apply the owned
    /// background or unpremultiply — the caller handles both.
    ///
    /// Requires the f32 path (panics on i16 paths). The f32 path is selected
    /// automatically for compositing-eligible configs (linear, 3ch, f32 I/O, u16 I/O),
    /// or can be forced with `.linear()` on the builder.
    ///
    /// Padding is NOT included in the returned slice — only resize content rows are
    /// returned. Use [`total_output_width()`](ResizeConfig::total_output_width) and
    /// the padding config to assemble the final output.
    ///
    /// # Two-stream compositing example
    ///
    /// ```ignore
    /// let mut fg = StreamingResize::new(&fg_config);
    /// let mut bg = StreamingResize::new(&bg_config);
    ///
    /// // Push rows to both resizers...
    ///
    /// // Pull and composite:
    /// let fg_row = fg.next_output_row_premul_linear_f32().unwrap();
    /// let mut composited = fg_row.to_vec();
    /// let bg_row = bg.next_output_row_premul_linear_f32().unwrap();
    /// composite_over_premul(&mut composited, bg_row, 4);
    /// unpremultiply_f32_row(&mut composited);
    /// // encode composited to u8...
    /// ```
    pub fn next_output_row_premul_linear_f32(&mut self) -> Option<&[f32]> {
        assert_eq!(
            self.path,
            StreamingPath::F32,
            "next_output_row_premul_linear_f32 requires f32 path (use .linear() on config)"
        );

        if !self.can_produce_next_output() {
            return None;
        }
        self.produce_next_f32_raw();
        let row_len = self.config.out_width as usize * self.channels;
        Some(&self.temp_output_f32[..row_len])
    }

    /// Pull the next output row, compositing the resized foreground over a
    /// caller-provided premultiplied linear f32 background row.
    ///
    /// This is the clean API for two-stream compositing: the caller manages the
    /// background stream independently and provides one row at a time.
    ///
    /// `bg_premul_linear` must have at least `out_width * channels` elements.
    /// The compositing uses Porter-Duff source-over in premultiplied linear space.
    ///
    /// Requires the f32 path.
    pub fn next_output_row_over(&mut self, bg_premul_linear: &[f32]) -> Option<&[u8]> {
        assert_eq!(
            self.path,
            StreamingPath::F32,
            "next_output_row_over requires f32 path"
        );

        if !self.can_produce_next_output() {
            return None;
        }
        self.produce_next_f32_raw();
        let row_len = self.config.out_width as usize * self.channels;

        // Composite fg over bg
        if self.channels == 4 {
            composite::composite_over_premul(
                &mut self.temp_output_f32[..row_len],
                &bg_premul_linear[..row_len],
                4,
            );
        }

        // Unpremultiply
        if self.needs_premul {
            simd::unpremultiply_alpha_row(&mut self.temp_output_f32[..row_len]);
        }

        // Encode to u8
        Self::encode_output_u8(
            &self.temp_output_f32[..row_len],
            &mut self.output_buf_u8[..row_len],
            self.config.effective_output_transfer(),
            self.channels,
            self.alpha_is_last,
        );
        Some(&self.output_buf_u8[..row_len])
    }

    // Non-padded content row helpers (factored out to avoid duplication)

    fn next_content_row_u8(&mut self) -> Option<&[u8]> {
        if !self.can_produce_next_output() {
            return None;
        }
        let row_len = self.config.out_width as usize * self.channels;
        match self.path {
            StreamingPath::I16Srgb | StreamingPath::I16Linear => {
                let mut tmp = core::mem::take(&mut self.output_buf_u8);
                match self.path {
                    StreamingPath::I16Srgb => self.produce_next_i16_srgb(&mut tmp[..row_len]),
                    StreamingPath::I16Linear => self.produce_next_i16_linear(&mut tmp[..row_len]),
                    StreamingPath::F32 => unreachable!(),
                }
                self.output_buf_u8 = tmp;
            }
            StreamingPath::F32 => {
                self.produce_next_f32();
                Self::encode_output_u8(
                    &self.temp_output_f32[..row_len],
                    &mut self.output_buf_u8[..row_len],
                    self.config.effective_output_transfer(),
                    self.channels,
                    self.alpha_is_last,
                );
            }
        }
        Some(&self.output_buf_u8[..row_len])
    }

    fn next_content_row_u8_into(&mut self, dst: &mut [u8]) -> bool {
        if !self.can_produce_next_output() {
            return false;
        }
        let row_len = self.config.out_width as usize * self.channels;
        match self.path {
            StreamingPath::I16Srgb => self.produce_next_i16_srgb(&mut dst[..row_len]),
            StreamingPath::I16Linear => self.produce_next_i16_linear(&mut dst[..row_len]),
            StreamingPath::F32 => {
                self.produce_next_f32();
                Self::encode_output_u8(
                    &self.temp_output_f32[..row_len],
                    &mut dst[..row_len],
                    self.config.effective_output_transfer(),
                    self.channels,
                    self.alpha_is_last,
                );
            }
        }
        true
    }

    // =========================================================================
    // Internal
    // =========================================================================

    /// Encode linear f32 to u8 using the specified transfer function.
    fn encode_output_u8(
        src: &[f32],
        dst: &mut [u8],
        tf: TransferFunction,
        channels: usize,
        alpha_is_last: bool,
    ) {
        match tf {
            TransferFunction::Srgb => {
                color::linear_f32_to_srgb_u8(src, dst, channels, alpha_is_last)
            }
            TransferFunction::Linear => simd::f32_to_u8_row(src, dst),
            TransferFunction::Bt709 => {
                Bt709.linear_f32_to_u8(src, dst, &(), channels, alpha_is_last, false)
            }
            TransferFunction::Pq => {
                Pq.linear_f32_to_u8(src, dst, &(), channels, alpha_is_last, false)
            }
            TransferFunction::Hlg => {
                Hlg.linear_f32_to_u8(src, dst, &(), channels, alpha_is_last, false)
            }
            _ => simd::f32_to_u8_row(src, dst),
        }
    }

    /// Encode linear f32 to u16 using the specified transfer function.
    fn encode_output_u16(
        src: &[f32],
        dst: &mut [u16],
        tf: TransferFunction,
        channels: usize,
        alpha_is_last: bool,
    ) {
        match tf {
            TransferFunction::Srgb => {
                Srgb.linear_f32_to_u16(src, dst, &(), channels, alpha_is_last, false)
            }
            TransferFunction::Linear => crate::transfer::NoTransfer.linear_f32_to_u16(
                src,
                dst,
                &(),
                channels,
                alpha_is_last,
                false,
            ),
            TransferFunction::Bt709 => {
                Bt709.linear_f32_to_u16(src, dst, &(), channels, alpha_is_last, false)
            }
            TransferFunction::Pq => {
                Pq.linear_f32_to_u16(src, dst, &(), channels, alpha_is_last, false)
            }
            TransferFunction::Hlg => {
                Hlg.linear_f32_to_u16(src, dst, &(), channels, alpha_is_last, false)
            }
            _ => crate::transfer::NoTransfer.linear_f32_to_u16(
                src,
                dst,
                &(),
                channels,
                alpha_is_last,
                false,
            ),
        }
    }

    fn first_output_row_max_input(&self) -> u32 {
        if self.v_weights.is_empty() {
            return 0;
        }
        let left = self.v_weights.left[0];
        let taps = self.v_weights.tap_count(0);
        let right = left + taps as i32 - 1;
        right.max(0) as u32
    }

    /// Check if the next output row can be produced (enough input rows available).
    fn can_produce_next_output(&self) -> bool {
        let out_y = self.output_rows_produced;
        if out_y >= self.config.out_height {
            return false;
        }

        let left = self.v_weights.left[out_y as usize];
        let tap_count = self.v_weights.tap_count(out_y as usize);
        let right = left + tap_count as i32 - 1;

        let resize_in_h = self.config.resize_in_height();
        let needed_max = right.min(resize_in_h as i32 - 1).max(0) as u32;
        needed_max < self.input_rows_received
    }

    /// V-filter → H-filter one output row into `temp_output_f32` (premultiplied linear).
    /// Does NOT apply composite or unpremultiply. Increments `output_rows_produced`.
    /// Caller must check `can_produce_next_output` first.
    fn produce_next_f32_raw(&mut self) {
        let out_y = self.output_rows_produced;

        let left = self.v_weights.left[out_y as usize];
        let tap_count = self.v_weights.tap_count(out_y as usize);
        let weights = self.v_weights.weights(out_y as usize);
        let resize_in_h = self.config.resize_in_height();
        let in_row_len = self.config.resize_in_width() as usize * self.channels;
        let out_row_len = self.config.out_width as usize * self.channels;

        // Step 1: V-filter from f16 v_cache into temp_v_output (f32, in_width-wide)
        with_v_rows(
            &self.v_cache,
            self.cache_size,
            left,
            tap_count,
            resize_in_h,
            in_row_len,
            |rows| simd::filter_v_row_f16(rows, &mut self.temp_v_output[..in_row_len], weights),
        );

        // Step 2: H-filter from temp_v_output into temp_output_f32 (out_width-wide)
        // temp_v_output has h_padding zeros beyond in_row_len for SIMD safety
        let h_weights = self.h_weights.as_ref().unwrap();
        simd::filter_h_row_f32(
            &self.temp_v_output,
            &mut self.temp_output_f32[..out_row_len],
            h_weights,
            self.channels,
        );

        self.output_rows_produced += 1;
    }

    /// V-filter → H-filter → composite → unpremultiply one output row.
    /// Result in `temp_output_f32`. Increments `output_rows_produced`.
    fn produce_next_f32(&mut self) {
        self.produce_next_f32_raw();

        let out_y = self.output_rows_produced - 1; // raw already incremented
        let out_row_len = self.config.out_width as usize * self.channels;

        // composite + unpremul (operates on temp_output_f32)
        composite::composite_dispatch(
            &mut self.temp_output_f32[..out_row_len],
            &mut self.background,
            &mut self.composite_bg_row,
            out_y,
            self.channels as u8,
            self.blend_mode,
        );

        if self.needs_premul {
            simd::unpremultiply_alpha_row(&mut self.temp_output_f32[..out_row_len]);
        }
    }

    /// Produce one output row via the I16Srgb H-first path.
    /// Ring buffer already contains H-filtered i16 rows (out_width wide).
    /// V-filter directly produces the output — no H-filter step needed.
    ///
    /// Paired-row optimization: when consecutive output rows share the same
    /// V-filter window, both are V-filtered back-to-back (L1-hot data).
    fn produce_next_i16_srgb(&mut self, dst: &mut [u8]) {
        let out_y = self.output_rows_produced as usize;
        let out_h = self.config.out_height as usize;
        let i16_v_weights = self.i16_v_weights.as_ref().unwrap();

        if self.paired_row_ready {
            let out_row_len = self.config.out_width as usize * self.channels;
            dst[..out_row_len].copy_from_slice(&self.paired_row_buf[..out_row_len]);
            self.paired_row_ready = false;
            self.output_rows_produced += 1;
            return;
        }

        let left = i16_v_weights.left[out_y];
        let tap_count = i16_v_weights.tap_count(out_y);
        let weights_a = i16_v_weights.weights(out_y);
        let resize_in_h = self.config.resize_in_height();
        let out_row_len = self.config.out_width as usize * self.channels;

        let can_pair = out_y + 1 < out_h
            && i16_v_weights.left[out_y + 1] == left
            && i16_v_weights.tap_count(out_y + 1) == tap_count;

        // V-filter row A from ring buffer (out_width wide) → temp i16
        with_v_rows(
            &self.i16_v_cache,
            self.cache_size,
            left,
            tap_count,
            resize_in_h,
            out_row_len,
            |rows| {
                simd::filter_v_row_i16(
                    rows,
                    &mut self.temp_v_output_i16[..out_row_len],
                    weights_a,
                );
            },
        );

        // Clamp i16 → u8 [0,255]
        for (&v, o) in self.temp_v_output_i16[..out_row_len]
            .iter()
            .zip(dst[..out_row_len].iter_mut())
        {
            *o = v.clamp(0, 255) as u8;
        }
        if self.needs_premul {
            simd::unpremultiply_u8_row(&mut dst[..out_row_len]);
        }
        self.output_rows_produced += 1;

        // Paired row B: same V-filter rows (L1-hot), different weights
        if can_pair {
            let weights_b = i16_v_weights.weights(out_y + 1);
            with_v_rows(
                &self.i16_v_cache,
                self.cache_size,
                left,
                tap_count,
                resize_in_h,
                out_row_len,
                |rows| {
                    simd::filter_v_row_i16(
                        rows,
                        &mut self.temp_v_output_i16[..out_row_len],
                        weights_b,
                    );
                },
            );

            for (&v, o) in self.temp_v_output_i16[..out_row_len]
                .iter()
                .zip(self.paired_row_buf[..out_row_len].iter_mut())
            {
                *o = v.clamp(0, 255) as u8;
            }
            if self.needs_premul {
                simd::unpremultiply_u8_row(&mut self.paired_row_buf[..out_row_len]);
            }
            self.paired_row_ready = true;
        }
    }

    /// Produce one output row via the I16Linear H-first path.
    /// Ring buffer contains H-filtered linear i16 rows (out_width wide).
    /// V-filter directly produces output — no H-filter step needed.
    fn produce_next_i16_linear(&mut self, dst: &mut [u8]) {
        let out_y = self.output_rows_produced as usize;
        let out_h = self.config.out_height as usize;
        let i16_v_weights = self.i16_v_weights.as_ref().unwrap();

        if self.paired_row_ready {
            let out_row_len = self.config.out_width as usize * self.channels;
            dst[..out_row_len].copy_from_slice(&self.paired_row_buf[..out_row_len]);
            self.paired_row_ready = false;
            self.output_rows_produced += 1;
            return;
        }

        let left = i16_v_weights.left[out_y];
        let tap_count = i16_v_weights.tap_count(out_y);
        let weights_a = i16_v_weights.weights(out_y);
        let resize_in_h = self.config.resize_in_height();
        let out_row_len = self.config.out_width as usize * self.channels;

        let can_pair = out_y + 1 < out_h
            && i16_v_weights.left[out_y + 1] == left
            && i16_v_weights.tap_count(out_y + 1) == tap_count;

        // V-filter row A from ring buffer (out_width wide) → temp i16
        with_v_rows(
            &self.i16_v_cache,
            self.cache_size,
            left,
            tap_count,
            resize_in_h,
            out_row_len,
            |rows| {
                simd::filter_v_row_i16(
                    rows,
                    &mut self.temp_v_output_i16[..out_row_len],
                    weights_a,
                );
            },
        );

        // Linear i12 → sRGB u8 (LUT clamps to [0,4095])
        color::linear_i12_to_srgb_u8_row(
            &self.temp_v_output_i16[..out_row_len],
            &mut dst[..out_row_len],
        );
        self.output_rows_produced += 1;

        // Paired row B
        if can_pair {
            let weights_b = i16_v_weights.weights(out_y + 1);
            with_v_rows(
                &self.i16_v_cache,
                self.cache_size,
                left,
                tap_count,
                resize_in_h,
                out_row_len,
                |rows| {
                    simd::filter_v_row_i16(
                        rows,
                        &mut self.temp_v_output_i16[..out_row_len],
                        weights_b,
                    );
                },
            );

            color::linear_i12_to_srgb_u8_row(
                &self.temp_v_output_i16[..out_row_len],
                &mut self.paired_row_buf[..out_row_len],
            );
            self.paired_row_ready = true;
        }
    }

    /// Check ring buffer overflow before any cache write.
    #[track_caller]
    fn check_ring_buffer(&self) -> Result<(), At<StreamingError>> {
        if self.output_rows_produced < self.config.out_height {
            let oldest_needed =
                self.v_weights.left[self.output_rows_produced as usize].max(0) as usize;
            if self.cache_write_idx > oldest_needed
                && self.cache_write_idx - oldest_needed >= self.cache_size
            {
                return Err(at!(StreamingError::RingBufferOverflow));
            }
        }
        Ok(())
    }

    /// Internal: convert f32 → f16 and cache the row into the f16 ring buffer.
    #[track_caller]
    fn push_row_internal(&mut self) -> Result<(), At<StreamingError>> {
        self.check_ring_buffer().at()?;

        let cache_slot = self.cache_write_idx % self.cache_size;
        let pixel_len = self.config.resize_in_width() as usize * self.channels;
        simd::f32_to_f16_row(
            &self.temp_input_f32[..pixel_len],
            &mut self.v_cache[cache_slot][..pixel_len],
        );

        self.cache_write_idx += 1;
        self.input_rows_received += 1;
        self.eagerly_capture_orient_rows();
        Ok(())
    }

    /// Internal: cache i16 row from `linearized_row_i16` into the i16 ring buffer (I16Linear path).
    #[track_caller]
    fn push_row_internal_i16(&mut self) -> Result<(), At<StreamingError>> {
        self.check_ring_buffer().at()?;

        let cache_slot = self.cache_write_idx % self.cache_size;
        let pixel_len = self.config.resize_in_width() as usize * self.channels;
        self.i16_v_cache[cache_slot][..pixel_len]
            .copy_from_slice(&self.linearized_row_i16[..pixel_len]);

        self.cache_write_idx += 1;
        self.input_rows_received += 1;
        self.eagerly_capture_orient_rows();
        Ok(())
    }

    /// Eagerly produce all available resize output rows into the orientation buffer.
    ///
    /// Only does work for non-row-local orientations. For Identity/FlipH, this is a no-op.
    /// This keeps the ring buffer from overflowing by consuming output rows as they
    /// become available, even though the caller won't see them until after `finish()`.
    fn eagerly_capture_orient_rows(&mut self) {
        if self.orient.is_row_local() {
            return;
        }
        let row_len = self.config.total_output_row_len();
        loop {
            let dst_offset = self.orient_captured as usize * row_len;
            if dst_offset + row_len > self.orient_buf.len() {
                break;
            }
            if !self.capture_one_orient_row(dst_offset, row_len) {
                break;
            }
            self.orient_captured += 1;
        }
    }

    /// Produce one unoriented output row and copy it into `orient_buf[dst_offset..]`.
    ///
    /// Returns `true` if a row was produced, `false` if no more available.
    /// Uses `next_output_row_unoriented()` → copy to avoid borrow aliasing.
    fn capture_one_orient_row(&mut self, dst_offset: usize, row_len: usize) -> bool {
        // Phase 1: top padding
        if self.pad_top_emitted < self.pad_top {
            self.pad_top_emitted += 1;
            self.orient_buf[dst_offset..dst_offset + row_len]
                .copy_from_slice(&self.pad_full_row_u8[..row_len]);
            return true;
        }

        // Phase 2: content rows
        if self.output_rows_produced < self.config.out_height && self.can_produce_next_output() {
            // Produce into output_buf_u8, then copy to orient_buf
            self.produce_content_row_to_output_buf(row_len);
            self.orient_buf[dst_offset..dst_offset + row_len]
                .copy_from_slice(&self.output_buf_u8[..row_len]);
            return true;
        }

        // Phase 3: bottom padding
        if self.pad_bottom_emitted < self.pad_bottom {
            self.pad_bottom_emitted += 1;
            self.orient_buf[dst_offset..dst_offset + row_len]
                .copy_from_slice(&self.pad_full_row_u8[..row_len]);
            return true;
        }

        false
    }

    /// Produce one content row (with optional padding) into `output_buf_u8`.
    fn produce_content_row_to_output_buf(&mut self, total_row_len: usize) {
        if self.has_padding {
            let content_len = self.config.out_width as usize * self.channels;
            let content_start = self.pad_left_elements;
            // Ensure output_buf_u8 is large enough for padded row
            if self.output_buf_u8.len() < total_row_len {
                self.output_buf_u8.resize(total_row_len, 0);
            }
            // Copy padding template
            self.output_buf_u8[..total_row_len]
                .copy_from_slice(&self.padded_output_buf_u8[..total_row_len]);
            // Produce content into the content region
            let mut tmp = core::mem::take(&mut self.output_buf_u8);
            match self.path {
                StreamingPath::I16Srgb => {
                    self.produce_next_i16_srgb(
                        &mut tmp[content_start..content_start + content_len],
                    );
                }
                StreamingPath::I16Linear => {
                    self.produce_next_i16_linear(
                        &mut tmp[content_start..content_start + content_len],
                    );
                }
                StreamingPath::F32 => {
                    self.produce_next_f32();
                    Self::encode_output_u8(
                        &self.temp_output_f32[..content_len],
                        &mut tmp[content_start..content_start + content_len],
                        self.config.effective_output_transfer(),
                        self.channels,
                        self.alpha_is_last,
                    );
                }
            }
            self.output_buf_u8 = tmp;
        } else {
            // No padding — produce content directly
            let mut tmp = core::mem::take(&mut self.output_buf_u8);
            match self.path {
                StreamingPath::I16Srgb => self.produce_next_i16_srgb(&mut tmp[..total_row_len]),
                StreamingPath::I16Linear => self.produce_next_i16_linear(&mut tmp[..total_row_len]),
                StreamingPath::F32 => {
                    self.produce_next_f32();
                    Self::encode_output_u8(
                        &self.temp_output_f32[..total_row_len],
                        &mut tmp[..total_row_len],
                        self.config.effective_output_transfer(),
                        self.channels,
                        self.alpha_is_last,
                    );
                }
            }
            self.output_buf_u8 = tmp;
        }
    }

    /// Produce the next output row WITHOUT orientation — raw resize + padding output.
    ///
    /// This is the core output pipeline that `next_output_row` delegates to.
    /// Returns `None` if no row is available (not enough input or all done).
    fn next_output_row_unoriented(&mut self) -> Option<&[u8]> {
        if !self.has_padding {
            return self.next_content_row_u8();
        }
        let total_row_len = self.config.total_output_row_len();

        // Phase 1: top padding
        if self.pad_top_emitted < self.pad_top {
            self.pad_top_emitted += 1;
            return Some(&self.pad_full_row_u8[..total_row_len]);
        }

        // Phase 2: content rows with left/right padding
        if self.output_rows_produced < self.config.out_height {
            if !self.can_produce_next_output() {
                return None;
            }
            let content_len = self.config.out_width as usize * self.channels;
            let content_start = self.pad_left_elements;
            let mut tmp = core::mem::take(&mut self.padded_output_buf_u8);
            match self.path {
                StreamingPath::I16Srgb => {
                    self.produce_next_i16_srgb(
                        &mut tmp[content_start..content_start + content_len],
                    );
                }
                StreamingPath::I16Linear => {
                    self.produce_next_i16_linear(
                        &mut tmp[content_start..content_start + content_len],
                    );
                }
                StreamingPath::F32 => {
                    self.produce_next_f32();
                    Self::encode_output_u8(
                        &self.temp_output_f32[..content_len],
                        &mut tmp[content_start..content_start + content_len],
                        self.config.effective_output_transfer(),
                        self.channels,
                        self.alpha_is_last,
                    );
                }
            }
            self.padded_output_buf_u8 = tmp;
            return Some(&self.padded_output_buf_u8[..total_row_len]);
        }

        // Phase 3: bottom padding
        if self.pad_bottom_emitted < self.pad_bottom {
            self.pad_bottom_emitted += 1;
            return Some(&self.pad_full_row_u8[..total_row_len]);
        }

        None
    }

    /// Accessor for the current orientation.
    pub fn orientation(&self) -> OrientOutput {
        self.orient
    }

    /// Apply the orientation transform to the buffered resize output.
    ///
    /// Called by `finish()` after all resize rows have been captured.
    fn apply_orient_transform(&mut self) {
        if self.orient.is_row_local() || self.orient_ready {
            return;
        }
        let src_h = self.config.total_output_height();
        let src_w = self.config.total_output_width();
        let ch = self.channels;
        let src_stride = src_w as usize * ch;
        let dst_w = self.oriented_w;
        let dst_stride = dst_w as usize * ch;

        // For non-row-local orientations, scatter source pixels to destination.
        // We need a separate destination buffer to avoid aliasing.
        let mut dst = vec![0u8; dst_stride * self.oriented_h as usize];

        for sy in 0..src_h {
            for sx in 0..src_w {
                let (dx, dy) = self.orient.forward_map(sx, sy, src_w, src_h);
                let src_off = sy as usize * src_stride + sx as usize * ch;
                let dst_off = dy as usize * dst_stride + dx as usize * ch;
                dst[dst_off..dst_off + ch].copy_from_slice(&self.orient_buf[src_off..src_off + ch]);
            }
        }

        self.orient_buf = dst;
        self.orient_ready = true;
    }
}

/// Reverse the pixel order of a u8 row in-place.
fn flip_h_row(row: &mut [u8], channels: usize) {
    let pixel_count = row.len() / channels;
    for i in 0..pixel_count / 2 {
        let j = pixel_count - 1 - i;
        let a = i * channels;
        let b = j * channels;
        for c in 0..channels {
            row.swap(a + c, b + c);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composite::SolidBackground;
    use crate::filter::Filter;
    use zenpixels::{AlphaMode, PixelDescriptor, TransferFunction};

    fn make_config(in_w: u32, in_h: u32, out_w: u32, out_h: u32) -> ResizeConfig {
        ResizeConfig::builder(in_w, in_h, out_w, out_h)
            .format(PixelDescriptor::RGBA8_SRGB)
            .srgb()
            .build()
    }

    /// Helper: push all input rows with interleaved drain, collect u8 output.
    fn push_drain_collect_u8(
        resizer: &mut StreamingResize<impl Background>,
        input_row: &[u8],
        in_h: u32,
    ) -> Vec<Vec<u8>> {
        let mut rows = Vec::new();
        for _ in 0..in_h {
            resizer.push_row(input_row).unwrap();
            while let Some(out) = resizer.next_output_row() {
                rows.push(out.to_vec());
            }
        }
        resizer.finish();
        while let Some(out) = resizer.next_output_row() {
            rows.push(out.to_vec());
        }
        rows
    }

    #[test]
    fn test_streaming_produces_correct_row_count() {
        let config = make_config(100, 100, 50, 50);
        let mut resizer = StreamingResize::new(&config);

        let row = vec![128u8; 100 * 4];
        let mut output_count = 0;
        for _ in 0..100 {
            resizer.push_row(&row).unwrap();
            while resizer.next_output_row().is_some() {
                output_count += 1;
            }
        }
        resizer.finish();
        while resizer.next_output_row().is_some() {
            output_count += 1;
        }

        assert_eq!(output_count, 50);
        assert_eq!(resizer.output_rows_produced(), 50);
    }

    #[test]
    fn test_streaming_constant_color() {
        let config = make_config(20, 20, 10, 10);
        let mut resizer = StreamingResize::new(&config);

        let mut row = vec![0u8; 20 * 4];
        for pixel in row.chunks_exact_mut(4) {
            pixel[0] = 128;
            pixel[1] = 128;
            pixel[2] = 128;
            pixel[3] = 255;
        }

        let mut total_rows = 0;
        for _ in 0..20 {
            resizer.push_row(&row).unwrap();
            while let Some(out_row) = resizer.next_output_row() {
                total_rows += 1;
                for pixel in out_row.chunks_exact(4) {
                    assert!(
                        (pixel[0] as i16 - 128).unsigned_abs() <= 2,
                        "R channel off: {}",
                        pixel[0]
                    );
                    assert!(
                        (pixel[3] as i16 - 255).unsigned_abs() <= 1,
                        "A channel off: {}",
                        pixel[3]
                    );
                }
            }
        }
        resizer.finish();
        while let Some(out_row) = resizer.next_output_row() {
            total_rows += 1;
            for pixel in out_row.chunks_exact(4) {
                assert!(
                    (pixel[0] as i16 - 128).unsigned_abs() <= 2,
                    "R channel off: {}",
                    pixel[0]
                );
                assert!(
                    (pixel[3] as i16 - 255).unsigned_abs() <= 1,
                    "A channel off: {}",
                    pixel[3]
                );
            }
        }
        assert_eq!(total_rows, 10);
    }

    #[test]
    fn test_streaming_upscale() {
        let config = make_config(10, 10, 20, 20);
        let mut resizer = StreamingResize::new(&config);

        let row = vec![200u8; 10 * 4];
        let mut count = 0;
        for _ in 0..10 {
            resizer.push_row(&row).unwrap();
            while resizer.next_output_row().is_some() {
                count += 1;
            }
        }
        resizer.finish();
        while resizer.next_output_row().is_some() {
            count += 1;
        }

        assert_eq!(count, 20);
        assert_eq!(resizer.output_rows_produced(), 20);
    }

    #[test]
    fn test_streaming_same_size() {
        let config = make_config(10, 10, 10, 10);
        let mut resizer = StreamingResize::new(&config);

        let row = vec![100u8; 10 * 4];
        let mut count = 0;
        for _ in 0..10 {
            resizer.push_row(&row).unwrap();
            while resizer.next_output_row().is_some() {
                count += 1;
            }
        }
        resizer.finish();
        while resizer.next_output_row().is_some() {
            count += 1;
        }

        assert_eq!(count, 10);
        assert_eq!(resizer.output_rows_produced(), 10);
    }

    #[test]
    fn test_builder_defaults() {
        let config = ResizeConfig::builder(100, 100, 50, 50).build();

        assert_eq!(config.filter, Filter::default());
        assert!(config.linear);
        assert_eq!(config.post_sharpen, 0.0);
        assert_eq!(config.in_stride, 0);
        assert_eq!(config.out_stride, 0);
    }

    #[test]
    fn test_input_output_row_len() {
        let config = ResizeConfig::builder(100, 100, 50, 50)
            .format(PixelDescriptor::RGBA8_SRGB)
            .build();
        let resizer = StreamingResize::new(&config);
        assert_eq!(resizer.input_row_len(), 100 * 4);
        assert_eq!(resizer.output_row_len(), 50 * 4);
    }

    #[test]
    fn test_next_output_row_into() {
        let config = make_config(20, 20, 10, 10);
        let mut resizer = StreamingResize::new(&config);

        let mut row = vec![0u8; 20 * 4];
        for pixel in row.chunks_exact_mut(4) {
            pixel[0] = 128;
            pixel[1] = 128;
            pixel[2] = 128;
            pixel[3] = 255;
        }

        let row_len = resizer.output_row_len();
        let mut output_buf = vec![0u8; row_len];
        let mut total_rows = 0;

        for _ in 0..20 {
            resizer.push_row(&row).unwrap();
            while resizer.next_output_row_into(&mut output_buf) {
                total_rows += 1;
                for pixel in output_buf.chunks_exact(4) {
                    assert!(
                        (pixel[0] as i16 - 128).unsigned_abs() <= 2,
                        "R channel off: {}",
                        pixel[0]
                    );
                }
            }
        }
        resizer.finish();
        while resizer.next_output_row_into(&mut output_buf) {
            total_rows += 1;
        }
        assert_eq!(total_rows, 10);
    }

    #[test]
    fn test_push_row_f32_with() {
        let config = ResizeConfig::builder(10, 10, 5, 5)
            .format(PixelDescriptor::RGBAF32_LINEAR)
            .build();
        let mut resizer = StreamingResize::new(&config);

        let mut count = 0;
        for _ in 0..10 {
            resizer
                .push_row_f32_with(|buf| {
                    for pixel in buf.chunks_exact_mut(4) {
                        pixel[0] = 0.5;
                        pixel[1] = 0.5;
                        pixel[2] = 0.5;
                        pixel[3] = 1.0;
                    }
                })
                .unwrap();
            while resizer.next_output_row_f32().is_some() {
                count += 1;
            }
        }
        resizer.finish();
        while resizer.next_output_row_f32().is_some() {
            count += 1;
        }
        assert_eq!(count, 5);
    }

    #[test]
    fn test_push_after_finish_returns_error() {
        let config = make_config(10, 10, 5, 5);
        let mut resizer = StreamingResize::new(&config);
        resizer.finish();

        let row = vec![128u8; 10 * 4];
        assert_eq!(
            resizer.push_row(&row).unwrap_err().into_inner(),
            StreamingError::AlreadyFinished
        );
    }

    #[test]
    fn test_input_too_short_returns_error() {
        let config = make_config(10, 10, 5, 5);
        let mut resizer = StreamingResize::new(&config);

        let short_row = vec![128u8; 10]; // needs 10*4=40
        assert_eq!(
            resizer.push_row(&short_row).unwrap_err().into_inner(),
            StreamingError::InputTooShort
        );
    }

    #[test]
    fn test_push_rows_batch_matches_single() {
        // Use with_batch_hint(8) to size the ring buffer for batch=8 pushes.
        let (in_w, in_h, out_w, out_h) = (400, 400, 100, 100);
        let config = make_config(in_w, in_h, out_w, out_h);
        let row = vec![128u8; in_w as usize * 4];

        // Collect output using single-row API
        let mut r1 = StreamingResize::new(&config);
        let single_output = push_drain_collect_u8(&mut r1, &row, in_h);

        // Collect output using batch API (push 8 rows at a time, like zenjpeg)
        let mut r2 = StreamingResize::with_batch_hint(&config, 8);
        let stride = in_w as usize * 4;
        let batch = 8usize;
        let buf: Vec<u8> = row.iter().copied().cycle().take(stride * batch).collect();
        let mut batch_output = Vec::new();
        for chunk_start in (0..in_h).step_by(batch) {
            let count = batch.min((in_h - chunk_start) as usize) as u32;
            let available = r2
                .push_rows(&buf[..stride * count as usize], stride, count)
                .unwrap();
            for _ in 0..available {
                batch_output.push(r2.next_output_row().unwrap().to_vec());
            }
        }
        let remaining = r2.finish();
        for _ in 0..remaining {
            batch_output.push(r2.next_output_row().unwrap().to_vec());
        }

        assert_eq!(single_output.len(), batch_output.len());
        assert_eq!(single_output.len(), out_h as usize);
        for (a, b) in single_output.iter().zip(batch_output.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn test_output_rows_available_accurate() {
        let config = make_config(20, 20, 10, 10);
        let mut resizer = StreamingResize::new(&config);
        let row = vec![128u8; 20 * 4];

        // Initially no output available
        assert_eq!(resizer.output_rows_available(), 0);

        // Push rows until some output becomes available
        let mut first_available = 0u32;
        for _ in 0..20 {
            resizer.push_row(&row).unwrap();
            let avail = resizer.output_rows_available();
            if avail > 0 && first_available == 0 {
                first_available = avail;
                // The count should match what next_output_row actually produces
                let mut actual = 0;
                while resizer.next_output_row().is_some() {
                    actual += 1;
                }
                assert_eq!(first_available, actual);
                break;
            }
        }
        assert!(first_available > 0, "should have produced output");
    }

    #[test]
    fn test_finish_returns_remaining_count() {
        let config = make_config(20, 20, 10, 10);
        let mut resizer = StreamingResize::new(&config);
        let row = vec![128u8; 20 * 4];

        let mut produced = 0u32;
        for _ in 0..20 {
            resizer.push_row(&row).unwrap();
            while resizer.next_output_row().is_some() {
                produced += 1;
            }
        }

        let remaining = resizer.finish();
        let mut after_finish = 0u32;
        while resizer.next_output_row().is_some() {
            after_finish += 1;
        }

        assert_eq!(remaining, after_finish);
        assert_eq!(produced + after_finish, 10);
    }

    // === Composite tests ===

    #[test]
    fn no_background_matches_new() {
        let config = make_config(20, 20, 10, 10);
        let mut row = vec![0u8; 20 * 4];
        for pixel in row.chunks_exact_mut(4) {
            pixel[0] = 128;
            pixel[1] = 64;
            pixel[2] = 32;
            pixel[3] = 200;
        }

        // Path A: new()
        let mut r1 = StreamingResize::new(&config);
        let rows1 = push_drain_collect_u8(&mut r1, &row, 20);

        // Path B: with_background(NoBackground)
        let mut r2 = StreamingResize::with_background(&config, NoBackground).unwrap();
        let rows2 = push_drain_collect_u8(&mut r2, &row, 20);

        assert_eq!(rows1, rows2);
    }

    #[test]
    fn transparent_background_matches_no_background() {
        let config = make_config(20, 20, 10, 10);
        let mut row = vec![0u8; 20 * 4];
        for pixel in row.chunks_exact_mut(4) {
            pixel[0] = 100;
            pixel[1] = 150;
            pixel[2] = 200;
            pixel[3] = 180;
        }

        let mut r1 = StreamingResize::new(&config);
        let rows1 = push_drain_collect_u8(&mut r1, &row, 20);

        let bg = SolidBackground::transparent(PixelDescriptor::RGBA8_SRGB);
        let mut r2 = StreamingResize::with_background(&config, bg).unwrap();
        let rows2 = push_drain_collect_u8(&mut r2, &row, 20);

        assert_eq!(rows1, rows2);
    }

    #[test]
    fn solid_opaque_bg_makes_output_opaque() {
        let config = ResizeConfig::builder(10, 10, 10, 10)
            .format(PixelDescriptor::RGBA8_SRGB)
            .linear()
            .build();

        // Semi-transparent input
        let mut row = vec![0u8; 10 * 4];
        for pixel in row.chunks_exact_mut(4) {
            pixel[0] = 128;
            pixel[1] = 64;
            pixel[2] = 32;
            pixel[3] = 128; // 50% alpha
        }

        let bg = SolidBackground::white(PixelDescriptor::RGBA8_SRGB);
        let mut resizer = StreamingResize::with_background(&config, bg).unwrap();

        for _ in 0..10 {
            resizer.push_row(&row).unwrap();
            while let Some(out_row) = resizer.next_output_row() {
                for pixel in out_row.chunks_exact(4) {
                    assert_eq!(pixel[3], 255, "output alpha must be 255 with opaque bg");
                    assert!(pixel[0] > 0, "R should have content");
                }
            }
        }
        resizer.finish();
        while let Some(out_row) = resizer.next_output_row() {
            for pixel in out_row.chunks_exact(4) {
                assert_eq!(pixel[3], 255, "output alpha must be 255 with opaque bg");
                assert!(pixel[0] > 0, "R should have content");
            }
        }
    }

    #[test]
    fn rejects_premultiplied_input() {
        let config = ResizeConfig::builder(10, 10, 5, 5)
            .format(PixelDescriptor::RGBA8_SRGB.with_alpha(Some(AlphaMode::Premultiplied)))
            .build();

        let bg = SolidBackground::white(
            PixelDescriptor::RGBA8_SRGB.with_alpha(Some(AlphaMode::Premultiplied)),
        );
        let result = StreamingResize::with_background(&config, bg);
        assert!(
            matches!(
                result.as_ref().map_err(|e| e.error()),
                Err(&CompositeError::PremultipliedInput)
            ),
            "expected PremultipliedInput error"
        );
    }

    // === u16 tests ===

    #[test]
    fn test_streaming_u16_constant_color() {
        let config = ResizeConfig::builder(20, 20, 10, 10)
            .filter(Filter::Lanczos)
            .format(PixelDescriptor::RGBA16_SRGB)
            .build();

        let mut resizer = StreamingResize::new(&config);

        let mut row = vec![0u16; 20 * 4];
        for pixel in row.chunks_exact_mut(4) {
            pixel[0] = 32768;
            pixel[1] = 32768;
            pixel[2] = 32768;
            pixel[3] = 65535;
        }

        let mut total_rows = 0;
        for _ in 0..20 {
            resizer.push_row_u16(&row).unwrap();
            while let Some(out_row) = resizer.next_output_row_u16() {
                total_rows += 1;
                for pixel in out_row.chunks_exact(4) {
                    assert!(
                        (pixel[0] as i32 - 32768).unsigned_abs() <= 100,
                        "R off: {} (expected ~32768)",
                        pixel[0]
                    );
                    assert!(
                        (pixel[3] as i32 - 65535).unsigned_abs() <= 1,
                        "A off: {} (expected 65535)",
                        pixel[3]
                    );
                }
            }
        }
        resizer.finish();
        while let Some(out_row) = resizer.next_output_row_u16() {
            total_rows += 1;
            for pixel in out_row.chunks_exact(4) {
                assert!(
                    (pixel[0] as i32 - 32768).unsigned_abs() <= 100,
                    "R off: {} (expected ~32768)",
                    pixel[0]
                );
                assert!(
                    (pixel[3] as i32 - 65535).unsigned_abs() <= 1,
                    "A off: {} (expected 65535)",
                    pixel[3]
                );
            }
        }
        assert_eq!(total_rows, 10);
    }

    #[test]
    fn test_streaming_u16_matches_fullframe() {
        let config = ResizeConfig::builder(40, 40, 20, 20)
            .filter(Filter::Lanczos)
            .format(PixelDescriptor::RGBA16_SRGB)
            .build();

        // Gradient input
        let mut input = vec![0u16; 40 * 40 * 4];
        for y in 0..40 {
            for x in 0..40 {
                let idx = (y * 40 + x) * 4;
                input[idx] = ((x * 65535) / 39) as u16;
                input[idx + 1] = ((y * 65535) / 39) as u16;
                input[idx + 2] = 32768;
                input[idx + 3] = 65535;
            }
        }

        // Fullframe
        let fullframe = {
            use crate::resize::Resizer;
            Resizer::new(&config).resize_u16(&input)
        };

        // Streaming with interleaved drain
        let mut resizer = StreamingResize::new(&config);
        let mut streaming = Vec::new();
        for y in 0..40 {
            let start = y * 40 * 4;
            let end = start + 40 * 4;
            resizer.push_row_u16(&input[start..end]).unwrap();
            while let Some(row) = resizer.next_output_row_u16() {
                streaming.extend_from_slice(row);
            }
        }
        resizer.finish();
        while let Some(row) = resizer.next_output_row_u16() {
            streaming.extend_from_slice(row);
        }

        assert_eq!(fullframe.len(), streaming.len());
        for (i, (&a, &b)) in fullframe.iter().zip(streaming.iter()).enumerate() {
            // ±16 tolerance: f16 quantization at different pipeline stages
            // (H-first fullframe vs V-first streaming) compounds quantization errors.
            // Observed max diff ~11 in u16 space (0.017%).
            assert!(
                (a as i32 - b as i32).unsigned_abs() <= 16,
                "mismatch at element {}: fullframe={}, streaming={}",
                i,
                a,
                b
            );
        }
    }

    #[test]
    fn test_streaming_u16_rgb_3ch() {
        let config = ResizeConfig::builder(16, 16, 8, 8)
            .filter(Filter::Lanczos)
            .format(PixelDescriptor::RGB16_SRGB)
            .build();

        let mut resizer = StreamingResize::new(&config);

        let mut row = vec![0u16; 16 * 3];
        for pixel in row.chunks_exact_mut(3) {
            pixel[0] = 40000;
            pixel[1] = 20000;
            pixel[2] = 60000;
        }

        let mut total_rows = 0;
        for _ in 0..16 {
            resizer.push_row_u16(&row).unwrap();
            while let Some(out_row) = resizer.next_output_row_u16() {
                total_rows += 1;
                for pixel in out_row.chunks_exact(3) {
                    assert!(
                        (pixel[0] as i32 - 40000).unsigned_abs() <= 200,
                        "R off: {}",
                        pixel[0]
                    );
                }
            }
        }
        resizer.finish();
        while let Some(out_row) = resizer.next_output_row_u16() {
            total_rows += 1;
            for pixel in out_row.chunks_exact(3) {
                assert!(
                    (pixel[0] as i32 - 40000).unsigned_abs() <= 200,
                    "R off: {}",
                    pixel[0]
                );
            }
        }
        assert_eq!(total_rows, 8);
    }

    // === i16 path tests ===

    #[test]
    fn test_i16_srgb_path_selected() {
        // Srgb8 + srgb() + Rgba = I16Srgb (no linearization, 4ch)
        let config = make_config(20, 20, 10, 10);
        let resizer = StreamingResize::new(&config);
        assert_eq!(resizer.path, StreamingPath::I16Srgb);
    }

    #[test]
    fn test_i16_linear_path_selected() {
        // Srgb8 + linear() + Rgbx = I16Linear (linearize, 4ch, no premul)
        let config = ResizeConfig::builder(20, 20, 10, 10)
            .format(PixelDescriptor::RGBX8_SRGB)
            .linear()
            .build();
        let resizer = StreamingResize::new(&config);
        assert_eq!(resizer.path, StreamingPath::I16Linear);
    }

    #[test]
    fn test_f32_path_for_rgba_linear() {
        // Srgb8 + linear() + Rgba = F32 (needs premul → not I16Linear)
        let config = ResizeConfig::builder(20, 20, 10, 10)
            .format(PixelDescriptor::RGBA8_SRGB)
            .linear()
            .build();
        let resizer = StreamingResize::new(&config);
        assert_eq!(resizer.path, StreamingPath::F32);
    }

    #[test]
    fn test_i16_linear_fallback_heavy_downscale() {
        // 4x downscale → max_taps ~25, exceeds I16Linear threshold → F32
        let config = ResizeConfig::builder(400, 400, 100, 100)
            .filter(Filter::Lanczos)
            .format(PixelDescriptor::RGBX8_SRGB)
            .linear()
            .build();
        let resizer = StreamingResize::new(&config);
        assert_eq!(resizer.path, StreamingPath::F32);
    }

    #[test]
    fn test_i16_linear_kept_mild_downscale() {
        // 2x downscale → max_taps ~13, within I16Linear threshold → I16Linear
        let config = ResizeConfig::builder(200, 200, 100, 100)
            .filter(Filter::Lanczos)
            .format(PixelDescriptor::RGBX8_SRGB)
            .linear()
            .build();
        let resizer = StreamingResize::new(&config);
        assert_eq!(resizer.path, StreamingPath::I16Linear);
    }

    #[test]
    fn test_i16_srgb_kept_moderate_downscale() {
        // 4x downscale → max_taps ~25, within I16Srgb threshold → I16Srgb
        let config = ResizeConfig::builder(400, 400, 100, 100)
            .filter(Filter::Lanczos)
            .format(PixelDescriptor::RGBA8_SRGB)
            .srgb()
            .build();
        let resizer = StreamingResize::new(&config);
        assert_eq!(resizer.path, StreamingPath::I16Srgb);
    }

    #[test]
    fn test_i16_srgb_fallback_extreme_downscale() {
        // 10x downscale → max_taps ~63, exceeds I16Srgb threshold → F32
        let config = ResizeConfig::builder(1000, 1000, 100, 100)
            .filter(Filter::Lanczos)
            .format(PixelDescriptor::RGBA8_SRGB)
            .srgb()
            .build();
        let resizer = StreamingResize::new(&config);
        assert_eq!(resizer.path, StreamingPath::F32);
    }

    #[test]
    fn test_f32_path_for_3ch() {
        // Srgb8 + srgb() + Rgb = F32 (3ch → not i16)
        let config = ResizeConfig::builder(20, 20, 10, 10)
            .format(PixelDescriptor::RGB8_SRGB)
            .srgb()
            .build();
        let resizer = StreamingResize::new(&config);
        assert_eq!(resizer.path, StreamingPath::F32);
    }

    #[test]
    fn test_i16_srgb_constant_color() {
        let config = make_config(20, 20, 10, 10); // I16Srgb
        let mut resizer = StreamingResize::new(&config);
        assert_eq!(resizer.path, StreamingPath::I16Srgb);

        let mut row = vec![0u8; 20 * 4];
        for pixel in row.chunks_exact_mut(4) {
            pixel[0] = 128;
            pixel[1] = 128;
            pixel[2] = 128;
            pixel[3] = 255;
        }

        let mut total_rows = 0;
        for _ in 0..20 {
            resizer.push_row(&row).unwrap();
            while let Some(out_row) = resizer.next_output_row() {
                total_rows += 1;
                for pixel in out_row.chunks_exact(4) {
                    assert!(
                        (pixel[0] as i16 - 128).unsigned_abs() <= 2,
                        "R channel off: {} (I16Srgb path)",
                        pixel[0]
                    );
                    assert!(
                        (pixel[3] as i16 - 255).unsigned_abs() <= 1,
                        "A channel off: {} (I16Srgb path)",
                        pixel[3]
                    );
                }
            }
        }
        resizer.finish();
        while resizer.next_output_row().is_some() {
            total_rows += 1;
        }
        assert_eq!(total_rows, 10);
    }

    #[test]
    fn test_i16_linear_constant_color() {
        // Rgbx + linear = I16Linear path
        let config = ResizeConfig::builder(20, 20, 10, 10)
            .format(PixelDescriptor::RGBX8_SRGB)
            .linear()
            .build();
        let mut resizer = StreamingResize::new(&config);
        assert_eq!(resizer.path, StreamingPath::I16Linear);

        let mut row = vec![0u8; 20 * 4];
        for pixel in row.chunks_exact_mut(4) {
            pixel[0] = 128;
            pixel[1] = 128;
            pixel[2] = 128;
            pixel[3] = 255;
        }

        let mut total_rows = 0;
        for _ in 0..20 {
            resizer.push_row(&row).unwrap();
            while let Some(out_row) = resizer.next_output_row() {
                total_rows += 1;
                for pixel in out_row.chunks_exact(4) {
                    assert!(
                        (pixel[0] as i16 - 128).unsigned_abs() <= 2,
                        "R channel off: {} (I16Linear path)",
                        pixel[0]
                    );
                }
            }
        }
        resizer.finish();
        while resizer.next_output_row().is_some() {
            total_rows += 1;
        }
        assert_eq!(total_rows, 10);
    }

    #[test]
    fn test_i16_srgb_matches_fullframe() {
        // Srgb8 + srgb + Rgba = both fullframe and streaming use i16 srgb path
        let config = ResizeConfig::builder(40, 40, 20, 20)
            .format(PixelDescriptor::RGBA8_SRGB)
            .srgb()
            .build();

        // Gradient input
        let mut input = vec![0u8; 40 * 40 * 4];
        for y in 0..40u32 {
            for x in 0..40u32 {
                let idx = (y * 40 + x) as usize * 4;
                input[idx] = ((x * 255) / 39) as u8;
                input[idx + 1] = ((y * 255) / 39) as u8;
                input[idx + 2] = 128;
                input[idx + 3] = 255;
            }
        }

        // Fullframe
        let fullframe = {
            use crate::resize::Resizer;
            Resizer::new(&config).resize(&input)
        };

        // Streaming
        let mut resizer = StreamingResize::new(&config);
        assert_eq!(resizer.path, StreamingPath::I16Srgb);
        let mut streaming = Vec::new();
        for y in 0..40 {
            let start = y * 40 * 4;
            let end = start + 40 * 4;
            resizer.push_row(&input[start..end]).unwrap();
            while let Some(row) = resizer.next_output_row() {
                streaming.extend_from_slice(row);
            }
        }
        resizer.finish();
        while let Some(row) = resizer.next_output_row() {
            streaming.extend_from_slice(row);
        }

        assert_eq!(fullframe.len(), streaming.len());
        let max_diff: u8 = fullframe
            .iter()
            .zip(streaming.iter())
            .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
            .max()
            .unwrap_or(0);
        assert!(
            max_diff <= 2,
            "i16_srgb streaming vs fullframe max diff {} exceeds tolerance 2",
            max_diff
        );
    }

    #[test]
    fn test_i16_linear_matches_fullframe() {
        // Srgb8 + linear + Rgbx = both fullframe and streaming use i16 linear path
        let config = ResizeConfig::builder(40, 40, 20, 20)
            .format(PixelDescriptor::RGBX8_SRGB)
            .linear()
            .build();

        // Gradient input
        let mut input = vec![0u8; 40 * 40 * 4];
        for y in 0..40u32 {
            for x in 0..40u32 {
                let idx = (y * 40 + x) as usize * 4;
                input[idx] = ((x * 255) / 39) as u8;
                input[idx + 1] = ((y * 255) / 39) as u8;
                input[idx + 2] = 128;
                input[idx + 3] = 255;
            }
        }

        // Fullframe
        let fullframe = {
            use crate::resize::Resizer;
            Resizer::new(&config).resize(&input)
        };

        // Streaming
        let mut resizer = StreamingResize::new(&config);
        assert_eq!(resizer.path, StreamingPath::I16Linear);
        let mut streaming = Vec::new();
        for y in 0..40 {
            let start = y * 40 * 4;
            let end = start + 40 * 4;
            resizer.push_row(&input[start..end]).unwrap();
            while let Some(row) = resizer.next_output_row() {
                streaming.extend_from_slice(row);
            }
        }
        resizer.finish();
        while let Some(row) = resizer.next_output_row() {
            streaming.extend_from_slice(row);
        }

        assert_eq!(fullframe.len(), streaming.len());
        let max_diff: u8 = fullframe
            .iter()
            .zip(streaming.iter())
            .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
            .max()
            .unwrap_or(0);
        assert!(
            max_diff <= 2,
            "i16_linear streaming vs fullframe max diff {} exceeds tolerance 2",
            max_diff
        );
    }

    // === WorkingFormat tests ===

    #[test]
    fn test_working_format_i16_srgb() {
        let config = make_config(20, 20, 10, 10); // srgb() + Rgba = I16Srgb
        let resizer = StreamingResize::new(&config);
        assert_eq!(resizer.working_format(), super::WorkingFormat::I16Srgb);
    }

    #[test]
    fn test_working_format_i16_linear() {
        let config = ResizeConfig::builder(20, 20, 10, 10)
            .format(PixelDescriptor::RGBX8_SRGB)
            .linear()
            .build();
        let resizer = StreamingResize::new(&config);
        assert_eq!(resizer.working_format(), super::WorkingFormat::I16Linear);
    }

    #[test]
    fn test_working_format_f32() {
        let config = ResizeConfig::builder(20, 20, 10, 10)
            .format(PixelDescriptor::RGBA8_SRGB)
            .linear()
            .build();
        let resizer = StreamingResize::new(&config);
        assert_eq!(resizer.working_format(), super::WorkingFormat::F32);
    }

    #[test]
    fn test_pq_forces_f32_path() {
        let pq = PixelDescriptor::RGBA8_SRGB.with_transfer(TransferFunction::Pq);
        let config = ResizeConfig::builder(20, 20, 10, 10)
            .input(pq)
            .output(pq)
            .srgb()
            .build();
        let resizer = StreamingResize::new(&config);
        assert_eq!(resizer.working_format(), super::WorkingFormat::F32);
    }

    #[test]
    fn test_hlg_forces_f32_path() {
        let hlg = PixelDescriptor::RGBX8_SRGB.with_transfer(TransferFunction::Hlg);
        let config = ResizeConfig::builder(20, 20, 10, 10)
            .input(hlg)
            .output(hlg)
            .linear()
            .build();
        let resizer = StreamingResize::new(&config);
        assert_eq!(resizer.working_format(), super::WorkingFormat::F32);
    }

    #[test]
    fn test_bt709_allows_f32_path() {
        // BT.709 is not identity/srgb so it goes to F32 anyway.
        let bt709 = PixelDescriptor::RGBA8_SRGB.with_transfer(TransferFunction::Bt709);
        let config = ResizeConfig::builder(20, 20, 10, 10)
            .input(bt709)
            .output(bt709)
            .build();
        let resizer = StreamingResize::new(&config);
        assert_eq!(resizer.working_format(), super::WorkingFormat::F32);
    }

    // === push_row_i16 tests ===

    #[test]
    fn test_push_row_i16_srgb_matches_push_row() {
        let config = make_config(20, 20, 10, 10); // I16Srgb path
        let mut row_u8 = vec![0u8; 20 * 4];
        for pixel in row_u8.chunks_exact_mut(4) {
            pixel[0] = 128;
            pixel[1] = 64;
            pixel[2] = 32;
            pixel[3] = 255;
        }

        // Via push_row (u8)
        let mut r1 = StreamingResize::new(&config);
        let rows1 = push_drain_collect_u8(&mut r1, &row_u8, 20);

        // Via push_row_i16 (zero-extend u8 to i16)
        let row_i16: Vec<i16> = row_u8.iter().map(|&v| v as i16).collect();
        let mut r2 = StreamingResize::new(&config);
        assert_eq!(r2.working_format(), super::WorkingFormat::I16Srgb);
        let mut rows2 = Vec::new();
        for _ in 0..20 {
            r2.push_row_i16(&row_i16).unwrap();
            while let Some(out) = r2.next_output_row() {
                rows2.push(out.to_vec());
            }
        }
        r2.finish();
        while let Some(out) = r2.next_output_row() {
            rows2.push(out.to_vec());
        }

        assert_eq!(rows1, rows2);
    }

    #[test]
    fn test_push_row_i16_linear_matches_push_row() {
        let config = ResizeConfig::builder(20, 20, 10, 10)
            .format(PixelDescriptor::RGBX8_SRGB)
            .linear()
            .build();

        let mut row_u8 = vec![0u8; 20 * 4];
        for pixel in row_u8.chunks_exact_mut(4) {
            pixel[0] = 128;
            pixel[1] = 64;
            pixel[2] = 32;
            pixel[3] = 255;
        }

        // Via push_row (u8 → sRGB LUT → i12)
        let mut r1 = StreamingResize::new(&config);
        assert_eq!(r1.working_format(), super::WorkingFormat::I16Linear);
        let rows1 = push_drain_collect_u8(&mut r1, &row_u8, 20);

        // Via push_row_i16 (pre-convert u8 → i12, push directly)
        let mut row_i16 = vec![0i16; 20 * 4];
        crate::color::srgb_u8_to_linear_i12_row(&row_u8, &mut row_i16);

        let mut r2 = StreamingResize::new(&config);
        let mut rows2 = Vec::new();
        for _ in 0..20 {
            r2.push_row_i16(&row_i16).unwrap();
            while let Some(out) = r2.next_output_row() {
                rows2.push(out.to_vec());
            }
        }
        r2.finish();
        while let Some(out) = r2.next_output_row() {
            rows2.push(out.to_vec());
        }

        assert_eq!(rows1, rows2);
    }

    // === push_row_linear_f32 tests ===

    #[test]
    fn test_push_row_linear_f32_matches_push_row() {
        // Use F32 path: Srgb8 + linear + Rgba
        let config = ResizeConfig::builder(20, 20, 10, 10)
            .format(PixelDescriptor::RGBA8_SRGB)
            .linear()
            .build();

        let mut row_u8 = vec![0u8; 20 * 4];
        for pixel in row_u8.chunks_exact_mut(4) {
            pixel[0] = 128;
            pixel[1] = 64;
            pixel[2] = 32;
            pixel[3] = 255;
        }

        // Via push_row (u8 → sRGB linearize → f32)
        let mut r1 = StreamingResize::new(&config);
        assert_eq!(r1.working_format(), super::WorkingFormat::F32);
        let rows1 = push_drain_collect_u8(&mut r1, &row_u8, 20);

        // Via push_row_linear_f32 (pre-linearize, push directly)
        let mut row_f32 = vec![0.0f32; 20 * 4];
        crate::color::srgb_u8_to_linear_f32(&row_u8, &mut row_f32, 4, true);

        let mut r2 = StreamingResize::new(&config);
        let mut rows2 = Vec::new();
        for _ in 0..20 {
            r2.push_row_linear_f32(&row_f32).unwrap();
            while let Some(out) = r2.next_output_row() {
                rows2.push(out.to_vec());
            }
        }
        r2.finish();
        while let Some(out) = r2.next_output_row() {
            rows2.push(out.to_vec());
        }

        assert_eq!(rows1, rows2);
    }

    // === Resize with new transfer functions ===

    #[test]
    fn test_bt709_resize_constant_color() {
        let bt709 = PixelDescriptor::RGBA8_SRGB.with_transfer(TransferFunction::Bt709);
        let config = ResizeConfig::builder(20, 20, 10, 10)
            .input(bt709)
            .output(bt709)
            .build();

        let mut resizer = StreamingResize::new(&config);
        let mut row = vec![0u8; 20 * 4];
        for pixel in row.chunks_exact_mut(4) {
            pixel[0] = 128;
            pixel[1] = 128;
            pixel[2] = 128;
            pixel[3] = 255;
        }

        let mut total_rows = 0;
        for _ in 0..20 {
            resizer.push_row(&row).unwrap();
            while let Some(out_row) = resizer.next_output_row() {
                total_rows += 1;
                for pixel in out_row.chunks_exact(4) {
                    assert!(
                        (pixel[0] as i16 - 128).unsigned_abs() <= 2,
                        "BT.709 R off: {}",
                        pixel[0]
                    );
                }
            }
        }
        resizer.finish();
        while resizer.next_output_row().is_some() {
            total_rows += 1;
        }
        assert_eq!(total_rows, 10);
    }

    #[test]
    fn test_pq_resize_streaming_matches_fullframe() {
        let pq = PixelDescriptor::RGBA8_SRGB.with_transfer(TransferFunction::Pq);
        let config = ResizeConfig::builder(20, 20, 10, 10)
            .input(pq)
            .output(pq)
            .build();

        let mut input = vec![0u8; 20 * 20 * 4];
        for pixel in input.chunks_exact_mut(4) {
            pixel[0] = 128;
            pixel[1] = 64;
            pixel[2] = 32;
            pixel[3] = 255;
        }

        // Fullframe
        let fullframe = {
            use crate::resize::Resizer;
            Resizer::new(&config).resize(&input)
        };

        // Streaming
        let mut resizer = StreamingResize::new(&config);
        let mut streaming = Vec::new();
        for y in 0..20 {
            let start = y * 20 * 4;
            let end = start + 20 * 4;
            resizer.push_row(&input[start..end]).unwrap();
            while let Some(row) = resizer.next_output_row() {
                streaming.extend_from_slice(row);
            }
        }
        resizer.finish();
        while let Some(row) = resizer.next_output_row() {
            streaming.extend_from_slice(row);
        }

        assert_eq!(fullframe.len(), streaming.len());
        for (i, (&a, &b)) in fullframe.iter().zip(streaming.iter()).enumerate() {
            assert!(
                (a as i16 - b as i16).unsigned_abs() <= 2,
                "PQ mismatch at {}: fullframe={}, streaming={}",
                i,
                a,
                b
            );
        }
    }

    #[test]
    fn test_hlg_resize_constant_color() {
        let hlg = PixelDescriptor::RGBA8_SRGB.with_transfer(TransferFunction::Hlg);
        let config = ResizeConfig::builder(20, 20, 10, 10)
            .input(hlg)
            .output(hlg)
            .build();

        let mut resizer = StreamingResize::new(&config);
        let mut row = vec![0u8; 20 * 4];
        for pixel in row.chunks_exact_mut(4) {
            pixel[0] = 128;
            pixel[1] = 128;
            pixel[2] = 128;
            pixel[3] = 255;
        }

        let mut total_rows = 0;
        for _ in 0..20 {
            resizer.push_row(&row).unwrap();
            while let Some(out_row) = resizer.next_output_row() {
                total_rows += 1;
                for pixel in out_row.chunks_exact(4) {
                    assert!(
                        (pixel[0] as i16 - 128).unsigned_abs() <= 2,
                        "HLG R off: {}",
                        pixel[0]
                    );
                }
            }
        }
        resizer.finish();
        while resizer.next_output_row().is_some() {
            total_rows += 1;
        }
        assert_eq!(total_rows, 10);
    }

    // =========================================================================
    // Crop tests
    // =========================================================================

    #[test]
    fn test_streaming_crop_basic() {
        // 100×100 input, crop a 50×50 region at (25,25), resize to 25×25
        let config = ResizeConfig::builder(100, 100, 25, 25)
            .format(PixelDescriptor::RGBA8_SRGB)
            .srgb()
            .crop(25, 25, 50, 50)
            .build();
        let mut resizer = StreamingResize::new(&config);

        // input_row_len should be full source width
        assert_eq!(resizer.input_row_len(), 100 * 4);

        let row = vec![200u8; 100 * 4];
        let mut output_count = 0;
        for _ in 0..100 {
            resizer.push_row(&row).unwrap();
            while resizer.next_output_row().is_some() {
                output_count += 1;
            }
        }
        resizer.finish();
        while resizer.next_output_row().is_some() {
            output_count += 1;
        }

        assert_eq!(output_count, 25);
        assert!(resizer.is_complete());
    }

    #[test]
    fn test_streaming_crop_constant_color() {
        // Crop a 20×20 region from a 40×40 constant-color image, resize to 10×10
        let config = ResizeConfig::builder(40, 40, 10, 10)
            .format(PixelDescriptor::RGBA8_SRGB)
            .srgb()
            .crop(10, 10, 20, 20)
            .build();
        let mut resizer = StreamingResize::new(&config);

        let mut row = vec![0u8; 40 * 4];
        for pixel in row.chunks_exact_mut(4) {
            pixel[0] = 100;
            pixel[1] = 100;
            pixel[2] = 100;
            pixel[3] = 255;
        }

        let rows = push_drain_collect_u8(&mut resizer, &row, 40);
        assert_eq!(rows.len(), 10);

        // All pixels should be close to the input value
        for out_row in &rows {
            for pixel in out_row.chunks_exact(4) {
                assert!(
                    (pixel[0] as i16 - 100).unsigned_abs() <= 2,
                    "crop constant color R off: {}",
                    pixel[0]
                );
            }
        }
    }

    #[test]
    fn test_streaming_crop_no_resize() {
        // Crop-only: 50×50 from 100×100, output same as crop size (1:1)
        let config = ResizeConfig::builder(100, 100, 50, 50)
            .format(PixelDescriptor::RGBA8_SRGB)
            .srgb()
            .crop(0, 0, 50, 50)
            .build();
        let mut resizer = StreamingResize::new(&config);

        let row = vec![128u8; 100 * 4];
        let rows = push_drain_collect_u8(&mut resizer, &row, 100);
        assert_eq!(rows.len(), 50);
    }

    // =========================================================================
    // Padding tests
    // =========================================================================

    #[test]
    fn test_streaming_padding_basic() {
        // 20×20 → 10×10 with 5px padding on all sides
        // Total output: 20×20
        let config = ResizeConfig::builder(20, 20, 10, 10)
            .format(PixelDescriptor::RGBA8_SRGB)
            .srgb()
            .padding_uniform(5)
            .padding_color([0.0, 0.0, 0.0, 1.0]) // black padding
            .build();
        let mut resizer = StreamingResize::new(&config);

        // output_row_len should be (5+10+5)*4 = 80
        assert_eq!(resizer.output_row_len(), 20 * 4);
        assert_eq!(resizer.total_output_height(), 20);

        let mut row = vec![0u8; 20 * 4];
        for pixel in row.chunks_exact_mut(4) {
            pixel[0] = 255;
            pixel[1] = 255;
            pixel[2] = 255;
            pixel[3] = 255;
        }

        let mut all_rows = Vec::new();
        for _ in 0..20 {
            resizer.push_row(&row).unwrap();
            while let Some(out) = resizer.next_output_row() {
                all_rows.push(out.to_vec());
            }
        }
        resizer.finish();
        while let Some(out) = resizer.next_output_row() {
            all_rows.push(out.to_vec());
        }

        assert_eq!(all_rows.len(), 20);
        assert!(resizer.is_complete());

        // First 5 rows should be black padding
        for row in &all_rows[..5] {
            for pixel in row.chunks_exact(4) {
                assert_eq!(pixel[0], 0, "top padding should be black");
                assert_eq!(pixel[3], 255, "top padding alpha");
            }
        }

        // Last 5 rows should be black padding
        for row in &all_rows[15..] {
            for pixel in row.chunks_exact(4) {
                assert_eq!(pixel[0], 0, "bottom padding should be black");
                assert_eq!(pixel[3], 255, "bottom padding alpha");
            }
        }

        // Content rows: first 5 pixels should be black (left pad)
        for row in &all_rows[5..15] {
            // Left padding
            for pixel in row[..20].chunks_exact(4) {
                assert_eq!(pixel[0], 0, "left padding should be black");
            }
            // Right padding
            for pixel in row[60..].chunks_exact(4) {
                assert_eq!(pixel[0], 0, "right padding should be black");
            }
        }
    }

    #[test]
    fn test_streaming_padding_no_resize() {
        // Pad-only: 10×10 → 10×10 with 2px padding
        let config = ResizeConfig::builder(10, 10, 10, 10)
            .format(PixelDescriptor::RGBA8_SRGB)
            .srgb()
            .padding(2, 2, 2, 2)
            .padding_color([1.0, 0.0, 0.0, 1.0]) // red padding
            .build();
        let mut resizer = StreamingResize::new(&config);

        assert_eq!(resizer.total_output_height(), 14);
        assert_eq!(resizer.output_row_len(), 14 * 4);

        let row = vec![128u8; 10 * 4];
        let rows = push_drain_collect_u8(&mut resizer, &row, 10);
        assert_eq!(rows.len(), 14);

        // Top padding rows should be red
        for pixel in rows[0].chunks_exact(4) {
            assert_eq!(pixel[0], 255, "red padding R");
            assert_eq!(pixel[1], 0, "red padding G");
            assert_eq!(pixel[2], 0, "red padding B");
        }
    }

    #[test]
    fn test_streaming_padding_top_available_immediately() {
        // With top padding, output should be available before any input
        let config = ResizeConfig::builder(20, 20, 10, 10)
            .format(PixelDescriptor::RGBA8_SRGB)
            .srgb()
            .padding(3, 0, 0, 0)
            .build();
        let resizer = StreamingResize::new(&config);

        assert_eq!(resizer.initial_input_rows_needed(), 0);
        assert_eq!(resizer.output_rows_available(), 3); // 3 top padding rows
    }

    // =========================================================================
    // Crop + Padding combined tests
    // =========================================================================

    #[test]
    fn test_streaming_crop_and_padding() {
        // Crop 30×30 from 50×50 at (10,10), resize to 15×15, pad 5px
        let config = ResizeConfig::builder(50, 50, 15, 15)
            .format(PixelDescriptor::RGBA8_SRGB)
            .srgb()
            .crop(10, 10, 30, 30)
            .padding_uniform(5)
            .padding_color([0.0, 0.0, 0.0, 1.0])
            .build();
        let mut resizer = StreamingResize::new(&config);

        assert_eq!(resizer.input_row_len(), 50 * 4);
        assert_eq!(resizer.output_row_len(), 25 * 4); // 5+15+5
        assert_eq!(resizer.total_output_height(), 25); // 5+15+5

        let row = vec![128u8; 50 * 4];
        let rows = push_drain_collect_u8(&mut resizer, &row, 50);
        assert_eq!(rows.len(), 25);
        assert!(resizer.is_complete());
    }

    #[test]
    fn test_streaming_crop_validation() {
        // Crop region exceeds input bounds — should panic on build
        let config = ResizeConfig::builder(100, 100, 50, 50)
            .format(PixelDescriptor::RGBA8_SRGB)
            .crop(80, 0, 50, 50) // x+w=130 > 100
            .build();
        let result = config.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_streaming_padding_f32_output() {
        // Test padding with f32 output
        let config = ResizeConfig::builder(10, 10, 5, 5)
            .filter(Filter::Lanczos)
            .format(PixelDescriptor::RGBAF32_LINEAR)
            .padding(2, 2, 2, 2)
            .padding_color([0.5, 0.5, 0.5, 1.0])
            .build();
        let mut resizer = StreamingResize::new(&config);

        assert_eq!(resizer.output_row_len(), 9 * 4); // (2+5+2)*4

        let row = vec![0.8f32; 10 * 4];
        let mut all_rows = Vec::new();
        for _ in 0..10 {
            resizer.push_row_f32(&row).unwrap();
            while let Some(out) = resizer.next_output_row_f32() {
                all_rows.push(out.to_vec());
            }
        }
        resizer.finish();
        while let Some(out) = resizer.next_output_row_f32() {
            all_rows.push(out.to_vec());
        }

        assert_eq!(all_rows.len(), 9); // 2+5+2

        // Top padding row should have pad color
        let first_row = &all_rows[0];
        for pixel in first_row.chunks_exact(4) {
            assert!((pixel[0] - 0.5).abs() < 0.01, "f32 padding R: {}", pixel[0]);
            assert!((pixel[3] - 1.0).abs() < 0.01, "f32 padding A: {}", pixel[3]);
        }
    }

    #[test]
    fn test_premul_linear_f32_output() {
        // Test that next_output_row_premul_linear_f32 returns premultiplied data
        let config = ResizeConfig::builder(4, 4, 2, 2)
            .filter(crate::Filter::Lanczos)
            .format(PixelDescriptor::RGBA8_SRGB)
            .linear() // force f32 path
            .build();

        let mut resizer = StreamingResize::new(&config);
        let row = vec![128u8; 4 * 4];
        for _ in 0..4 {
            resizer.push_row(&row).unwrap();
            while let Some(out) = resizer.next_output_row_premul_linear_f32() {
                // Should be premul linear f32, not u8
                assert_eq!(out.len(), 2 * 4);
                // Values should be in [0, 1] range (linear f32)
                for &v in out {
                    assert!(v >= 0.0 && v <= 1.1, "premul value out of range: {v}");
                }
            }
        }
        resizer.finish();
        while let Some(out) = resizer.next_output_row_premul_linear_f32() {
            assert_eq!(out.len(), 2 * 4);
        }
    }

    #[test]
    fn test_next_output_row_over_matches_with_background() {
        // Compare next_output_row_over() against with_background(SolidBackground)
        let config = ResizeConfig::builder(8, 8, 4, 4)
            .filter(crate::Filter::Lanczos)
            .format(PixelDescriptor::RGBA8_SRGB)
            .linear()
            .build();

        // Semi-transparent input
        let mut input = vec![0u8; 8 * 4];
        for pixel in input.chunks_exact_mut(4) {
            pixel[0] = 200; // R
            pixel[1] = 100; // G
            pixel[2] = 50; // B
            pixel[3] = 128; // A = 50%
        }

        // Path A: with_background(SolidBackground::white)
        let bg = crate::composite::SolidBackground::white(PixelDescriptor::RGBA8_SRGB);
        let mut resizer_a = StreamingResize::with_background(&config, bg).unwrap();
        let mut rows_a = Vec::new();
        for _ in 0..8 {
            resizer_a.push_row(&input).unwrap();
            while let Some(out) = resizer_a.next_output_row() {
                rows_a.push(out.to_vec());
            }
        }
        resizer_a.finish();
        while let Some(out) = resizer_a.next_output_row() {
            rows_a.push(out.to_vec());
        }

        // Path B: next_output_row_over with white bg row
        let mut resizer_b = StreamingResize::new(&config);
        // White in premul linear = [1.0, 1.0, 1.0, 1.0]
        let white_bg = vec![1.0f32; 4 * 4];
        let mut rows_b = Vec::new();
        for _ in 0..8 {
            resizer_b.push_row(&input).unwrap();
            while let Some(out) = resizer_b.next_output_row_over(&white_bg) {
                rows_b.push(out.to_vec());
            }
        }
        resizer_b.finish();
        while let Some(out) = resizer_b.next_output_row_over(&white_bg) {
            rows_b.push(out.to_vec());
        }

        assert_eq!(rows_a.len(), rows_b.len());
        for (i, (a, b)) in rows_a.iter().zip(rows_b.iter()).enumerate() {
            assert_eq!(
                a, b,
                "row {i} mismatch between with_background and next_output_row_over"
            );
        }
    }

    #[test]
    fn test_background_mut_accessible() {
        let bg = crate::composite::StreamedBackground::new(4, 8);
        let config = ResizeConfig::builder(4, 4, 2, 2)
            .filter(crate::Filter::Lanczos)
            .format(PixelDescriptor::RGBA8_SRGB)
            .linear()
            .build();
        let mut resizer = StreamingResize::with_background(&config, bg).unwrap();

        // Should be able to push rows via background_mut
        let bg_row = vec![0.5f32; 8];
        resizer.background_mut().push_row(&bg_row);
        assert_eq!(resizer.background_mut().rows_pushed(), 1);
    }

    // === Orientation tests ===

    #[test]
    fn test_orient_identity_passthrough() {
        let config = make_config(4, 4, 4, 4);
        let mut r1 = StreamingResize::new(&config);
        let mut r2 = StreamingResize::new(&config).with_orientation(OrientOutput::Identity);

        let row = vec![128u8; 4 * 4];
        let rows1 = push_drain_collect_u8(&mut r1, &row, 4);
        let rows2 = push_drain_collect_u8(&mut r2, &row, 4);
        assert_eq!(rows1, rows2);
    }

    #[test]
    fn test_orient_flip_h() {
        // 4x2 image with distinct columns — compare oriented vs unoriented
        let config = ResizeConfig::builder(4, 2, 4, 2)
            .format(PixelDescriptor::RGBA8_SRGB)
            .srgb()
            .build();

        let row = vec![
            10, 20, 30, 255, 40, 50, 60, 255, 70, 80, 90, 255, 100, 110, 120, 255,
        ];

        // Get unoriented output
        let mut plain = StreamingResize::new(&config);
        let plain_rows = push_drain_collect_u8(&mut plain, &row, 2);

        // Get FlipH output
        let mut flipped = StreamingResize::new(&config).with_orientation(OrientOutput::FlipH);
        let flip_rows = push_drain_collect_u8(&mut flipped, &row, 2);

        assert_eq!(flip_rows.len(), plain_rows.len());
        // Each row should have its pixels reversed relative to the unoriented output
        for (plain_row, flip_row) in plain_rows.iter().zip(flip_rows.iter()) {
            let plain_pixels: Vec<&[u8]> = plain_row.chunks_exact(4).collect();
            let flip_pixels: Vec<&[u8]> = flip_row.chunks_exact(4).collect();
            let reversed: Vec<&[u8]> = plain_pixels.iter().rev().copied().collect();
            assert_eq!(
                flip_pixels, reversed,
                "FlipH should reverse pixel order per row"
            );
        }
    }

    #[test]
    fn test_orient_flip_v() {
        // 2x4 image — compare oriented vs unoriented
        let config = ResizeConfig::builder(2, 4, 2, 4)
            .format(PixelDescriptor::RGBA8_SRGB)
            .srgb()
            .build();

        let rows_in = [
            vec![10u8, 0, 0, 255, 10, 0, 0, 255],
            vec![20, 0, 0, 255, 20, 0, 0, 255],
            vec![30, 0, 0, 255, 30, 0, 0, 255],
            vec![40, 0, 0, 255, 40, 0, 0, 255],
        ];

        // Get unoriented output
        let mut plain = StreamingResize::new(&config);
        for row in &rows_in {
            plain.push_row(row).unwrap();
        }
        plain.finish();
        let mut plain_rows = Vec::new();
        while let Some(row) = plain.next_output_row() {
            plain_rows.push(row.to_vec());
        }

        // Get FlipV output
        let mut flipped = StreamingResize::new(&config).with_orientation(OrientOutput::FlipV);
        for row in &rows_in {
            flipped.push_row(row).unwrap();
            // FlipV is non-row-local — no output until finish()
            assert!(flipped.next_output_row().is_none());
        }
        let remaining = flipped.finish();
        assert_eq!(remaining, 4);
        let mut flip_rows = Vec::new();
        while let Some(row) = flipped.next_output_row() {
            flip_rows.push(row.to_vec());
        }

        assert_eq!(flip_rows.len(), plain_rows.len());
        // Rows should be in reverse order relative to unoriented output
        for (i, flip_row) in flip_rows.iter().enumerate() {
            let plain_row = &plain_rows[plain_rows.len() - 1 - i];
            assert_eq!(
                flip_row, plain_row,
                "FlipV row {i} should match reversed plain row"
            );
        }
    }

    #[test]
    fn test_orient_rotate90() {
        // 4x2 identity resize with Rotate90 → output should be 2x4
        let config = ResizeConfig::builder(4, 2, 4, 2)
            .format(PixelDescriptor::RGBA8_SRGB)
            .srgb()
            .build();
        let mut resizer = StreamingResize::new(&config).with_orientation(OrientOutput::Rotate90);

        assert_eq!(resizer.output_row_len(), 2 * 4); // oriented width = 2
        assert_eq!(resizer.total_output_height(), 4); // oriented height = 4

        let row0 = vec![10, 0, 0, 255, 20, 0, 0, 255, 30, 0, 0, 255, 40, 0, 0, 255];
        let row1 = vec![50, 0, 0, 255, 60, 0, 0, 255, 70, 0, 0, 255, 80, 0, 0, 255];

        resizer.push_row(&row0).unwrap();
        resizer.push_row(&row1).unwrap();
        let remaining = resizer.finish();
        assert_eq!(remaining, 4);

        let mut output = Vec::new();
        while let Some(row) = resizer.next_output_row() {
            output.push(row.to_vec());
        }
        assert_eq!(output.len(), 4);
        assert!(resizer.is_complete());
    }

    #[test]
    fn test_orient_dimensions_reporting() {
        let config = ResizeConfig::builder(10, 20, 10, 20)
            .format(PixelDescriptor::RGBA8_SRGB)
            .srgb()
            .build();

        // Identity: 10x20
        let r = StreamingResize::new(&config).with_orientation(OrientOutput::Identity);
        assert_eq!(r.output_row_len(), 10 * 4);
        assert_eq!(r.total_output_height(), 20);

        // Rotate90: 20x10
        let r = StreamingResize::new(&config).with_orientation(OrientOutput::Rotate90);
        assert_eq!(r.output_row_len(), 20 * 4);
        assert_eq!(r.total_output_height(), 10);

        // FlipH: 10x20 (same dims)
        let r = StreamingResize::new(&config).with_orientation(OrientOutput::FlipH);
        assert_eq!(r.output_row_len(), 10 * 4);
        assert_eq!(r.total_output_height(), 20);
    }
}
