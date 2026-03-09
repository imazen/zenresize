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
    /// sRGB gamma i16 path: u8 → i16 V-filter → i16 H-filter → u8
    /// No linearization. 4ch only.
    I16Srgb,
    /// Linear i12 i16 path: sRGB u8 → i12 → i16 V-filter → i16 H-filter → sRGB u8
    /// 4ch, no premul (Rgbx or RgbaPremul).
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

    /// Background for compositing.
    background: B,
    /// Row buffer for non-solid backgrounds. Empty for NoBackground and SolidBackground.
    composite_bg_row: Vec<f32>,
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
        let v_weights = F32WeightTable::new(config.in_height, config.out_height, &filter);

        let channels = config.channels();
        let needs_premul = config.needs_premultiply();
        let alpha_is_last = config.input.has_alpha();

        // Path selection: mirrors fullframe Resizer paths 0/1.
        // I16Srgb: identity transfer (no linearization), u8 4ch.
        // I16Linear: sRGB→linear via compile-time LUTs, u8 4ch, no premul.
        // Both i16 paths hardcode their transfer functions, so they require
        // exact match — any other transfer (BT.709, PQ, HLG) goes to f32.
        let input_tf = config.effective_input_transfer();
        let output_tf = config.effective_output_transfer();
        let is_u8_format = config.input.channel_type() == ChannelType::U8;
        let path = if !active_composite
            && is_u8_format
            && input_tf == TransferFunction::Linear
            && output_tf == TransferFunction::Linear
            && channels == 4
        {
            StreamingPath::I16Srgb
        } else if !active_composite
            && is_u8_format
            && input_tf == TransferFunction::Srgb
            && output_tf == TransferFunction::Srgb
            && channels == 4
            && !needs_premul
        {
            StreamingPath::I16Linear
        } else {
            StreamingPath::F32
        };

        let extra_slack = if batch_hint > 0 {
            batch_hint as usize + 2
        } else {
            2
        };
        let cache_size = v_weights.max_taps + extra_slack;
        let in_row_len = config.in_width as usize * channels;
        let out_row_len = config.out_width as usize * channels;

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

        match path {
            StreamingPath::F32 => {
                let h_weights = F32WeightTable::new(config.in_width, config.out_width, &filter);
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
                    background,
                    composite_bg_row,
                }
            }
            StreamingPath::I16Srgb => {
                let i16_h_weights = I16WeightTable::new(config.in_width, config.out_width, &filter);
                let i16_v_weights =
                    I16WeightTable::new(config.in_height, config.out_height, &filter);

                // V-filter output needs H-filter SIMD padding (zeroed)
                let h_padding_bytes = i16_h_weights.groups4 * 16;
                let v_out_len = in_row_len + h_padding_bytes;

                let u8_v_cache = (0..cache_size).map(|_| vec![0u8; in_row_len]).collect();

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
                    u8_v_cache,
                    i16_v_cache: Vec::new(),
                    temp_v_output_u8: vec![0u8; v_out_len],
                    temp_v_output_i16: Vec::new(),
                    temp_h_output_i16: Vec::new(),
                    linearized_row_i16: Vec::new(),
                    input_rows_received: 0,
                    output_rows_produced: 0,
                    finished: false,
                    background,
                    composite_bg_row,
                }
            }
            StreamingPath::I16Linear => {
                let i16_h_weights = I16WeightTable::new(config.in_width, config.out_width, &filter);
                let i16_v_weights =
                    I16WeightTable::new(config.in_height, config.out_height, &filter);

                // H-filter SIMD padding for i16 data (in i16 elements)
                let h_padding_i16 = i16_h_weights.groups4 * 16;
                let v_out_i16_len = in_row_len + h_padding_i16;

                let i16_v_cache = (0..cache_size).map(|_| vec![0i16; in_row_len]).collect();

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
                    temp_v_output_i16: vec![0i16; v_out_i16_len],
                    temp_h_output_i16: vec![0i16; out_row_len],
                    linearized_row_i16: vec![0i16; in_row_len + h_padding_i16],
                    input_rows_received: 0,
                    output_rows_produced: 0,
                    finished: false,
                    background,
                    composite_bg_row,
                }
            }
        }
    }

    // =========================================================================
    // Info accessors
    // =========================================================================

    /// Number of elements per input row (for sizing decode buffers).
    ///
    /// Equal to `in_width * channels`.
    pub fn input_row_len(&self) -> usize {
        self.config.input_row_len()
    }

    /// Number of elements per output row (for sizing encode buffers).
    ///
    /// Equal to `out_width * channels`.
    pub fn output_row_len(&self) -> usize {
        self.config.output_row_len()
    }

    /// Mutable reference to the background (e.g., for pushing rows to [`crate::StreamedBackground`]).
    pub fn background_mut(&mut self) -> &mut B {
        &mut self.background
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

    /// How many input rows must be pushed before the first output row.
    pub fn initial_input_rows_needed(&self) -> u32 {
        let first_right = self.first_output_row_max_input();
        (first_right + 1).min(self.config.in_height)
    }

    /// Total output rows produced so far.
    pub fn output_rows_produced(&self) -> u32 {
        self.output_rows_produced
    }

    /// Check if all output rows have been produced.
    pub fn is_complete(&self) -> bool {
        self.output_rows_produced >= self.config.out_height
    }

    /// Count how many output rows can be produced right now without more input.
    ///
    /// This is the number of consecutive `next_output_row()` calls that will
    /// return `Some` before returning `None`.
    pub fn output_rows_available(&self) -> u32 {
        let mut count = 0u32;
        let mut probe_y = self.output_rows_produced;
        while probe_y < self.config.out_height {
            let left = self.v_weights.left[probe_y as usize];
            let tap_count = self.v_weights.tap_count(probe_y as usize);
            let right = left + tap_count as i32 - 1;
            let needed_max = right.min(self.config.in_height as i32 - 1).max(0) as u32;
            if needed_max >= self.input_rows_received {
                break;
            }
            count += 1;
            probe_y += 1;
        }
        count
    }

    // =========================================================================
    // Input methods
    // =========================================================================

    /// Push one row of u8 input pixels. Linearizes, premultiplies, and caches the row.
    ///
    /// Caller MUST drain all available output rows (via `next_output_row` or
    /// `next_output_row_into`) before pushing the next input row.
    ///
    /// `row` must contain at least `input_row_len()` elements.
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
        let pixel_len = self.config.input_row_len();
        let stride = self.config.effective_in_stride();
        if row.len() < pixel_len.min(stride) {
            return Err(at!(StreamingError::InputTooShort));
        }

        let pixel_data = &row[..pixel_len];

        match self.path {
            StreamingPath::I16Srgb => {
                // u8 → optional premul → cache u8 directly (no linearization)
                self.check_ring_buffer().at()?;
                let cache_slot = self.cache_write_idx % self.cache_size;
                if self.needs_premul {
                    simd::premultiply_u8_row(
                        pixel_data,
                        &mut self.u8_v_cache[cache_slot][..pixel_len],
                    );
                } else {
                    self.u8_v_cache[cache_slot][..pixel_len].copy_from_slice(pixel_data);
                }
                self.cache_write_idx += 1;
                self.input_rows_received += 1;
                return Ok(());
            }
            StreamingPath::I16Linear => {
                // sRGB u8 → linear i12 (LUT)
                color::srgb_u8_to_linear_i12_row(
                    pixel_data,
                    &mut self.linearized_row_i16[..pixel_len],
                );
                return self.push_row_internal_i16().at();
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
        let expected_len = if count == 0 {
            0
        } else {
            stride * (count as usize - 1) + self.config.input_row_len()
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
        let pixel_len = self.config.input_row_len();
        if row.len() < pixel_len {
            return Err(at!(StreamingError::InputTooShort));
        }

        self.temp_input_f32[..pixel_len].copy_from_slice(&row[..pixel_len]);

        if self.needs_premul {
            simd::premultiply_alpha_row(&mut self.temp_input_f32[..pixel_len]);
        }

        self.push_row_internal().at()
    }

    /// Push one row of f32 input by writing directly into the resizer's internal buffer.
    ///
    /// The closure receives `&mut [f32]` of length `input_row_len()`. Write your
    /// f32 pixel data into this slice. After the closure returns, premultiply and
    /// cache run without a `copy_from_slice` (saves one memcpy vs `push_row_f32`).
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
        let pixel_len = self.config.input_row_len();
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
        let pixel_len = self.config.input_row_len();
        let stride = self.config.effective_in_stride();
        if row.len() < pixel_len.min(stride) {
            return Err(at!(StreamingError::InputTooShort));
        }

        let pixel_data = &row[..pixel_len];

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
        let pixel_len = self.config.input_row_len();
        if row.len() < pixel_len {
            return Err(at!(StreamingError::InputTooShort));
        }

        match self.path {
            StreamingPath::I16Srgb => {
                // i16 → u8 cache (values are u8-range zero-extended)
                self.check_ring_buffer().at()?;
                let cache_slot = self.cache_write_idx % self.cache_size;
                for (s, d) in row[..pixel_len]
                    .iter()
                    .zip(self.u8_v_cache[cache_slot][..pixel_len].iter_mut())
                {
                    *d = (*s).clamp(0, 255) as u8;
                }
                self.cache_write_idx += 1;
                self.input_rows_received += 1;
                Ok(())
            }
            StreamingPath::I16Linear => {
                // Copy directly into the linearized_row_i16 buffer, then push
                self.linearized_row_i16[..pixel_len].copy_from_slice(&row[..pixel_len]);
                self.push_row_internal_i16().at()
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
        let pixel_len = self.config.input_row_len();
        if row.len() < pixel_len {
            return Err(at!(StreamingError::InputTooShort));
        }

        // Copy into temp buffer (skip transfer function)
        self.temp_input_f32[..pixel_len].copy_from_slice(&row[..pixel_len]);

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
        self.output_rows_available()
    }

    // =========================================================================
    // Output methods — lazy pull-based production
    // =========================================================================

    /// Pull the next output row as u8. Returns `None` if more input is needed
    /// or all output rows have been produced.
    ///
    /// Lazily produces one output row: V-filter → H-filter → composite → unpremultiply → u8 convert.
    /// The returned slice borrows from the resizer's internal buffer and is valid
    /// until the next method call on this resizer.
    pub fn next_output_row(&mut self) -> Option<&[u8]> {
        if !self.can_produce_next_output() {
            return None;
        }
        let row_len = self.config.out_width as usize * self.channels;
        match self.path {
            StreamingPath::I16Srgb | StreamingPath::I16Linear => {
                // i16 paths produce u8 directly — swap buffer out to avoid borrow conflict
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

    /// Pull the next output row directly into a caller-provided u8 buffer.
    /// Returns `true` if a row was produced, `false` if more input is needed.
    ///
    /// `dst` must be at least `output_row_len()` elements long.
    /// Skips the internal `output_buf_u8` — writes directly to the caller's buffer.
    pub fn next_output_row_into(&mut self, dst: &mut [u8]) -> bool {
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

    /// Pull the next output row as f32. Returns `None` if more input is needed.
    ///
    /// Returns a reference to the H-filter output directly (no format conversion).
    /// The returned slice borrows from the resizer's internal buffer.
    pub fn next_output_row_f32(&mut self) -> Option<&[f32]> {
        if !self.can_produce_next_output() {
            return None;
        }
        debug_assert_eq!(
            self.path,
            StreamingPath::F32,
            "next_output_row_f32 requires f32 path"
        );
        self.produce_next_f32();
        let row_len = self.config.out_width as usize * self.channels;
        Some(&self.temp_output_f32[..row_len])
    }

    /// Pull the next output row as u16. Returns `None` if more input is needed.
    ///
    /// Uses the sRGB transfer function to convert from linear f32 to encoded u16.
    pub fn next_output_row_u16(&mut self) -> Option<&[u16]> {
        if !self.can_produce_next_output() {
            return None;
        }
        debug_assert_eq!(
            self.path,
            StreamingPath::F32,
            "next_output_row_u16 requires f32 path"
        );
        self.produce_next_f32();
        let row_len = self.config.out_width as usize * self.channels;
        Self::encode_output_u16(
            &self.temp_output_f32[..row_len],
            &mut self.output_buf_u16[..row_len],
            self.config.effective_output_transfer(),
            self.channels,
            self.alpha_is_last,
        );
        Some(&self.output_buf_u16[..row_len])
    }

    /// Pull the next output row directly into a caller-provided u16 buffer.
    /// Returns `true` if a row was produced, `false` if more input is needed.
    ///
    /// `dst` must be at least `output_row_len()` elements long.
    pub fn next_output_row_u16_into(&mut self, dst: &mut [u16]) -> bool {
        if !self.can_produce_next_output() {
            return false;
        }
        debug_assert_eq!(
            self.path,
            StreamingPath::F32,
            "next_output_row_u16_into requires f32 path"
        );
        self.produce_next_f32();
        let row_len = self.config.out_width as usize * self.channels;
        Self::encode_output_u16(
            &self.temp_output_f32[..row_len],
            &mut dst[..row_len],
            self.config.effective_output_transfer(),
            self.channels,
            self.alpha_is_last,
        );
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

        let needed_max = right.min(self.config.in_height as i32 - 1).max(0) as u32;
        needed_max < self.input_rows_received
    }

    /// V-filter → H-filter one output row into `temp_output_f32`, apply composite and unpremultiply.
    /// Increments `output_rows_produced`. Caller must check `can_produce_next_output` first.
    /// Only called for the F32 path. Ring buffer stores f16 (u16), V-filter outputs f32.
    fn produce_next_f32(&mut self) {
        let out_y = self.output_rows_produced;

        let left = self.v_weights.left[out_y as usize];
        let tap_count = self.v_weights.tap_count(out_y as usize);
        let weights = self.v_weights.weights(out_y as usize);
        let in_row_len = self.config.in_width as usize * self.channels;
        let out_row_len = self.config.out_width as usize * self.channels;

        // Step 1: V-filter from f16 v_cache into temp_v_output (f32, in_width-wide)
        with_v_rows(
            &self.v_cache,
            self.cache_size,
            left,
            tap_count,
            self.config.in_height,
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

        // Step 3: composite + unpremul (operates on temp_output_f32)
        composite::composite_dispatch(
            &mut self.temp_output_f32[..out_row_len],
            &mut self.background,
            &mut self.composite_bg_row,
            out_y,
            self.channels as u8,
        );

        if self.needs_premul {
            simd::unpremultiply_alpha_row(&mut self.temp_output_f32[..out_row_len]);
        }

        self.output_rows_produced += 1;
    }

    /// Produce one output row via the I16Srgb path: u8 V-filter → u8 H-filter → u8 output.
    /// Result written to `output_buf_u8`. Increments `output_rows_produced`.
    fn produce_next_i16_srgb(&mut self, dst: &mut [u8]) {
        let out_y = self.output_rows_produced;
        let i16_v_weights = self.i16_v_weights.as_ref().unwrap();

        let left = i16_v_weights.left[out_y as usize];
        let tap_count = i16_v_weights.tap_count(out_y as usize);
        let weights = i16_v_weights.weights(out_y as usize);
        let in_row_len = self.config.in_width as usize * self.channels;
        let out_row_len = self.config.out_width as usize * self.channels;

        // Step 1: V-filter from u8_v_cache into temp_v_output_u8
        with_v_rows(
            &self.u8_v_cache,
            self.cache_size,
            left,
            tap_count,
            self.config.in_height,
            in_row_len,
            |rows| {
                simd::filter_v_row_u8_i16(rows, &mut self.temp_v_output_u8[..in_row_len], weights)
            },
        );

        // Step 2: H-filter u8 → u8 (via i16 weights)
        let i16_h_weights = self.i16_h_weights.as_ref().unwrap();
        simd::filter_h_u8_i16(
            &self.temp_v_output_u8,
            &mut dst[..out_row_len],
            i16_h_weights,
            self.channels,
        );

        // Step 3: unpremultiply if needed (in-place on u8 output)
        if self.needs_premul {
            simd::unpremultiply_u8_row(&mut dst[..out_row_len]);
        }

        self.output_rows_produced += 1;
    }

    /// Produce one output row via the I16Linear path: i16 V-filter → i16 H-filter → sRGB u8.
    /// Result written to `dst`. Increments `output_rows_produced`.
    fn produce_next_i16_linear(&mut self, dst: &mut [u8]) {
        let out_y = self.output_rows_produced;
        let i16_v_weights = self.i16_v_weights.as_ref().unwrap();

        let left = i16_v_weights.left[out_y as usize];
        let tap_count = i16_v_weights.tap_count(out_y as usize);
        let weights = i16_v_weights.weights(out_y as usize);
        let in_row_len = self.config.in_width as usize * self.channels;
        let out_row_len = self.config.out_width as usize * self.channels;

        // Step 1: V-filter from i16_v_cache into temp_v_output_i16
        with_v_rows(
            &self.i16_v_cache,
            self.cache_size,
            left,
            tap_count,
            self.config.in_height,
            in_row_len,
            |rows| simd::filter_v_row_i16(rows, &mut self.temp_v_output_i16[..in_row_len], weights),
        );

        // Step 2: H-filter i16 → i16 (via i16 weights)
        let i16_h_weights = self.i16_h_weights.as_ref().unwrap();
        simd::filter_h_i16_i16(
            &self.temp_v_output_i16,
            &mut self.temp_h_output_i16[..out_row_len],
            i16_h_weights,
            self.channels,
        );

        // Step 3: linear i12 → sRGB u8 (LUT)
        color::linear_i12_to_srgb_u8_row(
            &self.temp_h_output_i16[..out_row_len],
            &mut dst[..out_row_len],
        );

        self.output_rows_produced += 1;
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
        let pixel_len = self.config.in_width as usize * self.channels;
        simd::f32_to_f16_row(
            &self.temp_input_f32[..pixel_len],
            &mut self.v_cache[cache_slot][..pixel_len],
        );

        self.cache_write_idx += 1;
        self.input_rows_received += 1;
        Ok(())
    }

    /// Internal: cache i16 row from `linearized_row_i16` into the i16 ring buffer (I16Linear path).
    #[track_caller]
    fn push_row_internal_i16(&mut self) -> Result<(), At<StreamingError>> {
        self.check_ring_buffer().at()?;

        let cache_slot = self.cache_write_idx % self.cache_size;
        let pixel_len = self.config.in_width as usize * self.channels;
        self.i16_v_cache[cache_slot][..pixel_len]
            .copy_from_slice(&self.linearized_row_i16[..pixel_len]);

        self.cache_write_idx += 1;
        self.input_rows_received += 1;
        Ok(())
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
}
