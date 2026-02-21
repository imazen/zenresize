//! Row-at-a-time streaming resize state machine.
//!
//! The streaming resizer uses horizontal-first architecture with lazy pull-based
//! output production and natural backpressure:
//!
//! 1. `push_row()`: H-filter the input row and cache it in the ring buffer (no output produced)
//! 2. `next_output_row()`: Lazily V-filter one output row when enough cached rows exist
//! 3. Caller MUST drain all available output between `push_row` calls (ring buffer contract)
//!
//! # Backpressure Contract
//!
//! The H-cache ring buffer has `max_taps + 2` slots. The caller MUST drain all
//! available output rows between `push_row` calls.
//!
//! ```ignore
//! for row in input_rows {
//!     resizer.push_row(row)?;
//!     while let Some(output) = resizer.next_output_row() {
//!         encoder.write_row(output);
//!     }
//! }
//! resizer.finish();
//! while let Some(output) = resizer.next_output_row() {
//!     encoder.write_row(output);
//! }
//! ```

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::color;
use crate::composite::{self, Background, CompositeError, NoBackground};
use crate::filter::InterpolationDetails;
use crate::pixel::ResizeConfig;
use crate::simd;
use crate::transfer::{Srgb, TransferFunction};
use crate::weights::F32WeightTable;

/// Errors from streaming resize push operations.
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
    h_weights: F32WeightTable,
    v_weights: F32WeightTable,
    channels: usize,
    needs_premul: bool,
    alpha_is_last: bool,

    /// Ring buffer of horizontally-filtered rows (f32).
    h_cache: Vec<Vec<f32>>,
    /// Which ring buffer slot to write the next horizontally-filtered row into.
    cache_write_idx: usize,
    /// Total number of cache slots.
    cache_size: usize,

    /// Temporary buffer for format conversion (u8/u16 → f32).
    temp_input_f32: Vec<f32>,
    /// Temporary buffer for V-filter output (f32).
    temp_output_f32: Vec<f32>,

    /// Reusable u8 conversion buffer (allocated once, used by `next_output_row`).
    output_buf_u8: Vec<u8>,
    /// Reusable u16 conversion buffer (allocated once, used by `next_output_row_u16`).
    output_buf_u16: Vec<u16>,

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
        Self::new_inner(config, NoBackground, false)
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
    pub fn with_background(config: &ResizeConfig, background: B) -> Result<Self, CompositeError> {
        if config.input_format.layout().is_premultiplied() {
            return Err(CompositeError::PremultipliedInput);
        }
        Ok(Self::new_inner(config, background, true))
    }

    fn new_inner(config: &ResizeConfig, background: B, has_composite: bool) -> Self {
        config.validate().expect("invalid resize config");

        let mut config = config.clone();
        // Compositing requires linear f32 path
        if has_composite && !background.is_transparent() {
            config.linear = true;
        }

        let filter = InterpolationDetails::create(config.filter);
        let h_weights = F32WeightTable::new(config.in_width, config.out_width, &filter);
        let v_weights = F32WeightTable::new(config.in_height, config.out_height, &filter);

        let layout = config.input_format.layout();
        let channels = layout.channels() as usize;
        let needs_premul = layout.needs_premultiply();
        let alpha_is_last = layout.alpha_is_last_channel();

        let cache_size = v_weights.max_taps + 2;
        let row_len = config.out_width as usize * channels;

        let h_cache = (0..cache_size).map(|_| vec![0.0f32; row_len]).collect();

        // Pad with max_taps extra zero elements so SIMD H-pass reads
        // beyond the valid input range hit zeros. Zero-padded weights make these inert.
        let temp_input_f32 =
            vec![0.0f32; config.in_width as usize * channels + h_weights.max_taps * channels];
        let temp_output_f32 = vec![0.0f32; row_len];

        // Reusable output conversion buffers (allocated once)
        let output_buf_u8 = vec![0u8; row_len];
        let output_buf_u16 = vec![0u16; row_len];

        // Only allocate bg row buffer for non-solid, non-transparent backgrounds
        let needs_bg_row =
            has_composite && !background.is_transparent() && background.solid_pixel().is_none();
        let composite_bg_row = if needs_bg_row {
            vec![0.0f32; row_len]
        } else {
            Vec::new()
        };

        Self {
            config,
            h_weights,
            v_weights,
            channels,
            needs_premul,
            alpha_is_last,
            h_cache,
            cache_write_idx: 0,
            cache_size,
            temp_input_f32,
            temp_output_f32,
            output_buf_u8,
            output_buf_u16,
            input_rows_received: 0,
            output_rows_produced: 0,
            finished: false,
            background,
            composite_bg_row,
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

    /// Mutable reference to the background (e.g., for pushing rows to [`StreamedBackground`]).
    pub fn background_mut(&mut self) -> &mut B {
        &mut self.background
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

    // =========================================================================
    // Input methods
    // =========================================================================

    /// Push one row of u8 input pixels. H-filters and caches the row.
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
    pub fn push_row(&mut self, row: &[u8]) -> Result<(), StreamingError> {
        if self.finished {
            return Err(StreamingError::AlreadyFinished);
        }
        let pixel_len = self.config.input_row_len();
        let stride = self.config.effective_in_stride();
        if row.len() < pixel_len.min(stride) {
            return Err(StreamingError::InputTooShort);
        }

        let pixel_data = &row[..pixel_len];

        if self.config.needs_linearization() {
            color::srgb_u8_to_linear_f32(
                pixel_data,
                &mut self.temp_input_f32[..pixel_len],
                self.channels,
                self.alpha_is_last,
            );
        } else {
            simd::u8_to_f32_row(pixel_data, &mut self.temp_input_f32[..pixel_len]);
        }

        if self.needs_premul {
            simd::premultiply_alpha_row(&mut self.temp_input_f32[..pixel_len]);
        }

        self.push_row_internal()
    }

    /// Push one row of f32 input pixels. H-filters and caches the row.
    ///
    /// Caller MUST drain all available output rows before pushing the next input row.
    ///
    /// # Errors
    ///
    /// Returns [`StreamingError::AlreadyFinished`] if called after `finish()`.
    /// Returns [`StreamingError::InputTooShort`] if `row` is shorter than required.
    /// Returns [`StreamingError::RingBufferOverflow`] if output was not drained.
    pub fn push_row_f32(&mut self, row: &[f32]) -> Result<(), StreamingError> {
        if self.finished {
            return Err(StreamingError::AlreadyFinished);
        }
        let pixel_len = self.config.input_row_len();
        if row.len() < pixel_len {
            return Err(StreamingError::InputTooShort);
        }

        self.temp_input_f32[..pixel_len].copy_from_slice(&row[..pixel_len]);

        if self.needs_premul {
            simd::premultiply_alpha_row(&mut self.temp_input_f32[..pixel_len]);
        }

        self.push_row_internal()
    }

    /// Push one row of f32 input by writing directly into the resizer's internal buffer.
    ///
    /// The closure receives `&mut [f32]` of length `input_row_len()`. Write your
    /// f32 pixel data into this slice. After the closure returns, premultiply and
    /// H-filter run without a `copy_from_slice` (saves one memcpy vs `push_row_f32`).
    ///
    /// Caller MUST drain all available output rows before pushing the next input row.
    ///
    /// # Errors
    ///
    /// Returns [`StreamingError::AlreadyFinished`] if called after `finish()`.
    /// Returns [`StreamingError::RingBufferOverflow`] if output was not drained.
    pub fn push_row_f32_with<F: FnOnce(&mut [f32])>(&mut self, f: F) -> Result<(), StreamingError> {
        if self.finished {
            return Err(StreamingError::AlreadyFinished);
        }
        let pixel_len = self.config.input_row_len();
        f(&mut self.temp_input_f32[..pixel_len]);

        if self.needs_premul {
            simd::premultiply_alpha_row(&mut self.temp_input_f32[..pixel_len]);
        }

        self.push_row_internal()
    }

    /// Push one row of u16 input pixels. H-filters and caches the row.
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
    pub fn push_row_u16(&mut self, row: &[u16]) -> Result<(), StreamingError> {
        if self.finished {
            return Err(StreamingError::AlreadyFinished);
        }
        let pixel_len = self.config.input_row_len();
        let stride = self.config.effective_in_stride();
        if row.len() < pixel_len.min(stride) {
            return Err(StreamingError::InputTooShort);
        }

        let pixel_data = &row[..pixel_len];
        let tf = Srgb;

        tf.u16_to_linear_f32(
            pixel_data,
            &mut self.temp_input_f32[..pixel_len],
            &(),
            self.channels,
            self.alpha_is_last,
            self.needs_premul,
        );

        self.push_row_internal()
    }

    /// Signal end of input. No more rows may be pushed after this call.
    ///
    /// After calling `finish()`, drain remaining output rows:
    /// ```ignore
    /// resizer.finish();
    /// while let Some(row) = resizer.next_output_row() {
    ///     encoder.write_row(row);
    /// }
    /// ```
    pub fn finish(&mut self) {
        self.finished = true;
    }

    // =========================================================================
    // Output methods — lazy pull-based production
    // =========================================================================

    /// Pull the next output row as u8. Returns `None` if more input is needed
    /// or all output rows have been produced.
    ///
    /// Lazily produces one output row: V-filter → composite → unpremultiply → u8 convert.
    /// The returned slice borrows from the resizer's internal buffer and is valid
    /// until the next method call on this resizer.
    pub fn next_output_row(&mut self) -> Option<&[u8]> {
        if !self.can_produce_next_output() {
            return None;
        }
        self.produce_next_into_temp();
        let row_len = self.config.out_width as usize * self.channels;
        if self.config.needs_linearization() {
            color::linear_f32_to_srgb_u8(
                &self.temp_output_f32[..row_len],
                &mut self.output_buf_u8[..row_len],
                self.channels,
                self.alpha_is_last,
            );
        } else {
            simd::f32_to_u8_row(
                &self.temp_output_f32[..row_len],
                &mut self.output_buf_u8[..row_len],
            );
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
        self.produce_next_into_temp();
        let row_len = self.config.out_width as usize * self.channels;
        if self.config.needs_linearization() {
            color::linear_f32_to_srgb_u8(
                &self.temp_output_f32[..row_len],
                &mut dst[..row_len],
                self.channels,
                self.alpha_is_last,
            );
        } else {
            simd::f32_to_u8_row(&self.temp_output_f32[..row_len], &mut dst[..row_len]);
        }
        true
    }

    /// Pull the next output row as f32. Returns `None` if more input is needed.
    ///
    /// Returns a reference to the V-filter output directly (no format conversion).
    /// The returned slice borrows from the resizer's internal buffer.
    pub fn next_output_row_f32(&mut self) -> Option<&[f32]> {
        if !self.can_produce_next_output() {
            return None;
        }
        self.produce_next_into_temp();
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
        self.produce_next_into_temp();
        let row_len = self.config.out_width as usize * self.channels;
        let tf = Srgb;
        tf.linear_f32_to_u16(
            &self.temp_output_f32[..row_len],
            &mut self.output_buf_u16[..row_len],
            &(),
            self.channels,
            self.alpha_is_last,
            false, // unpremultiply already happened in produce_next_into_temp
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
        self.produce_next_into_temp();
        let row_len = self.config.out_width as usize * self.channels;
        let tf = Srgb;
        tf.linear_f32_to_u16(
            &self.temp_output_f32[..row_len],
            &mut dst[..row_len],
            &(),
            self.channels,
            self.alpha_is_last,
            false,
        );
        true
    }

    // =========================================================================
    // Internal
    // =========================================================================

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

    /// V-filter one output row into `temp_output_f32`, apply composite and unpremultiply.
    /// Increments `output_rows_produced`. Caller must check `can_produce_next_output` first.
    fn produce_next_into_temp(&mut self) {
        let out_y = self.output_rows_produced;

        let left = self.v_weights.left[out_y as usize];
        let tap_count = self.v_weights.tap_count(out_y as usize);
        let weights = self.v_weights.weights(out_y as usize);
        let row_len = self.config.out_width as usize * self.channels;

        let mut rows: Vec<&[f32]> = Vec::with_capacity(tap_count);
        for t in 0..tap_count {
            let input_y = (left + t as i32).clamp(0, self.config.in_height as i32 - 1) as u32;
            let cache_idx = input_y as usize % self.cache_size;
            rows.push(&self.h_cache[cache_idx][..row_len]);
        }

        simd::filter_v_row_f32(&rows, &mut self.temp_output_f32[..row_len], weights);

        // === Composite: source-over onto background ===
        composite::composite_dispatch(
            &mut self.temp_output_f32[..row_len],
            &mut self.background,
            &mut self.composite_bg_row,
            out_y,
            self.channels as u8,
        );

        if self.needs_premul {
            simd::unpremultiply_alpha_row(&mut self.temp_output_f32[..row_len]);
        }

        self.output_rows_produced += 1;
    }

    /// Internal: H-filter the row in `temp_input_f32` and cache it.
    fn push_row_internal(&mut self) -> Result<(), StreamingError> {
        // Verify ring buffer safety before writing
        if self.output_rows_produced < self.config.out_height {
            let oldest_needed =
                self.v_weights.left[self.output_rows_produced as usize].max(0) as usize;
            if self.cache_write_idx > oldest_needed
                && self.cache_write_idx - oldest_needed >= self.cache_size
            {
                return Err(StreamingError::RingBufferOverflow);
            }
        }

        let cache_slot = self.cache_write_idx % self.cache_size;
        simd::filter_h_row_f32(
            &self.temp_input_f32,
            &mut self.h_cache[cache_slot],
            &self.h_weights,
            self.channels,
        );

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
    use crate::pixel::{PixelFormat, PixelLayout};

    fn make_config(in_w: u32, in_h: u32, out_w: u32, out_h: u32) -> ResizeConfig {
        ResizeConfig::builder(in_w, in_h, out_w, out_h)
            .format(PixelFormat::Srgb8(PixelLayout::Rgba))
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
        assert_eq!(config.sharpen, 0.0);
        assert_eq!(config.in_stride, 0);
        assert_eq!(config.out_stride, 0);
    }

    #[test]
    fn test_input_output_row_len() {
        let config = ResizeConfig::builder(100, 100, 50, 50)
            .format(PixelFormat::Srgb8(PixelLayout::Rgba))
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
            .format(PixelFormat::LinearF32(PixelLayout::Rgba))
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
        assert_eq!(resizer.push_row(&row), Err(StreamingError::AlreadyFinished));
    }

    #[test]
    fn test_input_too_short_returns_error() {
        let config = make_config(10, 10, 5, 5);
        let mut resizer = StreamingResize::new(&config);

        let short_row = vec![128u8; 10]; // needs 10*4=40
        assert_eq!(
            resizer.push_row(&short_row),
            Err(StreamingError::InputTooShort)
        );
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

        let bg = SolidBackground::transparent(PixelLayout::Rgba);
        let mut r2 = StreamingResize::with_background(&config, bg).unwrap();
        let rows2 = push_drain_collect_u8(&mut r2, &row, 20);

        assert_eq!(rows1, rows2);
    }

    #[test]
    fn solid_opaque_bg_makes_output_opaque() {
        let config = ResizeConfig::builder(10, 10, 10, 10)
            .format(PixelFormat::Srgb8(PixelLayout::Rgba))
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

        let bg = SolidBackground::white(PixelLayout::Rgba);
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
            .format(PixelFormat::Srgb8(PixelLayout::RgbaPremul))
            .build();

        let bg = SolidBackground::white(PixelLayout::RgbaPremul);
        let result = StreamingResize::with_background(&config, bg);
        assert!(
            matches!(result, Err(CompositeError::PremultipliedInput)),
            "expected PremultipliedInput error"
        );
    }

    // === u16 tests ===

    #[test]
    fn test_streaming_u16_constant_color() {
        let config = ResizeConfig::builder(20, 20, 10, 10)
            .filter(Filter::Lanczos)
            .format(PixelFormat::Encoded16(PixelLayout::Rgba))
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
            .format(PixelFormat::Encoded16(PixelLayout::Rgba))
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
            assert!(
                (a as i32 - b as i32).unsigned_abs() <= 1,
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
            .format(PixelFormat::Encoded16(PixelLayout::Rgb))
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
}
