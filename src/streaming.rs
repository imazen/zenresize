//! Row-at-a-time streaming resize state machine.
//!
//! The streaming resizer uses horizontal-first architecture:
//! 1. Each input row is immediately horizontally filtered and cached
//! 2. When enough cached rows exist for the vertical filter window,
//!    vertical filtering produces output rows
//! 3. Old cached rows are evicted as processing advances

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::color;
use crate::filter::InterpolationDetails;
use crate::pixel::ResizeConfig;
use crate::simd;
use crate::weights::F32WeightTable;

/// Streaming resize state machine.
///
/// Push input rows one at a time, pull output rows as they become available.
pub struct StreamingResize {
    config: ResizeConfig,
    h_weights: F32WeightTable,
    v_weights: F32WeightTable,
    channels: usize,
    has_alpha: bool,

    /// Ring buffer of horizontally-filtered rows (f32 linear).
    /// Size: v_weights.max_taps rows × (out_width * channels) f32 values.
    h_cache: Vec<Vec<f32>>,
    /// Which ring buffer slot to write the next horizontally-filtered row into.
    cache_write_idx: usize,
    /// Total number of cache slots.
    cache_size: usize,

    /// Temporary buffer for format conversion (u8 → f32).
    temp_input_f32: Vec<f32>,
    /// Temporary buffer for output format conversion (f32 → u8).
    temp_output_f32: Vec<f32>,

    /// Number of input rows received.
    input_rows_received: u32,
    /// Number of output rows produced.
    output_rows_produced: u32,
    /// Queue of output rows ready to be pulled.
    output_queue: Vec<Vec<u8>>,
    /// Queue of f32 output rows (for f32 API).
    output_queue_f32: Vec<Vec<f32>>,
}

impl StreamingResize {
    /// Create a new streaming resizer.
    pub fn new(config: &ResizeConfig) -> Self {
        config.validate().expect("invalid resize config");

        let filter = InterpolationDetails::create(config.filter);
        let h_weights = F32WeightTable::new(config.in_width, config.out_width, &filter);
        let v_weights = F32WeightTable::new(config.in_height, config.out_height, &filter);

        let channels = config.input_format.channels() as usize;
        let has_alpha = config.input_format.has_alpha();

        // Cache needs enough rows for the max vertical filter window.
        // Add extra margin for edge handling.
        let cache_size = v_weights.max_taps + 2;
        let row_len = config.out_width as usize * channels;

        let h_cache = (0..cache_size).map(|_| vec![0.0f32; row_len]).collect();

        let temp_input_f32 = vec![0.0f32; config.in_width as usize * channels];
        let temp_output_f32 = vec![0.0f32; row_len];

        Self {
            config: config.clone(),
            h_weights,
            v_weights,
            channels,
            has_alpha,
            h_cache,
            cache_write_idx: 0,
            cache_size,
            temp_input_f32,
            temp_output_f32,
            input_rows_received: 0,
            output_rows_produced: 0,
            output_queue: Vec::new(),
            output_queue_f32: Vec::new(),
        }
    }

    /// How many input rows must be pushed before the first output row.
    pub fn initial_input_rows_needed(&self) -> u32 {
        // The first output pixel's vertical filter may need rows from
        // before the image (handled by clamping), but we need at least
        // enough rows to cover the filter window.
        let first_right = self.first_output_row_max_input();
        (first_right + 1).min(self.config.in_height)
    }

    /// Push one row of u8 input pixels. Returns number of output rows now available.
    pub fn push_row(&mut self, row: &[u8]) -> u32 {
        let expected_len = self.config.in_width as usize * self.channels;
        assert_eq!(row.len(), expected_len, "input row has wrong length");

        // Step 1: Convert u8 → f32 (with optional linearization)
        if self.config.needs_linearization() {
            color::srgb_u8_to_linear_f32(
                row,
                &mut self.temp_input_f32,
                self.channels,
                self.has_alpha,
            );
        } else {
            color::srgb_u8_to_f32(row, &mut self.temp_input_f32);
        }

        // Step 1b: Premultiply alpha if needed
        if self.has_alpha && self.channels == 4 {
            color::premultiply_alpha_f32(&mut self.temp_input_f32, self.channels);
        }

        self.push_row_f32_internal()
    }

    /// Push one row of f32 input pixels. Returns number of output rows now available.
    pub fn push_row_f32(&mut self, row: &[f32]) -> u32 {
        let expected_len = self.config.in_width as usize * self.channels;
        assert_eq!(row.len(), expected_len, "input row has wrong length");

        self.temp_input_f32.copy_from_slice(row);

        // Premultiply alpha if needed
        if self.has_alpha && self.channels == 4 {
            color::premultiply_alpha_f32(&mut self.temp_input_f32, self.channels);
        }

        self.push_row_f32_internal()
    }

    /// Internal: process the row in temp_input_f32.
    fn push_row_f32_internal(&mut self) -> u32 {
        // Step 2: Horizontal filter into cache
        let cache_slot = self.cache_write_idx % self.cache_size;
        simd::filter_h_row_f32(
            &self.temp_input_f32,
            &mut self.h_cache[cache_slot],
            &self.h_weights,
            self.channels,
        );

        self.cache_write_idx += 1;
        self.input_rows_received += 1;

        // Step 3: Check if any output rows are now ready
        self.produce_output_rows()
    }

    /// Signal end of input. Returns number of additional output rows produced.
    pub fn finish(&mut self) -> u32 {
        // All remaining output rows should now be producible
        // (using edge clamping for any missing bottom rows)
        let mut count = 0;
        while self.output_rows_produced < self.config.out_height {
            if self.try_produce_one_output_row() {
                count += 1;
            } else {
                break;
            }
        }
        count
    }

    /// Pull the next available u8 output row. Returns None if more input needed.
    pub fn next_output_row(&mut self) -> Option<Vec<u8>> {
        self.output_queue.pop()
    }

    /// Pull the next available f32 output row.
    pub fn next_output_row_f32(&mut self) -> Option<Vec<f32>> {
        self.output_queue_f32.pop()
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
    // Internal
    // =========================================================================

    /// Find the maximum input row index needed for the first output row.
    fn first_output_row_max_input(&self) -> u32 {
        if self.v_weights.is_empty() {
            return 0;
        }
        let left = self.v_weights.left[0];
        let taps = self.v_weights.tap_count(0);
        let right = left + taps as i32 - 1;
        right.max(0) as u32
    }

    /// Try to produce output rows from the current cache state.
    fn produce_output_rows(&mut self) -> u32 {
        let mut count = 0;
        while self.try_produce_one_output_row() {
            count += 1;
        }
        count
    }

    /// Try to produce one output row. Returns true if successful.
    fn try_produce_one_output_row(&mut self) -> bool {
        let out_y = self.output_rows_produced;
        if out_y >= self.config.out_height {
            return false;
        }

        let left = self.v_weights.left[out_y as usize];
        let tap_count = self.v_weights.tap_count(out_y as usize);
        let right = left + tap_count as i32 - 1;

        // Check if we have all needed input rows
        // (clamped to valid range)
        let needed_max = right.min(self.config.in_height as i32 - 1).max(0) as u32;
        if needed_max >= self.input_rows_received {
            return false;
        }

        // Gather row pointers from cache
        let weights = self.v_weights.weights(out_y as usize);
        let row_len = self.config.out_width as usize * self.channels;
        let mut rows: Vec<&[f32]> = Vec::with_capacity(tap_count);

        for t in 0..tap_count {
            let input_y = (left + t as i32).clamp(0, self.config.in_height as i32 - 1) as u32;
            // Map input_y to cache slot
            let cache_idx = input_y as usize % self.cache_size;
            rows.push(&self.h_cache[cache_idx][..row_len]);
        }

        // Vertical filter
        simd::filter_v_row_f32(&rows, &mut self.temp_output_f32[..row_len], weights);

        // Post-process: unpremultiply alpha, convert back to output format
        if self.has_alpha && self.channels == 4 {
            color::unpremultiply_alpha_f32(&mut self.temp_output_f32[..row_len], self.channels);
        }

        if self.config.output_format.is_u8() {
            let mut out_row = vec![0u8; row_len];
            if self.config.needs_linearization() {
                color::linear_f32_to_srgb_u8(
                    &self.temp_output_f32[..row_len],
                    &mut out_row,
                    self.channels,
                    self.has_alpha,
                );
            } else {
                color::f32_to_srgb_u8(&self.temp_output_f32[..row_len], &mut out_row);
            }
            // Push to front so pop() returns in order
            self.output_queue.insert(0, out_row);
        } else {
            let out_row = self.temp_output_f32[..row_len].to_vec();
            self.output_queue_f32.insert(0, out_row);
        }

        self.output_rows_produced += 1;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::Filter;
    use crate::pixel::{ColorSpace, PixelFormat};

    fn make_config(in_w: u32, in_h: u32, out_w: u32, out_h: u32) -> ResizeConfig {
        ResizeConfig {
            filter: Filter::Lanczos,
            in_width: in_w,
            in_height: in_h,
            out_width: out_w,
            out_height: out_h,
            input_format: PixelFormat::Srgb8 {
                channels: 4,
                has_alpha: true,
            },
            output_format: PixelFormat::Srgb8 {
                channels: 4,
                has_alpha: true,
            },
            sharpen: 0.0,
            color_space: ColorSpace::Srgb, // Use sRGB space for simpler testing
        }
    }

    #[test]
    fn test_streaming_produces_correct_row_count() {
        let config = make_config(100, 100, 50, 50);
        let mut resizer = StreamingResize::new(&config);

        let row = vec![128u8; 100 * 4];
        for _ in 0..100 {
            resizer.push_row(&row);
        }
        resizer.finish();

        assert_eq!(resizer.output_rows_produced(), 50);
    }

    #[test]
    fn test_streaming_constant_color() {
        let config = make_config(20, 20, 10, 10);
        let mut resizer = StreamingResize::new(&config);

        // All pixels are (128, 128, 128, 255)
        let row = vec![128u8; 20 * 4]
            .chunks_mut(4)
            .flat_map(|c| {
                c[0] = 128;
                c[1] = 128;
                c[2] = 128;
                c[3] = 255;
                c.iter().copied()
            })
            .collect::<Vec<u8>>();

        for _ in 0..20 {
            resizer.push_row(&row);
        }
        resizer.finish();

        let mut total_rows = 0;
        while let Some(out_row) = resizer.next_output_row() {
            total_rows += 1;
            // Check all pixels are approximately (128, 128, 128, 255)
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
        for _ in 0..10 {
            resizer.push_row(&row);
        }
        resizer.finish();

        assert_eq!(resizer.output_rows_produced(), 20);
    }

    #[test]
    fn test_streaming_same_size() {
        let config = make_config(10, 10, 10, 10);
        let mut resizer = StreamingResize::new(&config);

        let row = vec![100u8; 10 * 4];
        for _ in 0..10 {
            resizer.push_row(&row);
        }
        resizer.finish();

        assert_eq!(resizer.output_rows_produced(), 10);
    }
}
