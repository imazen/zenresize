//! Single-plane i16 image resizer.
//!
//! Provides a high-performance resize API for single-channel i16 data,
//! useful for YCbCr-domain processing where each plane (Y, Cb, Cr) is
//! resized independently without color conversion overhead.
//!
//! Values are expected in the i12 range [0, 4095]. For signed data (e.g.,
//! JPEG chroma coefficients in [-128, 127]), offset by +2048 before resize
//! and subtract after.
//!
//! # Example
//!
//! ```
//! use zenresize::{PlaneResizer, Filter};
//!
//! let src_w = 8;
//! let src_h = 8;
//! let dst_w = 4;
//! let dst_h = 4;
//!
//! let mut resizer = PlaneResizer::new(Filter::Robidoux, src_w, src_h, dst_w, dst_h);
//!
//! let input = vec![2048i16; src_w as usize * src_h as usize];
//! let mut output = vec![0i16; dst_w as usize * dst_h as usize];
//!
//! resizer.resize_plane(&input, src_w as usize, &mut output, dst_w as usize);
//! ```

#[cfg(not(feature = "std"))]
use alloc::vec;

use crate::filter::InterpolationDetails;
use crate::simd;
use crate::weights::I16WeightTable;

/// Single-channel i16 plane resizer with pre-computed weight tables.
///
/// Reusable across multiple resize operations with the same dimensions.
/// Uses SIMD-accelerated i16 convolution kernels internally.
///
/// Values are clamped to [0, 4095] (i12 range) by the convolution kernels.
pub struct PlaneResizer {
    h_weights: I16WeightTable,
    v_weights: I16WeightTable,
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
    /// Horizontal pass output: src_h rows of dst_w i16 values.
    intermediate: Vec<i16>,
}

impl PlaneResizer {
    /// Create a new plane resizer for the given dimensions and filter.
    ///
    /// Pre-computes weight tables. Reuse the resizer for multiple planes
    /// with the same dimensions to amortize this cost.
    pub fn new(filter: Filter, src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Self {
        assert!(src_w > 0 && src_h > 0, "source dimensions must be positive");
        assert!(dst_w > 0 && dst_h > 0, "target dimensions must be positive");

        let details = InterpolationDetails::create(filter);
        let h_weights = I16WeightTable::new(src_w, dst_w, &details);
        let v_weights = I16WeightTable::new(src_h, dst_h, &details);

        let intermediate = vec![0i16; src_h as usize * dst_w as usize];

        Self {
            h_weights,
            v_weights,
            src_w: src_w as usize,
            src_h: src_h as usize,
            dst_w: dst_w as usize,
            dst_h: dst_h as usize,
            intermediate,
        }
    }

    /// Resize a single i16 plane.
    ///
    /// Input: `src_h` rows of `src_w` values, laid out with `input_stride` elements per row.
    /// Output: `dst_h` rows of `dst_w` values, laid out with `output_stride` elements per row.
    ///
    /// Values are clamped to [0, 4095] by the convolution kernels.
    pub fn resize_plane(
        &mut self,
        input: &[i16],
        input_stride: usize,
        output: &mut [i16],
        output_stride: usize,
    ) {
        assert!(
            input_stride >= self.src_w,
            "input_stride ({input_stride}) must be >= src_w ({})",
            self.src_w
        );
        assert!(
            output_stride >= self.dst_w,
            "output_stride ({output_stride}) must be >= dst_w ({})",
            self.dst_w
        );
        assert!(
            input.len() >= (self.src_h - 1) * input_stride + self.src_w,
            "input buffer too small"
        );
        assert!(
            output.len() >= (self.dst_h - 1) * output_stride + self.dst_w,
            "output buffer too small"
        );

        // Horizontal pass: filter each input row into intermediate buffer.
        // intermediate layout: src_h rows × dst_w, contiguous (stride = dst_w).
        for y in 0..self.src_h {
            let in_start = y * input_stride;
            let in_row = &input[in_start..in_start + self.src_w];
            let out_start = y * self.dst_w;

            simd::filter_h_i16_i16(
                in_row,
                &mut self.intermediate[out_start..out_start + self.dst_w],
                &self.h_weights,
                1, // single channel
            );
        }

        // Vertical pass: filter all intermediate rows into output.
        // If output stride matches dst_w, write directly into output.
        if output_stride == self.dst_w {
            simd::filter_v_all_i16_i16(
                &self.intermediate,
                output,
                self.dst_w,
                self.src_h,
                self.dst_h,
                &self.v_weights,
            );
        } else {
            // Need a temporary contiguous buffer for the vertical pass,
            // then copy with stride.
            let mut temp = vec![0i16; self.dst_h * self.dst_w];
            simd::filter_v_all_i16_i16(
                &self.intermediate,
                &mut temp,
                self.dst_w,
                self.src_h,
                self.dst_h,
                &self.v_weights,
            );
            for y in 0..self.dst_h {
                let src_start = y * self.dst_w;
                let dst_start = y * output_stride;
                output[dst_start..dst_start + self.dst_w]
                    .copy_from_slice(&temp[src_start..src_start + self.dst_w]);
            }
        }
    }

    /// Source width.
    pub fn src_width(&self) -> usize {
        self.src_w
    }

    /// Source height.
    pub fn src_height(&self) -> usize {
        self.src_h
    }

    /// Destination width.
    pub fn dst_width(&self) -> usize {
        self.dst_w
    }

    /// Destination height.
    pub fn dst_height(&self) -> usize {
        self.dst_h
    }
}

// Re-export Filter here for ergonomics
use crate::filter::Filter;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_plane_preserves_value() {
        let src_w = 20u32;
        let src_h = 20;
        let dst_w = 10u32;
        let dst_h = 10;

        let mut resizer = PlaneResizer::new(Filter::Robidoux, src_w, src_h, dst_w, dst_h);

        let value = 2048i16;
        let input = vec![value; src_w as usize * src_h as usize];
        let mut output = vec![0i16; dst_w as usize * dst_h as usize];

        resizer.resize_plane(&input, src_w as usize, &mut output, dst_w as usize);

        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v - value).unsigned_abs() <= 2,
                "pixel {i}: got {v}, expected ~{value}"
            );
        }
    }

    #[test]
    fn upscale_preserves_value() {
        let src_w = 10u32;
        let src_h = 10;
        let dst_w = 20u32;
        let dst_h = 20;

        let mut resizer = PlaneResizer::new(Filter::Lanczos, src_w, src_h, dst_w, dst_h);

        let value = 1000i16;
        let input = vec![value; src_w as usize * src_h as usize];
        let mut output = vec![0i16; dst_w as usize * dst_h as usize];

        resizer.resize_plane(&input, src_w as usize, &mut output, dst_w as usize);

        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v - value).unsigned_abs() <= 2,
                "pixel {i}: got {v}, expected ~{value}"
            );
        }
    }

    #[test]
    fn resize_with_stride() {
        let src_w = 8u32;
        let src_h = 8;
        let dst_w = 4u32;
        let dst_h = 4;
        let in_stride = 16usize; // wider than src_w
        let out_stride = 8usize; // wider than dst_w

        let mut resizer = PlaneResizer::new(Filter::Robidoux, src_w, src_h, dst_w, dst_h);

        let value = 2000i16;
        let mut input = vec![0i16; src_h as usize * in_stride];
        for y in 0..src_h as usize {
            for x in 0..src_w as usize {
                input[y * in_stride + x] = value;
            }
        }

        let mut output = vec![0i16; dst_h as usize * out_stride];
        resizer.resize_plane(&input, in_stride, &mut output, out_stride);

        for y in 0..dst_h as usize {
            for x in 0..dst_w as usize {
                let v = output[y * out_stride + x];
                assert!(
                    (v - value).unsigned_abs() <= 2,
                    "({x},{y}): got {v}, expected ~{value}"
                );
            }
        }
    }

    #[test]
    fn identity_resize() {
        let w = 16u32;
        let h = 16;

        let mut resizer = PlaneResizer::new(Filter::Robidoux, w, h, w, h);

        // Gradient pattern
        let mut input = vec![0i16; w as usize * h as usize];
        for y in 0..h as usize {
            for x in 0..w as usize {
                input[y * w as usize + x] = ((x + y) * 256 / 32) as i16;
            }
        }

        let mut output = vec![0i16; w as usize * h as usize];
        resizer.resize_plane(&input, w as usize, &mut output, w as usize);

        // Identity resize should be very close to input.
        // i16 fixed-point rounding allows ±2 deviation.
        for i in 0..input.len() {
            assert!(
                (output[i] - input[i]).unsigned_abs() <= 2,
                "pixel {i}: input={}, output={}",
                input[i],
                output[i]
            );
        }
    }

    #[test]
    fn reuse_resizer() {
        let mut resizer = PlaneResizer::new(Filter::Robidoux, 20, 20, 10, 10);

        let input1 = vec![1000i16; 400];
        let mut output1 = vec![0i16; 100];
        resizer.resize_plane(&input1, 20, &mut output1, 10);

        let input2 = vec![2000i16; 400];
        let mut output2 = vec![0i16; 100];
        resizer.resize_plane(&input2, 20, &mut output2, 10);

        // Results should differ
        assert_ne!(output1, output2);

        // Re-running with same input should give same result
        let mut output1b = vec![0i16; 100];
        resizer.resize_plane(&input1, 20, &mut output1b, 10);
        assert_eq!(output1, output1b);
    }
}
