//! Independently specified two-pass fixed-point RGB cubic resampling.
//!
//! Source and target samples are encoded RGB8 values; there is no transfer
//! conversion or alpha channel. Coefficients use 22 fractional bits. Each pass
//! rounds once and clamps to u8. This arithmetic was qualified against Pillow's
//! public RGB resize API, without using its implementation source.
use crate::weights::fixed22_weights;
use crate::{InterpolationDetails, PixelDescriptor, ResizeConfig};
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Full-frame RGB8 cubic resize with separately rounded 22-bit integer passes.
///
/// This explicit path matches the measured Pillow RGB bicubic convention.
/// It uses antialiased Catmull–Rom, half-pixel centres and truncated edge taps.
/// It does not perform color conversion, alpha handling or compositing.
/// Input and output strides are in bytes and may include row padding.
#[derive(Debug, Clone)]
pub struct PillowCubicResizer {
    width: usize,
    height: usize,
    out_width: usize,
    out_height: usize,
    horizontal: Vec<(usize, Vec<i32>)>,
    vertical: Vec<(usize, Vec<i32>)>,
}
impl PillowCubicResizer {
    /// Validate dimensions and precompute the separable coefficient tables.
    ///
    /// Uses the same dimension and weight-memory limits as `ResizeConfig`.
    pub fn new(
        width: u32,
        height: u32,
        out_width: u32,
        out_height: u32,
    ) -> Result<Self, &'static str> {
        ResizeConfig::builder(width, height, out_width, out_height)
            .build()
            .validate()?;
        if u64::from(height) * u64::from(out_width) > 120_000_000 {
            return Err("horizontal intermediate exceeds 120 MP cap");
        }
        let filter = InterpolationDetails::cubic_bc(0.0, 0.5);
        Ok(Self {
            width: width as usize,
            height: height as usize,
            out_width: out_width as usize,
            out_height: out_height as usize,
            horizontal: fixed22_weights(width, out_width, &filter),
            vertical: fixed22_weights(height, out_height, &filter),
        })
    }

    /// Resize encoded RGB8 into caller-owned strided rows.
    ///
    /// The descriptor must be `PixelDescriptor::RGB8_SRGB`; other channel
    /// layouts, precisions and transfer functions are refused. Output is RGB8_SRGB.
    /// Returns an error for short buffers, invalid strides, size overflow or
    /// inability to allocate the horizontal u8 intermediate. Row padding is
    /// neither read nor written. All coefficient products accumulate in i64.
    pub fn resize_rgb8(
        &self,
        input: &[u8],
        input_format: PixelDescriptor,
        input_stride: usize,
        output: &mut [u8],
        output_stride: usize,
    ) -> Result<(), &'static str> {
        if input_format != PixelDescriptor::RGB8_SRGB {
            return Err("Pillow cubic input must be encoded RGB8_SRGB");
        }
        let row = self.width.checked_mul(3).ok_or("input row overflow")?;
        let out_row = self.out_width.checked_mul(3).ok_or("output row overflow")?;
        if input_stride < row || output_stride < out_row {
            return Err("stride is shorter than RGB row");
        }
        let input_len = (self.height - 1)
            .checked_mul(input_stride)
            .and_then(|n| n.checked_add(row))
            .ok_or("input size overflow")?;
        let output_len = (self.out_height - 1)
            .checked_mul(output_stride)
            .and_then(|n| n.checked_add(out_row))
            .ok_or("output size overflow")?;
        if input.len() < input_len || output.len() < output_len {
            return Err("RGB buffer is too short");
        }
        let length = self
            .height
            .checked_mul(out_row)
            .ok_or("intermediate size overflow")?;
        let mut intermediate = Vec::new();
        intermediate
            .try_reserve_exact(length)
            .map_err(|_| "cannot allocate resize intermediate")?;
        intermediate.resize(length, 0u8);
        fn rounded(value: i64) -> u8 {
            ((value + (1 << 21)) >> 22).clamp(0, 255) as u8
        }
        for y in 0..self.height {
            for (x, (left, weights)) in self.horizontal.iter().enumerate() {
                for c in 0..3 {
                    let mut sum = 0i64;
                    for (tap, &weight) in weights.iter().enumerate() {
                        sum += i64::from(input[y * input_stride + (left + tap) * 3 + c])
                            * i64::from(weight);
                    }
                    intermediate[y * out_row + x * 3 + c] = rounded(sum);
                }
            }
        }
        for (y, (top, weights)) in self.vertical.iter().enumerate() {
            for x in 0..out_row {
                let mut sum = 0i64;
                for (tap, &weight) in weights.iter().enumerate() {
                    sum += i64::from(intermediate[(top + tap) * out_row + x]) * i64::from(weight);
                }
                output[y * output_stride + x] = rounded(sum);
            }
        }
        Ok(())
    }
}
