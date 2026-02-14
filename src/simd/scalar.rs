//! Scalar fallback convolution kernels.
#![allow(clippy::too_many_arguments)]

use crate::weights::{F32WeightTable, I16_PRECISION, I16WeightTable};
use archmage::ScalarToken;

/// Horizontal convolution: filter one row of f32 pixels, scalar fallback.
pub(crate) fn filter_h_row_f32_scalar(
    _token: ScalarToken,
    input: &[f32],
    output: &mut [f32],
    weights: &F32WeightTable,
    channels: usize,
) {
    let out_width = weights.len();

    for out_x in 0..out_width {
        let left = weights.left[out_x] as usize;
        let w = weights.weights(out_x);
        let out_offset = out_x * channels;

        // Zero-initialize output pixel
        for c in 0..channels {
            output[out_offset + c] = 0.0;
        }

        // Accumulate weighted contributions
        for (t, &weight) in w.iter().enumerate() {
            let in_offset = (left + t) * channels;
            for c in 0..channels {
                output[out_offset + c] += input[in_offset + c] * weight;
            }
        }
    }
}

/// Vertical convolution: combine rows into one output row, scalar fallback.
pub(crate) fn filter_v_row_f32_scalar(
    _token: ScalarToken,
    rows: &[&[f32]],
    output: &mut [f32],
    weights: &[f32],
) {
    let width = output.len();
    debug_assert_eq!(rows.len(), weights.len());

    // Zero-initialize output
    for v in output.iter_mut() {
        *v = 0.0;
    }

    // Accumulate weighted rows
    for (row, &weight) in rows.iter().zip(weights.iter()) {
        debug_assert!(row.len() >= width);
        for x in 0..width {
            output[x] += row[x] * weight;
        }
    }
}

/// Convert u8 → f32 (divide by 255), scalar fallback.
pub(crate) fn u8_to_f32_row_scalar(_token: ScalarToken, input: &[u8], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());
    for (inp, out) in input.iter().zip(output.iter_mut()) {
        *out = *inp as f32 * (1.0 / 255.0);
    }
}

/// Convert f32 → u8 (multiply by 255, round, clamp), scalar fallback.
pub(crate) fn f32_to_u8_row_scalar(_token: ScalarToken, input: &[f32], output: &mut [u8]) {
    debug_assert_eq!(input.len(), output.len());
    for (inp, out) in input.iter().zip(output.iter_mut()) {
        *out = (*inp * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
    }
}

/// Premultiply alpha in-place, scalar fallback.
pub(crate) fn premultiply_alpha_row_scalar(_token: ScalarToken, row: &mut [f32]) {
    for pixel in row.chunks_exact_mut(4) {
        let a = pixel[3];
        pixel[0] *= a;
        pixel[1] *= a;
        pixel[2] *= a;
    }
}

/// Unpremultiply alpha in-place, scalar fallback.
pub(crate) fn unpremultiply_alpha_row_scalar(_token: ScalarToken, row: &mut [f32]) {
    for pixel in row.chunks_exact_mut(4) {
        let a = pixel[3];
        if a > 1.0 / 1024.0 {
            let inv_a = 1.0 / a;
            pixel[0] *= inv_a;
            pixel[1] *= inv_a;
            pixel[2] *= inv_a;
        }
    }
}

/// Integer horizontal convolution: u8 input → u8 output, scalar fallback.
pub(crate) fn filter_h_u8_i16_scalar(
    _token: ScalarToken,
    input: &[u8],
    output: &mut [u8],
    weights: &I16WeightTable,
    channels: usize,
) {
    let out_width = weights.len();

    for out_x in 0..out_width {
        let left = weights.left[out_x] as usize;
        let w = weights.weights(out_x);
        let out_base = out_x * channels;

        for c in 0..channels {
            let mut acc: i32 = 0;
            for (t, &weight) in w.iter().enumerate() {
                acc += input[(left + t) * channels + c] as i32 * weight as i32;
            }
            let rounded = (acc + (1 << (I16_PRECISION - 1))) >> I16_PRECISION;
            output[out_base + c] = rounded.clamp(0, 255) as u8;
        }
    }
}

/// 4-row batch horizontal convolution, scalar fallback.
/// Just calls filter_h_u8_i16_scalar 4 times.
pub(crate) fn filter_h_u8_i16_4rows_scalar(
    _token: ScalarToken,
    in0: &[u8],
    in1: &[u8],
    in2: &[u8],
    in3: &[u8],
    out0: &mut [u8],
    out1: &mut [u8],
    out2: &mut [u8],
    out3: &mut [u8],
    weights: &I16WeightTable,
) {
    let ch = 4;
    filter_h_u8_i16_scalar(_token, in0, out0, weights, ch);
    filter_h_u8_i16_scalar(_token, in1, out1, weights, ch);
    filter_h_u8_i16_scalar(_token, in2, out2, weights, ch);
    filter_h_u8_i16_scalar(_token, in3, out3, weights, ch);
}

/// Batch vertical filter for all output rows, scalar fallback.
pub(crate) fn filter_v_all_u8_i16_scalar(
    _token: ScalarToken,
    intermediate: &[u8],
    output: &mut [u8],
    h_row_len: usize,
    in_h: usize,
    out_h: usize,
    weights: &crate::weights::I16WeightTable,
) {
    for out_y in 0..out_h {
        let left = weights.left[out_y];
        let tap_count = weights.tap_count(out_y);
        let w = weights.weights(out_y);
        let out_start = out_y * h_row_len;

        for x in 0..h_row_len {
            let mut acc: i32 = 0;
            for (t, &weight) in w[..tap_count].iter().enumerate() {
                let in_y = (left + t as i32).clamp(0, in_h as i32 - 1) as usize;
                acc += intermediate[in_y * h_row_len + x] as i32 * weight as i32;
            }
            let rounded = (acc + (1 << (I16_PRECISION - 1))) >> I16_PRECISION;
            output[out_start + x] = rounded.clamp(0, 255) as u8;
        }
    }
}

// ============================================================================
// Integer i16→i16 path (linear-light i12 values 0-4095)
// ============================================================================

/// Integer horizontal convolution: i16 input → i16 output, scalar fallback.
/// For linear-light i12 path (values 0-4095).
pub(crate) fn filter_h_i16_i16_scalar(
    _token: ScalarToken,
    input: &[i16],
    output: &mut [i16],
    weights: &I16WeightTable,
    channels: usize,
) {
    let out_width = weights.len();

    for out_x in 0..out_width {
        let left = weights.left[out_x] as usize;
        let w = weights.weights(out_x);
        let out_base = out_x * channels;

        for c in 0..channels {
            let mut acc: i32 = 0;
            for (t, &weight) in w.iter().enumerate() {
                acc += input[(left + t) * channels + c] as i32 * weight as i32;
            }
            let rounded = (acc + (1 << (I16_PRECISION - 1))) >> I16_PRECISION;
            output[out_base + c] = rounded.clamp(0, 4095) as i16;
        }
    }
}

/// Batch vertical filter for all output rows (i16 intermediate → i16 output), scalar fallback.
/// For linear-light i12 path (values 0-4095).
pub(crate) fn filter_v_all_i16_i16_scalar(
    _token: ScalarToken,
    intermediate: &[i16],
    output: &mut [i16],
    h_row_len: usize,
    in_h: usize,
    out_h: usize,
    weights: &crate::weights::I16WeightTable,
) {
    for out_y in 0..out_h {
        let left = weights.left[out_y];
        let tap_count = weights.tap_count(out_y);
        let w = weights.weights(out_y);
        let out_start = out_y * h_row_len;

        for x in 0..h_row_len {
            let mut acc: i32 = 0;
            for (t, &weight) in w.iter().enumerate().take(tap_count) {
                let in_y = (left + t as i32).clamp(0, in_h as i32 - 1) as usize;
                acc += intermediate[in_y * h_row_len + x] as i32 * weight as i32;
            }
            let rounded = (acc + (1 << (I16_PRECISION - 1))) >> I16_PRECISION;
            output[out_start + x] = rounded.clamp(0, 4095) as i16;
        }
    }
}

/// Premultiply alpha on RGBA u8 row: input → output, scalar fallback.
pub(crate) fn premultiply_u8_row_scalar(_token: ScalarToken, input: &[u8], output: &mut [u8]) {
    debug_assert_eq!(input.len(), output.len());
    for (inp, out) in input.chunks_exact(4).zip(output.chunks_exact_mut(4)) {
        let a = inp[3] as u16;
        // (c * a + 127) / 255 — exact for all u8 inputs
        out[0] = ((inp[0] as u16 * a + 127) / 255) as u8;
        out[1] = ((inp[1] as u16 * a + 127) / 255) as u8;
        out[2] = ((inp[2] as u16 * a + 127) / 255) as u8;
        out[3] = inp[3];
    }
}

/// Unpremultiply alpha in-place on RGBA u8 row, scalar fallback.
pub(crate) fn unpremultiply_u8_row_scalar(_token: ScalarToken, row: &mut [u8]) {
    for pixel in row.chunks_exact_mut(4) {
        let a = pixel[3];
        if a == 0 {
            pixel[0] = 0;
            pixel[1] = 0;
            pixel[2] = 0;
        } else if a < 255 {
            let a16 = a as u16;
            pixel[0] = ((pixel[0] as u16 * 255 + a16 / 2) / a16).min(255) as u8;
            pixel[1] = ((pixel[1] as u16 * 255 + a16 / 2) / a16).min(255) as u8;
            pixel[2] = ((pixel[2] as u16 * 255 + a16 / 2) / a16).min(255) as u8;
        }
        // a == 255: no change needed
    }
}

/// Convert sRGB u8 → linear f32 using LUT.
pub(crate) fn srgb_u8_to_linear_f32_scalar(
    _token: ScalarToken,
    input: &[u8],
    output: &mut [f32],
    channels: usize,
    has_alpha: bool,
) {
    crate::color::srgb_u8_to_linear_f32_impl(input, output, channels, has_alpha);
}

/// Convert linear f32 → sRGB u8 using LUT.
pub(crate) fn linear_f32_to_srgb_u8_scalar(
    _token: ScalarToken,
    input: &[f32],
    output: &mut [u8],
    channels: usize,
    has_alpha: bool,
) {
    crate::color::linear_f32_to_srgb_u8_impl(input, output, channels, has_alpha);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::{Filter, InterpolationDetails};

    #[test]
    fn test_horizontal_convolution_identity() {
        let filter = InterpolationDetails::create(Filter::Box);
        let width = 10;
        let channels = 4;
        let weights = F32WeightTable::new(width, width, &filter);

        let input: Vec<f32> = (0..width as usize * channels)
            .map(|i| (i as f32) / (width as f32 * channels as f32))
            .collect();
        let mut output = vec![0.0f32; width as usize * channels];

        filter_h_row_f32_scalar(ScalarToken, &input, &mut output, &weights, channels);

        for i in 0..input.len() {
            assert!(
                (input[i] - output[i]).abs() < 0.01,
                "Mismatch at {}: {} vs {}",
                i,
                input[i],
                output[i]
            );
        }
    }

    #[test]
    fn test_vertical_convolution_single_row() {
        let width = 40;
        let row: Vec<f32> = (0..width).map(|i| i as f32).collect();
        let mut output = vec![0.0f32; width];

        filter_v_row_f32_scalar(ScalarToken, &[&row], &mut output, &[1.0]);

        for i in 0..width {
            assert!(
                (row[i] - output[i]).abs() < 1e-6,
                "Mismatch at {}: {} vs {}",
                i,
                row[i],
                output[i]
            );
        }
    }

    #[test]
    fn test_vertical_convolution_average() {
        let width = 8;
        let row_a: Vec<f32> = vec![1.0; width];
        let row_b: Vec<f32> = vec![3.0; width];
        let mut output = vec![0.0f32; width];

        filter_v_row_f32_scalar(ScalarToken, &[&row_a, &row_b], &mut output, &[0.5, 0.5]);

        for v in &output {
            assert!((*v - 2.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_horizontal_downscale() {
        let filter = InterpolationDetails::create(Filter::Lanczos);
        let in_width = 100u32;
        let out_width = 50u32;
        let channels = 4;
        let weights = F32WeightTable::new(in_width, out_width, &filter);

        let input = vec![0.5f32; in_width as usize * channels];
        let mut output = vec![0.0f32; out_width as usize * channels];

        filter_h_row_f32_scalar(ScalarToken, &input, &mut output, &weights, channels);

        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v - 0.5).abs() < 0.01,
                "Constant input should produce constant output at {}: {}",
                i,
                v
            );
        }
    }
}
