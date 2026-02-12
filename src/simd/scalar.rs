//! Scalar fallback convolution kernels.

use crate::weights::F32WeightTable;
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
