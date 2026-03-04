//! Post-resize Gaussian blur.
//!
//! Separable Gaussian blur that reuses the SIMD convolution infrastructure
//! from the resize pipeline. Applied after resize as a compression-optimized
//! preprocessing step.

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::simd;
use crate::weights::F32WeightTable;

/// Maximum blur radius in pixels. Limits kernel to 61 taps (radius=30).
const MAX_RADIUS: usize = 30;

/// Build an [`F32WeightTable`] with Gaussian weights for same-size convolution.
///
/// Each output pixel gets a symmetric Gaussian kernel centered on the
/// corresponding input pixel, clamped at image edges with renormalization.
pub(crate) fn gaussian_weight_table(size: u32, sigma: f32) -> F32WeightTable {
    assert!(size > 0, "size must be positive");
    assert!(sigma > 0.0, "sigma must be positive");

    let radius = ((sigma * 3.0).ceil() as usize).min(MAX_RADIUS);
    let max_taps = 2 * radius + 1;
    let n = size as usize;

    let mut left = Vec::with_capacity(n);
    let mut tap_counts = Vec::with_capacity(n);
    let mut weights_flat = vec![0.0f32; n * max_taps];

    let inv_2sigma2 = 1.0 / (2.0 * sigma as f64 * sigma as f64);

    for pixel in 0..n {
        let first = (pixel as i32) - (radius as i32);
        let clamped_first = first.max(0) as usize;
        let clamped_last = (pixel + radius).min(n - 1);
        let taps = clamped_last - clamped_first + 1;

        left.push(clamped_first as i32);
        tap_counts.push(taps as u16);

        let offset = pixel * max_taps;
        let mut sum = 0.0f64;
        for t in 0..taps {
            let input_pixel = clamped_first + t;
            let d = input_pixel as f64 - pixel as f64;
            let w = (-d * d * inv_2sigma2).exp();
            weights_flat[offset + t] = w as f32;
            sum += w;
        }

        // Renormalize so weights sum to 1.0
        if sum > 0.0 {
            let inv = 1.0 / sum as f32;
            for t in 0..taps {
                weights_flat[offset + t] *= inv;
            }
        }
    }

    F32WeightTable::from_parts(left, weights_flat, tap_counts, max_taps)
}

/// In-place Gaussian blur of an f32 buffer (width × height × channels).
///
/// Uses separable H+V passes with SIMD-accelerated convolution.
pub(crate) fn blur_f32(data: &mut [f32], width: u32, height: u32, channels: usize, sigma: f32) {
    if sigma <= 0.0 || width == 0 || height == 0 {
        return;
    }
    let w = width as usize;
    let h = height as usize;
    let row_len = w * channels;

    let h_weights = gaussian_weight_table(width, sigma);
    let v_weights = gaussian_weight_table(height, sigma);

    // SIMD padding: the AVX2 4ch path reads max_taps*channels f32 values from
    // the input starting at `left*channels`. For the last output pixels, this
    // window can extend past the actual row data. Pad with zeros so the reads
    // are valid (the padded weight entries are zero, so the values don't matter).
    let h_pad = h_weights.max_taps * channels;

    // Padded input row for H-pass (copied per row to add trailing zeros).
    let mut padded_row = vec![0.0f32; row_len + h_pad];

    // Intermediate buffer holds the H-pass output (same dimensions as input).
    let mut intermediate = vec![0.0f32; h * row_len];

    // === Horizontal pass ===
    for y in 0..h {
        let in_start = y * row_len;
        padded_row[..row_len].copy_from_slice(&data[in_start..in_start + row_len]);
        // Trailing padding is already zero from allocation (and stays zero).
        simd::filter_h_row_f32(
            &padded_row,
            &mut intermediate[y * row_len..y * row_len + row_len],
            &h_weights,
            channels,
        );
    }

    // === Vertical pass ===
    let max_taps = v_weights.max_taps;
    let mut row_ptrs: Vec<&[f32]> = Vec::with_capacity(max_taps);

    for out_y in 0..h {
        let left = v_weights.left[out_y];
        let tap_count = v_weights.tap_count(out_y);
        let weights = v_weights.weights(out_y);

        row_ptrs.clear();
        for t in 0..tap_count {
            let in_y = (left + t as i32) as usize;
            let start = in_y * row_len;
            row_ptrs.push(&intermediate[start..start + row_len]);
        }

        let out_start = out_y * row_len;
        simd::filter_v_row_f32(
            &row_ptrs,
            &mut data[out_start..out_start + row_len],
            weights,
        );
    }
}

/// In-place Gaussian blur of a u8 buffer (width × height × channels).
///
/// Converts to f32, blurs, converts back. The blur operates in the same
/// color space as the input (typically sRGB gamma).
pub(crate) fn blur_u8(data: &mut [u8], width: u32, height: u32, channels: usize, sigma: f32) {
    if sigma <= 0.0 || width == 0 || height == 0 {
        return;
    }
    let len = width as usize * height as usize * channels;

    // Convert u8 → f32
    let mut f32_buf = vec![0.0f32; len];
    simd::u8_to_f32_row(&data[..len], &mut f32_buf);

    // Blur in f32
    blur_f32(&mut f32_buf, width, height, channels, sigma);

    // Convert f32 → u8
    simd::f32_to_u8_row(&f32_buf, &mut data[..len]);
}

/// In-place unsharp mask of a u8 buffer (width × height × channels).
///
/// `sharp = original + amount * (original - gaussian_blur(original, sigma))`
///
/// This enhances edges by adding back the high-frequency detail that the
/// Gaussian blur removes. `amount` controls the strength (1.0 = standard,
/// higher = more aggressive). `sigma` controls the blur radius.
pub(crate) fn unsharp_mask_u8(
    data: &mut [u8],
    width: u32,
    height: u32,
    channels: usize,
    amount: f32,
    sigma: f32,
) {
    if amount <= 0.0 || sigma <= 0.0 || width == 0 || height == 0 {
        return;
    }
    let len = width as usize * height as usize * channels;

    // Convert u8 → f32
    let mut original = vec![0.0f32; len];
    simd::u8_to_f32_row(&data[..len], &mut original);

    // Blur a copy
    let mut blurred = original.clone();
    blur_f32(&mut blurred, width, height, channels, sigma);

    // sharp = original + amount * (original - blurred), clamped to [0, 1]
    for i in 0..len {
        let v = original[i] + amount * (original[i] - blurred[i]);
        original[i] = v.clamp(0.0, 1.0);
    }

    // Convert f32 → u8
    simd::f32_to_u8_row(&original, &mut data[..len]);
}

/// In-place unsharp mask of an f32 buffer (width × height × channels).
///
/// Values are clamped to [0, 1] after sharpening.
pub(crate) fn unsharp_mask_f32(
    data: &mut [f32],
    width: u32,
    height: u32,
    channels: usize,
    amount: f32,
    sigma: f32,
) {
    if amount <= 0.0 || sigma <= 0.0 || width == 0 || height == 0 {
        return;
    }
    let len = width as usize * height as usize * channels;

    // Blur a copy
    let mut blurred = data[..len].to_vec();
    blur_f32(&mut blurred, width, height, channels, sigma);

    // sharp = original + amount * (original - blurred)
    for i in 0..len {
        data[i] = (data[i] + amount * (data[i] - blurred[i])).clamp(0.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_weight_table_sums_to_one() {
        let table = gaussian_weight_table(100, 2.0);
        for pixel in 0..100 {
            let weights = table.weights(pixel);
            let sum: f32 = weights.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "pixel {pixel}: sum = {sum}");
        }
    }

    #[test]
    fn gaussian_weight_table_symmetric_center() {
        let table = gaussian_weight_table(100, 2.0);
        // Center pixel (50) should have symmetric weights
        let weights = table.weights(50);
        let n = weights.len();
        for i in 0..n / 2 {
            let diff = (weights[i] - weights[n - 1 - i]).abs();
            assert!(
                diff < 1e-6,
                "asymmetry at {i}: {:.8} vs {:.8}",
                weights[i],
                weights[n - 1 - i]
            );
        }
    }

    #[test]
    fn blur_f32_identity_sigma_near_zero() {
        // Very small sigma should produce near-identity (1 tap essentially)
        let w = 8;
        let h = 8;
        let ch = 3;
        let mut data: Vec<f32> = (0..w * h * ch).map(|i| (i as f32) / 255.0).collect();
        let original = data.clone();

        // sigma=0.1 means radius=ceil(0.3)=1, but center weight dominates
        blur_f32(&mut data, w as u32, h as u32, ch, 0.1);

        let max_diff: f32 = data
            .iter()
            .zip(original.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 0.02, "max_diff = {max_diff}");
    }

    #[test]
    fn blur_u8_does_not_panic() {
        let w = 16u32;
        let h = 16u32;
        let ch = 4usize;
        let mut data = vec![128u8; w as usize * h as usize * ch];
        blur_u8(&mut data, w, h, ch, 1.5);
        // Uniform input should stay uniform (or very close)
        for &v in &data {
            assert!(
                (v as i32 - 128).unsigned_abs() <= 1,
                "uniform input changed to {v}"
            );
        }
    }
}
