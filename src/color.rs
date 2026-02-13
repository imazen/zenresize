//! Color space conversion and alpha handling.
//!
//! Provides sRGB↔linear conversion and alpha premultiply/unpremultiply
//! for use in the resize pipeline.

/// Convert a row of sRGB u8 pixels to linear f32.
///
/// Uses LUT-based batch conversion (~20x faster than per-pixel scalar powf).
/// Alpha channels (if `has_alpha` is true and `channels` is 4) are copied
/// as-is (divided by 255.0 but not gamma-converted).
pub fn srgb_u8_to_linear_f32(input: &[u8], output: &mut [f32], channels: usize, has_alpha: bool) {
    debug_assert_eq!(input.len(), output.len());

    // Batch LUT conversion: all channels including alpha go through sRGB curve
    linear_srgb::default::srgb_u8_to_linear_slice(input, output);

    // Fix alpha: should be linear scale (v/255), not sRGB curve
    if has_alpha && channels == 4 {
        for (chunk_in, chunk_out) in input.chunks_exact(4).zip(output.chunks_exact_mut(4)) {
            chunk_out[3] = chunk_in[3] as f32 / 255.0;
        }
    }
}

/// Convert a row of linear f32 pixels to sRGB u8.
///
/// Uses SIMD-accelerated batch conversion.
/// Alpha channels are copied as-is (scaled by 255 but not gamma-converted).
pub fn linear_f32_to_srgb_u8(input: &[f32], output: &mut [u8], channels: usize, has_alpha: bool) {
    debug_assert_eq!(input.len(), output.len());

    // SIMD batch conversion: all channels including alpha go through sRGB curve
    linear_srgb::default::linear_to_srgb_u8_slice(input, output);

    // Fix alpha: should be linear scale (v*255+0.5), not sRGB curve
    if has_alpha && channels == 4 {
        for (chunk_in, chunk_out) in input.chunks_exact(4).zip(output.chunks_exact_mut(4)) {
            chunk_out[3] = (chunk_in[3] * 255.0 + 0.5) as u8;
        }
    }
}

/// Convert a row of sRGB u8 pixels to f32 without gamma conversion (sRGB space).
pub fn srgb_u8_to_f32(input: &[u8], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());
    for (inp, out) in input.iter().zip(output.iter_mut()) {
        *out = *inp as f32 / 255.0;
    }
}

/// Convert a row of f32 pixels to u8 without gamma conversion (sRGB space).
pub fn f32_to_srgb_u8(input: &[f32], output: &mut [u8]) {
    debug_assert_eq!(input.len(), output.len());
    for (inp, out) in input.iter().zip(output.iter_mut()) {
        *out = (*inp * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
    }
}

/// Premultiply alpha in-place on a row of f32 RGBA pixels.
///
/// Multiplies RGB channels by the alpha value.
pub fn premultiply_alpha_f32(row: &mut [f32], channels: usize) {
    debug_assert_eq!(channels, 4);
    for pixel in row.chunks_exact_mut(4) {
        let a = pixel[3];
        pixel[0] *= a;
        pixel[1] *= a;
        pixel[2] *= a;
    }
}

/// Unpremultiply alpha in-place on a row of f32 RGBA pixels.
///
/// Divides RGB channels by the alpha value. Transparent pixels (alpha ≈ 0) are left as-is.
pub fn unpremultiply_alpha_f32(row: &mut [f32], channels: usize) {
    debug_assert_eq!(channels, 4);
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

    #[test]
    fn test_srgb_u8_roundtrip() {
        let input: Vec<u8> = (0..=255).collect();
        let mut linear = vec![0.0f32; 256];
        let mut output = vec![0u8; 256];

        srgb_u8_to_linear_f32(&input, &mut linear, 1, false);
        linear_f32_to_srgb_u8(&linear, &mut output, 1, false);

        for i in 0..256 {
            let diff = (input[i] as i16 - output[i] as i16).unsigned_abs();
            assert!(
                diff <= 1,
                "Roundtrip mismatch at {}: {} -> {} -> {}",
                i,
                input[i],
                linear[i],
                output[i]
            );
        }
    }

    #[test]
    fn test_alpha_premultiply_roundtrip() {
        let mut row = [0.5f32, 0.3, 0.8, 0.5];
        let original = row;

        premultiply_alpha_f32(&mut row, 4);
        assert!((row[0] - 0.25).abs() < 1e-6);
        assert!((row[3] - 0.5).abs() < 1e-6);

        unpremultiply_alpha_f32(&mut row, 4);
        for i in 0..4 {
            assert!(
                (row[i] - original[i]).abs() < 1e-5,
                "Channel {} mismatch: {} vs {}",
                i,
                row[i],
                original[i]
            );
        }
    }

    #[test]
    fn test_transparent_pixel_unpremultiply() {
        let mut row = [0.0f32, 0.0, 0.0, 0.0];
        unpremultiply_alpha_f32(&mut row, 4);
        // Should not produce NaN or Inf
        for v in &row {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_srgb_space_roundtrip() {
        let input: Vec<u8> = (0..=255).collect();
        let mut f32_buf = vec![0.0f32; 256];
        let mut output = vec![0u8; 256];

        srgb_u8_to_f32(&input, &mut f32_buf);
        f32_to_srgb_u8(&f32_buf, &mut output);

        for i in 0..256 {
            assert_eq!(
                input[i], output[i],
                "sRGB space roundtrip mismatch at {}",
                i
            );
        }
    }
}
