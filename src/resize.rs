//! Full-frame resize API (convenience wrappers around [`StreamingResize`]).

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::pixel::ResizeConfig;
use crate::streaming::StreamingResize;

/// Resize an entire u8 image. Allocates and returns output buffer.
///
/// # Panics
/// Panics if the config is invalid or input length doesn't match
/// `in_width * in_height * channels`.
pub fn resize(config: &ResizeConfig, input: &[u8]) -> Vec<u8> {
    let channels = config.input_format.channels() as usize;
    let expected = config.in_width as usize * config.in_height as usize * channels;
    assert_eq!(input.len(), expected, "input length mismatch");

    let mut resizer = StreamingResize::new(config);
    let row_len = config.in_width as usize * channels;

    for y in 0..config.in_height {
        let start = y as usize * row_len;
        resizer.push_row(&input[start..start + row_len]);
    }
    resizer.finish();

    let out_row_len = config.out_width as usize * channels;
    let mut output = vec![0u8; config.out_height as usize * out_row_len];
    let mut row_idx = 0;
    while let Some(row) = resizer.next_output_row() {
        let start = row_idx * out_row_len;
        output[start..start + out_row_len].copy_from_slice(&row);
        row_idx += 1;
    }

    output
}

/// Resize a u8 image into a caller-provided buffer.
///
/// # Panics
/// Panics if input/output lengths don't match the config.
pub fn resize_into(config: &ResizeConfig, input: &[u8], output: &mut [u8]) {
    let channels = config.input_format.channels() as usize;
    let in_expected = config.in_width as usize * config.in_height as usize * channels;
    let out_expected = config.out_width as usize * config.out_height as usize * channels;
    assert_eq!(input.len(), in_expected, "input length mismatch");
    assert_eq!(output.len(), out_expected, "output length mismatch");

    let mut resizer = StreamingResize::new(config);
    let row_len = config.in_width as usize * channels;

    for y in 0..config.in_height {
        let start = y as usize * row_len;
        resizer.push_row(&input[start..start + row_len]);
    }
    resizer.finish();

    let out_row_len = config.out_width as usize * channels;
    let mut row_idx = 0;
    while let Some(row) = resizer.next_output_row() {
        let start = row_idx * out_row_len;
        output[start..start + out_row_len].copy_from_slice(&row);
        row_idx += 1;
    }
}

/// Resize an f32 image. Allocates and returns output buffer.
pub fn resize_f32(config: &ResizeConfig, input: &[f32]) -> Vec<f32> {
    let channels = config.input_format.channels() as usize;
    let expected = config.in_width as usize * config.in_height as usize * channels;
    assert_eq!(input.len(), expected, "input length mismatch");

    let mut resizer = StreamingResize::new(config);
    let row_len = config.in_width as usize * channels;

    for y in 0..config.in_height {
        let start = y as usize * row_len;
        resizer.push_row_f32(&input[start..start + row_len]);
    }
    resizer.finish();

    let out_row_len = config.out_width as usize * channels;
    let mut output = vec![0.0f32; config.out_height as usize * out_row_len];
    let mut row_idx = 0;
    while let Some(row) = resizer.next_output_row_f32() {
        let start = row_idx * out_row_len;
        output[start..start + out_row_len].copy_from_slice(&row);
        row_idx += 1;
    }

    output
}

/// Resize an f32 image into a caller-provided buffer.
pub fn resize_f32_into(config: &ResizeConfig, input: &[f32], output: &mut [f32]) {
    let channels = config.input_format.channels() as usize;
    let in_expected = config.in_width as usize * config.in_height as usize * channels;
    let out_expected = config.out_width as usize * config.out_height as usize * channels;
    assert_eq!(input.len(), in_expected, "input length mismatch");
    assert_eq!(output.len(), out_expected, "output length mismatch");

    let mut resizer = StreamingResize::new(config);
    let row_len = config.in_width as usize * channels;

    for y in 0..config.in_height {
        let start = y as usize * row_len;
        resizer.push_row_f32(&input[start..start + row_len]);
    }
    resizer.finish();

    let out_row_len = config.out_width as usize * channels;
    let mut row_idx = 0;
    while let Some(row) = resizer.next_output_row_f32() {
        let start = row_idx * out_row_len;
        output[start..start + out_row_len].copy_from_slice(&row);
        row_idx += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::Filter;
    use crate::pixel::{ColorSpace, PixelFormat};

    fn test_config(in_w: u32, in_h: u32, out_w: u32, out_h: u32) -> ResizeConfig {
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
            color_space: ColorSpace::Srgb,
        }
    }

    #[test]
    fn test_resize_constant_color() {
        let config = test_config(20, 20, 10, 10);
        let mut input = vec![0u8; 20 * 20 * 4];
        for pixel in input.chunks_exact_mut(4) {
            pixel[0] = 128;
            pixel[1] = 128;
            pixel[2] = 128;
            pixel[3] = 255;
        }

        let output = resize(&config, &input);
        assert_eq!(output.len(), 10 * 10 * 4);

        for pixel in output.chunks_exact(4) {
            assert!(
                (pixel[0] as i16 - 128).unsigned_abs() <= 2,
                "R off: {}",
                pixel[0]
            );
            assert!(
                (pixel[3] as i16 - 255).unsigned_abs() <= 1,
                "A off: {}",
                pixel[3]
            );
        }
    }

    #[test]
    fn test_resize_into_matches_resize() {
        let config = test_config(20, 20, 10, 10);
        let input = vec![100u8; 20 * 20 * 4];

        let output_alloc = resize(&config, &input);
        let mut output_into = vec![0u8; 10 * 10 * 4];
        resize_into(&config, &input, &mut output_into);

        assert_eq!(output_alloc, output_into);
    }

    #[test]
    fn test_resize_upscale() {
        let config = test_config(10, 10, 20, 20);
        let input = vec![200u8; 10 * 10 * 4];

        let output = resize(&config, &input);
        assert_eq!(output.len(), 20 * 20 * 4);
    }

    #[test]
    fn test_resize_1x1() {
        let config = test_config(1, 1, 1, 1);
        let input = vec![128, 64, 32, 255];

        let output = resize(&config, &input);
        assert_eq!(output.len(), 4);
        // Should approximately preserve the single pixel
        assert!((output[0] as i16 - 128).unsigned_abs() <= 2);
    }
}
