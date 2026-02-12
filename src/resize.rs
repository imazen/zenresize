//! Full-frame resize API.
//!
//! Uses an optimized two-pass approach for full-frame resizes:
//! 1. Horizontal pass: convert u8→f32, premultiply alpha, horizontal filter → intermediate
//! 2. Vertical pass: vertical filter → unpremultiply alpha, convert f32→u8 → output
//!
//! This eliminates the per-row allocation overhead of the streaming path.
//! The streaming API ([`crate::streaming::StreamingResize`]) is still available
//! for pipeline integration where the full image isn't in memory.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::color;
use crate::filter::InterpolationDetails;
use crate::pixel::ResizeConfig;
use crate::simd;
use crate::weights::F32WeightTable;

/// Resize an entire u8 image. Allocates and returns output buffer.
///
/// Input may be tightly packed or strided (set `in_stride` in config).
/// Output is always tightly packed.
///
/// # Panics
/// Panics if the config is invalid or input is too short.
pub fn resize(config: &ResizeConfig, input: &[u8]) -> Vec<u8> {
    let out_row_len = config.output_row_len();
    let mut output = vec![0u8; config.out_height as usize * out_row_len];
    resize_into(config, input, &mut output);
    output
}

/// Resize a u8 image into a caller-provided buffer.
///
/// Output buffer must be tightly packed: `out_width * out_height * channels`.
///
/// # Panics
/// Panics if input is too short or output length doesn't match.
pub fn resize_into(config: &ResizeConfig, input: &[u8], output: &mut [u8]) {
    config.validate().expect("invalid resize config");

    let in_stride = config.effective_in_stride();
    let in_row_len = config.input_row_len();
    let in_expected = if config.in_height > 0 {
        (config.in_height as usize - 1) * in_stride + in_row_len
    } else {
        0
    };
    let out_row_len = config.output_row_len();
    let out_expected = config.out_height as usize * out_row_len;
    assert!(input.len() >= in_expected, "input too short");
    assert_eq!(output.len(), out_expected, "output length mismatch");

    let channels = config.input_format.channels() as usize;
    let has_alpha = config.input_format.has_alpha();
    let in_w = config.in_width as usize;
    let in_h = config.in_height as usize;
    let out_w = config.out_width as usize;
    let out_h = config.out_height as usize;
    let linearize = config.needs_linearization();

    let filter = InterpolationDetails::create(config.filter);
    let h_weights = F32WeightTable::new(config.in_width, config.out_width, &filter);
    let v_weights = F32WeightTable::new(config.in_height, config.out_height, &filter);

    let h_row_len = out_w * channels;

    // Flat intermediate buffer for horizontally-filtered rows.
    // For the vertical pass, only ~12 rows are accessed at a time, and
    // sequential output rows share most of their input rows.
    let mut intermediate = vec![0.0f32; h_row_len * in_h];

    // Temporary buffer for u8→f32 conversion of one input row.
    let mut temp_row = vec![0.0f32; in_w * channels];

    // === Horizontal pass ===
    for y in 0..in_h {
        let in_start = y * in_stride;
        let in_row = &input[in_start..in_start + in_row_len];
        let out_start = y * h_row_len;

        if linearize {
            // Linear path: separate u8→linear f32 conversion + f32 filter
            color::srgb_u8_to_linear_f32(in_row, &mut temp_row, channels, has_alpha);
            if has_alpha && channels == 4 {
                simd::premultiply_alpha_row(&mut temp_row);
            }
            simd::filter_h_row_f32(
                &temp_row,
                &mut intermediate[out_start..out_start + h_row_len],
                &h_weights,
                channels,
            );
        } else if has_alpha && channels == 4 {
            // sRGB with alpha: need premultiply, so convert then filter
            simd::u8_to_f32_row(in_row, &mut temp_row);
            simd::premultiply_alpha_row(&mut temp_row);
            simd::filter_h_row_f32(
                &temp_row,
                &mut intermediate[out_start..out_start + h_row_len],
                &h_weights,
                channels,
            );
        } else {
            // sRGB without alpha: fused u8→f32+filter (fastest path)
            simd::filter_h_row_u8_srgb(
                in_row,
                &mut intermediate[out_start..out_start + h_row_len],
                &h_weights,
                channels,
            );
        }
    }

    // === Vertical pass ===
    let max_taps = v_weights.max_taps;
    let mut row_ptrs: Vec<&[f32]> = Vec::with_capacity(max_taps);
    // Only needed for linear or alpha paths.
    let needs_temp = linearize || (has_alpha && channels == 4);
    let mut temp_output = if needs_temp {
        vec![0.0f32; h_row_len]
    } else {
        Vec::new()
    };

    for out_y in 0..out_h {
        let left = v_weights.left[out_y];
        let tap_count = v_weights.tap_count(out_y);
        let weights = v_weights.weights(out_y);

        // Gather row pointers (reuse vec, no allocation).
        row_ptrs.clear();
        for t in 0..tap_count {
            let in_y = (left + t as i32).clamp(0, in_h as i32 - 1) as usize;
            let start = in_y * h_row_len;
            row_ptrs.push(&intermediate[start..start + h_row_len]);
        }

        let out_start = out_y * out_row_len;
        let out_slice = &mut output[out_start..out_start + out_row_len];

        if linearize {
            // Linear: vertical → f32 temp → unpremul → linear→sRGB → u8
            simd::filter_v_row_f32(&row_ptrs, &mut temp_output[..h_row_len], weights);
            if has_alpha && channels == 4 {
                simd::unpremultiply_alpha_row(&mut temp_output[..h_row_len]);
            }
            color::linear_f32_to_srgb_u8(
                &temp_output[..h_row_len],
                out_slice,
                channels,
                has_alpha,
            );
        } else if has_alpha && channels == 4 {
            // sRGB with alpha: vertical → f32 temp → unpremul → u8
            simd::filter_v_row_f32(&row_ptrs, &mut temp_output[..h_row_len], weights);
            simd::unpremultiply_alpha_row(&mut temp_output[..h_row_len]);
            simd::f32_to_u8_row(&temp_output[..h_row_len], out_slice);
        } else {
            // sRGB without alpha: fused vertical → u8 (fastest path)
            simd::filter_v_row_f32_to_u8(&row_ptrs, out_slice, weights);
        }
    }
}

/// Resize an f32 image. Allocates and returns output buffer.
pub fn resize_f32(config: &ResizeConfig, input: &[f32]) -> Vec<f32> {
    let out_row_len = config.output_row_len();
    let mut output = vec![0.0f32; config.out_height as usize * out_row_len];
    resize_f32_into(config, input, &mut output);
    output
}

/// Resize an f32 image into a caller-provided buffer.
pub fn resize_f32_into(config: &ResizeConfig, input: &[f32], output: &mut [f32]) {
    config.validate().expect("invalid resize config");

    let in_stride = config.effective_in_stride();
    let in_row_len = config.input_row_len();
    let in_expected = if config.in_height > 0 {
        (config.in_height as usize - 1) * in_stride + in_row_len
    } else {
        0
    };
    let out_row_len = config.output_row_len();
    let out_expected = config.out_height as usize * out_row_len;
    assert!(input.len() >= in_expected, "input too short");
    assert_eq!(output.len(), out_expected, "output length mismatch");

    let channels = config.input_format.channels() as usize;
    let has_alpha = config.input_format.has_alpha();
    let in_w = config.in_width as usize;
    let in_h = config.in_height as usize;
    let out_w = config.out_width as usize;
    let out_h = config.out_height as usize;

    let filter = InterpolationDetails::create(config.filter);
    let h_weights = F32WeightTable::new(config.in_width, config.out_width, &filter);
    let v_weights = F32WeightTable::new(config.in_height, config.out_height, &filter);

    let h_row_len = out_w * channels;
    let mut intermediate = vec![0.0f32; h_row_len * in_h];
    let mut temp_row = vec![0.0f32; in_w * channels];

    // === Horizontal pass ===
    for y in 0..in_h {
        let in_start = y * in_stride;
        temp_row[..in_row_len].copy_from_slice(&input[in_start..in_start + in_row_len]);

        if has_alpha && channels == 4 {
            simd::premultiply_alpha_row(&mut temp_row[..in_row_len]);
        }

        let out_start = y * h_row_len;
        simd::filter_h_row_f32(
            &temp_row[..in_row_len],
            &mut intermediate[out_start..out_start + h_row_len],
            &h_weights,
            channels,
        );
    }

    // === Vertical pass ===
    let max_taps = v_weights.max_taps;
    let mut row_ptrs: Vec<&[f32]> = Vec::with_capacity(max_taps);

    for out_y in 0..out_h {
        let left = v_weights.left[out_y];
        let tap_count = v_weights.tap_count(out_y);
        let weights = v_weights.weights(out_y);

        row_ptrs.clear();
        for t in 0..tap_count {
            let in_y = (left + t as i32).clamp(0, in_h as i32 - 1) as usize;
            let start = in_y * h_row_len;
            row_ptrs.push(&intermediate[start..start + h_row_len]);
        }

        let out_start = out_y * out_row_len;
        let out_slice = &mut output[out_start..out_start + out_row_len];
        simd::filter_v_row_f32(&row_ptrs, out_slice, weights);

        if has_alpha && channels == 4 {
            simd::unpremultiply_alpha_row(out_slice);
        }
    }
}

// =============================================================================
// imgref integration
// =============================================================================

// =============================================================================
// imgref + rgb crate integration
// =============================================================================
//
// All 4-channel pixel types (RGBA, BGRA, ARGB, ABGR) work identically because
// the pipeline is channel-order-agnostic. The `rgb` crate's ComponentSlice trait
// gives us zero-copy &[u8] access to any pixel type.

#[cfg(feature = "imgref")]
mod imgref_impl {
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;

    use crate::pixel::{PixelFormat, ResizeConfig};
    use crate::streaming::StreamingResize;
    use imgref::{Img, ImgRef, ImgVec};
    use rgb::ComponentSlice;

    /// Resize a 4-channel u8 image. Works with any pixel type that implements
    /// `ComponentSlice` (RGBA, BGRA, ARGB, ABGR from the `rgb` crate).
    ///
    /// Channel order is preserved — the pipeline doesn't care whether
    /// the bytes represent RGBA or BGRA.
    pub fn resize_4ch<P>(
        img: ImgRef<P>,
        out_width: u32,
        out_height: u32,
        has_alpha: bool,
        config: &ResizeConfig,
    ) -> ImgVec<P>
    where
        P: Copy + ComponentSlice<u8> + Default,
    {
        assert_eq!(core::mem::size_of::<P>(), 4, "pixel type must be 4 bytes");

        let mut cfg = config.clone();
        cfg.in_width = img.width() as u32;
        cfg.in_height = img.height() as u32;
        cfg.out_width = out_width;
        cfg.out_height = out_height;
        cfg.input_format = PixelFormat::Srgb8 { channels: 4, has_alpha };
        cfg.output_format = PixelFormat::Srgb8 { channels: 4, has_alpha };
        cfg.in_stride = 0;

        let mut resizer = StreamingResize::new(&cfg);

        // Reusable buffer: flatten pixel row to &[u8] without per-row allocation.
        // For repr(C) pixel types (rgb::RGBA etc.) the compiler optimizes
        // the per-pixel copy into a single memcpy.
        let w = img.width();
        let mut row_buf = vec![0u8; w * 4];
        for row in img.rows() {
            for (px, chunk) in row.iter().zip(row_buf.chunks_exact_mut(4)) {
                chunk.copy_from_slice(px.as_slice());
            }
            resizer.push_row(&row_buf);
        }
        resizer.finish();

        let out_row_len = cfg.output_row_len();
        let mut out_pixels = Vec::with_capacity(out_width as usize * out_height as usize);
        while let Some(row) = resizer.next_output_row() {
            debug_assert_eq!(row.len(), out_row_len);
            for chunk in row.chunks_exact(4) {
                let mut px = P::default();
                px.as_mut_slice().copy_from_slice(chunk);
                out_pixels.push(px);
            }
        }

        Img::new(out_pixels, out_width as usize, out_height as usize)
    }

    /// Resize a 3-channel u8 image. Works with `rgb::RGB<u8>`, `rgb::BGR<u8>`, etc.
    pub fn resize_3ch<P>(
        img: ImgRef<P>,
        out_width: u32,
        out_height: u32,
        config: &ResizeConfig,
    ) -> ImgVec<P>
    where
        P: Copy + ComponentSlice<u8> + Default,
    {
        assert_eq!(core::mem::size_of::<P>(), 3, "pixel type must be 3 bytes");

        let mut cfg = config.clone();
        cfg.in_width = img.width() as u32;
        cfg.in_height = img.height() as u32;
        cfg.out_width = out_width;
        cfg.out_height = out_height;
        cfg.input_format = PixelFormat::Srgb8 { channels: 3, has_alpha: false };
        cfg.output_format = PixelFormat::Srgb8 { channels: 3, has_alpha: false };
        cfg.in_stride = 0;

        let mut resizer = StreamingResize::new(&cfg);

        let w = img.width();
        let mut row_buf = vec![0u8; w * 3];
        for row in img.rows() {
            for (px, chunk) in row.iter().zip(row_buf.chunks_exact_mut(3)) {
                chunk.copy_from_slice(px.as_slice());
            }
            resizer.push_row(&row_buf);
        }
        resizer.finish();

        let out_row_len = cfg.output_row_len();
        let mut out_pixels = Vec::with_capacity(out_width as usize * out_height as usize);
        while let Some(row) = resizer.next_output_row() {
            debug_assert_eq!(row.len(), out_row_len);
            for chunk in row.chunks_exact(3) {
                let mut px = P::default();
                px.as_mut_slice().copy_from_slice(chunk);
                out_pixels.push(px);
            }
        }

        Img::new(out_pixels, out_width as usize, out_height as usize)
    }

    /// Resize a grayscale u8 image.
    pub fn resize_gray8(
        img: ImgRef<u8>,
        out_width: u32,
        out_height: u32,
        config: &ResizeConfig,
    ) -> ImgVec<u8> {
        let mut cfg = config.clone();
        cfg.in_width = img.width() as u32;
        cfg.in_height = img.height() as u32;
        cfg.out_width = out_width;
        cfg.out_height = out_height;
        cfg.input_format = PixelFormat::Srgb8 { channels: 1, has_alpha: false };
        cfg.output_format = PixelFormat::Srgb8 { channels: 1, has_alpha: false };
        cfg.in_stride = 0;

        let mut resizer = StreamingResize::new(&cfg);

        for row in img.rows() {
            resizer.push_row(row);
        }
        resizer.finish();

        let mut out_buf = Vec::with_capacity(out_width as usize * out_height as usize);
        while let Some(row) = resizer.next_output_row() {
            out_buf.extend_from_slice(&row);
        }

        Img::new(out_buf, out_width as usize, out_height as usize)
    }
}

#[cfg(feature = "imgref")]
pub use imgref_impl::{resize_3ch, resize_4ch, resize_gray8};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::Filter;
    use crate::pixel::PixelFormat;

    fn test_config(in_w: u32, in_h: u32, out_w: u32, out_h: u32) -> ResizeConfig {
        ResizeConfig::builder(in_w, in_h, out_w, out_h)
            .filter(Filter::Lanczos)
            .format(PixelFormat::Srgb8 {
                channels: 4,
                has_alpha: true,
            })
            .srgb()
            .build()
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

    #[test]
    fn test_resize_with_stride() {
        // Create an image with extra padding bytes per row
        let config = ResizeConfig::builder(10, 10, 5, 5)
            .format(PixelFormat::Srgb8 { channels: 4, has_alpha: true })
            .srgb()
            .in_stride(10 * 4 + 8) // 8 bytes padding per row
            .build();

        let stride = 10 * 4 + 8;
        let mut input = vec![0u8; 10 * stride];
        // Fill only the pixel data (first 40 bytes of each row)
        for y in 0..10 {
            for x in 0..10 * 4 {
                input[y * stride + x] = 128;
            }
        }

        let output = resize(&config, &input);
        assert_eq!(output.len(), 5 * 5 * 4);
    }

    #[cfg(feature = "imgref")]
    #[test]
    fn test_resize_imgref_rgba() {
        use crate::resize::resize_4ch;
        use imgref::Img;
        use rgb::RGBA;

        let config = ResizeConfig::builder(20, 20, 10, 10)
            .filter(Filter::Lanczos)
            .srgb()
            .build();

        let pixels = vec![RGBA::new(128u8, 128, 128, 255); 20 * 20];
        let img = Img::new(pixels, 20, 20);

        let out = resize_4ch(img.as_ref(), 10, 10, true, &config);
        assert_eq!(out.width(), 10);
        assert_eq!(out.height(), 10);

        for px in out.pixels() {
            assert!(
                (px.r as i16 - 128).unsigned_abs() <= 2,
                "R off: {}",
                px.r
            );
            assert!(
                (px.a as i16 - 255).unsigned_abs() <= 1,
                "A off: {}",
                px.a
            );
        }
    }
}
