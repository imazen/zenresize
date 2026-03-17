//! Integration tests for zenresize.
//!
//! Tests cover: streaming vs full-frame parity, all filter types,
//! edge cases, stride handling, and format combinations.

use zenresize::{AlphaMode, Filter, PixelDescriptor, ResizeConfig, Resizer, StreamingResize};

fn config_srgb(in_w: u32, in_h: u32, out_w: u32, out_h: u32) -> ResizeConfig {
    ResizeConfig::builder(in_w, in_h, out_w, out_h)
        .format(PixelDescriptor::RGBA8_SRGB)
        .srgb()
        .build()
}

fn config_linear(in_w: u32, in_h: u32, out_w: u32, out_h: u32) -> ResizeConfig {
    ResizeConfig::builder(in_w, in_h, out_w, out_h)
        .format(PixelDescriptor::RGBA8_SRGB)
        .linear()
        .build()
}

/// Generate a gradient test image (RGBA u8).
fn gradient_image(w: u32, h: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity(w as usize * h as usize * 4);
    for y in 0..h {
        for x in 0..w {
            let r = ((x as f32 / w as f32) * 255.0) as u8;
            let g = ((y as f32 / h as f32) * 255.0) as u8;
            let b = (((x + y) as f32 / (w + h) as f32) * 255.0) as u8;
            buf.extend_from_slice(&[r, g, b, 255]);
        }
    }
    buf
}

/// Generate a gradient test image (RGB u8, 3 channels).
fn gradient_image_3ch(w: u32, h: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity(w as usize * h as usize * 3);
    for y in 0..h {
        for x in 0..w {
            let r = ((x as f32 / w as f32) * 255.0) as u8;
            let g = ((y as f32 / h as f32) * 255.0) as u8;
            let b = (((x + y) as f32 / (w + h) as f32) * 255.0) as u8;
            buf.extend_from_slice(&[r, g, b]);
        }
    }
    buf
}

/// Helper: run streaming resize with interleaved push/drain, collect u8 output.
fn streaming_collect(config: &ResizeConfig, input: &[u8]) -> Vec<u8> {
    let in_w = config.in_width as usize;
    let in_h = config.in_height as usize;
    let channels = config.input.channels();
    let row_len = in_w * channels;

    let mut resizer = StreamingResize::new(config);
    let mut output = Vec::new();
    for y in 0..in_h {
        resizer
            .push_row(&input[y * row_len..(y + 1) * row_len])
            .unwrap();
        while let Some(row) = resizer.next_output_row() {
            output.extend_from_slice(row);
        }
    }
    resizer.finish();
    while let Some(row) = resizer.next_output_row() {
        output.extend_from_slice(row);
    }
    output
}

// =============================================================================
// Streaming vs full-frame parity
// =============================================================================

#[test]
fn streaming_matches_fullframe_downscale() {
    let config = config_srgb(40, 40, 20, 20);
    let input = gradient_image(40, 40);

    // Full-frame
    let full_output = Resizer::new(&config).resize(&input);

    // Streaming
    let stream_output = streaming_collect(&config, &input);

    let max_diff: u8 = full_output
        .iter()
        .zip(stream_output.iter())
        .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);
    assert!(
        max_diff <= 1,
        "streaming vs full-frame max diff {} exceeds tolerance 1",
        max_diff
    );
}

#[test]
fn streaming_matches_fullframe_upscale() {
    let config = config_srgb(10, 10, 30, 30);
    let input = gradient_image(10, 10);

    let full_output = Resizer::new(&config).resize(&input);
    let stream_output = streaming_collect(&config, &input);

    let max_diff: u8 = full_output
        .iter()
        .zip(stream_output.iter())
        .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);
    assert!(
        max_diff <= 1,
        "streaming vs full-frame max diff {} exceeds tolerance 1",
        max_diff
    );
}

#[test]
fn streaming_matches_fullframe_linear() {
    let config = config_linear(30, 30, 15, 15);
    let input = gradient_image(30, 30);

    let full_output = Resizer::new(&config).resize(&input);
    let stream_output = streaming_collect(&config, &input);

    let max_diff: u8 = full_output
        .iter()
        .zip(stream_output.iter())
        .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);
    assert!(
        max_diff <= 1,
        "streaming vs full-frame linear max diff {} exceeds tolerance 1",
        max_diff
    );
}

// =============================================================================
// Streaming vs full-frame parity at realistic sizes
// =============================================================================

#[test]
fn streaming_matches_fullframe_srgb_1024() {
    let config = ResizeConfig::builder(1024, 1024, 512, 512)
        .filter(Filter::Lanczos)
        .format(PixelDescriptor::RGBA8_SRGB)
        .srgb()
        .build();
    let input = gradient_image(1024, 1024);
    let full_output = Resizer::new(&config).resize(&input);
    let stream_output = streaming_collect(&config, &input);
    let max_diff: u8 = full_output
        .iter()
        .zip(stream_output.iter())
        .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);
    assert!(max_diff <= 1, "max diff {} exceeds 1", max_diff);
}

#[test]
fn streaming_matches_fullframe_linear_1024() {
    let config = ResizeConfig::builder(1024, 1024, 512, 512)
        .filter(Filter::Lanczos)
        .format(PixelDescriptor::RGBX8_SRGB)
        .linear()
        .build();
    let input = gradient_image(1024, 1024);
    let full_output = Resizer::new(&config).resize(&input);
    let stream_output = streaming_collect(&config, &input);
    let max_diff: u8 = full_output
        .iter()
        .zip(stream_output.iter())
        .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);
    assert!(max_diff <= 1, "max diff {} exceeds 1", max_diff);
}

#[test]
fn streaming_matches_fullframe_srgb_upscale() {
    let config = ResizeConfig::builder(512, 512, 1024, 1024)
        .filter(Filter::Lanczos)
        .format(PixelDescriptor::RGBA8_SRGB)
        .srgb()
        .build();
    let input = gradient_image(512, 512);
    let full_output = Resizer::new(&config).resize(&input);
    let stream_output = streaming_collect(&config, &input);
    let max_diff: u8 = full_output
        .iter()
        .zip(stream_output.iter())
        .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);
    assert!(max_diff <= 1, "max diff {} exceeds 1", max_diff);
}

#[test]
fn streaming_matches_fullframe_linear_upscale() {
    let config = ResizeConfig::builder(512, 512, 1024, 1024)
        .filter(Filter::Lanczos)
        .format(PixelDescriptor::RGBX8_SRGB)
        .linear()
        .build();
    let input = gradient_image(512, 512);
    let full_output = Resizer::new(&config).resize(&input);
    let stream_output = streaming_collect(&config, &input);
    let max_diff: u8 = full_output
        .iter()
        .zip(stream_output.iter())
        .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);
    assert!(max_diff <= 1, "max diff {} exceeds 1", max_diff);
}

#[test]
fn streaming_matches_fullframe_srgb_10x() {
    let config = ResizeConfig::builder(2000, 1500, 200, 150)
        .filter(Filter::Lanczos)
        .format(PixelDescriptor::RGBA8_SRGB)
        .srgb()
        .build();
    let input = gradient_image(2000, 1500);
    let full_output = Resizer::new(&config).resize(&input);
    let stream_output = streaming_collect(&config, &input);
    let max_diff: u8 = full_output
        .iter()
        .zip(stream_output.iter())
        .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);
    assert!(max_diff <= 1, "max diff {} exceeds 1", max_diff);
}

#[test]
fn streaming_matches_fullframe_linear_10x() {
    let config = ResizeConfig::builder(2000, 1500, 200, 150)
        .filter(Filter::Lanczos)
        .format(PixelDescriptor::RGBX8_SRGB)
        .linear()
        .build();
    let input = gradient_image(2000, 1500);
    let full_output = Resizer::new(&config).resize(&input);
    let stream_output = streaming_collect(&config, &input);
    let max_diff: u8 = full_output
        .iter()
        .zip(stream_output.iter())
        .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);
    assert!(max_diff <= 1, "max diff {} exceeds 1", max_diff);
}

#[test]
fn streaming_matches_fullframe_f32_1024() {
    let config = ResizeConfig::builder(1024, 1024, 512, 512)
        .filter(Filter::Lanczos)
        .format(PixelDescriptor::RGB8_SRGB)
        .linear()
        .build();
    let input = gradient_image_3ch(1024, 1024);
    let full_output = Resizer::new(&config).resize(&input);
    let stream_output = streaming_collect(&config, &input);
    let max_diff: u8 = full_output
        .iter()
        .zip(stream_output.iter())
        .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);
    assert!(max_diff <= 1, "max diff {} exceeds 1", max_diff);
}

// =============================================================================
// All filter types produce valid output
// =============================================================================

#[test]
fn all_filters_produce_valid_output() {
    let filters = [
        Filter::Robidoux,
        Filter::RobidouxSharp,
        Filter::RobidouxFast,
        Filter::Mitchell,
        Filter::CatmullRom,
        Filter::Lanczos,
        Filter::LanczosSharp,
        Filter::Lanczos2,
        Filter::Lanczos2Sharp,
        Filter::Fastest,
        Filter::NCubic,
        Filter::NCubicSharp,
        Filter::Box,
        Filter::Triangle,
        Filter::Hermite,
        Filter::CubicBSpline,
        Filter::Ginseng,
        Filter::GinsengSharp,
        Filter::Jinc,
        Filter::RawLanczos3,
        Filter::RawLanczos3Sharp,
        Filter::RawLanczos2,
        Filter::RawLanczos2Sharp,
        Filter::CubicFast,
        Filter::Cubic,
        Filter::CubicSharp,
        Filter::CatmullRomFast,
        Filter::CatmullRomFastSharp,
        Filter::MitchellFast,
        Filter::Linear,
    ];

    let input = gradient_image(32, 32);

    for filter in &filters {
        let config = ResizeConfig::builder(32, 32, 16, 16)
            .filter(*filter)
            .format(PixelDescriptor::RGBA8_SRGB)
            .srgb()
            .build();

        let output = Resizer::new(&config).resize(&input);
        assert_eq!(
            output.len(),
            16 * 16 * 4,
            "Wrong output size for filter {:?}",
            filter
        );

        // Verify output is reasonable (not all zeros, not all 255s)
        let sum: u64 = output.iter().map(|&b| b as u64).sum();
        assert!(
            sum > 0 && sum < output.len() as u64 * 255,
            "Filter {:?} produced degenerate output (sum={})",
            filter,
            sum
        );
    }
}

// =============================================================================
// Edge cases
// =============================================================================

#[test]
fn resize_1x1_to_1x1() {
    let config = config_srgb(1, 1, 1, 1);
    let input = vec![128, 64, 32, 255];
    let output = Resizer::new(&config).resize(&input);
    assert_eq!(output.len(), 4);
    // Single pixel should be approximately preserved
    assert!((output[0] as i16 - 128).unsigned_abs() <= 2);
    assert!((output[1] as i16 - 64).unsigned_abs() <= 2);
    assert!((output[2] as i16 - 32).unsigned_abs() <= 2);
    assert!((output[3] as i16 - 255).unsigned_abs() <= 1);
}

#[test]
fn resize_1xn() {
    let config = config_srgb(1, 10, 1, 5);
    let input = vec![128u8; 10 * 4];
    let output = Resizer::new(&config).resize(&input);
    assert_eq!(output.len(), 5 * 4);
}

#[test]
fn resize_nx1() {
    let config = config_srgb(10, 1, 5, 1);
    let input = vec![128u8; 10 * 4];
    let output = Resizer::new(&config).resize(&input);
    assert_eq!(output.len(), 5 * 4);
}

#[test]
fn resize_same_size() {
    let config = config_srgb(20, 20, 20, 20);
    let input = gradient_image(20, 20);
    let output = Resizer::new(&config).resize(&input);
    assert_eq!(output.len(), input.len());

    // Same-size resize should be very close to identity
    let max_diff: u8 = input
        .iter()
        .zip(output.iter())
        .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);
    assert!(
        max_diff <= 3,
        "Same-size resize drifted too much: max diff {}",
        max_diff
    );
}

#[test]
fn resize_non_square() {
    let config = config_srgb(100, 50, 30, 75);
    let input = gradient_image(100, 50);
    let output = Resizer::new(&config).resize(&input);
    assert_eq!(output.len(), 30 * 75 * 4);
}

#[test]
fn resize_large_downscale() {
    let config = config_srgb(200, 200, 10, 10);
    let input = vec![128u8; 200 * 200 * 4];
    let output = Resizer::new(&config).resize(&input);
    assert_eq!(output.len(), 10 * 10 * 4);

    // Constant input → output should be close to constant
    for px in output.chunks_exact(4) {
        assert!(
            (px[0] as i16 - 128).unsigned_abs() <= 3,
            "pixel value drifted: {}",
            px[0]
        );
    }
}

#[test]
fn resize_large_upscale() {
    let config = config_srgb(5, 5, 100, 100);
    let input = vec![100u8; 5 * 5 * 4];
    let output = Resizer::new(&config).resize(&input);
    assert_eq!(output.len(), 100 * 100 * 4);
}

// =============================================================================
// Channel count variations
// =============================================================================

#[test]
fn resize_rgb_3ch() {
    let config = ResizeConfig::builder(20, 20, 10, 10)
        .format(PixelDescriptor::RGB8_SRGB)
        .srgb()
        .build();

    let input = vec![128u8; 20 * 20 * 3];
    let output = Resizer::new(&config).resize(&input);
    assert_eq!(output.len(), 10 * 10 * 3);
}

#[test]
fn resize_gray_1ch() {
    let config = ResizeConfig::builder(20, 20, 10, 10)
        .format(PixelDescriptor::GRAY8_SRGB)
        .srgb()
        .build();

    let input = vec![128u8; 20 * 20];
    let output = Resizer::new(&config).resize(&input);
    assert_eq!(output.len(), 10 * 10);
}

#[test]
fn resize_rgbx_no_premul() {
    // RGBX: 4 channels, no alpha premultiply/unpremultiply
    let config = ResizeConfig::builder(20, 20, 10, 10)
        .format(PixelDescriptor::RGBX8_SRGB)
        .srgb()
        .build();

    let mut input = vec![0u8; 20 * 20 * 4];
    for px in input.chunks_exact_mut(4) {
        px[0] = 128;
        px[1] = 64;
        px[2] = 32;
        px[3] = 0; // padding
    }
    let output = Resizer::new(&config).resize(&input);
    assert_eq!(output.len(), 10 * 10 * 4);
}

#[test]
fn resize_premultiplied_alpha() {
    // RgbaPremul: 4 channels, already premultiplied — skip premul/unpremul
    let config = ResizeConfig::builder(20, 20, 10, 10)
        .format(PixelDescriptor::RGBA8_SRGB.with_alpha(Some(AlphaMode::Premultiplied)))
        .srgb()
        .build();

    let mut input = vec![0u8; 20 * 20 * 4];
    for px in input.chunks_exact_mut(4) {
        px[0] = 64; // R * A/255 = 128 * 128/255 ≈ 64
        px[1] = 32;
        px[2] = 16;
        px[3] = 128; // alpha
    }
    let output = Resizer::new(&config).resize(&input);
    assert_eq!(output.len(), 10 * 10 * 4);

    // Constant input → output should be close to constant
    for px in output.chunks_exact(4) {
        assert!((px[0] as i16 - 64).unsigned_abs() <= 3, "R: {}", px[0]);
        assert!((px[3] as i16 - 128).unsigned_abs() <= 2, "A: {}", px[3]);
    }
}

// =============================================================================
// f32 path
// =============================================================================

#[test]
fn resize_f32_constant() {
    let config = ResizeConfig::builder(20, 20, 10, 10)
        .format(PixelDescriptor::RGBAF32_LINEAR)
        .linear()
        .build();

    let input = vec![0.5f32; 20 * 20 * 4];
    let output = Resizer::new(&config).resize_f32(&input);
    assert_eq!(output.len(), 10 * 10 * 4);

    for &v in &output {
        assert!((v - 0.5).abs() < 0.02, "f32 constant value drifted: {}", v);
    }
}

#[test]
fn resize_f32_gradient() {
    let config = ResizeConfig::builder(30, 30, 15, 15)
        .format(PixelDescriptor::RGBAF32_LINEAR.with_alpha(Some(AlphaMode::Undefined)))
        .linear()
        .build();

    let mut input = Vec::with_capacity(30 * 30 * 4);
    for y in 0..30 {
        for x in 0..30 {
            input.push(x as f32 / 29.0);
            input.push(y as f32 / 29.0);
            input.push(0.5);
            input.push(1.0);
        }
    }

    let output = Resizer::new(&config).resize_f32(&input);
    assert_eq!(output.len(), 15 * 15 * 4);

    // All values should be in [0, 1] range (approximately)
    for &v in &output {
        assert!((-0.1..=1.1).contains(&v), "f32 output out of range: {}", v);
    }
}

// =============================================================================
// Linear vs sRGB color space
// =============================================================================

#[test]
fn linear_and_srgb_produce_different_results() {
    let input = gradient_image(30, 30);

    let config_lin = config_linear(30, 30, 15, 15);
    let config_s = config_srgb(30, 30, 15, 15);

    let out_lin = Resizer::new(&config_lin).resize(&input);
    let out_srgb = Resizer::new(&config_s).resize(&input);

    // They should produce different results (linear is gamma-correct)
    assert_ne!(
        out_lin, out_srgb,
        "linear and sRGB resize should differ on gradients"
    );
}

// =============================================================================
// Stride handling
// =============================================================================

#[test]
fn strided_input_matches_tight() {
    let w = 20u32;
    let h = 20u32;
    let channels = 4;

    // Tight input
    let tight_input = gradient_image(w, h);
    let config_tight = config_srgb(w, h, 10, 10);
    let out_tight = Resizer::new(&config_tight).resize(&tight_input);

    // Strided input (16 bytes of padding per row)
    let padding = 16;
    let tight_stride = w as usize * channels;
    let padded_stride = tight_stride + padding;
    let mut strided_input = vec![0u8; h as usize * padded_stride];
    for y in 0..h as usize {
        let src = &tight_input[y * tight_stride..(y + 1) * tight_stride];
        let dst = &mut strided_input[y * padded_stride..y * padded_stride + tight_stride];
        dst.copy_from_slice(src);
    }

    let config_strided = ResizeConfig::builder(w, h, 10, 10)
        .format(PixelDescriptor::RGBA8_SRGB)
        .srgb()
        .in_stride(padded_stride)
        .build();

    let out_strided = Resizer::new(&config_strided).resize(&strided_input);
    assert_eq!(
        out_tight, out_strided,
        "strided input should match tight input"
    );
}

// =============================================================================
// Interleaved push/drain matches batch-then-drain (via streaming_collect)
// =============================================================================

#[test]
fn interleaved_push_drain_produces_correct_output() {
    let config = config_srgb(20, 20, 10, 10);
    let input = gradient_image(20, 20);

    // Use streaming_collect helper (interleaved push/drain)
    let output = streaming_collect(&config, &input);
    assert_eq!(output.len(), 10 * 10 * 4);

    // Compare with fullframe
    let fullframe = Resizer::new(&config).resize(&input);
    let max_diff: u8 = fullframe
        .iter()
        .zip(output.iter())
        .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);
    assert!(
        max_diff <= 1,
        "interleaved push/drain vs fullframe max diff {} exceeds tolerance 1",
        max_diff
    );
}

// =============================================================================
// Linear i16 vs f32 parity
// =============================================================================

/// The linear i16 path (4ch, no alpha, linearize) should produce output
/// within ±1 of the f32 linear path (streaming, which always uses f32).
#[test]
fn linear_i16_matches_f32_downscale() {
    let config = ResizeConfig::builder(64, 64, 32, 32)
        .filter(Filter::Lanczos)
        .format(PixelDescriptor::RGBX8_SRGB)
        .linear()
        .build();

    let input = gradient_image(64, 64);

    // Full-frame: uses i16 linear path (linearize + 4ch + !has_alpha)
    let i16_output = Resizer::new(&config).resize(&input);

    // Streaming: always uses f32 path
    let f32_output = streaming_collect(&config, &input);

    let max_diff: u8 = i16_output
        .iter()
        .zip(f32_output.iter())
        .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);
    assert!(
        max_diff <= 1,
        "linear i16 vs f32 max diff {} exceeds tolerance 1",
        max_diff
    );
}

/// Parity test with upscale to catch different edge cases.
#[test]
fn linear_i16_matches_f32_upscale() {
    let config = ResizeConfig::builder(16, 16, 48, 48)
        .filter(Filter::Lanczos)
        .format(PixelDescriptor::RGBX8_SRGB)
        .linear()
        .build();

    let input = gradient_image(16, 16);

    let i16_output = Resizer::new(&config).resize(&input);
    let f32_output = streaming_collect(&config, &input);

    let max_diff: u8 = i16_output
        .iter()
        .zip(f32_output.iter())
        .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);
    assert!(
        max_diff <= 2,
        "linear i16 vs f32 upscale max diff {} exceeds tolerance 2",
        max_diff
    );
}

// =============================================================================
// Comprehensive regression: all filter × scale × path combinations
// =============================================================================

/// Compare fullframe (i16 path) vs streaming (f32 path) across all filters,
/// multiple scales, and path configurations. Catches weight normalization bugs.
#[test]
fn no_catastrophic_errors_across_all_combinations() {
    let scales: &[(u32, u32, &str)] = &[
        (200, 100, "2x_down"),
        (200, 50, "4x_down"),
        (200, 25, "8x_down"),
        (50, 100, "2x_up"),
        (50, 200, "4x_up"),
        (50, 400, "8x_up"),
    ];

    // Path configs: (format, color_space_fn, label)
    let path_configs: Vec<(PixelDescriptor, bool, &str)> = vec![
        (PixelDescriptor::RGBX8_SRGB, false, "srgb-noalpha"),
        (PixelDescriptor::RGBA8_SRGB, false, "srgb-alpha"),
        (PixelDescriptor::RGBX8_SRGB, true, "linear-noalpha"),
        (PixelDescriptor::RGBA8_SRGB, true, "linear-alpha"),
    ];

    let mut failures: Vec<String> = Vec::new();

    for &filter in Filter::all() {
        for &(in_size, out_size, scale_name) in scales {
            for &(format, linearize, path_name) in &path_configs {
                let config = {
                    let mut b = ResizeConfig::builder(in_size, in_size, out_size, out_size)
                        .filter(filter)
                        .format(format);
                    if linearize {
                        b = b.linear();
                    } else {
                        b = b.srgb();
                    }
                    b.build()
                };

                let input = gradient_image(in_size, in_size);

                // Full-frame
                let full_output = Resizer::new(&config).resize(&input);

                // Streaming (always f32 weights with f64 normalization)
                let stream_output = streaming_collect(&config, &input);

                assert_eq!(
                    full_output.len(),
                    stream_output.len(),
                    "size mismatch: {:?} {} {}",
                    filter,
                    scale_name,
                    path_name
                );

                let max_diff: u8 = full_output
                    .iter()
                    .zip(stream_output.iter())
                    .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
                    .max()
                    .unwrap_or(0);

                // Threshold: ±2 for i16 vs f32 quantization differences
                if max_diff > 2 {
                    failures.push(format!(
                        "  {:20?} {:8} {:16} max_diff={}",
                        filter, scale_name, path_name, max_diff,
                    ));
                }
            }
        }
    }

    if !failures.is_empty() {
        panic!(
            "Catastrophic errors found in {} combinations:\n{}",
            failures.len(),
            failures.join("\n")
        );
    }
}

// =============================================================================
// BGRA / BGRX: native zero-swizzle support
// =============================================================================
// The pipeline is channel-order-agnostic. sRGB transfer is identical for R/G/B,
// and convolution kernels just multiply N floats. BGRA data passes through
// unchanged — use RGBA8_SRGB, same as RGBA.

#[test]
fn bgra_preserves_channel_order() {
    // BGRA input should come out as BGRA — no swizzle, no reordering.
    let w = 20u32;
    let h = 20u32;
    let mut bgra_input = vec![0u8; (w * h * 4) as usize];
    for px in bgra_input.chunks_exact_mut(4) {
        px[0] = 200; // B
        px[1] = 100; // G
        px[2] = 50; // R
        px[3] = 255; // A
    }

    // Same config as RGBA8_SRGB — pipeline doesn't care about channel names
    let config = ResizeConfig::builder(w, h, 10, 10)
        .format(PixelDescriptor::RGBA8_SRGB)
        .srgb()
        .build();

    let output = Resizer::new(&config).resize(&bgra_input);
    assert_eq!(output.len(), 10 * 10 * 4);

    // Output preserves BGRA order
    for px in output.chunks_exact(4) {
        assert!((px[0] as i16 - 200).unsigned_abs() <= 3, "B: {}", px[0]);
        assert!((px[1] as i16 - 100).unsigned_abs() <= 3, "G: {}", px[1]);
        assert!((px[2] as i16 - 50).unsigned_abs() <= 3, "R: {}", px[2]);
        assert!((px[3] as i16 - 255).unsigned_abs() <= 1, "A: {}", px[3]);
    }
}

#[test]
fn bgrx_as_4ch_no_alpha() {
    // BGRX: 4 bytes, no alpha. Use RGBX8_SRGB.
    // The X channel is just another data channel — passes through like any other.
    let w = 20u32;
    let h = 20u32;
    let mut bgrx_input = vec![0u8; (w * h * 4) as usize];
    for px in bgrx_input.chunks_exact_mut(4) {
        px[0] = 200; // B
        px[1] = 100; // G
        px[2] = 50; // R
        px[3] = 0; // X (padding)
    }

    let config = ResizeConfig::builder(w, h, 10, 10)
        .format(PixelDescriptor::RGBX8_SRGB)
        .srgb()
        .build();

    let output = Resizer::new(&config).resize(&bgrx_input);
    assert_eq!(output.len(), 10 * 10 * 4);

    // Channel order preserved, X stays ~0 (it was 0 in constant input)
    for px in output.chunks_exact(4) {
        assert!((px[0] as i16 - 200).unsigned_abs() <= 3, "B: {}", px[0]);
        assert!((px[1] as i16 - 100).unsigned_abs() <= 3, "G: {}", px[1]);
        assert!((px[2] as i16 - 50).unsigned_abs() <= 3, "R: {}", px[2]);
        assert!(px[3] <= 3, "X should stay ~0, got {}", px[3]);
    }
}

#[test]
fn bgra_linear_preserves_order() {
    // Even with linear-light processing, channel order is preserved.
    // sRGB→linear curve is the same for channels 0, 1, and 2 regardless of
    // whether they represent R,G,B or B,G,R.
    let w = 20u32;
    let h = 20u32;
    let mut bgra_input = vec![0u8; (w * h * 4) as usize];
    for px in bgra_input.chunks_exact_mut(4) {
        px[0] = 200; // B
        px[1] = 100; // G
        px[2] = 50; // R
        px[3] = 255; // A
    }

    let config = ResizeConfig::builder(w, h, 10, 10)
        .format(PixelDescriptor::RGBA8_SRGB)
        .linear() // linear-light processing
        .build();

    let output = Resizer::new(&config).resize(&bgra_input);

    for px in output.chunks_exact(4) {
        assert!((px[0] as i16 - 200).unsigned_abs() <= 3, "B: {}", px[0]);
        assert!((px[1] as i16 - 100).unsigned_abs() <= 3, "G: {}", px[1]);
        assert!((px[2] as i16 - 50).unsigned_abs() <= 3, "R: {}", px[2]);
        assert!((px[3] as i16 - 255).unsigned_abs() <= 1, "A: {}", px[3]);
    }
}

// =============================================================================
// Post-resize blur tests
// =============================================================================

#[test]
fn identity_resize_no_blur_is_bit_identical() {
    let w = 64u32;
    let h = 64u32;
    let input = gradient_image(w, h);

    let config = ResizeConfig::builder(w, h, w, h)
        .format(PixelDescriptor::RGBA8_SRGB)
        .srgb()
        .build();

    let output = Resizer::new(&config).resize(&input);
    assert_eq!(input, output, "ratio=1 sRGB path should be bit-identical");
}

#[test]
fn identity_resize_linear_no_blur_max_diff_1() {
    let w = 64u32;
    let h = 64u32;
    let input = gradient_image(w, h);

    let config = ResizeConfig::builder(w, h, w, h)
        .format(PixelDescriptor::RGBA8_SRGB)
        .linear()
        .build();

    let output = Resizer::new(&config).resize(&input);

    let max_diff: u8 = input
        .iter()
        .zip(output.iter())
        .map(|(&a, &b)| a.abs_diff(b))
        .max()
        .unwrap_or(0);
    assert!(
        max_diff <= 1,
        "linear identity max diff = {max_diff}, expected <= 1"
    );
}

#[test]
fn blur_reduces_high_frequency_content() {
    let w = 64u32;
    let h = 64u32;

    let mut input = vec![0u8; w as usize * h as usize * 4];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let idx = (y * w as usize + x) * 4;
            let val = if (x + y) % 2 == 0 { 200u8 } else { 50u8 };
            input[idx] = val;
            input[idx + 1] = val;
            input[idx + 2] = val;
            input[idx + 3] = 255;
        }
    }

    let config = ResizeConfig::builder(w, h, w, h)
        .format(PixelDescriptor::RGBA8_SRGB)
        .srgb()
        .post_blur(1.5)
        .build();

    let output = Resizer::new(&config).resize(&input);

    let mut input_diff_sum = 0u64;
    let mut output_diff_sum = 0u64;
    for y in 0..h as usize {
        for x in 1..w as usize {
            let idx = (y * w as usize + x) * 4;
            let prev = (y * w as usize + x - 1) * 4;
            input_diff_sum += (input[idx] as i32 - input[prev] as i32).unsigned_abs() as u64;
            output_diff_sum += (output[idx] as i32 - output[prev] as i32).unsigned_abs() as u64;
        }
    }

    assert!(
        output_diff_sum < input_diff_sum / 2,
        "blur should reduce neighbor diffs: input={input_diff_sum}, output={output_diff_sum}"
    );
}

#[test]
fn blur_uniform_image_stays_uniform() {
    let w = 32u32;
    let h = 32u32;
    let input = vec![128u8; w as usize * h as usize * 4];

    let config = ResizeConfig::builder(w, h, w, h)
        .format(PixelDescriptor::RGBA8_SRGB)
        .srgb()
        .post_blur(2.0)
        .build();

    let output = Resizer::new(&config).resize(&input);

    let max_diff: u8 = input
        .iter()
        .zip(output.iter())
        .map(|(&a, &b)| a.abs_diff(b))
        .max()
        .unwrap_or(0);
    assert!(
        max_diff <= 1,
        "uniform image should stay uniform, max diff = {max_diff}"
    );
}

#[test]
fn blur_f32_reduces_variance() {
    let w = 32u32;
    let h = 32u32;
    let channels = 4;
    let len = w as usize * h as usize * channels;

    let mut input = vec![0.0f32; len];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let idx = (y * w as usize + x) * channels;
            let val = if (x + y) % 2 == 0 { 0.8 } else { 0.2 };
            input[idx] = val;
            input[idx + 1] = val;
            input[idx + 2] = val;
            input[idx + 3] = 1.0;
        }
    }

    let config = ResizeConfig::builder(w, h, w, h)
        .format(PixelDescriptor::RGBAF32_LINEAR)
        .post_blur(1.0)
        .build();

    let output = Resizer::new(&config).resize_f32(&input);

    let max_deviation: f32 = output
        .chunks(channels)
        .flat_map(|px| px[..3].iter())
        .map(|&v| (v - 0.5).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_deviation < 0.15,
        "blurred pattern should converge toward mean, max_deviation = {max_deviation}"
    );
}
