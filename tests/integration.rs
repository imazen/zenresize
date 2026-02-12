//! Integration tests for zenresize.
//!
//! Tests cover: streaming vs full-frame parity, all filter types,
//! edge cases, stride handling, and format combinations.

use zenresize::filter::Filter;
use zenresize::pixel::{ChannelOrder, PixelFormat, ResizeConfig};
use zenresize::resize::{resize, resize_f32};
use zenresize::streaming::StreamingResize;

fn config_srgb(in_w: u32, in_h: u32, out_w: u32, out_h: u32) -> ResizeConfig {
    ResizeConfig::builder(in_w, in_h, out_w, out_h)
        .format(PixelFormat::Srgb8 {
            channels: 4,
            has_alpha: true,
        })
        .srgb()
        .build()
}

fn config_linear(in_w: u32, in_h: u32, out_w: u32, out_h: u32) -> ResizeConfig {
    ResizeConfig::builder(in_w, in_h, out_w, out_h)
        .format(PixelFormat::Srgb8 {
            channels: 4,
            has_alpha: true,
        })
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

// =============================================================================
// Streaming vs full-frame parity
// =============================================================================

#[test]
fn streaming_matches_fullframe_downscale() {
    let config = config_srgb(40, 40, 20, 20);
    let input = gradient_image(40, 40);

    // Full-frame
    let full_output = resize(&config, &input);

    // Streaming
    let mut resizer = StreamingResize::new(&config);
    let row_len = 40 * 4;
    for y in 0..40 {
        resizer.push_row(&input[y * row_len..(y + 1) * row_len]);
    }
    resizer.finish();

    let mut stream_output = vec![0u8; 20 * 20 * 4];
    let out_row_len = 20 * 4;
    let mut idx = 0;
    while let Some(row) = resizer.next_output_row() {
        stream_output[idx * out_row_len..(idx + 1) * out_row_len].copy_from_slice(&row);
        idx += 1;
    }

    assert_eq!(full_output, stream_output, "streaming and full-frame outputs must be identical");
}

#[test]
fn streaming_matches_fullframe_upscale() {
    let config = config_srgb(10, 10, 30, 30);
    let input = gradient_image(10, 10);

    let full_output = resize(&config, &input);

    let mut resizer = StreamingResize::new(&config);
    let row_len = 10 * 4;
    for y in 0..10 {
        resizer.push_row(&input[y * row_len..(y + 1) * row_len]);
    }
    resizer.finish();

    let mut stream_output = vec![0u8; 30 * 30 * 4];
    let out_row_len = 30 * 4;
    let mut idx = 0;
    while let Some(row) = resizer.next_output_row() {
        stream_output[idx * out_row_len..(idx + 1) * out_row_len].copy_from_slice(&row);
        idx += 1;
    }

    assert_eq!(full_output, stream_output);
}

#[test]
fn streaming_matches_fullframe_linear() {
    let config = config_linear(30, 30, 15, 15);
    let input = gradient_image(30, 30);

    let full_output = resize(&config, &input);

    let mut resizer = StreamingResize::new(&config);
    let row_len = 30 * 4;
    for y in 0..30 {
        resizer.push_row(&input[y * row_len..(y + 1) * row_len]);
    }
    resizer.finish();

    let mut stream_output = vec![0u8; 15 * 15 * 4];
    let out_row_len = 15 * 4;
    let mut idx = 0;
    while let Some(row) = resizer.next_output_row() {
        stream_output[idx * out_row_len..(idx + 1) * out_row_len].copy_from_slice(&row);
        idx += 1;
    }

    assert_eq!(full_output, stream_output);
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
            .format(PixelFormat::Srgb8 {
                channels: 4,
                has_alpha: true,
            })
            .srgb()
            .build();

        let output = resize(&config, &input);
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
    let output = resize(&config, &input);
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
    let input = vec![128u8; 1 * 10 * 4];
    let output = resize(&config, &input);
    assert_eq!(output.len(), 1 * 5 * 4);
}

#[test]
fn resize_nx1() {
    let config = config_srgb(10, 1, 5, 1);
    let input = vec![128u8; 10 * 1 * 4];
    let output = resize(&config, &input);
    assert_eq!(output.len(), 5 * 1 * 4);
}

#[test]
fn resize_same_size() {
    let config = config_srgb(20, 20, 20, 20);
    let input = gradient_image(20, 20);
    let output = resize(&config, &input);
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
    let output = resize(&config, &input);
    assert_eq!(output.len(), 30 * 75 * 4);
}

#[test]
fn resize_large_downscale() {
    let config = config_srgb(200, 200, 10, 10);
    let input = vec![128u8; 200 * 200 * 4];
    let output = resize(&config, &input);
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
    let output = resize(&config, &input);
    assert_eq!(output.len(), 100 * 100 * 4);
}

// =============================================================================
// Channel count variations
// =============================================================================

#[test]
fn resize_rgb_3ch() {
    let config = ResizeConfig::builder(20, 20, 10, 10)
        .format(PixelFormat::Srgb8 {
            channels: 3,
            has_alpha: false,
        })
        .srgb()
        .build();

    let input = vec![128u8; 20 * 20 * 3];
    let output = resize(&config, &input);
    assert_eq!(output.len(), 10 * 10 * 3);
}

#[test]
fn resize_gray_1ch() {
    let config = ResizeConfig::builder(20, 20, 10, 10)
        .format(PixelFormat::Srgb8 {
            channels: 1,
            has_alpha: false,
        })
        .srgb()
        .build();

    let input = vec![128u8; 20 * 20];
    let output = resize(&config, &input);
    assert_eq!(output.len(), 10 * 10);
}

#[test]
fn resize_gray_alpha_2ch() {
    let config = ResizeConfig::builder(20, 20, 10, 10)
        .format(PixelFormat::Srgb8 {
            channels: 2,
            has_alpha: true,
        })
        .srgb()
        .build();

    let mut input = vec![0u8; 20 * 20 * 2];
    for px in input.chunks_exact_mut(2) {
        px[0] = 128;
        px[1] = 255;
    }
    let output = resize(&config, &input);
    assert_eq!(output.len(), 10 * 10 * 2);
}

// =============================================================================
// f32 path
// =============================================================================

#[test]
fn resize_f32_constant() {
    let config = ResizeConfig::builder(20, 20, 10, 10)
        .format(PixelFormat::LinearF32 {
            channels: 4,
            has_alpha: true,
        })
        .linear()
        .build();

    let input = vec![0.5f32; 20 * 20 * 4];
    let output = resize_f32(&config, &input);
    assert_eq!(output.len(), 10 * 10 * 4);

    for &v in &output {
        assert!(
            (v - 0.5).abs() < 0.02,
            "f32 constant value drifted: {}",
            v
        );
    }
}

#[test]
fn resize_f32_gradient() {
    let config = ResizeConfig::builder(30, 30, 15, 15)
        .format(PixelFormat::LinearF32 {
            channels: 4,
            has_alpha: false,
        })
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

    let output = resize_f32(&config, &input);
    assert_eq!(output.len(), 15 * 15 * 4);

    // All values should be in [0, 1] range (approximately)
    for &v in &output {
        assert!(
            v >= -0.1 && v <= 1.1,
            "f32 output out of range: {}",
            v
        );
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

    let out_lin = resize(&config_lin, &input);
    let out_srgb = resize(&config_s, &input);

    // They should produce different results (linear is gamma-correct)
    assert_ne!(out_lin, out_srgb, "linear and sRGB resize should differ on gradients");
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
    let out_tight = resize(&config_tight, &tight_input);

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
        .format(PixelFormat::Srgb8 {
            channels: 4,
            has_alpha: true,
        })
        .srgb()
        .in_stride(padded_stride)
        .build();

    let out_strided = resize(&config_strided, &strided_input);
    assert_eq!(out_tight, out_strided, "strided input should match tight input");
}

// =============================================================================
// push_rows batch method
// =============================================================================

#[test]
fn push_rows_matches_individual() {
    let config = config_srgb(20, 20, 10, 10);
    let input = gradient_image(20, 20);
    let row_len = 20 * 4;

    // Individual pushes
    let mut resizer1 = StreamingResize::new(&config);
    for y in 0..20 {
        resizer1.push_row(&input[y * row_len..(y + 1) * row_len]);
    }
    resizer1.finish();

    let mut out1 = Vec::new();
    while let Some(row) = resizer1.next_output_row() {
        out1.extend_from_slice(&row);
    }

    // Batch push
    let mut resizer2 = StreamingResize::new(&config);
    resizer2.push_rows(&input, row_len, 20);
    resizer2.finish();

    let mut out2 = Vec::new();
    while let Some(row) = resizer2.next_output_row() {
        out2.extend_from_slice(&row);
    }

    assert_eq!(out1, out2, "push_rows batch should match individual push_row");
}

// =============================================================================
// BGRA / BGRX channel order
// =============================================================================

#[test]
fn bgra_input_produces_correct_output() {
    // Create BGRA input (B=200, G=100, R=50, A=255)
    let w = 20u32;
    let h = 20u32;
    let mut bgra_input = vec![0u8; (w * h * 4) as usize];
    for px in bgra_input.chunks_exact_mut(4) {
        px[0] = 200; // B
        px[1] = 100; // G
        px[2] = 50;  // R
        px[3] = 255; // A
    }

    // Resize with BGRA input → RGBA output
    let config = ResizeConfig::builder(w, h, 10, 10)
        .format(PixelFormat::Srgb8 { channels: 4, has_alpha: true })
        .in_channel_order(ChannelOrder::Bgra)
        .out_channel_order(ChannelOrder::Rgba)
        .srgb()
        .build();

    let output = resize(&config, &bgra_input);
    assert_eq!(output.len(), 10 * 10 * 4);

    // Output should be RGBA: R≈50, G≈100, B≈200, A≈255
    for px in output.chunks_exact(4) {
        assert!((px[0] as i16 - 50).unsigned_abs() <= 3, "R: {}", px[0]);
        assert!((px[1] as i16 - 100).unsigned_abs() <= 3, "G: {}", px[1]);
        assert!((px[2] as i16 - 200).unsigned_abs() <= 3, "B: {}", px[2]);
        assert!((px[3] as i16 - 255).unsigned_abs() <= 1, "A: {}", px[3]);
    }
}

#[test]
fn bgra_roundtrip() {
    // BGRA input → BGRA output should preserve channel positions
    let w = 20u32;
    let h = 20u32;
    let mut bgra_input = vec![0u8; (w * h * 4) as usize];
    for px in bgra_input.chunks_exact_mut(4) {
        px[0] = 200; // B
        px[1] = 100; // G
        px[2] = 50;  // R
        px[3] = 255; // A
    }

    let config = ResizeConfig::builder(w, h, 10, 10)
        .format(PixelFormat::Srgb8 { channels: 4, has_alpha: true })
        .channel_order(ChannelOrder::Bgra) // both input and output BGRA
        .srgb()
        .build();

    let output = resize(&config, &bgra_input);

    // Output should still be BGRA: B≈200, G≈100, R≈50, A≈255
    for px in output.chunks_exact(4) {
        assert!((px[0] as i16 - 200).unsigned_abs() <= 3, "B: {}", px[0]);
        assert!((px[1] as i16 - 100).unsigned_abs() <= 3, "G: {}", px[1]);
        assert!((px[2] as i16 - 50).unsigned_abs() <= 3, "R: {}", px[2]);
        assert!((px[3] as i16 - 255).unsigned_abs() <= 1, "A: {}", px[3]);
    }
}

#[test]
fn bgrx_input_no_alpha() {
    // BGRX: 4 bytes but no alpha (X is padding)
    let w = 20u32;
    let h = 20u32;
    let mut bgrx_input = vec![0u8; (w * h * 4) as usize];
    for px in bgrx_input.chunks_exact_mut(4) {
        px[0] = 200; // B
        px[1] = 100; // G
        px[2] = 50;  // R
        px[3] = 0;   // X (garbage/padding)
    }

    let config = ResizeConfig::builder(w, h, 10, 10)
        .format(PixelFormat::Srgb8 { channels: 4, has_alpha: false })
        .channel_order(ChannelOrder::Bgrx)
        .srgb()
        .build();

    let output = resize(&config, &bgrx_input);
    assert_eq!(output.len(), 10 * 10 * 4);

    // Output should be BGRX with X=255
    for px in output.chunks_exact(4) {
        assert!((px[0] as i16 - 200).unsigned_abs() <= 3, "B: {}", px[0]);
        assert!((px[1] as i16 - 100).unsigned_abs() <= 3, "G: {}", px[1]);
        assert!((px[2] as i16 - 50).unsigned_abs() <= 3, "R: {}", px[2]);
        assert_eq!(px[3], 255, "X should be 255");
    }
}

#[test]
fn bgra_to_rgba_conversion() {
    // Test cross-format: BGRA input → RGBA output
    let w = 10u32;
    let h = 10u32;
    let mut bgra = vec![0u8; (w * h * 4) as usize];
    for px in bgra.chunks_exact_mut(4) {
        px[0] = 255; // B
        px[1] = 0;   // G
        px[2] = 0;   // R
        px[3] = 255; // A
    }

    let config = ResizeConfig::builder(w, h, w, h) // same size = identity-ish
        .format(PixelFormat::Srgb8 { channels: 4, has_alpha: true })
        .in_channel_order(ChannelOrder::Bgra)
        .out_channel_order(ChannelOrder::Rgba)
        .srgb()
        .build();

    let output = resize(&config, &bgra);

    // Input was pure blue in BGRA (B=255,G=0,R=0,A=255)
    // Output as RGBA should be (R=0,G=0,B=255,A=255)
    for px in output.chunks_exact(4) {
        assert!(px[0] <= 3, "R should be ~0, got {}", px[0]);
        assert!(px[1] <= 3, "G should be ~0, got {}", px[1]);
        assert!(px[2] >= 252, "B should be ~255, got {}", px[2]);
        assert!(px[3] >= 254, "A should be ~255, got {}", px[3]);
    }
}
