//! Cross-format resize test matrix.
//!
//! Tests all 9 format pairs {Srgb8, LinearF32, Encoded16} × {Srgb8, LinearF32, Encoded16}
//! with both transfer functions {Srgb, None} on input and output (where applicable).

use zenresize::{
    Filter, PixelFormat, PixelLayout, ResizeConfig, Resizer, StreamingResize, Transfer,
};

const IN_W: u32 = 40;
const IN_H: u32 = 40;
const OUT_W: u32 = 20;
const OUT_H: u32 = 20;
const LAYOUT: PixelLayout = PixelLayout::Rgba;
const CH: usize = 4;

// =============================================================================
// Helper: make constant-color test images
// =============================================================================

fn make_u8_image(r: u8, g: u8, b: u8, a: u8) -> Vec<u8> {
    let mut img = vec![0u8; IN_W as usize * IN_H as usize * CH];
    for pixel in img.chunks_exact_mut(CH) {
        pixel[0] = r;
        pixel[1] = g;
        pixel[2] = b;
        pixel[3] = a;
    }
    img
}

fn make_f32_image(r: f32, g: f32, b: f32, a: f32) -> Vec<f32> {
    let mut img = vec![0.0f32; IN_W as usize * IN_H as usize * CH];
    for pixel in img.chunks_exact_mut(CH) {
        pixel[0] = r;
        pixel[1] = g;
        pixel[2] = b;
        pixel[3] = a;
    }
    img
}

fn make_u16_image(r: u16, g: u16, b: u16, a: u16) -> Vec<u16> {
    let mut img = vec![0u16; IN_W as usize * IN_H as usize * CH];
    for pixel in img.chunks_exact_mut(CH) {
        pixel[0] = r;
        pixel[1] = g;
        pixel[2] = b;
        pixel[3] = a;
    }
    img
}

// =============================================================================
// 1. Same-format constant-color consistency (sanity check)
// =============================================================================

#[test]
fn same_format_u8_srgb() {
    let config = ResizeConfig::builder(IN_W, IN_H, OUT_W, OUT_H)
        .filter(Filter::Lanczos)
        .format(PixelFormat::Srgb8(LAYOUT))
        .linear()
        .build();
    let input = make_u8_image(128, 64, 32, 255);
    let output = Resizer::new(&config).resize(&input);
    assert_eq!(output.len(), OUT_W as usize * OUT_H as usize * CH);
    for pixel in output.chunks_exact(CH) {
        assert!(
            (pixel[0] as i16 - 128).unsigned_abs() <= 2,
            "R: {}",
            pixel[0]
        );
        assert!(
            (pixel[1] as i16 - 64).unsigned_abs() <= 2,
            "G: {}",
            pixel[1]
        );
        assert!(
            (pixel[2] as i16 - 32).unsigned_abs() <= 2,
            "B: {}",
            pixel[2]
        );
        assert_eq!(pixel[3], 255, "A: {}", pixel[3]);
    }
}

#[test]
fn same_format_f32_linear() {
    let config = ResizeConfig::builder(IN_W, IN_H, OUT_W, OUT_H)
        .filter(Filter::Lanczos)
        .format(PixelFormat::LinearF32(LAYOUT))
        .build();
    let input = make_f32_image(0.5, 0.3, 0.1, 1.0);
    let output = Resizer::new(&config).resize_f32(&input);
    for pixel in output.chunks_exact(CH) {
        assert!((pixel[0] - 0.5).abs() < 0.02, "R: {}", pixel[0]);
        assert!((pixel[1] - 0.3).abs() < 0.02, "G: {}", pixel[1]);
        assert!((pixel[2] - 0.1).abs() < 0.02, "B: {}", pixel[2]);
        assert!((pixel[3] - 1.0).abs() < 0.01, "A: {}", pixel[3]);
    }
}

#[test]
fn same_format_u16_encoded() {
    let config = ResizeConfig::builder(IN_W, IN_H, OUT_W, OUT_H)
        .filter(Filter::Lanczos)
        .format(PixelFormat::Encoded16(LAYOUT))
        .build();
    let input = make_u16_image(32768, 16384, 8192, 65535);
    let output = Resizer::new(&config).resize_u16(&input);
    for pixel in output.chunks_exact(CH) {
        assert!(
            (pixel[0] as i32 - 32768).unsigned_abs() <= 100,
            "R: {}",
            pixel[0]
        );
        assert!(
            (pixel[1] as i32 - 16384).unsigned_abs() <= 100,
            "G: {}",
            pixel[1]
        );
        assert!(
            (pixel[2] as i32 - 8192).unsigned_abs() <= 100,
            "B: {}",
            pixel[2]
        );
        assert!(
            (pixel[3] as i32 - 65535).unsigned_abs() <= 1,
            "A: {}",
            pixel[3]
        );
    }
}

// =============================================================================
// 2. Cross-format constant-color tests (all 6 cross-type pairs)
// =============================================================================

#[test]
fn cross_u8_to_f32() {
    let config = ResizeConfig::builder(IN_W, IN_H, OUT_W, OUT_H)
        .filter(Filter::Lanczos)
        .input_format(PixelFormat::Srgb8(LAYOUT))
        .output_format(PixelFormat::LinearF32(LAYOUT))
        .linear()
        .build();
    let input = make_u8_image(128, 64, 32, 255);
    let output = Resizer::new(&config).resize_u8_to_f32(&input);
    assert_eq!(output.len(), OUT_W as usize * OUT_H as usize * CH);
    // 128 sRGB ≈ 0.216 linear
    for pixel in output.chunks_exact(CH) {
        assert!(pixel[0] > 0.19 && pixel[0] < 0.24, "R: {}", pixel[0]);
        assert!((pixel[3] - 1.0).abs() < 0.01, "A: {}", pixel[3]);
    }
}

#[test]
fn cross_f32_to_u8() {
    let config = ResizeConfig::builder(IN_W, IN_H, OUT_W, OUT_H)
        .filter(Filter::Lanczos)
        .input_format(PixelFormat::LinearF32(LAYOUT))
        .output_format(PixelFormat::Srgb8(LAYOUT))
        .linear()
        .build();
    // 0.216 linear ≈ 128 sRGB, 0.0514 ≈ 64, 0.01444 ≈ 32
    let input = make_f32_image(0.2158605, 0.05126946, 0.01444384, 1.0);
    let output = Resizer::new(&config).resize_f32_to_u8(&input);
    assert_eq!(output.len(), OUT_W as usize * OUT_H as usize * CH);
    for pixel in output.chunks_exact(CH) {
        assert!(
            (pixel[0] as i16 - 128).unsigned_abs() <= 2,
            "R: {}",
            pixel[0]
        );
        assert!(
            (pixel[1] as i16 - 64).unsigned_abs() <= 2,
            "G: {}",
            pixel[1]
        );
        assert!(
            (pixel[2] as i16 - 32).unsigned_abs() <= 2,
            "B: {}",
            pixel[2]
        );
        assert_eq!(pixel[3], 255, "A: {}", pixel[3]);
    }
}

#[test]
fn cross_u8_to_u16() {
    let config = ResizeConfig::builder(IN_W, IN_H, OUT_W, OUT_H)
        .filter(Filter::Lanczos)
        .input_format(PixelFormat::Srgb8(LAYOUT))
        .output_format(PixelFormat::Encoded16(LAYOUT))
        .linear()
        .build();
    let input = make_u8_image(128, 128, 128, 255);
    let output = Resizer::new(&config).resize_u8_to_u16(&input);
    assert_eq!(output.len(), OUT_W as usize * OUT_H as usize * CH);
    // sRGB 128 ≈ Encoded16 32768 (both ~50% in their encoded space)
    for pixel in output.chunks_exact(CH) {
        assert!(
            (pixel[0] as i32 - 32768).unsigned_abs() <= 300,
            "R: {} (expected ~32768)",
            pixel[0]
        );
        assert!(
            (pixel[3] as i32 - 65535).unsigned_abs() <= 1,
            "A: {}",
            pixel[3]
        );
    }
}

#[test]
fn cross_u16_to_u8() {
    let config = ResizeConfig::builder(IN_W, IN_H, OUT_W, OUT_H)
        .filter(Filter::Lanczos)
        .input_format(PixelFormat::Encoded16(LAYOUT))
        .output_format(PixelFormat::Srgb8(LAYOUT))
        .linear()
        .build();
    let input = make_u16_image(32768, 32768, 32768, 65535);
    let output = Resizer::new(&config).resize_u16_to_u8(&input);
    assert_eq!(output.len(), OUT_W as usize * OUT_H as usize * CH);
    // Encoded16 32768 ≈ sRGB 128
    for pixel in output.chunks_exact(CH) {
        assert!(
            (pixel[0] as i16 - 128).unsigned_abs() <= 2,
            "R: {} (expected ~128)",
            pixel[0]
        );
        assert_eq!(pixel[3], 255, "A: {}", pixel[3]);
    }
}

#[test]
fn cross_u16_to_f32() {
    let config = ResizeConfig::builder(IN_W, IN_H, OUT_W, OUT_H)
        .filter(Filter::Lanczos)
        .input_format(PixelFormat::Encoded16(LAYOUT))
        .output_format(PixelFormat::LinearF32(LAYOUT))
        .build();
    let input = make_u16_image(32768, 0, 65535, 65535);
    let output = Resizer::new(&config).resize_u16_to_f32(&input);
    assert_eq!(output.len(), OUT_W as usize * OUT_H as usize * CH);
    // 32768/65535 ≈ 0.5 encoded → ~0.214 linear (sRGB)
    for pixel in output.chunks_exact(CH) {
        assert!(pixel[0] > 0.2 && pixel[0] < 0.25, "R: {}", pixel[0]);
        assert!(pixel[1].abs() < 0.01, "G: {}", pixel[1]);
        assert!((pixel[2] - 1.0).abs() < 0.02, "B: {}", pixel[2]);
        assert!((pixel[3] - 1.0).abs() < 0.01, "A: {}", pixel[3]);
    }
}

#[test]
fn cross_f32_to_u16() {
    let config = ResizeConfig::builder(IN_W, IN_H, OUT_W, OUT_H)
        .filter(Filter::Lanczos)
        .input_format(PixelFormat::LinearF32(LAYOUT))
        .output_format(PixelFormat::Encoded16(LAYOUT))
        .build();
    // 0.5 linear → ~0.735 sRGB → ~48163 u16
    let input = make_f32_image(0.5, 0.0, 1.0, 1.0);
    let output = Resizer::new(&config).resize_f32_to_u16(&input);
    assert_eq!(output.len(), OUT_W as usize * OUT_H as usize * CH);
    for pixel in output.chunks_exact(CH) {
        assert!(
            pixel[0] > 45000 && pixel[0] < 50000,
            "R: {} (expected ~48163)",
            pixel[0]
        );
        assert_eq!(pixel[2], 65535, "B: {} (expected 65535)", pixel[2]);
        assert!(
            (pixel[3] as i32 - 65535).unsigned_abs() <= 1,
            "A: {}",
            pixel[3]
        );
    }
}

// =============================================================================
// 3. Roundtrip: u8 → resize → f32 → resize_back → u8 ≈ direct u8 → resize → u8
// =============================================================================

#[test]
fn roundtrip_u8_f32_u8() {
    // Direct u8 → u8
    let direct_config = ResizeConfig::builder(IN_W, IN_H, OUT_W, OUT_H)
        .filter(Filter::Lanczos)
        .format(PixelFormat::Srgb8(LAYOUT))
        .linear()
        .build();
    let input = make_u8_image(200, 100, 50, 255);
    let direct_output = Resizer::new(&direct_config).resize(&input);

    // u8 → f32
    let to_f32_config = ResizeConfig::builder(IN_W, IN_H, OUT_W, OUT_H)
        .filter(Filter::Lanczos)
        .input_format(PixelFormat::Srgb8(LAYOUT))
        .output_format(PixelFormat::LinearF32(LAYOUT))
        .linear()
        .build();
    let f32_output = Resizer::new(&to_f32_config).resize_u8_to_f32(&input);

    // f32 → u8 (identity resize, same dimensions)
    let to_u8_config = ResizeConfig::builder(OUT_W, OUT_H, OUT_W, OUT_H)
        .filter(Filter::Lanczos)
        .input_format(PixelFormat::LinearF32(LAYOUT))
        .output_format(PixelFormat::Srgb8(LAYOUT))
        .linear()
        .build();
    let roundtrip_output = Resizer::new(&to_u8_config).resize_f32_to_u8(&f32_output);

    assert_eq!(direct_output.len(), roundtrip_output.len());
    for (i, (&a, &b)) in direct_output
        .iter()
        .zip(roundtrip_output.iter())
        .enumerate()
    {
        assert!(
            (a as i16 - b as i16).unsigned_abs() <= 2,
            "roundtrip mismatch at byte {}: direct={}, roundtrip={}",
            i,
            a,
            b
        );
    }
}

#[test]
fn roundtrip_u8_u16_u8() {
    // Direct u8 → u8
    let direct_config = ResizeConfig::builder(IN_W, IN_H, OUT_W, OUT_H)
        .filter(Filter::Lanczos)
        .format(PixelFormat::Srgb8(LAYOUT))
        .linear()
        .build();
    let input = make_u8_image(200, 100, 50, 255);
    let direct_output = Resizer::new(&direct_config).resize(&input);

    // u8 → u16
    let to_u16_config = ResizeConfig::builder(IN_W, IN_H, OUT_W, OUT_H)
        .filter(Filter::Lanczos)
        .input_format(PixelFormat::Srgb8(LAYOUT))
        .output_format(PixelFormat::Encoded16(LAYOUT))
        .linear()
        .build();
    let u16_output = Resizer::new(&to_u16_config).resize_u8_to_u16(&input);

    // u16 → u8 (identity resize)
    let to_u8_config = ResizeConfig::builder(OUT_W, OUT_H, OUT_W, OUT_H)
        .filter(Filter::Lanczos)
        .input_format(PixelFormat::Encoded16(LAYOUT))
        .output_format(PixelFormat::Srgb8(LAYOUT))
        .linear()
        .build();
    let roundtrip_output = Resizer::new(&to_u8_config).resize_u16_to_u8(&u16_output);

    assert_eq!(direct_output.len(), roundtrip_output.len());
    for (i, (&a, &b)) in direct_output
        .iter()
        .zip(roundtrip_output.iter())
        .enumerate()
    {
        assert!(
            (a as i16 - b as i16).unsigned_abs() <= 2,
            "roundtrip mismatch at byte {}: direct={}, roundtrip={}",
            i,
            a,
            b
        );
    }
}

// =============================================================================
// 4. Transfer function independence
// =============================================================================

#[test]
fn tf_srgb_decode_linear_output() {
    // sRGB u8 → linear f32: sRGB decode, no output TF needed (f32 is always identity)
    let config = ResizeConfig::builder(IN_W, IN_H, OUT_W, OUT_H)
        .filter(Filter::Lanczos)
        .input_format(PixelFormat::Srgb8(LAYOUT))
        .output_format(PixelFormat::LinearF32(LAYOUT))
        .input_transfer(Transfer::Srgb)
        .output_transfer(Transfer::None)
        .build();
    let input = make_u8_image(128, 128, 128, 255);
    let output = Resizer::new(&config).resize_u8_to_f32(&input);

    // 128 sRGB → ~0.216 linear
    for pixel in output.chunks_exact(CH) {
        assert!(pixel[0] > 0.19 && pixel[0] < 0.24, "R: {}", pixel[0]);
    }
}

#[test]
fn tf_linear_input_srgb_encode() {
    // linear f32 → sRGB u8: no input TF, sRGB encode
    let config = ResizeConfig::builder(IN_W, IN_H, OUT_W, OUT_H)
        .filter(Filter::Lanczos)
        .input_format(PixelFormat::LinearF32(LAYOUT))
        .output_format(PixelFormat::Srgb8(LAYOUT))
        .input_transfer(Transfer::None)
        .output_transfer(Transfer::Srgb)
        .build();
    // 0.216 linear → ~128 sRGB
    let input = make_f32_image(0.2158605, 0.2158605, 0.2158605, 1.0);
    let output = Resizer::new(&config).resize_f32_to_u8(&input);

    for pixel in output.chunks_exact(CH) {
        assert!(
            (pixel[0] as i16 - 128).unsigned_abs() <= 2,
            "R: {} (expected ~128)",
            pixel[0]
        );
    }
}

#[test]
fn tf_none_decode_srgb_encode_shifts_brightness() {
    // u8 with Transfer::None input → u8 with Transfer::Srgb output
    // This means: u8 values are treated as already-linear (just /255),
    // then sRGB-encoded on output. Should produce visibly different results
    // from a normal sRGB→sRGB resize.
    let config_none_srgb = ResizeConfig::builder(IN_W, IN_H, OUT_W, OUT_H)
        .filter(Filter::Lanczos)
        .format(PixelFormat::Srgb8(LAYOUT))
        .input_transfer(Transfer::None)
        .output_transfer(Transfer::Srgb)
        .build();

    let config_srgb_srgb = ResizeConfig::builder(IN_W, IN_H, OUT_W, OUT_H)
        .filter(Filter::Lanczos)
        .format(PixelFormat::Srgb8(LAYOUT))
        .input_transfer(Transfer::Srgb)
        .output_transfer(Transfer::Srgb)
        .build();

    let input = make_u8_image(128, 128, 128, 255);
    let output_none_srgb = Resizer::new(&config_none_srgb).resize(&input);
    let output_srgb_srgb = Resizer::new(&config_srgb_srgb).resize(&input);

    // They should produce DIFFERENT results because the input decode is different
    assert_ne!(
        output_none_srgb, output_srgb_srgb,
        "different input TFs should produce different output"
    );
}

#[test]
fn tf_u16_srgb_to_u8_srgb() {
    // Encoded16(sRGB) → Srgb8(sRGB): just precision loss
    let config = ResizeConfig::builder(IN_W, IN_H, OUT_W, OUT_H)
        .filter(Filter::Lanczos)
        .input_format(PixelFormat::Encoded16(LAYOUT))
        .output_format(PixelFormat::Srgb8(LAYOUT))
        .input_transfer(Transfer::Srgb)
        .output_transfer(Transfer::Srgb)
        .build();
    // u16 32768 ≈ sRGB 128
    let input = make_u16_image(32768, 32768, 32768, 65535);
    let output = Resizer::new(&config).resize_u16_to_u8(&input);

    for pixel in output.chunks_exact(CH) {
        assert!(
            (pixel[0] as i16 - 128).unsigned_abs() <= 2,
            "R: {} (expected ~128)",
            pixel[0]
        );
        assert_eq!(pixel[3], 255, "A: {}", pixel[3]);
    }
}

#[test]
fn tf_u16_none_to_u8_none() {
    // Encoded16 with Transfer::None → Srgb8 with Transfer::None
    // Both sides are identity: u16 v/65535 → linear f32 → u8 clamp+round
    let config = ResizeConfig::builder(IN_W, IN_H, OUT_W, OUT_H)
        .filter(Filter::Lanczos)
        .input_format(PixelFormat::Encoded16(LAYOUT))
        .output_format(PixelFormat::Srgb8(LAYOUT))
        .input_transfer(Transfer::None)
        .output_transfer(Transfer::None)
        .build();
    // u16 32768 ≈ 0.5 → u8 128 (identity both ways)
    let input = make_u16_image(32768, 16384, 65535, 65535);
    let output = Resizer::new(&config).resize_u16_to_u8(&input);

    for pixel in output.chunks_exact(CH) {
        assert!(
            (pixel[0] as i16 - 128).unsigned_abs() <= 2,
            "R: {} (expected ~128 from identity)",
            pixel[0]
        );
        assert!(
            (pixel[1] as i16 - 64).unsigned_abs() <= 2,
            "G: {} (expected ~64 from identity)",
            pixel[1]
        );
        assert_eq!(pixel[3], 255, "A: {}", pixel[3]);
    }
}

// =============================================================================
// 5. Streaming matches fullframe for cross-format
// =============================================================================

/// Helper: run streaming resize for u8→f32 and compare to fullframe.
#[test]
fn streaming_matches_fullframe_u8_to_f32() {
    let config = ResizeConfig::builder(IN_W, IN_H, OUT_W, OUT_H)
        .filter(Filter::Lanczos)
        .input_format(PixelFormat::Srgb8(LAYOUT))
        .output_format(PixelFormat::LinearF32(LAYOUT))
        .linear()
        .build();

    let input = make_u8_image(180, 90, 45, 255);

    // Fullframe
    let fullframe = Resizer::new(&config).resize_u8_to_f32(&input);

    // Streaming: push u8 rows, pull f32 rows
    let mut resizer = StreamingResize::new(&config);
    let mut streaming = Vec::new();
    let row_len = IN_W as usize * CH;
    for y in 0..IN_H as usize {
        let start = y * row_len;
        resizer.push_row(&input[start..start + row_len]).unwrap();
        while let Some(row) = resizer.next_output_row_f32() {
            streaming.extend_from_slice(row);
        }
    }
    resizer.finish();
    while let Some(row) = resizer.next_output_row_f32() {
        streaming.extend_from_slice(row);
    }

    assert_eq!(fullframe.len(), streaming.len());
    for (i, (&a, &b)) in fullframe.iter().zip(streaming.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-4,
            "mismatch at element {}: fullframe={}, streaming={}",
            i,
            a,
            b
        );
    }
}

/// Helper: run streaming resize for u16→u8 and compare to fullframe.
#[test]
fn streaming_matches_fullframe_u16_to_u8() {
    let config = ResizeConfig::builder(IN_W, IN_H, OUT_W, OUT_H)
        .filter(Filter::Lanczos)
        .input_format(PixelFormat::Encoded16(LAYOUT))
        .output_format(PixelFormat::Srgb8(LAYOUT))
        .build();

    let input = make_u16_image(40000, 20000, 10000, 65535);

    // Fullframe
    let fullframe = Resizer::new(&config).resize_u16_to_u8(&input);

    // Streaming: push u16 rows, pull u8 rows
    let mut resizer = StreamingResize::new(&config);
    let mut streaming = Vec::new();
    let row_len = IN_W as usize * CH;
    for y in 0..IN_H as usize {
        let start = y * row_len;
        resizer
            .push_row_u16(&input[start..start + row_len])
            .unwrap();
        while let Some(row) = resizer.next_output_row() {
            streaming.extend_from_slice(row);
        }
    }
    resizer.finish();
    while let Some(row) = resizer.next_output_row() {
        streaming.extend_from_slice(row);
    }

    assert_eq!(fullframe.len(), streaming.len());
    for (i, (&a, &b)) in fullframe.iter().zip(streaming.iter()).enumerate() {
        assert!(
            (a as i16 - b as i16).unsigned_abs() <= 2,
            "mismatch at byte {}: fullframe={}, streaming={}",
            i,
            a,
            b
        );
    }
}

// =============================================================================
// 6. All 9 format pairs smoke test (constant color, verify dimensions)
// =============================================================================

#[test]
fn all_9_pairs_smoke_test() {
    let pairs: &[(PixelFormat, PixelFormat, &str)] = &[
        (
            PixelFormat::Srgb8(LAYOUT),
            PixelFormat::Srgb8(LAYOUT),
            "u8→u8",
        ),
        (
            PixelFormat::Srgb8(LAYOUT),
            PixelFormat::LinearF32(LAYOUT),
            "u8→f32",
        ),
        (
            PixelFormat::Srgb8(LAYOUT),
            PixelFormat::Encoded16(LAYOUT),
            "u8→u16",
        ),
        (
            PixelFormat::LinearF32(LAYOUT),
            PixelFormat::Srgb8(LAYOUT),
            "f32→u8",
        ),
        (
            PixelFormat::LinearF32(LAYOUT),
            PixelFormat::LinearF32(LAYOUT),
            "f32→f32",
        ),
        (
            PixelFormat::LinearF32(LAYOUT),
            PixelFormat::Encoded16(LAYOUT),
            "f32→u16",
        ),
        (
            PixelFormat::Encoded16(LAYOUT),
            PixelFormat::Srgb8(LAYOUT),
            "u16→u8",
        ),
        (
            PixelFormat::Encoded16(LAYOUT),
            PixelFormat::LinearF32(LAYOUT),
            "u16→f32",
        ),
        (
            PixelFormat::Encoded16(LAYOUT),
            PixelFormat::Encoded16(LAYOUT),
            "u16→u16",
        ),
    ];

    for (in_fmt, out_fmt, label) in pairs {
        let config = ResizeConfig::builder(IN_W, IN_H, OUT_W, OUT_H)
            .filter(Filter::Lanczos)
            .input_format(*in_fmt)
            .output_format(*out_fmt)
            .linear()
            .build();
        let mut resizer = Resizer::new(&config);
        let out_len = OUT_W as usize * OUT_H as usize * CH;

        match (in_fmt.is_u8(), in_fmt.is_f32(), in_fmt.is_u16()) {
            (true, _, _) => {
                let input = make_u8_image(128, 128, 128, 255);
                if out_fmt.is_u8() {
                    let output = resizer.resize(&input);
                    assert_eq!(output.len(), out_len, "{label}: wrong u8 output length");
                } else if out_fmt.is_f32() {
                    let output = resizer.resize_u8_to_f32(&input);
                    assert_eq!(output.len(), out_len, "{label}: wrong f32 output length");
                } else {
                    let output = resizer.resize_u8_to_u16(&input);
                    assert_eq!(output.len(), out_len, "{label}: wrong u16 output length");
                }
            }
            (_, true, _) => {
                let input = make_f32_image(0.5, 0.5, 0.5, 1.0);
                if out_fmt.is_u8() {
                    let output = resizer.resize_f32_to_u8(&input);
                    assert_eq!(output.len(), out_len, "{label}: wrong u8 output length");
                } else if out_fmt.is_f32() {
                    let output = resizer.resize_f32(&input);
                    assert_eq!(output.len(), out_len, "{label}: wrong f32 output length");
                } else {
                    let output = resizer.resize_f32_to_u16(&input);
                    assert_eq!(output.len(), out_len, "{label}: wrong u16 output length");
                }
            }
            (_, _, true) => {
                let input = make_u16_image(32768, 32768, 32768, 65535);
                if out_fmt.is_u8() {
                    let output = resizer.resize_u16_to_u8(&input);
                    assert_eq!(output.len(), out_len, "{label}: wrong u8 output length");
                } else if out_fmt.is_f32() {
                    let output = resizer.resize_u16_to_f32(&input);
                    assert_eq!(output.len(), out_len, "{label}: wrong f32 output length");
                } else {
                    let output = resizer.resize_u16(&input);
                    assert_eq!(output.len(), out_len, "{label}: wrong u16 output length");
                }
            }
            _ => unreachable!(),
        }
    }
}

// =============================================================================
// 7. Transfer function matrix (Srgb × None on input/output)
// =============================================================================

#[test]
fn transfer_matrix_u8_to_u8() {
    let tfs = [(Transfer::Srgb, "Srgb"), (Transfer::None, "None")];

    for (in_tf, in_name) in &tfs {
        for (out_tf, out_name) in &tfs {
            let config = ResizeConfig::builder(IN_W, IN_H, OUT_W, OUT_H)
                .filter(Filter::Lanczos)
                .format(PixelFormat::Srgb8(LAYOUT))
                .input_transfer(*in_tf)
                .output_transfer(*out_tf)
                .build();
            let input = make_u8_image(128, 128, 128, 255);
            let output = Resizer::new(&config).resize(&input);
            assert_eq!(
                output.len(),
                OUT_W as usize * OUT_H as usize * CH,
                "TF {in_name}→{out_name}: wrong output length"
            );
            // All output pixels should be consistent (constant color in → constant color out)
            let first = &output[..CH];
            for (px_idx, pixel) in output.chunks_exact(CH).enumerate() {
                for c in 0..CH {
                    assert!(
                        (pixel[c] as i16 - first[c] as i16).unsigned_abs() <= 2,
                        "TF {in_name}→{out_name}: inconsistent pixel at {px_idx} ch {c}: {} vs {}",
                        pixel[c],
                        first[c]
                    );
                }
            }
        }
    }
}

#[test]
fn transfer_matrix_u16_to_u16() {
    let tfs = [(Transfer::Srgb, "Srgb"), (Transfer::None, "None")];

    for (in_tf, in_name) in &tfs {
        for (out_tf, out_name) in &tfs {
            let config = ResizeConfig::builder(IN_W, IN_H, OUT_W, OUT_H)
                .filter(Filter::Lanczos)
                .format(PixelFormat::Encoded16(LAYOUT))
                .input_transfer(*in_tf)
                .output_transfer(*out_tf)
                .build();
            let input = make_u16_image(32768, 32768, 32768, 65535);
            let output = Resizer::new(&config).resize_u16(&input);
            assert_eq!(
                output.len(),
                OUT_W as usize * OUT_H as usize * CH,
                "TF {in_name}→{out_name}: wrong output length"
            );
            // Verify consistency
            let first = &output[..CH];
            for (px_idx, pixel) in output.chunks_exact(CH).enumerate() {
                for c in 0..CH {
                    assert!(
                        (pixel[c] as i32 - first[c] as i32).unsigned_abs() <= 100,
                        "TF {in_name}→{out_name}: inconsistent pixel at {px_idx} ch {c}: {} vs {}",
                        pixel[c],
                        first[c]
                    );
                }
            }
        }
    }
}

// =============================================================================
// 8. _into variants match allocating variants
// =============================================================================

#[test]
fn into_matches_alloc_u8_to_f32() {
    let config = ResizeConfig::builder(IN_W, IN_H, OUT_W, OUT_H)
        .filter(Filter::Lanczos)
        .input_format(PixelFormat::Srgb8(LAYOUT))
        .output_format(PixelFormat::LinearF32(LAYOUT))
        .linear()
        .build();
    let input = make_u8_image(128, 64, 32, 255);

    let alloc_output = Resizer::new(&config).resize_u8_to_f32(&input);
    let mut into_output = vec![0.0f32; alloc_output.len()];
    Resizer::new(&config).resize_u8_to_f32_into(&input, &mut into_output);

    assert_eq!(alloc_output, into_output);
}

#[test]
fn into_matches_alloc_f32_to_u8() {
    let config = ResizeConfig::builder(IN_W, IN_H, OUT_W, OUT_H)
        .filter(Filter::Lanczos)
        .input_format(PixelFormat::LinearF32(LAYOUT))
        .output_format(PixelFormat::Srgb8(LAYOUT))
        .linear()
        .build();
    let input = make_f32_image(0.5, 0.3, 0.1, 1.0);

    let alloc_output = Resizer::new(&config).resize_f32_to_u8(&input);
    let mut into_output = vec![0u8; alloc_output.len()];
    Resizer::new(&config).resize_f32_to_u8_into(&input, &mut into_output);

    assert_eq!(alloc_output, into_output);
}
