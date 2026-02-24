//! Golden output tests: capture byte-exact resize outputs for all paths.
//!
//! Run once to generate baselines in test_outputs/, then verify they
//! don't change after refactoring. Each test saves raw bytes and a
//! CRC32-like checksum.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use zenresize::{Filter, PixelFormat, PixelLayout, ResizeConfig, Resizer, StreamingResize};

fn hash_bytes(data: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    data.hash(&mut hasher);
    hasher.finish()
}

fn hash_f32(data: &[f32]) -> u64 {
    let mut hasher = DefaultHasher::new();
    for v in data {
        v.to_bits().hash(&mut hasher);
    }
    hasher.finish()
}

/// Create a deterministic RGBA test image.
fn make_rgba_gradient(w: usize, h: usize) -> Vec<u8> {
    let mut img = vec![0u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 4;
            img[idx] = ((x * 255) / w.max(1)) as u8;
            img[idx + 1] = ((y * 255) / h.max(1)) as u8;
            img[idx + 2] = (((x + y) * 128) / (w + h).max(1)) as u8;
            img[idx + 3] = 255; // opaque
        }
    }
    img
}

/// Create a deterministic RGBA test image with varying alpha.
fn make_rgba_alpha_gradient(w: usize, h: usize) -> Vec<u8> {
    let mut img = vec![0u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 4;
            img[idx] = ((x * 200) / w.max(1)) as u8;
            img[idx + 1] = ((y * 150) / h.max(1)) as u8;
            img[idx + 2] = 100;
            img[idx + 3] = ((x * 255) / w.max(1)) as u8; // alpha varies
        }
    }
    img
}

/// Create a deterministic Rgbx test image.
fn make_rgbx_gradient(w: usize, h: usize) -> Vec<u8> {
    let mut img = vec![0u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 4;
            img[idx] = ((x * 255) / w.max(1)) as u8;
            img[idx + 1] = ((y * 255) / h.max(1)) as u8;
            img[idx + 2] = (((x + y) * 128) / (w + h).max(1)) as u8;
            img[idx + 3] = 0xFF; // padding
        }
    }
    img
}

/// Create a deterministic linear f32 RGBA test image.
fn make_f32_gradient(w: usize, h: usize) -> Vec<f32> {
    let mut img = vec![0.0f32; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 4;
            img[idx] = x as f32 / w.max(1) as f32;
            img[idx + 1] = y as f32 / h.max(1) as f32;
            img[idx + 2] = (x + y) as f32 / (w + h).max(1) as f32 * 0.5;
            img[idx + 3] = 1.0;
        }
    }
    img
}

/// Create a deterministic Gray test image.
fn make_gray_gradient(w: usize, h: usize) -> Vec<u8> {
    let mut img = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            img[y * w + x] = ((x * 255) / w.max(1)) as u8;
        }
    }
    img
}

/// Create a deterministic RGB test image.
fn make_rgb_gradient(w: usize, h: usize) -> Vec<u8> {
    let mut img = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            img[idx] = ((x * 255) / w.max(1)) as u8;
            img[idx + 1] = ((y * 255) / h.max(1)) as u8;
            img[idx + 2] = 128;
        }
    }
    img
}

/// Create a deterministic u16 RGBA test image.
fn make_u16_rgba_gradient(w: usize, h: usize) -> Vec<u16> {
    let mut img = vec![0u16; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 4;
            img[idx] = ((x * 65535) / w.max(1)) as u16;
            img[idx + 1] = ((y * 65535) / h.max(1)) as u16;
            img[idx + 2] = (((x + y) * 32768) / (w + h).max(1)) as u16;
            img[idx + 3] = 65535; // opaque
        }
    }
    img
}

/// Create a deterministic u16 RGB test image.
fn make_u16_rgb_gradient(w: usize, h: usize) -> Vec<u16> {
    let mut img = vec![0u16; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            img[idx] = ((x * 65535) / w.max(1)) as u16;
            img[idx + 1] = ((y * 65535) / h.max(1)) as u16;
            img[idx + 2] = 32768;
        }
    }
    img
}

/// Create a deterministic u16 Gray test image.
fn make_u16_gray_gradient(w: usize, h: usize) -> Vec<u16> {
    let mut img = vec![0u16; w * h];
    for y in 0..h {
        for x in 0..w {
            img[y * w + x] = ((x * 65535) / w.max(1)) as u16;
        }
    }
    img
}

fn save_golden(name: &str, data: &[u8]) {
    let dir = std::path::Path::new("test_outputs");
    std::fs::create_dir_all(dir).unwrap();
    let path = dir.join(format!("{}.raw", name));
    std::fs::write(&path, data).unwrap();
}

fn save_golden_f32(name: &str, data: &[f32]) {
    let dir = std::path::Path::new("test_outputs");
    std::fs::create_dir_all(dir).unwrap();
    let path = dir.join(format!("{}.raw", name));
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&path, bytes).unwrap();
}

fn check_golden(name: &str, data: &[u8]) {
    let dir = std::path::Path::new("test_outputs");
    let path = dir.join(format!("{}.raw", name));
    if path.exists() {
        let existing = std::fs::read(&path).unwrap();
        assert_eq!(
            data,
            existing.as_slice(),
            "Golden output mismatch for {}! hash_new={}, hash_old={}",
            name,
            hash_bytes(data),
            hash_bytes(&existing),
        );
    } else {
        save_golden(name, data);
    }
}

fn check_golden_f32(name: &str, data: &[f32]) {
    let dir = std::path::Path::new("test_outputs");
    let path = dir.join(format!("{}.raw", name));
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    if path.exists() {
        let existing = std::fs::read(&path).unwrap();
        assert_eq!(
            bytes,
            existing.as_slice(),
            "Golden f32 output mismatch for {}! hash_new={}, hash_old={}",
            name,
            hash_f32(data),
            hash_bytes(&existing),
        );
    } else {
        save_golden_f32(name, data);
    }
}

// ============================================================================
// Path 0: sRGB i16 fast path (no linearization, 4ch, no premul)
// Config: Srgb8(Rgba), srgb mode → 4ch opaque triggers batch path
// ============================================================================

#[test]
fn golden_path0_srgb_i16_downscale() {
    let config = ResizeConfig::builder(100, 100, 50, 50)
        .filter(Filter::Lanczos)
        .format(PixelFormat::Srgb8(PixelLayout::Rgba))
        .srgb()
        .build();
    let input = make_rgba_gradient(100, 100);
    let output = Resizer::new(&config).resize(&input);
    check_golden("path0_srgb_i16_down_100x100_50x50", &output);
}

#[test]
fn golden_path0_srgb_i16_upscale() {
    let config = ResizeConfig::builder(50, 50, 100, 100)
        .filter(Filter::Lanczos)
        .format(PixelFormat::Srgb8(PixelLayout::Rgba))
        .srgb()
        .build();
    let input = make_rgba_gradient(50, 50);
    let output = Resizer::new(&config).resize(&input);
    check_golden("path0_srgb_i16_up_50x50_100x100", &output);
}

#[test]
fn golden_path0_srgb_i16_identity() {
    let config = ResizeConfig::builder(32, 32, 32, 32)
        .filter(Filter::Lanczos)
        .format(PixelFormat::Srgb8(PixelLayout::Rgba))
        .srgb()
        .build();
    let input = make_rgba_gradient(32, 32);
    let output = Resizer::new(&config).resize(&input);
    check_golden("path0_srgb_i16_identity_32x32", &output);
}

#[test]
fn golden_path0_with_alpha() {
    // Rgba with varying alpha in sRGB mode — needs premul so may take different subpath
    let config = ResizeConfig::builder(80, 60, 40, 30)
        .filter(Filter::Robidoux)
        .format(PixelFormat::Srgb8(PixelLayout::Rgba))
        .srgb()
        .build();
    let input = make_rgba_alpha_gradient(80, 60);
    let output = Resizer::new(&config).resize(&input);
    check_golden("path0_alpha_down_80x60_40x30", &output);
}

// ============================================================================
// Path 1: linear-light i16 (Rgbx, linear mode, 4ch, no premul)
// ============================================================================

#[test]
fn golden_path1_linear_i16_downscale() {
    let config = ResizeConfig::builder(100, 100, 50, 50)
        .filter(Filter::Lanczos)
        .format(PixelFormat::Srgb8(PixelLayout::Rgbx))
        .linear()
        .build();
    let input = make_rgbx_gradient(100, 100);
    let output = Resizer::new(&config).resize(&input);
    check_golden("path1_linear_i16_down_100x100_50x50", &output);
}

#[test]
fn golden_path1_linear_i16_upscale() {
    let config = ResizeConfig::builder(40, 40, 80, 80)
        .filter(Filter::Lanczos)
        .format(PixelFormat::Srgb8(PixelLayout::Rgbx))
        .linear()
        .build();
    let input = make_rgbx_gradient(40, 40);
    let output = Resizer::new(&config).resize(&input);
    check_golden("path1_linear_i16_up_40x40_80x80", &output);
}

// ============================================================================
// Path 2: f32 with u8 I/O (Rgba, linear mode — premul forces f32)
// ============================================================================

#[test]
fn golden_path2_f32_u8io_downscale() {
    let config = ResizeConfig::builder(100, 100, 50, 50)
        .filter(Filter::Lanczos)
        .format(PixelFormat::Srgb8(PixelLayout::Rgba))
        .linear()
        .build();
    let input = make_rgba_gradient(100, 100);
    let output = Resizer::new(&config).resize(&input);
    check_golden("path2_f32_u8io_down_100x100_50x50", &output);
}

#[test]
fn golden_path2_f32_u8io_upscale() {
    let config = ResizeConfig::builder(50, 50, 100, 100)
        .filter(Filter::Lanczos)
        .format(PixelFormat::Srgb8(PixelLayout::Rgba))
        .linear()
        .build();
    let input = make_rgba_gradient(50, 50);
    let output = Resizer::new(&config).resize(&input);
    check_golden("path2_f32_u8io_up_50x50_100x100", &output);
}

#[test]
fn golden_path2_with_alpha() {
    let config = ResizeConfig::builder(80, 60, 40, 30)
        .filter(Filter::Robidoux)
        .format(PixelFormat::Srgb8(PixelLayout::Rgba))
        .linear()
        .build();
    let input = make_rgba_alpha_gradient(80, 60);
    let output = Resizer::new(&config).resize(&input);
    check_golden("path2_alpha_down_80x60_40x30", &output);
}

// ============================================================================
// Path 2 variants: Gray (1ch) and RGB (3ch) — also use f32 path
// ============================================================================

#[test]
fn golden_gray_downscale() {
    let config = ResizeConfig::builder(100, 100, 50, 50)
        .filter(Filter::Lanczos)
        .format(PixelFormat::Srgb8(PixelLayout::Gray))
        .linear()
        .build();
    let input = make_gray_gradient(100, 100);
    let output = Resizer::new(&config).resize(&input);
    check_golden("gray_down_100x100_50x50", &output);
}

#[test]
fn golden_rgb_downscale() {
    let config = ResizeConfig::builder(100, 100, 50, 50)
        .filter(Filter::Lanczos)
        .format(PixelFormat::Srgb8(PixelLayout::Rgb))
        .linear()
        .build();
    let input = make_rgb_gradient(100, 100);
    let output = Resizer::new(&config).resize(&input);
    check_golden("rgb_down_100x100_50x50", &output);
}

// ============================================================================
// Path 3: f32 native (LinearF32 input)
// ============================================================================

#[test]
fn golden_path3_f32_native_downscale() {
    let config = ResizeConfig::builder(100, 100, 50, 50)
        .filter(Filter::Lanczos)
        .format(PixelFormat::LinearF32(PixelLayout::Rgba))
        .build();
    let input = make_f32_gradient(100, 100);
    let output = Resizer::new(&config).resize_f32(&input);
    check_golden_f32("path3_f32_native_down_100x100_50x50", &output);
}

#[test]
fn golden_path3_f32_native_upscale() {
    let config = ResizeConfig::builder(40, 40, 80, 80)
        .filter(Filter::Lanczos)
        .format(PixelFormat::LinearF32(PixelLayout::Rgba))
        .build();
    let input = make_f32_gradient(40, 40);
    let output = Resizer::new(&config).resize_f32(&input);
    check_golden_f32("path3_f32_native_up_40x40_80x80", &output);
}

// ============================================================================
// Streaming vs full-frame: must match exactly
// ============================================================================

fn streaming_resize_u8(config: &ResizeConfig, input: &[u8]) -> Vec<u8> {
    let in_w = config.in_width as usize;
    let channels = config.input_format.channels() as usize;
    let row_len = in_w * channels;
    let in_h = config.in_height as usize;

    let mut streamer = StreamingResize::new(config);
    let mut output = Vec::new();
    for y in 0..in_h {
        let start = y * row_len;
        streamer.push_row(&input[start..start + row_len]).unwrap();
        while let Some(row) = streamer.next_output_row() {
            output.extend_from_slice(row);
        }
    }
    streamer.finish();
    while let Some(row) = streamer.next_output_row() {
        output.extend_from_slice(row);
    }
    output
}

#[test]
fn golden_streaming_matches_fullframe_path2() {
    let config = ResizeConfig::builder(100, 100, 50, 50)
        .filter(Filter::Lanczos)
        .format(PixelFormat::Srgb8(PixelLayout::Rgba))
        .linear()
        .build();
    let input = make_rgba_gradient(100, 100);

    let fullframe = Resizer::new(&config).resize(&input);
    let streaming = streaming_resize_u8(&config, &input);

    // Both paths use f16 intermediates but at different stages (H-first vs V-first),
    // so f16 quantization produces small differences (typically ≤1 u8 LSB).
    assert_eq!(fullframe.len(), streaming.len());
    for (i, (&a, &b)) in fullframe.iter().zip(streaming.iter()).enumerate() {
        assert!(
            (a as i16 - b as i16).unsigned_abs() <= 1,
            "mismatch at element {}: fullframe={}, streaming={}",
            i,
            a,
            b
        );
    }
}

// ============================================================================
// Various filters — ensure different filters produce deterministic output
// ============================================================================

#[test]
fn golden_filters() {
    let filters = [
        (Filter::Lanczos, "lanczos"),
        (Filter::Robidoux, "robidoux"),
        (Filter::RobidouxSharp, "robidoux_sharp"),
        (Filter::CatmullRom, "catmull_rom"),
        (Filter::Mitchell, "mitchell"),
        (Filter::Triangle, "triangle"),
    ];

    let input = make_rgba_gradient(80, 60);

    for (filter, name) in &filters {
        let config = ResizeConfig::builder(80, 60, 40, 30)
            .filter(*filter)
            .format(PixelFormat::Srgb8(PixelLayout::Rgba))
            .linear()
            .build();
        let output = Resizer::new(&config).resize(&input);
        check_golden(&format!("filter_{}_down_80x60_40x30", name), &output);
    }
}

// ============================================================================
// Path 4: u16 Encoded16 path (sRGB transfer, f32 working space)
// ============================================================================

#[test]
fn golden_u16_rgba_downscale() {
    let config = ResizeConfig::builder(100, 100, 50, 50)
        .filter(Filter::Lanczos)
        .format(PixelFormat::Encoded16(PixelLayout::Rgba))
        .build();
    let input = make_u16_rgba_gradient(100, 100);
    let output = Resizer::new(&config).resize_u16(&input);
    let bytes: Vec<u8> = output.iter().flat_map(|v| v.to_le_bytes()).collect();
    check_golden("u16_rgba_down_100x100_50x50", &bytes);
}

#[test]
fn golden_u16_rgba_upscale() {
    let config = ResizeConfig::builder(50, 50, 100, 100)
        .filter(Filter::Lanczos)
        .format(PixelFormat::Encoded16(PixelLayout::Rgba))
        .build();
    let input = make_u16_rgba_gradient(50, 50);
    let output = Resizer::new(&config).resize_u16(&input);
    let bytes: Vec<u8> = output.iter().flat_map(|v| v.to_le_bytes()).collect();
    check_golden("u16_rgba_up_50x50_100x100", &bytes);
}

#[test]
fn golden_u16_rgb_downscale() {
    let config = ResizeConfig::builder(100, 100, 50, 50)
        .filter(Filter::Lanczos)
        .format(PixelFormat::Encoded16(PixelLayout::Rgb))
        .build();
    let input = make_u16_rgb_gradient(100, 100);
    let output = Resizer::new(&config).resize_u16(&input);
    let bytes: Vec<u8> = output.iter().flat_map(|v| v.to_le_bytes()).collect();
    check_golden("u16_rgb_down_100x100_50x50", &bytes);
}

#[test]
fn golden_u16_gray_downscale() {
    let config = ResizeConfig::builder(100, 100, 50, 50)
        .filter(Filter::Lanczos)
        .format(PixelFormat::Encoded16(PixelLayout::Gray))
        .build();
    let input = make_u16_gray_gradient(100, 100);
    let output = Resizer::new(&config).resize_u16(&input);
    let bytes: Vec<u8> = output.iter().flat_map(|v| v.to_le_bytes()).collect();
    check_golden("u16_gray_down_100x100_50x50", &bytes);
}

// ============================================================================
// Non-square ratios
// ============================================================================

#[test]
fn golden_nonsquare() {
    let config = ResizeConfig::builder(200, 50, 40, 100)
        .filter(Filter::Lanczos)
        .format(PixelFormat::Srgb8(PixelLayout::Rgba))
        .linear()
        .build();
    let input = make_rgba_gradient(200, 50);
    let output = Resizer::new(&config).resize(&input);
    check_golden("nonsquare_200x50_to_40x100", &output);
}

// ============================================================================
// Summary: print all checksums
// ============================================================================

#[test]
fn golden_print_checksums() {
    let dir = std::path::Path::new("test_outputs");
    if !dir.exists() {
        return;
    }

    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "raw"))
        .collect();
    entries.sort_by_key(|e| e.file_name());

    println!("\n=== Golden Output Checksums ===");
    for entry in entries {
        let data = std::fs::read(entry.path()).unwrap();
        let hash = hash_bytes(&data);
        println!(
            "  {} : {:016x} ({} bytes)",
            entry.file_name().to_string_lossy(),
            hash,
            data.len()
        );
    }
    println!("===============================\n");
}
