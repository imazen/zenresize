# zenresize

High-quality image resampling with 31 filters, streaming API, and SIMD acceleration. Extracted from the imageflow image processing pipeline.

## Features

- 31 resampling filters (Lanczos, Mitchell, Robidoux, Ginseng, etc.)
- sRGB-aware linear-light processing for correct gamma handling
- Row-at-a-time streaming API for pipeline integration
- `Resizer` struct for amortizing weight computation across repeated resizes
- Alpha premultiply/unpremultiply built into the pipeline
- Channel-order-agnostic: RGBA, BGRA, ARGB, BGRX all work without swizzling
- u8 and f32 pixel format support
- `no_std` + `alloc` compatible (std optional)
- SIMD-accelerated via [archmage](https://github.com/imazen/archmage) (AVX2+FMA on x86, NEON on ARM, scalar fallback)

## Quick Start

```rust
use zenresize::{resize, ResizeConfig, Filter, PixelFormat};

let input = vec![128u8; 1024 * 768 * 4]; // RGBA pixels

let config = ResizeConfig::builder(1024, 768, 512, 384)
    .filter(Filter::Lanczos)
    .format(PixelFormat::Srgb8 { channels: 4, has_alpha: true })
    .linear() // resize in linear light (default, recommended)
    .build();

let output = resize(&config, &input);
assert_eq!(output.len(), 512 * 384 * 4);
```

## API Overview

### One-Shot Functions

For single resize operations where you don't need to reuse weight tables.

```rust
use zenresize::{resize, resize_into, resize_f32, resize_f32_into};
use zenresize::{ResizeConfig, Filter, PixelFormat};

let config = ResizeConfig::builder(100, 100, 50, 50)
    .filter(Filter::Lanczos)
    .format(PixelFormat::Srgb8 { channels: 4, has_alpha: true })
    .build();

// Allocating: returns a new Vec<u8>
let output = resize(&config, &input_u8);

// Non-allocating: writes into your buffer
let mut buf = vec![0u8; 50 * 50 * 4];
resize_into(&config, &input_u8, &mut buf);

// f32 variants for linear-light pipelines
let config_f32 = ResizeConfig::builder(100, 100, 50, 50)
    .format(PixelFormat::LinearF32 { channels: 4, has_alpha: true })
    .build();
let output_f32 = resize_f32(&config_f32, &input_f32);
```

### Resizer (Cached Weights)

When resizing many images with the same dimensions and filter, `Resizer` pre-computes weight tables once and reuses them.

```rust
use zenresize::Resizer;

let config = ResizeConfig::builder(1024, 1024, 512, 512)
    .filter(Filter::Lanczos)
    .format(PixelFormat::Srgb8 { channels: 4, has_alpha: true })
    .srgb()
    .build();

let mut resizer = Resizer::new(&config);

// Resize multiple images with the same dimensions
for frame in frames {
    let output = resizer.resize(&frame);
    // process output...
}
```

### StreamingResize

Push input rows one at a time, pull output rows as they become available. Useful for pipeline integration where the full image isn't in memory.

```rust
use zenresize::StreamingResize;

let config = ResizeConfig::builder(1000, 800, 500, 400)
    .filter(Filter::Lanczos)
    .format(PixelFormat::Srgb8 { channels: 4, has_alpha: true })
    .linear()
    .build();

let mut resizer = StreamingResize::new(&config);

// Push input rows
for y in 0..800 {
    let row = &input_data[y * 4000..(y + 1) * 4000];
    resizer.push_row(row);

    // Pull output rows as they become available
    while let Some(out_row) = resizer.next_output_row() {
        // process output row...
    }
}
resizer.finish();

// Drain remaining output rows
while let Some(out_row) = resizer.next_output_row() {
    // ...
}
```

Batch variants `push_rows()` and `push_rows_f32()` push multiple rows from a contiguous buffer with stride.

### ResizeConfig Builder

```rust
let config = ResizeConfig::builder(in_w, in_h, out_w, out_h)
    .filter(Filter::Lanczos)        // resampling filter (default: Robidoux)
    .format(pixel_format)            // sets both input and output format
    .input_format(fmt_in)            // or set them separately
    .output_format(fmt_out)
    .linear()                        // resize in linear light (default)
    .srgb()                          // resize in sRGB space (faster, slight quality loss)
    .sharpen(0.0)                    // sharpening amount (default: 0.0)
    .in_stride(stride)               // input row stride in elements (default: tightly packed)
    .out_stride(stride)              // output row stride in elements (default: tightly packed)
    .build();
```

### Pixel Formats

```rust
use zenresize::PixelFormat;

// sRGB gamma-encoded u8 (most common)
PixelFormat::Srgb8 { channels: 4, has_alpha: true }   // RGBA
PixelFormat::Srgb8 { channels: 4, has_alpha: false }   // RGBX (padding channel)
PixelFormat::Srgb8 { channels: 3, has_alpha: false }   // RGB
PixelFormat::Srgb8 { channels: 1, has_alpha: false }   // Grayscale

// Linear light f32 (for pipelines already in linear space)
PixelFormat::LinearF32 { channels: 4, has_alpha: true }

// sRGB f32 (for fast sRGB-space resize with f32 data)
PixelFormat::SrgbF32 { channels: 4, has_alpha: false }

// Linear light u8 (rare, for pre-linearized u8 data)
PixelFormat::Linear8 { channels: 4, has_alpha: true }
```

The pipeline is channel-order-agnostic. BGRA, ARGB, ABGR all work identically — the sRGB transfer function is the same for R, G, and B, and the convolution kernels operate on N floats per pixel. Pass BGRA data as `Srgb8 { channels: 4, has_alpha: true }`, same as RGBA.

### Color Space

- `ColorSpace::Linear` (default): Converts sRGB u8 to linear light f32 before resampling, then converts back. Produces correct results on gradients and avoids darkening halos. Uses f32 intermediate buffers.
- `ColorSpace::Srgb`: Resamples directly in sRGB gamma space. Faster (uses i16 integer path for 4-channel u8), but can produce slightly incorrect gradients. Good enough for thumbnails and previews.

### Filters

31 filters from imageflow, covering a range of sharpness/smoothness tradeoffs:

| Filter | Category | Window | Notes |
|--------|----------|--------|-------|
| `Lanczos` | Sinc | 3.0 | Sharp, some ringing. Good for photos. |
| `Lanczos2` | Sinc | 2.0 | Less ringing than Lanczos-3. |
| `Robidoux` | Cubic | 2.0 | **Default.** Balanced sharpness/smoothness. |
| `RobidouxSharp` | Cubic | 2.0 | More detail, slight ringing. |
| `Mitchell` | Cubic | 2.0 | Mitchell-Netravali (B=1/3, C=1/3). Balanced blur/ringing. |
| `CatmullRom` | Cubic | 2.0 | Catmull-Rom spline (B=0, C=0.5). |
| `Ginseng` | Jinc-sinc | 3.0 | Jinc-windowed sinc. Excellent for upscaling. |
| `Hermite` | Cubic | 1.0 | Smooth interpolation. |
| `CubicBSpline` | Cubic | 2.0 | Very smooth, blurs. B-spline (B=1, C=0). |
| `Triangle` | Linear | 1.0 | Bilinear interpolation. |
| `Box` | Nearest | 0.5 | Nearest neighbor. Fastest, blocky. |
| `Fastest` | Cubic | 0.74 | Minimal quality, maximum speed. |

Plus `LanczosSharp`, `Lanczos2Sharp`, `RobidouxFast`, `GinsengSharp`, `CubicFast`, `Cubic`, `CubicSharp`, `CatmullRomFast`, `CatmullRomFastSharp`, `MitchellFast`, `NCubic`, `NCubicSharp`, `RawLanczos2`, `RawLanczos2Sharp`, `RawLanczos3`, `RawLanczos3Sharp`, `Jinc`, `Linear`, `LegacyIDCTFilter`.

Sharp variants use a slightly reduced blur factor for tighter kernels. Fast variants use smaller windows.

### imgref Integration

With the `imgref` feature, you get typed wrappers for the `imgref` + `rgb` crates:

```rust
use zenresize::{resize_4ch, resize_3ch, resize_gray8};

// Works with rgb::RGBA, rgb::BGRA, or any 4-byte pixel type implementing ComponentSlice
let output: ImgVec<RGBA8> = resize_4ch(img.as_ref(), 512, 384, true, &config);

let output_rgb: ImgVec<RGB8> = resize_3ch(img_rgb.as_ref(), 512, 384, &config);
let output_gray: ImgVec<u8> = resize_gray8(img_gray.as_ref(), 512, 384, &config);
```

## Performance

Benchmark: 1024x1024 RGBA → 512x512, Lanczos3, sRGB color space, single-threaded (Intel i7-12700, Ubuntu 22.04 WSL2, Rust 1.85):

| Method | Time | Throughput |
|--------|------|------------|
| **zenresize Resizer** (cached weights) | 0.95ms | 1109 MP/s |
| **zenresize one-shot** | 1.20ms | 871 MP/s |
| pic-scale 0.6 | 1.29ms | 814 MP/s |
| fast_image_resize 6 | 1.77ms | 593 MP/s |
| resize 0.8 | 3.90ms | 269 MP/s |

The sRGB color space path uses an i16 integer pipeline with 14-bit fixed-point weights, halving SIMD width requirements compared to f32 and quartering intermediate buffer size. The linear-light path uses f32 intermediates for precision.

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `std` | yes | Enables std library. Disable for `no_std` + `alloc`. |
| `imgref` | no | Adds `resize_4ch`, `resize_3ch`, `resize_gray8` using the `imgref` and `rgb` crates. |

## License

AGPL-3.0-or-later
