# zenresize

High-quality image resampling with 31 filters, streaming API, and SIMD acceleration.

## Quick Start

```rust
use zenresize::{Resizer, ResizeConfig, Filter, PixelFormat, PixelLayout};

let input = vec![128u8; 1024 * 768 * 4]; // RGBA pixels

let config = ResizeConfig::builder(1024, 768, 512, 384)
    .filter(Filter::Lanczos)
    .format(PixelFormat::Srgb8(PixelLayout::Rgba))
    .build();

let output = Resizer::new(&config).resize(&input);
assert_eq!(output.len(), 512 * 384 * 4);
```

## Features

- 31 resampling filters (Lanczos, Mitchell, Robidoux, Ginseng, etc.)
- sRGB-aware linear-light processing for correct gamma handling
- Row-at-a-time streaming API for pipeline integration
- `Resizer` struct for amortizing weight computation across repeated resizes
- Alpha premultiply/unpremultiply built into the pipeline
- Premultiplied alpha passthrough for compositing pipelines
- Channel-order-agnostic: RGBA, BGRA, ARGB, BGRX all work without swizzling
- u8 and f32 pixel format support
- `no_std` + `alloc` compatible (std optional)
- SIMD-accelerated via [archmage](https://github.com/imazen/archmage) (AVX2+FMA 256-bit on x86, NEON on ARM, scalar fallback)

## Resizer

`Resizer` pre-computes weight tables from the config. Creating one is cheap, and reusing it across images with the same dimensions and filter saves ~27% per call.

```rust
use zenresize::{Resizer, ResizeConfig, Filter, PixelFormat, PixelLayout};

let config = ResizeConfig::builder(1024, 1024, 512, 512)
    .filter(Filter::Lanczos)
    .format(PixelFormat::Srgb8(PixelLayout::Rgba))
    .build();

let mut resizer = Resizer::new(&config);

// Allocating — returns a new Vec<u8>
let output: Vec<u8> = resizer.resize(&input);

// Non-allocating — writes into your buffer
let mut buf = vec![0u8; 512 * 512 * 4];
resizer.resize_into(&input, &mut buf);
```

For pipelines that already work in linear f32:

```rust
let config_f32 = ResizeConfig::builder(1024, 1024, 512, 512)
    .filter(Filter::Lanczos)
    .format(PixelFormat::LinearF32(PixelLayout::Rgba))
    .build();

let mut resizer = Resizer::new(&config_f32);

let output_f32: Vec<f32> = resizer.resize_f32(&input_f32);

let mut buf_f32 = vec![0.0f32; 512 * 512 * 4];
resizer.resize_f32_into(&input_f32, &mut buf_f32);
```

## StreamingResize

Push input rows one at a time, pull output rows as they become available. Uses a V-first pipeline: input rows are cached in the ring buffer, then V-filter and H-filter run on demand when output rows are pulled. For downscaling, this runs the H-filter only `out_height` times instead of `in_height` times, making streaming ~30% faster than fullframe for the f32 linear path.

```rust
use zenresize::{StreamingResize, ResizeConfig, Filter, PixelFormat, PixelLayout};

let config = ResizeConfig::builder(1000, 800, 500, 400)
    .filter(Filter::Lanczos)
    .format(PixelFormat::Srgb8(PixelLayout::Rgba))
    .build();

let mut stream = StreamingResize::new(&config);

// Interleaved push/drain — MUST drain between pushes (backpressure contract)
for y in 0..800 {
    let row = &input_data[y * 4000..(y + 1) * 4000];
    stream.push_row(row);

    while let Some(out_row) = stream.next_output_row() {
        // out_row is &[u8], width * channels bytes (zero-alloc)
    }
}
stream.finish();

// Drain remaining output rows
while let Some(out_row) = stream.next_output_row() {
    // ...
}

assert!(stream.is_complete());
assert_eq!(stream.output_rows_produced(), 400);
```

### Zero-copy output into caller buffer

Write output directly into an encoder's buffer (skips internal buffer):

```rust
let row_len = stream.output_row_len();
let mut enc_buf = vec![0u8; row_len];
// In the push/drain loop:
while stream.next_output_row_into(&mut enc_buf) {
    encoder.write_row(&enc_buf);
}
```

### f32 streaming

```rust
stream.push_row_f32(&f32_row);

// Or write directly into the resizer's internal buffer (saves a memcpy):
stream.push_row_f32_with(|buf| {
    // fill buf with f32 pixel data
});

while let Some(out_row) = stream.next_output_row_f32() {
    // out_row is &[f32] (zero-alloc)
}
```

### Query methods

```rust
// How many input rows needed before the first output row appears
let needed = stream.initial_input_rows_needed();

// How many output rows produced so far
let produced = stream.output_rows_produced();

// Whether all output rows have been produced
let done = stream.is_complete();
```

## ResizeConfig

All resize operations take a `ResizeConfig` built with the builder pattern.

```rust
use zenresize::{ResizeConfig, Filter, PixelFormat, PixelLayout};

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

### Defaults

If you call `.build()` with no other methods, you get:

- Filter: `Robidoux`
- Format: `Srgb8(Rgba)` for both input and output
- Linear: `true` (sRGB u8 → linear f32 → resize → sRGB u8)
- Sharpen: `0.0`
- Stride: tightly packed (width * channels)

### Config fields

`ResizeConfig` fields are public:

```rust
config.filter           // Filter
config.in_width         // u32
config.in_height        // u32
config.out_width        // u32
config.out_height       // u32
config.input_format     // PixelFormat
config.output_format    // PixelFormat
config.linear           // bool
config.sharpen          // f32
config.in_stride        // usize (0 = tightly packed)
config.out_stride       // usize (0 = tightly packed)
```

### Config methods

```rust
config.validate()?;                  // Err if dimensions are 0 or layouts mismatch
config.effective_in_stride();        // Actual stride (resolves 0 → width * channels)
config.effective_out_stride();
config.input_row_len();              // width * channels (no padding)
config.output_row_len();
config.needs_linearization();        // true if linear + sRGB input + not premultiplied
```

## Pixel Formats and Color Space

A `PixelFormat` pairs a data type with a `PixelLayout`:

```rust
use zenresize::{PixelFormat, PixelLayout};

// PixelLayout describes channel count and alpha semantics
//   Gray        — 1 channel
//   Rgb         — 3 channels (RGB, BGR, etc.)
//   Rgbx        — 4 channels, padding byte (RGBX, BGRX)
//   Rgba        — 4 channels, straight alpha (pipeline premultiplies/unpremultiplies)
//   RgbaPremul  — 4 channels, premultiplied alpha (passthrough, no conversion)

// PixelFormat wraps a layout with a data type
PixelFormat::Srgb8(PixelLayout::Rgba)         // sRGB u8, straight alpha (common case)
PixelFormat::Srgb8(PixelLayout::Rgbx)         // sRGB u8, 4ch no alpha
PixelFormat::Srgb8(PixelLayout::Rgb)          // sRGB u8, 3ch
PixelFormat::Srgb8(PixelLayout::Gray)         // sRGB u8, grayscale
PixelFormat::Srgb8(PixelLayout::RgbaPremul)   // sRGB u8, premultiplied alpha
PixelFormat::LinearF32(PixelLayout::Rgba)     // linear f32, straight alpha
```

**Channel order doesn't matter.** The sRGB transfer function is the same for R, G, and B, and the convolution kernels operate on N floats per pixel. Pass BGRA data as `Srgb8(Rgba)` — no swizzling needed.

**Premultiplied alpha** (`RgbaPremul`): skips premultiply/unpremultiply and skips linearization (linearizing premultiplied sRGB is mathematically incorrect). Always takes the fast i16 integer path.

### Color space (`.linear()` / `.srgb()`)

The builder's `.linear()` (default) and `.srgb()` control how the resize computation happens:

- **Linear**: sRGB u8 → linear f32 → resize → sRGB u8. Correct on gradients, avoids darkening halos. Uses f32 intermediate buffers.
- **Srgb**: Resize directly in gamma space. Uses an i16 integer pipeline with 14-bit fixed-point weights for 4-channel formats, halving SIMD width requirements. Slightly incorrect on gradients; good enough for thumbnails.

### Query methods

Both `PixelLayout` and `PixelFormat` expose query methods. `PixelFormat` delegates to its layout for channel/alpha queries and adds type queries:

```rust
let layout = PixelLayout::Rgba;
layout.channels()              // 4
layout.has_alpha()             // true
layout.needs_premultiply()     // true (only straight alpha)
layout.is_premultiplied()      // false

let fmt = PixelFormat::Srgb8(layout);
fmt.layout()                   // PixelLayout::Rgba
fmt.channels()                 // 4 (delegates to layout)
fmt.is_u8()                    // true
fmt.is_f32()                   // false
fmt.is_srgb()                  // true
fmt.is_linear()                // false
```

## Filters

31 filters covering a range of sharpness/smoothness tradeoffs:

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

```rust
use zenresize::Filter;

let f = Filter::default();      // Robidoux
let all = Filter::all();        // &[Filter; 31] — all variants
```

## imgref Integration

With the `imgref` feature, you get typed wrappers for the `imgref` + `rgb` crates. These accept any pixel type implementing `ComponentSlice` (RGBA, BGRA, etc. from the `rgb` crate).

```rust
use zenresize::{resize_4ch, resize_3ch, resize_gray8};
use zenresize::{ResizeConfig, Filter, PixelLayout};
use imgref::ImgVec;
use rgb::RGBA8;

let config = ResizeConfig::builder(0, 0, 0, 0) // dimensions overridden by imgref
    .filter(Filter::Lanczos)
    .build();

// 4-channel: pass layout to control alpha handling
let output: ImgVec<RGBA8> = resize_4ch(
    img.as_ref(),    // ImgRef<RGBA8>
    512, 384,        // output dimensions
    PixelLayout::Rgba,
    &config,
);

// 3-channel: always Rgb layout
let output_rgb: ImgVec<RGB8> = resize_3ch(img_rgb.as_ref(), 512, 384, &config);

// Grayscale
let output_gray: ImgVec<u8> = resize_gray8(img_gray.as_ref(), 512, 384, &config);
```

The imgref functions override the config's dimensions, formats, and stride — they take those from the `ImgRef` and output size parameters. Filter, linear mode, and sharpen are preserved from the config.

## Performance

Benchmark: 1024x1024 RGBA → 512x512, Lanczos3, sRGB, single-threaded (Intel i7-12700, Ubuntu 22.04 WSL2, Rust 1.85):

| Method | Time | Throughput |
|--------|------|------------|
| **zenresize Resizer** (cached weights) | 0.95ms | 1109 MP/s |
| **zenresize** (single use) | 1.20ms | 871 MP/s |
| pic-scale 0.6 | 1.29ms | 814 MP/s |
| fast_image_resize 6 | 1.77ms | 593 MP/s |
| resize 0.8 | 3.90ms | 269 MP/s |

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `std` | yes | Enables std library. Disable for `no_std` + `alloc`. |
| `imgref` | no | Adds `resize_4ch`, `resize_3ch`, `resize_gray8` using the `imgref` and `rgb` crates. |

## License

AGPL-3.0-or-later
