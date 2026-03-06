# zenresize

High-quality image resampling with 31 filters, streaming API, and SIMD acceleration.

## Quick Start

```rust
use zenresize::{Resizer, ResizeConfig, Filter, PixelDescriptor};

let input = vec![128u8; 1024 * 768 * 4]; // RGBA pixels

let config = ResizeConfig::builder(1024, 768, 512, 384)
    .filter(Filter::Lanczos)
    .format(PixelDescriptor::RGBA8_SRGB)
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
- Channel-order-agnostic: RGBA, BGRA, ARGB, BGRX all work without swizzling
- u8, u16, and f32 pixel I/O; cross-format resize (e.g., u8 in, f32 out)
- `no_std` + `alloc` compatible (std optional)
- SIMD-accelerated via [archmage](https://crates.io/crates/archmage): AVX2+FMA on x86-64, NEON on ARM, WASM SIMD, scalar fallback
- Optional AVX-512 V-filter kernel (`avx512` feature)

## Resizer

`Resizer` pre-computes weight tables from the config. Reusing one across images with the same dimensions and filter saves the weight computation cost.

```rust
use zenresize::{Resizer, ResizeConfig, Filter, PixelDescriptor};

let config = ResizeConfig::builder(1024, 1024, 512, 512)
    .filter(Filter::Lanczos)
    .format(PixelDescriptor::RGBA8_SRGB)
    .build();

let mut resizer = Resizer::new(&config);

// Allocating -- returns a new Vec<u8>
let output: Vec<u8> = resizer.resize(&input);

// Non-allocating -- writes into your buffer
let mut buf = vec![0u8; 512 * 512 * 4];
resizer.resize_into(&input, &mut buf);
```

For pipelines that already work in linear f32:

```rust
let config = ResizeConfig::builder(1024, 1024, 512, 512)
    .filter(Filter::Lanczos)
    .format(PixelDescriptor::RGBAF32_LINEAR)
    .build();

let mut resizer = Resizer::new(&config);
let output_f32: Vec<f32> = resizer.resize_f32(&input_f32);
```

Cross-format resizing (u8 sRGB input, f32 linear output, or any combination):

```rust
let mut resizer = Resizer::new(&ResizeConfig::builder(w, h, out_w, out_h)
    .filter(Filter::Lanczos)
    .input(PixelDescriptor::RGBA8_SRGB)
    .output(PixelDescriptor::RGBAF32_LINEAR)
    .build());

let output_f32: Vec<f32> = resizer.resize_u8_to_f32(&input_u8);
```

## StreamingResize

Push input rows one at a time, pull output rows as they become available. Uses a V-first pipeline internally: the H-filter runs only `out_height` times (once per output row) instead of `in_height` times.

```rust
use zenresize::{StreamingResize, ResizeConfig, Filter, PixelDescriptor};

let config = ResizeConfig::builder(1000, 800, 500, 400)
    .filter(Filter::Lanczos)
    .format(PixelDescriptor::RGBA8_SRGB)
    .build();

let mut stream = StreamingResize::new(&config);

for y in 0..800 {
    let row = &input_data[y * 4000..(y + 1) * 4000];
    stream.push_row(row).unwrap();

    // Drain output rows as they become available
    while let Some(out_row) = stream.next_output_row() {
        // out_row is &[u8], width * channels bytes
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

### Zero-copy output

Write output directly into an encoder's buffer:

```rust
let row_len = stream.output_row_len();
let mut enc_buf = vec![0u8; row_len];
while stream.next_output_row_into(&mut enc_buf) {
    encoder.write_row(&enc_buf);
}
```

### f32 streaming

```rust
stream.push_row_f32(&f32_row).unwrap();

// Or write directly into the resizer's internal buffer (saves a memcpy):
stream.push_row_f32_with(|buf| {
    // fill buf with f32 pixel data
}).unwrap();

while let Some(out_row) = stream.next_output_row_f32() {
    // out_row is &[f32]
}
```

## ResizeConfig

All resize operations take a `ResizeConfig` built with the builder pattern.

```rust
use zenresize::{ResizeConfig, Filter, PixelDescriptor};

let config = ResizeConfig::builder(in_w, in_h, out_w, out_h)
    .filter(Filter::Lanczos)        // resampling filter (default: Robidoux)
    .format(PixelDescriptor::RGBA8_SRGB)  // sets both input and output format
    .input(PixelDescriptor::RGBA8_SRGB)   // or set them separately
    .output(PixelDescriptor::RGBA8_SRGB)
    .linear()                        // resize in linear light (default)
    .srgb()                          // resize in sRGB space (faster, slight quality loss)
    .resize_sharpen(15.0)            // sharpen during resampling (% negative lobe, default: 0)
    .post_sharpen(0.0)               // post-resize unsharp mask (default: 0.0)
    .in_stride(stride)               // input row stride in elements (default: tightly packed)
    .out_stride(stride)              // output row stride in elements (default: tightly packed)
    .build();
```

### Defaults

If you call `.build()` with no other methods:

- Filter: `Robidoux`
- Format: `RGBA8_SRGB` for both input and output
- Linear: `true` (sRGB u8 -> linear f32 -> resize -> sRGB u8)
- Resize sharpen: `0.0` (natural filter ratio)
- Post sharpen: `0.0`
- Stride: tightly packed (width * channels)

### Config fields

`ResizeConfig` fields are public (`#[non_exhaustive]`):

```rust
config.filter           // Filter
config.in_width         // u32
config.in_height        // u32
config.out_width        // u32
config.out_height       // u32
config.input            // PixelDescriptor
config.output           // PixelDescriptor
config.linear           // bool
config.post_sharpen     // f32
config.post_blur_sigma  // f32
config.kernel_width_scale // Option<f64>
config.lobe_ratio       // LobeRatio
config.in_stride        // usize (0 = tightly packed)
config.out_stride       // usize (0 = tightly packed)
```

## Pixel Descriptors

`PixelDescriptor` (from [zenpixels](https://crates.io/crates/zenpixels)) describes pixel format, channel layout, alpha mode, and transfer function in one value. Use the provided constants:

```rust
use zenresize::PixelDescriptor;

// sRGB u8
PixelDescriptor::RGBA8_SRGB    // 4ch, straight alpha
PixelDescriptor::RGBX8_SRGB    // 4ch, no alpha (padding byte)
PixelDescriptor::RGB8_SRGB     // 3ch
PixelDescriptor::GRAY8_SRGB    // 1ch grayscale
PixelDescriptor::BGRA8_SRGB    // 4ch, BGR byte order

// Linear f32
PixelDescriptor::RGBAF32_LINEAR
PixelDescriptor::RGBF32_LINEAR

// sRGB u16
PixelDescriptor::RGBA16_SRGB
PixelDescriptor::RGB16_SRGB
```

**Channel order doesn't matter.** The sRGB transfer function is the same for R, G, and B, and the convolution kernels operate on N floats per pixel. Pass BGRA data as `RGBA8_SRGB` -- no swizzling needed. (Use `BGRA8_SRGB` if you want the descriptor to be semantically accurate, but the resize output is identical either way.)

### Color space (`.linear()` / `.srgb()`)

- **Linear** (default): sRGB u8 -> linear f32 -> resize -> sRGB u8. Correct on gradients, avoids darkening halos. Uses f32 intermediate buffers.
- **sRGB**: Resize directly in gamma space. Uses an i16 integer pipeline with 14-bit fixed-point weights for 4-channel formats. Faster; slightly incorrect on gradients; good enough for thumbnails.

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
let all = Filter::all();        // &[Filter] -- all 31 variants
```

## imgref Integration

Typed wrappers for the [imgref](https://crates.io/crates/imgref) + [rgb](https://crates.io/crates/rgb) crates. These accept any pixel type implementing `ComponentSlice` (RGBA, BGRA, etc. from the `rgb` crate).

```rust
use zenresize::{resize_4ch, resize_3ch, resize_gray8};
use zenresize::{ResizeConfig, Filter, PixelDescriptor};
use imgref::ImgVec;
use rgb::RGBA8;

let config = ResizeConfig::builder(0, 0, 0, 0) // dimensions overridden by imgref
    .filter(Filter::Lanczos)
    .build();

// 4-channel: pass a PixelDescriptor to control alpha handling
let output: ImgVec<RGBA8> = resize_4ch(
    img.as_ref(),                   // ImgRef<RGBA8>
    512, 384,                       // output dimensions
    PixelDescriptor::RGBA8_SRGB,
    &config,
);

// 3-channel
let output_rgb: ImgVec<RGB8> = resize_3ch(img_rgb.as_ref(), 512, 384, &config);

// Grayscale
let output_gray: ImgVec<u8> = resize_gray8(img_gray.as_ref(), 512, 384, &config);
```

The imgref functions override the config's dimensions, formats, and stride. Filter, linear mode, and sharpen are preserved.

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `std` | yes | Enables std library. Disable for `no_std` + `alloc`. |
| `layout` | yes | Layout negotiation and pipeline execution via [zenlayout](https://crates.io/crates/zenlayout). |
| `avx512` | no | Native AVX-512 V-filter kernel (x86-64 only). |

## License

AGPL-3.0-or-later
