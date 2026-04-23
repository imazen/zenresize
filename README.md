# zenresize [![ci](https://img.shields.io/github/actions/workflow/status/imazen/zenresize/ci.yml?branch=main&style=flat-square)](https://github.com/imazen/zenresize/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/zenresize?style=flat-square)](https://crates.io/crates/zenresize) [![docs.rs](https://img.shields.io/docsrs/zenresize?style=flat-square)](https://docs.rs/zenresize) [![msrv](https://img.shields.io/badge/MSRV-1.93-blue?style=flat-square)](https://doc.rust-lang.org/cargo/reference/manifest.html#the-rust-version-field) [![license](https://img.shields.io/crates/l/zenresize?style=flat-square)](https://github.com/imazen/zenresize#license)

zenresize is a SIMD-accelerated image resampling library with crop, resize, and canvas padding in streaming or fullframe modes.

```toml
[dependencies]
zenresize = "0.1"
```

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

## Operations

All operations work in the streaming API. Crop and padding also work independently (without resize) by setting output dimensions equal to crop/content dimensions.

| Operation | What it does | Builder method |
|-----------|-------------|----------------|
| **Resize** | Resample to new dimensions with a choice of 31 filters | `.filter(Filter::Lanczos)` |
| **Fit** | Aspect-preserving resize to a target box | `.fit(FitMode::Fit, max_w, max_h)` |
| **Crop** | Extract a rectangular region from the input | `.crop(x, y, w, h)` |
| **Pad** | Add solid-color border around the output | `.padding(top, right, bottom, left)` |
| **Orient** | Apply EXIF orientation (rotate/flip) post-resize | `stream.with_orientation(OrientOutput::Rotate90)` |
| **Crop + Resize** | Extract region, then resize it | `.crop(...)` on a config with different output dims |
| **Resize + Pad** | Resize, then add padding | `.padding(...)` on a config with different input/output dims |
| **Crop + Resize + Pad** | All three in sequence | `.crop(...)` + `.padding(...)` |

The pipeline order is always: **crop** (input side) -> **resize** -> **pad** (output side).

## Features

- **Crop, resize, and pad** -- independently or combined, streaming or fullframe
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

## Compositing

Resize foreground images onto a background in a single pass. Compositing happens in premultiplied linear f32 space between the vertical filter and unpremultiply -- no extra buffer copy.

```rust
use zenresize::{StreamingResize, ResizeConfig, Filter, PixelDescriptor, SolidBackground, BlendMode};

let config = ResizeConfig::builder(800, 600, 400, 300)
    .filter(Filter::Lanczos)
    .format(PixelDescriptor::RGBA8_SRGB)
    .build();

let bg = SolidBackground::white(PixelDescriptor::RGBA8_SRGB);
let mut stream = StreamingResize::with_background(&config, bg)
    .expect("compositing config")
    .with_blend_mode(BlendMode::SrcOver); // default; 31 modes available

for y in 0..600 {
    stream.push_row(&input[y * 3200..(y + 1) * 3200]).unwrap();
    while let Some(out) = stream.next_output_row() {
        // composited output rows
    }
}
```

Background types: `SolidBackground` (constant color), `SliceBackground` (borrow a buffer), `StreamedBackground` (push rows), or implement the `Background` trait yourself. `NoBackground` (the default) eliminates all composite code at compile time.

### Masking

Apply per-pixel masks to control where the foreground is visible. Masks are applied between resize and compositing, so rounded corners over a white background produce white corners (not transparent-over-black).

```rust
use zenresize::{StreamingResize, ResizeConfig, PixelDescriptor, SolidBackground, RoundedRectMask};

let config = ResizeConfig::builder(800, 600, 400, 300)
    .format(PixelDescriptor::RGBA8_SRGB)
    .build();

let bg = SolidBackground::white(PixelDescriptor::RGBA8_SRGB);
let mask = RoundedRectMask::new(400, 300, 20.0);
let stream = StreamingResize::with_background(&config, bg)
    .expect("compositing config")
    .with_mask(mask);
```

Mask types re-exported from [zenblend](https://crates.io/crates/zenblend): `RoundedRectMask`, `LinearGradientMask`, `RadialGradientMask`, or implement `MaskSource`.

## Source Region (Crop)

Extract a rectangular region from the input before resizing. The streaming API accepts full-width input rows; the resizer skips rows outside the vertical range and extracts the horizontal region internally.

```rust
use zenresize::{StreamingResize, ResizeConfig, Filter, PixelDescriptor};

// Crop a 400x300 region starting at (100, 50), resize to 200x150
let config = ResizeConfig::builder(1000, 800, 200, 150)
    .filter(Filter::Lanczos)
    .format(PixelDescriptor::RGBA8_SRGB)
    .crop(100, 50, 400, 300)
    .build();

let mut stream = StreamingResize::new(&config);

// Push full-width rows -- rows outside [50..350) are skipped automatically
for y in 0..800 {
    stream.push_row(&source_rows[y]).unwrap();
    while let Some(out) = stream.next_output_row() {
        // 200 * 4 bytes per row
    }
}
```

Crop without resize (extract only):

```rust
// Extract 400x300 at (100, 50), no resize
let config = ResizeConfig::builder(1000, 800, 400, 300)
    .format(PixelDescriptor::RGBA8_SRGB)
    .crop(100, 50, 400, 300)
    .build();
```

## Fit Modes (Aspect-Ratio Constraints)

Four common ways to fit an input into a target box, preserving aspect ratio
where appropriate. One call sets `out_width`/`out_height` (and, for `Cover`,
a center-anchored source crop) without reaching for a separate layout crate.

| Mode | Behavior | Typical use |
|------|----------|-------------|
| `FitMode::Fit` | Aspect-preserving, fit entirely inside bounds. Output `≤` bounds on both axes, `==` on one. May up- or down-scale. | Thumbnail letterbox |
| `FitMode::Within` | Like `Fit`, but never upscales past input size. | Thumbnails that stay sharp when source is small |
| `FitMode::Cover` | Aspect-preserving, fills the bounds exactly. Source is center-cropped to target aspect, then resized. Output is exactly `max_w × max_h`. | Hero images, cover art, imageflow `fit=crop` |
| `FitMode::Stretch` | Ignores aspect, stretches to exact bounds. | Non-photo UI assets |

```rust
use zenresize::{FitMode, ResizeConfig, Filter, PixelDescriptor};

// 1600×900 source, fit into 800×600 letterbox → 800×450, no crop.
let config = ResizeConfig::builder(1600, 900, 0, 0)
    .filter(Filter::Lanczos)
    .format(PixelDescriptor::RGBA8_SRGB)
    .fit(FitMode::Fit, 800, 600)
    .build();
assert_eq!((config.out_width, config.out_height), (800, 450));

// Same source, Cover: center-cropped to 4:3, output exactly 800×600.
let config = ResizeConfig::builder(1600, 900, 0, 0)
    .format(PixelDescriptor::RGBA8_SRGB)
    .fit(FitMode::Cover, 800, 600)
    .build();
assert_eq!((config.out_width, config.out_height), (800, 600));
// `.fit(Cover, ...)` also sets `source_region` for the crop — no extra call.
```

For raw dimension math without the builder:

```rust
use zenresize::{FitMode, fit_dims, fit_cover_source_crop};

// What output dims would FitMode produce?
assert_eq!(fit_dims(1600, 900, 800, 600, FitMode::Fit),   (800, 450));
assert_eq!(fit_dims(1600, 900, 800, 600, FitMode::Cover), (800, 600));
assert_eq!(fit_dims(400, 300,  800, 600, FitMode::Within), (400, 300));

// What source crop does Cover apply?
// Target 4:3 from 16:9 source → crop to 1200×900 centered.
assert_eq!(fit_cover_source_crop(1600, 900, 800, 600), (200, 0, 1200, 900));
```

The math is a port of [`zenlayout`](https://crates.io/crates/zenlayout)'s
`fit_inside` / `crop_to_aspect` including snap-to-target rounding — verified
byte-identical across a ~6M-case brute-force sweep
(`tests/vs_zenlayout.rs`). Callers migrating from `zenlayout` for simple
fit/within/cover cases see no pixel-level drift.

## EXIF Orientation

`OrientOutput` is the 8-element D4 dihedral group (EXIF orientations 1–8),
applied post-resize by the streaming pipeline. If you already hold a
[`zenpixels::Orientation`](https://crates.io/crates/zenpixels) from metadata
parsing, it converts directly:

```rust
use zenresize::{OrientOutput, Orientation, StreamingResize};

let exif_tag: u8 = 6;  // Rotate 90° CW
let orient = Orientation::from_exif(exif_tag).unwrap_or_default();
let mut resizer = StreamingResize::new(&config).with_orientation(orient.into());
```

`Orientation` (re-exported from `zenpixels`) has the full group algebra —
`compose`, `inverse`, `from_exif`, `to_exif`, `swaps_axes` — so you can
build up composed transforms (e.g. EXIF orient + explicit 180°) and hand
the result to zenresize with one `.into()`.

## Output Padding

Add a solid-color border around the resized output. The total output becomes `(left + width + right)` by `(top + height + bottom)`.

```rust
use zenresize::{StreamingResize, ResizeConfig, Filter, PixelDescriptor};

// Resize 1000x800 -> 500x400, then add 20px black border
let config = ResizeConfig::builder(1000, 800, 500, 400)
    .filter(Filter::Lanczos)
    .format(PixelDescriptor::RGBA8_SRGB)
    .padding_uniform(20)
    .padding_color([0.0, 0.0, 0.0, 1.0])
    .build();

let mut stream = StreamingResize::new(&config);

// output_row_len() is (20 + 500 + 20) * 4 = 2160
// total_output_height() is 20 + 400 + 20 = 440
// Top padding rows are available before any input is pushed

for y in 0..800 {
    stream.push_row(&source_rows[y]).unwrap();
    while let Some(out) = stream.next_output_row() {
        // First 20 rows: solid black
        // Next 400 rows: 20px black + 500px content + 20px black
        // Last 20 rows: solid black
    }
}
```

Asymmetric letterboxing:

```rust
let config = ResizeConfig::builder(1000, 800, 500, 400)
    .format(PixelDescriptor::RGBA8_SRGB)
    .padding(40, 0, 40, 0)              // 40px top/bottom only
    .padding_color([0.0, 0.0, 0.0, 1.0])
    .build();
// Total output: 500 x 480
```

Padding without resize:

```rust
let config = ResizeConfig::builder(500, 400, 500, 400)
    .format(PixelDescriptor::RGBA8_SRGB)
    .padding_uniform(10)
    .padding_color([1.0, 1.0, 1.0, 1.0]) // white border
    .build();
// Total output: 520 x 420
```

### Padding color

The `padding_color` values are 0.0-1.0 in the output's color space. For sRGB u8 output, 0.5 maps to value 128. For linear f32, 0.5 maps to 0.5. Only the first N channels are used (N = channel count of the output format).

Works with all output types: u8 (`next_output_row`), f32 (`next_output_row_f32`), u16 (`next_output_row_u16`).

## Crop + Resize + Pad

All three operations compose naturally:

```rust
// Extract 800x600 region, resize to 400x300, add 10px white border
let config = ResizeConfig::builder(2000, 1500, 400, 300)
    .filter(Filter::Lanczos)
    .format(PixelDescriptor::RGBA8_SRGB)
    .crop(200, 100, 800, 600)
    .padding_uniform(10)
    .padding_color([1.0, 1.0, 1.0, 1.0])
    .build();

// Pipeline: crop 800x600 -> resize to 400x300 -> pad to 420x320
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
    .crop(x, y, w, h)               // source region (default: full input)
    .padding(top, right, bottom, left)  // output padding (default: none)
    .padding_color([0.0, 0.0, 0.0, 1.0])  // padding fill color
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
config.in_width         // u32 (full source width)
config.in_height        // u32 (full source height)
config.out_width        // u32 (content output width, before padding)
config.out_height       // u32 (content output height, before padding)
config.input            // PixelDescriptor
config.output           // PixelDescriptor
config.linear           // bool
config.post_sharpen     // f32
config.post_blur_sigma  // f32
config.kernel_width_scale // Option<f64>
config.lobe_ratio       // LobeRatio
config.in_stride        // usize (0 = tightly packed)
config.out_stride       // usize (0 = tightly packed)
config.source_region    // Option<SourceRegion> (crop rectangle)
config.padding          // Option<Padding> (output padding)
```

Helper methods:

```rust
config.resize_in_width()     // crop width if set, else in_width
config.resize_in_height()    // crop height if set, else in_height
config.total_output_width()  // out_width + left + right padding
config.total_output_height() // out_height + top + bottom padding
config.total_output_row_len() // total_output_width * channels
```

## Pixel Formats

`PixelDescriptor` (from [zenpixels](https://crates.io/crates/zenpixels)) describes pixel format, channel layout, alpha mode, and transfer function in one value.

### Supported formats

| Format | Channels | Type | Transfer | Constant |
|--------|----------|------|----------|----------|
| RGBA sRGB | 4 (straight alpha) | u8 | sRGB | `RGBA8_SRGB` |
| RGBX sRGB | 4 (no alpha) | u8 | sRGB | `RGBX8_SRGB` |
| RGB sRGB | 3 | u8 | sRGB | `RGB8_SRGB` |
| Gray sRGB | 1 | u8 | sRGB | `GRAY8_SRGB` |
| BGRA sRGB | 4 (straight alpha) | u8 | sRGB | `BGRA8_SRGB` |
| RGBA linear | 4 (straight alpha) | f32 | Linear | `RGBAF32_LINEAR` |
| RGB linear | 3 | f32 | Linear | `RGBF32_LINEAR` |
| RGBA sRGB | 4 (straight alpha) | u16 | sRGB | `RGBA16_SRGB` |
| RGB sRGB | 3 | u16 | sRGB | `RGB16_SRGB` |

Cross-format resize is supported: any input type to any output type (u8 <-> u16 <-> f32).

### Transfer functions

All five transfer functions work with all channel types and layouts:

| Transfer | Description |
|----------|-------------|
| `Srgb` | Standard sRGB gamma (default) |
| `Linear` | Linear light (identity) |
| `Bt709` | BT.709 broadcast gamma |
| `Pq` | HDR10 Perceptual Quantizer |
| `Hlg` | Hybrid Log-Gamma (HDR) |

### Channel order

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
| `zennode` | no | Self-documenting node definitions for [zennode](https://crates.io/crates/zennode) pipeline integration. |
| `pretty-safe` | no | Replaces bounds-checked indexing with `get_unchecked` in SIMD kernels where bounds are proven by prior guards. ~17% fewer instructions on x86-64. Introduces `unsafe`; the default build is `#![forbid(unsafe_code)]`. |

## Benchmarks

The `benches/` directory contains 19 benchmark binaries covering throughput, precision, and profiling:

| Benchmark | What it measures |
|-----------|-----------------|
| `paired_bench` | Interleaved paired comparison against pic-scale, fast_image_resize, resize. Statistical diff with 95% CI. |
| `resize_bench` | Criterion throughput at 50%, 25%, and 200% scale across image sizes. |
| `tango_bench` | Regression detection across code changes. |
| `sweep_bench` | Performance across sizes (64–7680 px) and ratios (12.5%–300%). CSV output. |
| `precision` | f32/u8 accuracy vs f64 reference and cross-library comparison. |
| `transfer_bench` | sRGB/BT.709/PQ/HLG transfer function speed vs powf and colorutils-rs. |
| `planar_bench` | Interleaved vs planar resize strategies at 0.5–24 MP. |
| `profile_*` | Minimal binaries for callgrind/perf (sRGB, linear, f32, f16, streaming). |

```bash
cargo bench --bench paired_bench    # quick paired comparison
cargo bench --bench resize_bench    # full criterion suite (HTML reports in target/criterion/)
```

The `bench-simd-competitors` feature enables SIMD on pic-scale for fair comparison (off by default, so pic-scale runs scalar-only).

## Limitations

- No f16 channel type (f32 and u16 cover HDR use cases)
- No narrow/video signal range -- full range only
- Premultiplied input is incompatible with compositing (unpremultiply first, or the pipeline returns `CompositeError::PremultipliedInput`)
- GrayAlpha and Oklab pixel layouts are not supported

## Image tech I maintain

| | |
|:--|:--|
| State of the art codecs* | [zenjpeg] · [zenpng] · [zenwebp] · [zengif] · [zenavif] ([rav1d-safe] · [zenrav1e] · [zenavif-parse] · [zenavif-serialize]) · [zenjxl] ([jxl-encoder] · [zenjxl-decoder]) · [zentiff] · [zenbitmaps] · [heic] · [zenraw] · [zenpdf] · [ultrahdr] · [mozjpeg-rs] · [webpx] |
| Compression | [zenflate] · [zenzop] |
| Processing | **zenresize** · [zenfilters] · [zenquant] · [zenblend] |
| Metrics | [zensim] · [fast-ssim2] · [butteraugli] · [resamplescope-rs] · [codec-eval] · [codec-corpus] |
| Pixel types & color | [zenpixels] · [zenpixels-convert] · [linear-srgb] · [garb] |
| Pipeline | [zenpipe] · [zencodec] · [zencodecs] · [zenlayout] · [zennode] |
| ImageResizer | [ImageResizer] (C#) — 24M+ NuGet downloads across all packages |
| [Imageflow][] | Image optimization engine (Rust) — [.NET][imageflow-dotnet] · [node][imageflow-node] · [go][imageflow-go] — 9M+ NuGet downloads across all packages |
| [Imageflow Server][] | [The fast, safe image server](https://www.imazen.io/) (Rust+C#) — 552K+ NuGet downloads, deployed by Fortune 500s and major brands |

<sub>* as of 2026</sub>

### General Rust awesomeness

[archmage] · [magetypes] · [enough] · [whereat] · [zenbench] · [cargo-copter]

[And other projects](https://www.imazen.io/open-source) · [GitHub @imazen](https://github.com/imazen) · [GitHub @lilith](https://github.com/lilith) · [lib.rs/~lilith](https://lib.rs/~lilith) · [NuGet](https://www.nuget.org/profiles/imazen) (over 30 million downloads / 87 packages)

## License

Dual-licensed: [AGPL-3.0](LICENSE-AGPL3) or [commercial](LICENSE-COMMERCIAL).

I've maintained and developed open-source image server software — and the 40+
library ecosystem it depends on — full-time since 2011. Fifteen years of
continual maintenance, backwards compatibility, support, and the (very rare)
security patch. That kind of stability requires sustainable funding, and
dual-licensing is how we make it work without venture capital or rug-pulls.
Support sustainable and secure software; swap patch tuesday for patch leap-year.

[Our open-source products](https://www.imazen.io/open-source)

**Your options:**

- **Startup license** — $1 if your company has under $1M revenue and fewer
  than 5 employees. [Get a key →](https://www.imazen.io/pricing)
- **Commercial subscription** — Governed by the Imazen Site-wide Subscription
  License v1.1 or later. Apache 2.0-like terms, no source-sharing requirement.
  Sliding scale by company size.
  [Pricing & 60-day free trial →](https://www.imazen.io/pricing)
- **AGPL v3** — Free and open. Share your source if you distribute.

See [LICENSE-COMMERCIAL](LICENSE-COMMERCIAL) for details.

[zenjpeg]: https://github.com/imazen/zenjpeg
[zenpng]: https://github.com/imazen/zenpng
[zenwebp]: https://github.com/imazen/zenwebp
[zengif]: https://github.com/imazen/zengif
[zenavif]: https://github.com/imazen/zenavif
[zenjxl]: https://github.com/imazen/zenjxl
[zentiff]: https://github.com/imazen/zentiff
[zenbitmaps]: https://github.com/imazen/zenbitmaps
[heic]: https://github.com/imazen/heic-decoder-rs
[zenraw]: https://github.com/imazen/zenraw
[zenpdf]: https://github.com/imazen/zenpdf
[ultrahdr]: https://github.com/imazen/ultrahdr
[jxl-encoder]: https://github.com/imazen/jxl-encoder
[zenjxl-decoder]: https://github.com/imazen/zenjxl-decoder
[rav1d-safe]: https://github.com/imazen/rav1d-safe
[zenrav1e]: https://github.com/imazen/zenrav1e
[mozjpeg-rs]: https://github.com/imazen/mozjpeg-rs
[zenavif-parse]: https://github.com/imazen/zenavif-parse
[zenavif-serialize]: https://github.com/imazen/zenavif-serialize
[webpx]: https://github.com/imazen/webpx
[zenflate]: https://github.com/imazen/zenflate
[zenzop]: https://github.com/imazen/zenzop
[zenfilters]: https://github.com/imazen/zenfilters
[zenquant]: https://github.com/imazen/zenquant
[zenblend]: https://github.com/imazen/zenblend
[zensim]: https://github.com/imazen/zensim
[fast-ssim2]: https://github.com/imazen/fast-ssim2
[butteraugli]: https://github.com/imazen/butteraugli
[zenpixels]: https://github.com/imazen/zenpixels
[zenpixels-convert]: https://github.com/imazen/zenpixels
[linear-srgb]: https://github.com/imazen/linear-srgb
[garb]: https://github.com/imazen/garb
[zenpipe]: https://github.com/imazen/zenpipe
[zencodec]: https://github.com/imazen/zencodec
[zencodecs]: https://github.com/imazen/zencodecs
[zenlayout]: https://github.com/imazen/zenlayout
[zennode]: https://github.com/imazen/zennode
[Imageflow]: https://github.com/imazen/imageflow
[Imageflow Server]: https://github.com/imazen/imageflow-server
[imageflow-dotnet]: https://github.com/imazen/imageflow-dotnet
[imageflow-node]: https://github.com/imazen/imageflow-node
[imageflow-go]: https://github.com/imazen/imageflow-go
[ImageResizer]: https://github.com/imazen/resizer
[archmage]: https://github.com/imazen/archmage
[magetypes]: https://github.com/imazen/archmage
[enough]: https://github.com/imazen/enough
[whereat]: https://github.com/lilith/whereat
[zenbench]: https://github.com/imazen/zenbench
[cargo-copter]: https://github.com/imazen/cargo-copter
[resamplescope-rs]: https://github.com/imazen/resamplescope-rs
[codec-eval]: https://github.com/imazen/codec-eval
[codec-corpus]: https://github.com/imazen/codec-corpus
