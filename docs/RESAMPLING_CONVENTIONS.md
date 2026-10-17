# Resampling conventions

Four `ResizeConfig` settings describe sampling geometry and arithmetic. They
apply to every `Filter`, and their defaults reproduce earlier releases byte for
byte.

| Builder method | Default | Controls |
|---|---|---|
| `antialias(bool)` | `true` | whether kernel support widens by `in/out` when downscaling |
| `sample_mapping(SampleMapping)` | `HalfPixel` | where output sample `i` lands in input coordinates |
| `boundary(BoundaryMode)` | `Truncate` | what happens to kernel taps outside the image |
| `arithmetic(Arithmetic)` | `Auto` | numeric representation of the two passes |

Three filter presets cover kernels that reference implementations use:
`Filter::Nearest` (point sampling), `Filter::Gaussian` (sigma 0.5, radius 1.5)
and `Filter::CubicA075` (Keys cubic, a = −0.75).

```rust
use zenresize::{Arithmetic, BoundaryMode, Filter, PixelDescriptor, ResizeConfig};

// Bicubic with fixed support and replicated edges, in f32 throughout.
let config = ResizeConfig::builder(641, 479, 224, 224)
    .format(PixelDescriptor::RGBF32_LINEAR)
    .filter(Filter::CubicA075)
    .antialias(false)
    .boundary(BoundaryMode::Clamp)
    .arithmetic(Arithmetic::F32)
    .build();
```

## Matching other libraries

Measured on a 548-case panel (shapes 1–40 px, four patterns, 227,236 pixels per
row) against pinned public APIs; see the
[qualification record](../benchmarks/conventions_2026-10-15/README.md).
"±1 on" is the share of pixels whose rounded 8-bit value differs, never by
more than 1. The two references for the same kernel disagree with each other
at that level too, so no single f32 arithmetic matches both bit for bit.

| Reference | Settings | Result |
|---|---|---|
| OpenCV `INTER_CUBIC` (f32) | `CubicA075`, `antialias(false)`, `Clamp`, `F32` | ±1 on 0.04% |
| OpenCV `INTER_LINEAR` (f32) | `Triangle`, `antialias(false)`, `Clamp`, `F32` | ±1 on 0.40% |
| OpenCV `INTER_NEAREST` | `Nearest`, `AsymmetricFloat64` | exact |
| PyTorch `bicubic` | `CubicA075`, `antialias(false)`, `Clamp`, `F32` | ±1 on 0.10% (0.07% with `AlignCorners`) |
| PyTorch `bilinear` | `Triangle`, `antialias(false)`, `Clamp`, `F32` | ±1 on 0.86% (0.35% with `AlignCorners`) |
| PyTorch `bicubic`, `antialias=True` | `CatmullRom`, `F32` | ±1 on 0.08%, except output width 1 (below) |
| PyTorch `nearest` / `nearest-exact` | `Nearest` with `AsymmetricFloat32` / `HalfPixel` | exact |
| TensorFlow `gaussian` | `Gaussian`, `HalfPixelFloat32`, `F32`, `antialias` as the caller sets it | ±1 on 0.08% (0.01% antialiased) |
| Pillow `BICUBIC` (RGB) | `CatmullRom`, `srgb()`, `Fixed22` | exact |

Not covered: OpenCV's 8-bit fixed-point cubic and linear (the independently
written models left ±1 on 0.0035% and 19% of pixels, so no mode ships), PyTorch
antialiased bicubic at output width 1 (PyTorch returns a flat column), and
Pillow's RGBA path (it premultiplies in 8 bits first). `Fixed22` with filters
other than `CatmullRom` uses Pillow's arithmetic but has not been compared to
Pillow's other kernels.

## Antialiasing

On downscale, the kernel normally stretches by `in/out` so it averages every
input pixel it covers. `antialias(false)` keeps the kernel at its own radius, so
a downscale samples rather than averages. `Filter::Nearest` never widens.

## Sample mapping

`in` and `out` are the axis lengths, `i` the output index.

| `SampleMapping` | Source position | `Nearest` picks |
|---|---|---|
| `HalfPixel` (default) | `(i + 0.5)·in/out − 0.5` | `floor((i + 0.5)·in/out)`, exact integer |
| `HalfPixelFloat32` | same, in f32: `s = 1/(out/in)`, sample `(i + 0.5)·s`, tap `j` at `k·(j + 0.5 − sample)` | floor of the f32 sample |
| `AlignCorners` | `i·(in − 1)/(out − 1)`; one output maps to 0 | nearest sample, ties round up |
| `Asymmetric` | `i·in/out` | `floor(i·in/out)`, exact integer |
| `AsymmetricFloat32` | `i · (in/out)` in f32 | floor of the f32 product |
| `AsymmetricFloat64` | `i · (1/(out/in))` in f64 | floor of the f64 product |

In `HalfPixelFloat32`, `k` is `1/s` when antialiasing widens the support and 1
otherwise. The floating-point variants exist because rounding decides which
sample is chosen at exact integer positions (unit tests pin 26→22 at i=11 for
`AsymmetricFloat32` and 6→34 at i=17 for `AsymmetricFloat64`) and which taps
fall inside a hard kernel cutoff such as the Gaussian's (dropping the f32
arithmetic raises the TensorFlow ±1 share from 0.08% to 0.53%).

`AlignCorners` with antialiasing on is refused for weighted filters: widened
support has no defined align-corners geometry. Use `antialias(false)`.

## Boundary

`Truncate` (default) drops taps outside the image and renormalizes the rest.
`Clamp` adds each outside tap's weight to the nearest edge sample, as if the
edge pixels repeated indefinitely.

## Arithmetic

- **`Auto`** (default): the fastest representation within the crate's accuracy
  targets: 14-bit fixed point for eligible 4-channel u8 inputs, otherwise f32
  arithmetic with input rows cached as f16.
- **`F32`**: f32 arithmetic and f32 storage for every intermediate. Use it when
  results must track an f32 reference. With `Filter::Nearest` it copies f32
  samples exactly, which f16 storage cannot.
- **`Fixed22`**: Pillow's arithmetic. u8 samples, horizontal pass then vertical
  pass, coefficients with 22 fractional bits, and each pass rounds half up and
  clamps to `0..=255`. It works for any channel count and honors strides, crop
  and padding. It requires u8 input and output with no transfer
  conversion (call `srgb()` for gamma-encoded data) and no straight alpha.
  Background compositing returns `CompositeError::UnsupportedArithmetic`,
  `with_mask` panics, and the `resize_hfirst_streaming*` helpers return an error.

## Validation

`ResizeConfig::validate` (and every constructor) refuses weighted
`AlignCorners` with antialiasing on, and `Fixed22` outside the formats above.
The weight-table memory bound is unchanged: every preset, the new ones
included, fits its radius-8 estimate.

Weight tables store tap counts as u32. Earlier releases stored u16, so
downscales needing more than 65,535 taps (for example Lanczos from 2,000,000 to
100) lost taps and came out dark. Rows that wide also sum their f32
coefficients in f64 before renormalizing; narrower rows keep their previous
arithmetic.
