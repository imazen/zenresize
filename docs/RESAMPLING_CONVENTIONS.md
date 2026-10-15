# Resampling conventions

Custom kernels, support scaling and coordinates are additive. Named-filter
configuration keeps its existing output bits. These options specify sampling
geometry; they do not promise every framework's floating-point accumulation bits.
The [qualification record](../benchmarks/conventions_2026-10-15/README.md)
reports strict differences, including the remaining nonzero targets.

## Custom interpolation

```rust
use zenresize::{BoundaryMode, InterpolationDetails, PixelDescriptor, ResizeConfig};

let cubic = InterpolationDetails::cubic_bc(0.0, 0.75)
    .with_antialias(false)
    .with_boundary_mode(BoundaryMode::Clamp);
let config = ResizeConfig::builder(641, 479, 224, 224)
    .format(PixelDescriptor::RGBF32_LINEAR)
    .interpolation(cubic)
    .build();
```

`cubic_bc(b,c)` is Mitchell–Netravali. Keys parameter `a` is `-c` with
`b=0`: `c=0.5` gives Catmull–Rom/Pillow/MATLAB's mathematical kernel;
`c=0.75` gives torch/OpenCV's non-antialiased cubic kernel.

Antialiasing defaults to true: on downscale, kernel support expands by
`input/output`. `with_antialias(false)` keeps support fixed. Nearest always
selects one source sample and does not widen support. `BoundaryMode::Truncate`
(default) removes out-of-image taps and renormalizes. `Clamp` folds their
weights into the closest edge sample.

`ResizeConfigBuilder::interpolation` overrides both named-filter presets.
A later `.filter(...)` clears the custom override. `.up_filter(...)` does not:
custom interpolation takes precedence regardless of its call order relative to
`.up_filter(...)`. Existing blur/lobe modifiers
are applied to the selected descriptor. Invalid custom coefficients, support,
blur and allocation bounds are refused by `ResizeConfig::validate`. Support must
be at least 0.5. Validation checks every sampled custom row for finite normalized
weights representable by the integer path, after applying blur/lobe modifiers.
Curve blur changes coefficient values, not support width; the allocation guard
bounds support divided by the antialias scale, independently of curve blur.

Custom interpolation uses native f32 row caching in the floating pipeline;
existing named-filter configurations keep their f16 cache and arithmetic.
Consequently `.filter(Filter::CatmullRom)` and
`.interpolation(InterpolationDetails::cubic_bc(0.0, 0.5))` can produce different
output bits despite sharing the same mathematical kernel. Choose a preset to
retain its historical cache precision, or custom interpolation for f32 caching.
Nearest copies selected f32 samples directly, preserving non-half-representable
values and signed zero on the RGBF32 path. The usual format/alpha/compositing
contract still applies when callers request those operations.

## Coordinates

`with_sample_mapping` selects the mapping independently of the kernel.
`AlignCorners` with antialiasing enabled is refused for weighted kernels:
that support convention has not been qualified. Use `.with_antialias(false)`.
Nearest has no widened support and remains supported with either flag.

| Mapping | Source coordinate / nearest index |
|---|---|
| `HalfPixel` (default) | `(i+0.5)*input/output-0.5`; nearest uses exact integer `floor((i+0.5)*input/output)` |
| `AlignCorners` | `i*(input-1)/(output-1)`; output length one maps to zero |
| `Asymmetric` | exact-rational nearest `floor(i*input/output)` |
| `AsymmetricFloat32` | legacy torch nearest: floor of the f32 product `i*(input/output)` |
| `AsymmetricFloat64` | OpenCV nearest: floor of the f64 product `i*(1/(output/input))` |

The last three agree mathematically but can select different samples at integer
boundaries because of their floating-point operations. Public-oracle probes of
all 1-D size pairs 1..=64 identify the OpenCV reciprocal variant; unit tests pin
its `6→34, i=17` boundary and torch's `26→22, i=11` boundary.

```rust
use zenresize::{InterpolationDetails, SampleMapping};
let torch_nearest = InterpolationDetails::nearest()
    .with_sample_mapping(SampleMapping::AsymmetricFloat32);
let opencv_nearest = InterpolationDetails::nearest()
    .with_sample_mapping(SampleMapping::AsymmetricFloat64);
```

## TensorFlow Gaussian

`InterpolationDetails::gaussian_tf()` supplies sigma 0.5 and support radius
1.5. These parameters and hard support boundaries were measured with basis-image
impulses through [tf.image.resize](https://www.tensorflow.org/api_docs/python/tf/image/resize),
TensorFlow CPU 2.21.0, with and without antialiasing. Its default half-pixel
mapping uses reciprocal f32 scale and f32 source-centre arithmetic before the
hard cutoff. Kernel normalization remains f64; weighted outputs retain small
nonzero floating differences from TensorFlow. This is not a bit-exact TensorFlow
arithmetic implementation. Override antialiasing explicitly to match the caller;
TensorFlow's public resize default is false.

## Pillow RGB integer passes

```rust
use zenresize::{PillowCubicResizer, PixelDescriptor};
let resize = PillowCubicResizer::new(7, 5, 3, 2).unwrap();
let input = vec![128u8; 7 * 5 * 3];
let mut output = vec![0u8; 3 * 2 * 3];
resize.resize_rgb8(&input, PixelDescriptor::RGB8_SRGB, 7 * 3, &mut output, 3 * 3).unwrap();
```

This dedicated full-frame path operates on encoded RGB8 samples. It performs
horizontal then vertical cubic passes, using 22-bit coefficients, i64 products,
and a rounded/clipped u8 intermediate. There is no color conversion, alpha
handling, crop or compositing. Input/output row strides are in bytes; padding
is untouched. The horizontal intermediate is capped at 120 MP and allocated
fallibly. It matches every measured Pillow RGB bicubic byte in the dense panel;
exact pins are regenerated by `benchmarks/conventions_2026-10-15/pins.py`.
[Pillow's public resize API](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.resize)
is the oracle; its implementation source was not used.

The separate OpenCV 11-bit feasibility models do not qualify as exact, so no
OpenCV integer compatibility path is installed. Nor does this dedicated RGB
path claim Pillow RGBA's alpha behavior. See the qualification record for all
residuals, versions and reproducible commands.

## Typed Pillow input and validation

`PillowCubicResizer::resize_rgb8` requires an explicit
`PixelDescriptor::RGB8_SRGB` input descriptor in addition to its byte buffer and
stride. RGBA, other channel types and linear transfer descriptors return an
error. The output is encoded RGB8 with the caller's byte stride. The resizer
implements `Debug` and `Clone`.

Very wide kernels retain u32 tap counts instead of truncating at 65,535. This
corrects pre-existing dark output on extreme downscales. Rows beyond that old
limit also use an f64 sum for f32 coefficient renormalization; narrower named
kernels retain their previous arithmetic.
