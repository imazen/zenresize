# Magic Kernel Sharp 2013 and 2021

`Filter::MagicKernelSharp` and `Filter::MagicKernelSharp2021` are opt-in direct resampling kernels. The unqualified name denotes the 2013 formula, which Costella recommends for most applications.
Mitchell remains the comparison baseline; the library default remains Robidoux.

## Provenance and derivation

Written independently from the mathematical definitions on
[John Costella's Magic Kernel page](https://johncostella.com/magic/).
No reference implementation source was consulted or copied.

For t = |x|, the quadratic B-spline m(x) is 3/4 - t² for t < 1/2,
(3/2 - t)² / 2 for 1/2 ≤ t < 3/2, and zero elsewhere.
The 2013 correction is K(x) = [6m(x) - m(x-1) - m(x+1)] / 4.
Expanding gives:

- t < 1/2: 17/16 - 7t²/4
- 1/2 ≤ t < 3/2: (t - 1)(t - 7/4)
- 3/2 ≤ t < 5/2: -(t - 5/2)²/8
- otherwise zero

Production evaluates these quadratics; tests independently compose the spline
and discrete correction. The identity-scale samples are [-1, 0, 34, 0, -1]/32,
so identity resizing is not an exact copy.

The existing weight builder supplies center alignment, widened downsampling
support, truncated-and-renormalized boundaries, and fixed-point quantization.
No separate sharpening pass or SIMD implementation is added. This does not
claim equivalence to a multi-pass reference at boundaries or after rounding.

## Validation

Run `just magic-kernel-test`. Tests cover the formula on a dense grid and near
piece boundaries, partition of unity, a dense f64 image oracle for tiny,
identity, fractional up/downscale and mixed-axis dimensions, and exact constant
preservation through both u8 APIs. The f32 APIs use an f16 cache, so their
comparison with the f64 image oracle allows 0.001 absolute error on the tested
normalized samples. This is not a cross-tier bit-exactness claim.

Measured 2026-11-10 by that test, sampling the unit-step reconstruction at
1/1024-pixel intervals from -4 to 4:

| Filter | Minimum | Maximum |
|---|---:|---:|
| Mitchell | -0.036281173 | 1.036281173 |
| Magic Kernel Sharp 2013 | -0.142857075 | 1.142857075 |

These are unclipped mathematical responses, not a perceptual quality ranking.
Magic has stronger overshoot here. No speed advantage or photographic-quality
advantage has been established.

[resamplescope-rs](https://github.com/imazen/resamplescope-rs) can reconstruct
resizer responses; its documented known-filter matcher includes Mitchell but
not Magic Kernel Sharp. No resamplescope run was used for the numbers above.
Multi-pass variants are outside this addition.

Validation on 2026-11-10: `cargo test --lib --tests`, strict clippy for the
library and new integration test, `cargo test --no-default-features --lib`,
`cargo test --doc`, and format checking passed. Existing ignored doctests
were unchanged. Dedicated Magic cross-tier and photographic comparisons
have not been run.

## Mitchell with resize_sharpen(15)

Measured 2026-11-10 using `just magic-kernel-test`: actual normalized f32
weight tables, every interior output sampling phase and every integer step
boundary under its support. Image-edge-truncated rows are excluded. Values
below are maximum response to an unclipped unit step (1 is the bright plateau).
This differs from the raw-kernel sampling above: sharpening operates on the
sampled weights, not on `InterpolationDetails::filter`.

| Input → output | Mitchell | Mitchell sharpen 15 | Magic Sharp 2013 |
|---|---:|---:|---:|
| 257 → 1021 | 1.036281006 | 1.176470712 | 1.142857015 |
| 257 → 257 | 1.000000000 | 1.000000000 | 1.031250000 |
| 257 → 127 | 1.020198603 | 1.108607737 | 1.101290702 |
| 257 → 63 | 1.018188544 | 1.092833095 | 1.095590078 |

Sharpened Mitchell overshoots more in the tested enlargement and 257→127
reduction, and slightly less at 257→63. Identity-grid Mitchell has no negative
sampled weights to amplify; sharpening does not make it an interpolating kernel.
This comparison measures ringing, not photographic preference or alias rejection.

An existing path discrepancy was also verified: `I16WeightTable::new` does not
apply `lobe_ratio_goal`. At 257→1021, **0 of 1021** Mitchell weight rows changed
when requesting sharpen 15. Consequently this table describes f32-path
sharpening, not current i16-path behavior. The diagnostic reports the number
without asserting that the bug must remain. Fixing it is outside this comparison.

## Magic Kernel Sharp 2021

The independently expanded 2021 formula composes m with
[-1, 6, -35, 204, -35, 6, -1]/144. Production uses five quadratic pieces;
the test oracle sums seven shifted splines. Its support radius is 4.5,
compared with 2.5 for 2013. No implementation source was consulted or copied.

Identity-grid samples are [-1, 0, 0, 0, 1154, 0, 0, 0, -1]/1152.
This is closer to identity than 2013, but not mathematically exact identity.
The wider support can require more convolution taps; speed has not been measured.

Measured 2026-11-10 with the same normalized f32 weight-table procedure:

| Input → output | Mitchell sharpen 15 | Magic 2013 | Magic 2021 |
|---|---:|---:|---:|
| 257 → 1021 | 1.176470712 | 1.142857015 | 1.104601771 |
| 257 → 257 | 1.000000000 | 1.031250000 | 1.000868109 |
| 257 → 127 | 1.108607737 | 1.101290702 | 1.065072305 |
| 257 → 63 | 1.092833095 | 1.095590078 | 1.059846469 |

2021 has less step overshoot in these cases. This does not establish a
photographic quality ranking. Both Magic variants now run through the dense
image oracle and integer constant-preservation tests. The 2021 formula test
covers every piece boundary, dense sampling, and partition of unity.

Both production functions in `src/filter.rs` explain their derivations,
including the choice a = 1/4 for the 2013 correction and the 2021 residual
cancellation. These names do not change the library default (Robidoux).

## Mitchell with resize_sharpen(12)

Measured 2026-11-10 with `just magic-kernel-test`, using the same interior
normalized f32 weight-table step-response procedure. Maximum response:

| Input → output | Mitchell sharpen 12 | Mitchell sharpen 15 | MagicKernelSharp | MagicKernelSharp2021 |
|---|---:|---:|---:|---:|
| 257 → 1021 | 1.136363745 | 1.176470712 | 1.142857015 | 1.104601771 |
| 257 → 257 | 1.000000000 | 1.000000000 | 1.031250000 | 1.000868109 |
| 257 → 127 | 1.083924182 | 1.108607737 | 1.101290702 | 1.065072305 |
| 257 → 63 | 1.071734626 | 1.092833095 | 1.095590078 | 1.059846469 |

Sharpen 12 has less overshoot than 2013 and more than 2021 in all three
nonidentity cases tested. This is a ringing measurement, not a photographic
quality ranking. The i16 diagnostic also found 0/1021 changed weight rows
for sharpen 12, confirming that the existing ignored-adjustment bug applies
to this setting as well.

## Integer sharpening fix

The stacked fix shares lobe adjustment math between f32 and i16 builders.
The earlier 0/1021 changed-row observations above describe the pre-fix state.
Tests now require sharpen 12 and 15 to change i16 weights and compare adjusted
coefficients to f32 within one Q14 step while checking exact weight sums.
Pixel tests exercise I16Srgb and I16Linear through both resize APIs.
Requested lobe ratios near 0.5 select f32: positive coefficients can reach 2,
which is not representable in signed Q14. Direct I16WeightTable construction
rejects an unrepresentable adjusted coefficient rather than saturating it.

### Weight-construction review

The PR #28 follow-up keeps `F32WeightTable`'s original implementation
byte-for-byte unchanged and adds integer-side adjustment without moving the
f32 normalization code. No additional row allocation is required.

The executable-region coverage audit is recorded in
[`benchmarks/weight_coverage_2026-11-10.json`](../benchmarks/weight_coverage_2026-11-10.json).
Both constructors have 100% region coverage before and after; the new integer
adjustment helper also has 100%. This is LLVM region coverage, not a branch
or MC/DC claim. The audit exercises narrow support, empty rows, all-negative
flattening, positive-only rows, odd/even padding, and unrepresentable adjusted
coefficients, in addition to the normal filter matrix.

`just weight-fingerprint OUTPUT` records all table fields for all 33 filters,
42 dimension pairs and 10 controls. The 15,246 f32/unadjusted-i16 table
fingerprints match the pre-PR source byte-for-byte. The report records their
SHA-256 and source identities. Requested integer adjustments are tested
separately against f32 coefficients within one Q14 step.

The ratio cutoff reserves one Q14 step for largest-remainder correction:
`r < 1 - 16384/32766`. Direct integer table construction checks the rounded
coefficient with that same one-step headroom. A ratio just below 0.5 can
otherwise round to the signed limit before correction; regression tests
exercise this boundary as well as the larger ratios.

## Pixel-path coverage and alpha overshoot

`just magic-paths-test` runs both Magic formulas against an independent dense
f64 convolution, with analytic sRGB decoding/encoding and premultiply math.
It checks 9/9 u8/u16/f32 sample-type pairs, strided source rows with sentinels,
RGB three-channel input, and RGBX linear-light integer processing, at three
up/downscale sizes. All 3/3 working formats (I16Srgb, I16Linear, F32) are
asserted explicitly, including the existing 2021 >14-tap linear precision
fallback. Fullframe and streaming output must agree exactly for each case.

The oracle permits finite-precision differences: five u8 codes for integer
straight-alpha output with source alpha >=0.5, two for opaque integer output,
one for f32-to-u8 output, and 0.001 for other f32-path output. These bounds
account for premultiplication, intermediate rounding and f16 input caching;
this is not a bit-exact f64-reference claim.

The matrix exposed a real integer output bug: negative lobes can push filtered
alpha above 255. Clamping that alpha before unpremultiplication brightened
RGB. Both direct and paired output rows now retain the unclamped signed alpha
as the divisor and clamp the final result. In-range alpha keeps the existing
SIMD implementation. The varying-alpha oracle fails without this repair.

Native dispatch and forced scalar both pass locally on x86-64. CI runs the
same oracle with native and forced-scalar dispatch on supported integration
lanes. Dedicated AVX-512, NEON and WASM execution has not been measured locally;
Windows ARM currently runs library tests only under the existing CI policy.
Mask/compositing behavior has existing generic tests; this matrix does not
add a separate Magic-specific mask/compositing oracle.
