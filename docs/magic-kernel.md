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
Requested lobe ratios >= 0.5 select f32: positive coefficients can reach 2,
which is not representable in signed Q14. Direct I16WeightTable construction
rejects an unrepresentable adjusted coefficient rather than saturating it.
