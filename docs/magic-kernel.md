# Magic Kernel Sharp 2013

`Filter::MagicKernelSharp2013` is an opt-in direct resampling kernel. Mitchell
remains the comparison baseline; the library default remains Robidoux.

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
2021 and multi-pass variants are outside this addition.

Validation on 2026-11-10: `cargo test --lib --tests`, strict clippy for the
library and new integration test, `cargo test --no-default-features --lib`,
`cargo test --doc`, and format checking passed. Existing ignored doctests
were unchanged. Dedicated Magic cross-tier and photographic comparisons
have not been run.
