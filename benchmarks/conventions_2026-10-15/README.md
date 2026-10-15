# Convention qualification — 2026-10-15

Unmet exactness targets: weighted cubic/linear f32 outputs are not bit-exact to
all framework oracles. Exact OpenCV u8 fixed-point cubic/linear paths are not
installed: the independently specified arithmetic models retain nonzero errors.
Nearest and the dedicated Pillow RGB path qualify with zero differences.

The implementation is based on zenresize main e3975fb9; core change 09f6045d.
All code is independently written or uses the existing zenresize kernels.
No OpenCV/Pillow implementation source, TensorFlow source beyond public API docs,
LGPL/GPL port, disassembly or pyiqa was used.

## Oracle panel

Pinned versions are in [requirements.txt](requirements.txt): torch CPU 2.14.1+cpu,
TensorFlow CPU 2.21.0, OpenCV headless 5.0.0.93, Pillow 12.3.0, NumPy 2.5.4.
OpenCV IPP is disabled by environment and API. All frameworks use one CPU thread;
TensorFlow oneDNN optimizations are disabled. Installed under
`~/tmp/aic-variants-scratch/zenresize/venv` on the measured host i265.

Nine explicit size tuples cover identity, odd dimensions, one-pixel input/output
axes and noninteger up/down scales. Another 128 tuples draw each dimension from
1..40 with NumPy seed 20261015. Each has texture, impulse, ramp and fractional-f32
patterns: 548 cases, 227,236 output pixels per convention. Fractional-f32 cases
reuse the texture input for u8 comparisons. A pixel differs if any RGB channel
differs. f32 comparisons are strict, without tolerance; rounded-u8 comparisons
first scale fractional-f32 fixtures by 255, then use ties-to-even and clamp
to [0,255]. Other fixtures already use byte units. Native u8 paths compare
actual bytes.

| Oracle / native convention | Maximum native error | Different pixels % | Maximum rounded-u8 error | Rounded-u8 pixels differing % |
|---|---:|---:|---:|---:|
| cv-nearest-f32 | 0.0 | 0.0 | 0 | 0.0 |
| cv-cubic-f32 | 0.00048828125 | 73.80960763259343 | 1 | 0.040486542625288245 |
| cv-linear-f32 | 0.000396728515625 | 56.78193596085127 | 1 | 0.3995845728669753 |
| torch-cubic-f32 | 0.00079345703125 | 75.7173159182524 | 1 | 0.09769578763928251 |
| torch-cubic-align-f32 | 0.0007476806640625 | 71.10801105458643 | 1 | 0.06513052509285501 |
| torch-linear-f32 | 0.0006866455078125 | 61.553627066133885 | 1 | 0.8594588885563907 |
| torch-linear-align-f32 | 0.0006561279296875 | 59.251174989878365 | 1 | 0.3502966079318418 |
| torch-cubic-aa-f32 | 241.2944679260254 | 75.682110229013 | 242 | 0.07789258744213065 |
| torch-nearest-f32 | 0.0 | 0.0 | 0 | 0.0 |
| torch-nearest-exact-f32 | 0.0 | 0.0 | 0 | 0.0 |
| cv-cubic-u8 | 1.0 | 6.9007551620341845 | 1 | 6.9007551620341845 |
| cv-linear-u8 | 1.0 | 19.51407347427344 | 1 | 19.51407347427344 |
| pillow22-cubic-u8 | 0.0 | 0.0 | 0 | 0.0 |
| pillow-cubic-u8 | 27.0 | 30.157193402453835 | 27 | 30.157193402453835 |
| tf-gaussian-f32 | 0.0001068115234375 | 62.84303543452622 | 1 | 0.08229329859705328 |
| tf-gaussian-aa-f32 | 0.0001373291015625 | 64.51134503335739 | 1 | 0.011001777887306588 |

`summary.json` retains exact values; all per-case arrays and captures remain
outside git with hashes in `validation.meta`. No case is excluded from this table.

## Coordinate and Gaussian probes

Public OpenCV nearest probes cover every 1-D input/output size pair in 1..=64.
Integer rational floor differs at 140 indices; direct f64 input/output at 143;
division by f64 output/input at 121. Multiplication by its f64 reciprocal has
zero differences. Torch legacy nearest requires a direct f32 ratio/product.
Exact half-pixel nearest uses integer arithmetic to avoid cancellation.

TensorFlow Gaussian basis-image probes identify sigma 0.5 and support radius
1.5. Across 7,832 coefficients with AA on/off, reciprocal f32 scale gives zero
support-mask differences and maximum coefficient error 9.309383597333465e-8.
Direct input/output division gives eight support-mask differences. Exact hard-support unit pins include the 18→39 and 14→29 coordinate
boundaries (public TF generator `pin_gaussian_support.py`). The hard
cutoff therefore uses the measured f32 coordinate operations before f64 kernel
normalization. Sources: [tf.image.resize public docs](https://www.tensorflow.org/api_docs/python/tf/image/resize),
and the committed public-operation probe drivers. No TF implementation source
was read.

## Floating arithmetic limits

OpenCV and torch cubic reference outputs themselves differ by maximum
0.001251220703125 (74.87105916316077% of pixels). A single choice of mathematical
kernel/support/mapping cannot be bit-identical to both; framework-specific
floating arithmetic/pass-order controls remain outside these sampling knobs.

Torch antialiased bicubic has a separate one-column compatibility gap in the
public operation: 24×16→1×23 returns channel0=6.050907135009766 for every row.
The separable native result runs from 6.050908088684082 to 247.31326293945312.
The maximum all-channel error is 241.29446411132812. The observation repeats
with contiguous NCHW and channels-last input; the layout probe and arrays are
retained. No geometry-specific workaround was added to the kernel.

## Fixed-point study

The initial 27-case independent model sweep tested 11/14/16/22 coefficient bits
and three rounding choices. Pillow RGB cubic qualified only at 22 bits in that
sweep. Two rounded/clipped integer passes then match every byte in the dense
panel. Production uses i64 accumulators and native row strides. Exact unit pins
come from [Pillow Image.resize](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.resize);
the 16×12→23×9 pin distinguishes the tested 16-bit and 22-bit arithmetic.

OpenCV models use the requested documented scale 2048, an integer horizontal
intermediate and one final shift by 22. They test f32/f64 coefficients, direct
cubic evaluation/partition-of-unity and half-up/ties-even output rounding.
The best cubic model still differs at 8/227236 pixels (0.0035205689239381083%),
maximum 1 u8. Best linear differs at 43196/227236 pixels (19.009311904803816%),
maximum 1 u8. These models do not qualify as exact; no partial OpenCV mode is
installed. Exact coefficient/dispatch/rounding behavior remains unresolved;
no implementation structure was copied to close that gap.

## Reproduce

```sh
export TMPDIR=$HOME/tmp/aic-variants-scratch/zenresize
export CARGO_TARGET_DIR=$HOME/tmp/zenresize-conventions-target
# Serialize every heavy command with flock ~/tmp/i265-heavy.lock and run-heavy --mem 16G.
uv venv --python 3.12 "$TMPDIR/venv"
uv pip install --python "$TMPDIR/venv/bin/python" --torch-backend cpu -r benchmarks/conventions_2026-10-15/requirements.txt
just conventions-build
just conventions-compare "$TMPDIR/venv/bin/python" NEW_LABEL --dense
just conventions-coordinate-probe "$TMPDIR/venv/bin/python"
just conventions-fixed-study "$TMPDIR/venv/bin/python" NEW_LABEL
just conventions-cv-fixed-probe "$TMPDIR/venv/bin/python" NEW_LABEL
just conventions-pins "$TMPDIR/venv/bin/python"
just conventions-ci
just api-doc-check
```

Output labels must be new: captures are protected from accidental replacement.
The complete baseline and post-change default byte captures are each 438,960
bytes and have SHA256
`765c3d0f737f79ee495e17ed76c11134b78d6c330c2a7b6fd1f86ed60bc38442`.
They cover all 31 existing filters, u8 and f32, at four resize tuples.
Existing golden and stress tests also pass. Final gates and resource receipts
are recorded in `validation.meta`. Two no_std unused-import warnings already
exist in the base transfer/working modules; default clippy adds no warnings.
