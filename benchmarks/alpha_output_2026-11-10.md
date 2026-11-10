# Alpha output and convolution audit — 2026-11-10

Code: `95c329d6`, stacked on the reviewed integer-weight fix. Baseline:
`581fc9a90e6f`, before the signed-alpha correction. The subsequent Q14-headroom
fix changes plan construction, not the untimed benchmark's unadjusted tables.
Toolchain, source hashes, lock hash, CPU/OS/RAM and artifact location are in
[the metadata](alpha_output_2026-11-10.json).

## Result and arithmetic

The separate overshoot-repair pass is removed. Conversion uses the signed
filtered alpha until the final byte store, with AVX2/WASM packing, NEON channel
deinterleaving and an exact integer scalar reciprocal. The exhaustive oracle
checks every nonnegative i16 color/alpha pair, signed lanes, mixed opaque
blocks and vector tails. It runs natively, with forced scalar dispatch, and
under WASM SIMD. The dense image oracle covers both formulas, all 9 sample
conversion pairs and all 3 working formats, including strided rows and exact
fullframe/streaming agreement.

The portable integer V-filter exposes eight independent lanes. The scalar
float H-filter exposes four independent channel accumulators; its arithmetic
is bit-identical to the original generic loop. AVX2 integer V filtering uses
16/32/128-tap scratch capacities and reuses row references and paired weights
across 32 components. Short f16 V rows retain four vector accumulators across
taps, avoiding per-tap output reloads/stores. Each component keeps its previous
FMA order; wider f16 rows retain their row-major loop. Existing golden values
are unchanged.

## Selected paired measurements

These are individual measurements, not projections to other sizes or CPUs.
`partial` has alpha 160; `ringing` alternates source alpha 128 and 255.
Dimensions describe square images. Percentages are Tango's printed ratios;
its displayed times are rounded.

| Formula / CPU / tier | Case | Before | After | Change |
|---|---|---:|---:|---:|
| 2021 / 270K Plus / AVX2 | 256→512 partial | 802.7 µs | 625.9 µs | −22.03% |
| 2021 / 270K Plus / AVX2 | 1920→960 ringing | 8.4 ms | 7.7 ms | −8.32% |
| 2021 / 270K Plus / AVX2 | 64→32 linear RGBX | 12.1 µs | 10.9 µs | −10.37% |
| 2021 / N1 / scalar | 512→256 linear RGBX | 33.3 ms | 27.0 ms | −19.13% |

The x86 rows above are from `vertical-four-block-gate.log`; the ARM row is
from `fixed-float-arm.log`. The former run's resource footer is recorded with
its result log; the ARM run reports peak RSS 0.01 GiB, 90 seconds, exit 0.
Full final native matrices and scalar controls are recorded alongside this
report. Failed pilot measurements are retained in the artifact directory;
they are not measurements of the final implementation. In particular, the
f32 fallback controls were retained and optimized when repeat runs confirmed
a slowdown; no expectation or significance threshold was relaxed.

## Coverage scope

PR #28's original f32 constructor is byte-for-byte unchanged. All 15,246
fingerprints of adjusted/unadjusted f32 and unadjusted i16 tables match the
pre-PR source. The constructors and new adjustment helper have 100% LLVM
executable-region coverage before/after, including their closures. See the
[weight audit](weight_coverage_2026-11-10.json).

The portable integer V body has 26/26 regions before and 89/89 after; float H
has 29/29 before and 59/59 after, using the same oracle fixtures on each source
revision. See the [scalar audit](scalar_coverage_2026-11-10.json). LLVM omits
archmage target-feature bodies from this export. Their scratch, block and tail
boundaries are checked by i64 dot-product and explicit-FMA oracles, plus the
unchanged golden outputs. These are region counts, not branch or MC/DC claims.

## Reproduction

Build `benches/alpha_output.rs` in both source exports using their own target
directories. The baseline production source is unchanged; only the identical
benchmark harness is added. The lock files match. Compare the executables:

```sh
just alpha-output-bench compare /path/to/baseline -s 100 --seed 261
ZENRESIZE_MAGIC_2021=1 just alpha-output-bench compare /path/to/baseline -s 100 --seed 261
ZENRESIZE_MAGIC_SCALAR=1 just alpha-output-bench compare /path/to/baseline -s 30 --seed 191
just alpha-output-exact
just magic-paths-test
just magic-wasm-test
just scalar-coverage coverage.json
```

Use the resource wrapper and CPU affinity shown in the committed run logs.
Tango marks a result when z ≥ 2.6 and the mean difference exceeds 0.5%; this is
not a 95% confidence interval or a guarantee of zero slowdown on every CPU.
No Apple M4 or dedicated AVX-512 timing is claimed. AVX-512 output dispatches
to AVX2; the new native kernels here are AVX2 and NEON, with WASM SIMD execution
checked under Wasmtime.
