# Alpha output and convolution audit — 2026-11-10

Code: `a5d1d442`, stacked on the reviewed integer-weight fix. Baseline:
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
| 2021 / 270K Plus / AVX2 | 1920→960 ringing | 8.5 ms | 7.5 ms | −11.36% |
| 2021 / 270K Plus / AVX2 | 64→32 linear RGBX | 12.6 µs | 10.8 µs | −14.26% |
| 2021 / N1 / NEON | 1920→960 ringing | 157.3 ms | 123.7 ms | −21.38% |
| 2021 / N1 / scalar | 1920→960 ringing | 206.5 ms | 134.3 ms | −34.97% |

All 280 final printed ratios are nonpositive. Every case is retained in the
[native x86 CSV](alpha_output_native_2026-11-10.csv),
[scalar x86 CSV](alpha_output_scalar_2026-11-10.csv), and
[final ARM CSV](alpha_output_arm_final_2026-11-10.csv).
Both formulas run at seven square-image sizes, with opaque, partial-alpha,
ringing-alpha, RGBX and linear-RGBX controls, in native and forced-scalar
processes on each CPU. Forced scalar selects portable entry points;
compiler autovectorization remains enabled, including baseline NEON on ARM.

Combined x86 test/build/benchmark job: `run-heavy: done rc=0 176s | peak-RSS
0.43GiB | min-avail 24920MiB | peak-load 1.48`.
ARM full-matrix job: `run-heavy: done rc=0 406s | peak-RSS 0.06GiB |
min-avail 7098MiB | peak-load 2.77`.
These are job peaks, not per-image allocation measurements.
Failed pilot measurements are retained in the artifact directory and are not
measurements of the final implementation. No expectation or significance
threshold was relaxed.

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

The signed-i12 lookup follow-up keeps eight independent lookups in flight.
The compiler unrolls this loop on the measured x86 build; packed min/max is
not claimed. All 65,536 signed inputs match the original formula, with 100%
region coverage before (16/16) and after (40/40 plus 2/2 closure regions).
See [the LUT audit](lut_coverage_2026-11-10.json).
