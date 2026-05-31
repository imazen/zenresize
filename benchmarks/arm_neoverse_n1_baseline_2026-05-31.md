# zenresize — ARM Neoverse-N1 baseline + f16-decode SIMD optimization (2026-05-31)

Box: Hetzner CAX21, Ampere Altra **Neoverse-N1**, 4 cores, 8 GB, Ubuntu.
Toolchain: rustc 1.96, `RUSTFLAGS=-C target-cpu=neoverse-n1` (pinned) unless
noted. Harness: `cargo bench --bench competitors_zen` (zenbench, interleaved,
resource-gated). Cross-codec competitors: pic-scale (scalar), pic-scale-safe,
fast_image_resize (fir), resize crate.

x86 cross-check: AMD Ryzen 9 7950X, runtime dispatch (`RUSTFLAGS` unset).

## 1. Baseline (Neoverse-N1, before optimization)

`1024_2x_down`, RGBA, Lanczos. Throughput = input MiB/s.

| Path | zenresize | best competitor | gap |
|---|--:|--:|--:|
| sRGB (RGBA8, `.srgb()`) | 124 | ps_scalar 507 | 4.1× slower |
| sRGB i16 (RGBX8) | 115 | — | (integer path) |
| linear (RGBA8, `.linear()`) | 46.9 | fir 113 | 2.4× slower |
| linear i16 (RGBX8) | 122 | fir 113 | **wins** |
| **f32** (RGBAF32) | **201** | **fir 677** | **3.4× slower** |

The **integer (i16) paths are competitive** on ARM (linear_i16 122 ties/beats
fir 113). The **f32 / linear-f32 paths were 2.4–3.4× slower** than fir.

### x86 cross-check (7950X) — the smoking gun

Same `1024_2x_down`:

| Path | x86 zenresize | ARM zenresize | x86/ARM ratio |
|---|--:|--:|--:|
| f32 | 4.19 GiB/s | 201 MiB/s | **~21×** |
| linear | 887 MiB/s | 46.9 MiB/s | **~19×** |
| sRGB | 1.44 GiB/s | 124 MiB/s | ~12× |

On x86 zenresize_f32 **wins** (4.19 G vs fir 2.66 G) and linear **wins** (887 vs
fir 526). The ~21× x86/ARM ratio on f32 (far beyond the ~4× clock/width gap)
flagged a pathological ARM-only cost in the f32 path.

## 2. Profiling (samply + perf, `profile_f32`, 1024→512 f32)

| Function | self-time |
|---|--:|
| `simd::filter_v_row_f16` | **73.2 %** |
| `simd::f32_to_f16_row` | 17.4 % |
| `simd::filter_h_row_f32` (actual convolution) | 5.0 % |
| memcpy | 2.7 % |

**~90 % of the f32 path was scalar f16↔f32 software conversion.** The f32 path
stores its V-cache as f16 (a memory optimization). On x86 the f16↔f32 conversion
uses the F16C hardware intrinsics (`vcvtps2ph`/`vcvtph2ps`) and is nearly free
(`x86.rs` line 2359: "F16C is guaranteed by X64V3Token"). On ARM/WASM the
magetypes generic SIMD API exposes no f16 vector type, so the kernels fell back
to the scalar `f16_to_f32_soft` — a branchy bit-twiddle (subnormal normalization
loop + Inf/NaN branches) that LLVM cannot auto-vectorize — invoked per element
per tap inside the V-filter.

## 3. Optimization — branchless SIMD f16→f32 (won, shipped)

Hypothesis: vectorize the f16→f32 decode within the existing magetypes generic
SIMD API (NEON/WASM128), bit-identically, to remove the scalar bottleneck.

`magetypes` 0.9.20 has **no f16 vector type** and no widening multiply, so a
hardware-`fcvt` path is unreachable in safe stable Rust (`core::f16` is still
nightly-gated; `#![forbid(unsafe_code)]` bans intrinsics). Instead used Fabian
Giesen's **magic-multiply** method expressed entirely in `i32x4`/`f32x4`:
shift the 15 magnitude bits into the f32 exponent/mantissa field, reinterpret,
multiply by 2^112 (rescales the exponent, denormalizes subnormals in one step),
branchless Inf/NaN select, OR in the sign.

**Bit-exactness verified exhaustively over all 65 536 f16 values** vs the scalar
`f16_to_f32_soft` (0 mismatches; NaN inputs map to NaN, never occur in resize
intermediates). Applied to `filter_v_row_f16_impl` (16-lane, 4 f32x4
accumulators) and `f16_to_f32_row_impl`. `f32_to_f16_row` (the push-side
direction) was left scalar — see §5.

### Measured win (Neoverse-N1, `competitors_zen`, MiB/s)

| Scenario | path | before | after | speedup |
|---|---|--:|--:|--:|
| 1024_2x_down | f32 | 201 | 320 | **+59 %** |
| 4k_2x_down | f32 | 199 | 313 | **+57 %** |
| 576_2x_up | f32 | 90.1 | 133 | **+48 %** |
| 1024_2x_down | linear | 46.9 | 71.4 | **+52 %** |
| 4k_2x_down | linear | 47.0 | 72.1 | **+53 %** |
| 576_2x_up | linear | 18.9 | 25.4 | **+34 %** |

### No-regression across downscale ratio (standalone best-of-15, low noise)

The `competitors_zen` 4k_10x cell had high variance (5–8 rounds, ±6 % MAD) and
showed a spurious −9 %; a clean best-of-15 standalone driver across the ratio
sweep (same bench binary, same scenario) shows the SIMD version **wins at every
ratio**, including extreme downscale:

| Scenario (2048→, f32) | ~taps | pristine | SIMD | speedup |
|---|--:|--:|--:|--:|
| 2x | ~12 | 309.0 ms | 197.7 ms | **+56 %** |
| 3x | ~18 | 303.4 ms | 191.3 ms | **+59 %** |
| 4x | ~24 | 299.2 ms | 200.2 ms | **+49 %** |
| 6x | ~36 | 296.0 ms | 210.8 ms | **+40 %** |
| 10x (4000→400) | ~60 | 839.4 ms | 804.8 ms | **+4.3 %** |

A 2-accumulator (8-lane) variant was tried and **regressed** the high-tap case
(993.9 ms at 10x vs 4-acc 804.8 ms) — 4 accumulators is the right unroll.

### Paths affected

The win lands on the **F32 streaming path**, which uses the f16 V-cache:
f32 I/O (`resize_f32`), u16 resize, BT.709/PQ/HLG transfers, RGBA needing
premultiply/composite, heavy downscale (>14–50 taps), 3-channel, and masks. The
`I16Srgb` / `I16Linear` integer fast paths (sRGB/linear u8 RGBA at low downscale,
no premul) don't touch f16 and are unchanged.

## 4. Correctness

- SIMD f16→f32 is **bit-identical** to the scalar path (exhaustive 65 536-value
  proof). Output is byte-for-byte unchanged vs prior ARM behavior.
- Full suite on ARM: **all 320 tests pass**, except 4 `golden_outputs` cases
  (`golden_path0_with_alpha`, `golden_path3_f32_native_{down,up}scale`,
  `golden_u16_rgba_downscale`). **These 4 fail identically on the pristine,
  unmodified code on ARM** (verified: same hashes with and without this change)
  — a **pre-existing** divergence: the committed golden `.raw` files were
  generated on x86 (hardware F16C) and differ from ARM's soft-f16 output. This
  optimization does not introduce or worsen it. (The cross-platform f16
  determinism question is separate and out of scope here.)

## 5. Next hypotheses (not yet shipped)

1. **Vectorize the push-side `f32_to_f16_row` (now ~27 % of the f32 path).**
   Giesen's branchless f32→f16-RTNE is SIMD-friendly but its rounding for
   **f32-subnormal inputs** differs from `f32_to_f16_soft` (which flushes f32
   subnormals to f16 zero). Resize intermediates are premultiplied linear values
   in ~[0,1], far above f32-subnormal range, so the divergence is unreachable in
   practice — but a bit-identical guarantee needs the flush-to-zero branch
   matched first. Stacking this should push the f32 path past +80 %.
2. **f32 V-cache (eliminate f16 entirely on this path).** Would remove both
   conversions and route through the already-vectorized `filter_v_row_f32`,
   plus kill the redundant cross-output-row re-decode that dominates extreme
   downscale. This is a **precision-changing** (f16 quantization removed →
   strictly more accurate) and **memory-doubling** (ring buffer) tradeoff that
   alters golden f32 output bytes — a design decision for the maintainer, not a
   drop-in.
3. **Cross-platform f16 determinism.** The 4 golden mismatches are x86-F16C vs
   ARM-soft. Making f16 conversion bit-identical across x86/ARM/WASM (or
   regenerating goldens per-arch) would close the pre-existing gap.

## Files

- Optimization: `src/simd/wide_kernels.rs` — `f16x4_to_f32x4`,
  `load_f16x4_as_f32x4`, `filter_v_row_f16_impl`, `f16_to_f32_row_impl`.
- Raw bench logs: `zenresize-baseline-neoverse-n1.txt` (before),
  `zenresize-after-f16simd-neon-n1.txt` (after) on the box at
  `~/work/zen-arm-bench/`.
