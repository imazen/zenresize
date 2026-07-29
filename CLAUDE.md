# zenresize — Project Notes for Claude

## Architecture

### Resize Pipelines

Two resize APIs with different pipeline architectures:

**Fullframe `Resizer` (H-first):** `src/resize.rs`
- Processes full image at once: H-filter all rows → intermediate buffer → V-filter all rows
- Uses i16 integer path for sRGB 4ch, f32 for linear/other layouts
- Batch kernels: `filter_h_u8_i16_4rows`, `filter_v_all_u8_i16`, `filter_v_all_i16_i16`

**Streaming `StreamingResize` (V-first):** `src/streaming.rs`
- Row-at-a-time: push input rows, pull output rows
- V-first pipeline: `push_row` caches rows in ring buffer, output production runs V-filter → H-filter → composite → unpremul
- Ring buffer is `v_weights.max_taps + 2` slots, each `in_width * ch + h_padding` wide
- H-filter runs only `out_height` times (once per output row), not `in_height` times
- Three internal paths (selected automatically in `new_inner()`):
  - **F32**: Full f32 pipeline with linearization/premul. Used for compositing, 3ch, f32 I/O, u16 I/O.
  - **I16Srgb**: u8 ring buffer → `filter_v_row_u8_i16` → `filter_h_u8_i16` → u8. For sRGB 4ch without linearization. ~2× faster.
  - **I16Linear**: i16 ring buffer → `filter_v_row_i16` → `filter_h_i16_i16` → `linear_i12_to_srgb_u8`. For Rgbx 4ch with linearization, no premul.

### SIMD Kernels

Located in `src/simd/`:
- `x86.rs` — AVX2+FMA kernels via archmage `X64V3Token`
- `scalar.rs` — Portable fallback
- `wide_kernels.rs` — Portable SIMD via `magetypes` generic types (`#[magetypes(neon, wasm128)]`)
- `neon.rs`, `wasm128.rs` — Token wrappers calling `_impl_neon`/`_impl_wasm128` from wide_kernels
- `mod.rs` — Dispatch via `archmage::incant!`

### Compositing / Blending

Blend math lives in `~/work/zen/zenblend/` (MIT OR Apache-2.0). zenresize re-exports
`BlendMode` from zenblend and delegates all blend operations to it. Both `StreamingResize`
and `Resizer` expose a `with_blend_mode()` builder. The `pretty-safe` feature replaces
bounds-checked indexing with `get_unchecked` in SIMD kernels where bounds are proven by
prior guards — the default build is `#![forbid(unsafe_code)]`.

**H-filter `filter_h_4ch` (f32 path):** AVX2 256-bit, 8 taps per iteration with 4 accumulators. Uses `vpermps` for weight broadcasting — loads 8 weights at once, permutes to create per-tap lane broadcasts. SSE 128-bit remainder for 0-7 leftover taps.

**H-filter `filter_h_u8_4ch` (i16 path):** AVX2 256-bit using `madd_epi16` for paired tap accumulation with pre-expanded weight tables.

**V-filter `filter_v_row_f32`:** AVX2 256-bit, row-major accumulation with 4x8-float ILP blocks.

### Weight Tables

`src/weights.rs`:
- `F32WeightTable` — flat layout, `max_taps` zero-padded per output pixel
- `I16WeightTable` — flat layout + pre-expanded 4ch weights for `madd_epi16`
- `weights_padded(out_x)` returns `max_taps` elements (zero-padded)
- `weights(out_x)` returns only actual (non-zero) taps

### Safe Indexing

`src/proven.rs` — `idx()`, `idx_mut()`, `sub()` with debug_assert bounds checks. When `pretty-safe` feature is on, these use `get_unchecked`.

### Golden Tests

`tests/golden_outputs.rs` — Exact-match checksums for all paths. Stored in `test_outputs/` (gitignored). Must regenerate after any change to FMA accumulation order or numerical behavior in f32/4ch kernels.

## Build Rules

**NEVER compile with `-Ctarget-cpu=native` except for diagnostics (e.g., `cargo asm`).** We will never deploy "native" binaries. All production and benchmark builds must use dynamic dispatch via `incant!`/`arcane`/`rite`. Performance must be verified on every token tier using `dangerously_disable_token_process_wide(true)` — not just the highest tier the machine supports. Use `testable_dispatch` feature in dev-dependencies if needed to override compile-time feature guarantees.

## TODO

### Compositing composability [DONE]
Blend math extracted into `~/work/zen/zenblend/` (MIT OR Apache-2.0).
zenresize and zenpipe both delegate to zenblend for all blend operations.
`BlendMode` enum re-exported from zenresize, `with_blend_mode()` builder on both
`StreamingResize` and `Resizer`. SrcOver has AVX2+FMA SIMD kernel (2 pixels/iter).
Remaining: per-row i16→f32→composite→u8 in i16 paths (avoids full f32 pipeline).

### zenblend blend modes + mask pipeline [DONE]
**Phase 1:** 9 new separable blend modes: LinearBurn, LinearDodge, VividLight,
LinearLight, PinLight, HardMix, Divide, Subtract, Plus. Total: 31 modes.
Plus operates on premultiplied values directly (SVG/CSS semantics).

**Phase 2:** `mask_row_rgb` (RGB×mask, alpha untouched) and `lerp_row`
(per-pixel interpolation between two rows). Full SIMD: AVX2+FMA (2 px/iter),
wide f32x4 (NEON/WASM128), scalar.

**Phase 3:** `LinearGradientMask` and `RadialGradientMask` implementing MaskSource.
No new dependencies. Pure math with MaskFill hints for uniform-row optimization.

**Phase 4:** `StreamingResize::with_mask()` builder. Pipeline: resize → mask
→ composite → unpremultiply. Mask before composite so rounded corners + white
background → JPEG gets white corners (not transparent-over-black). Forces f32
path when mask present. Re-exports MaskSource, MaskFill, RoundedRectMask,
LinearGradientMask, RadialGradientMask from zenblend.

**Phase 5:** `zenpipe::sources::MaskTransformSource` for standalone no-resize masking.
Requires RGBAF32_LINEAR_PREMUL upstream.

### Native AVX-512 for remaining hot-path kernels
`filter_v_row_i16` has a native AVX-512 kernel (32 i16/iter). Still delegating to AVX2:
- `filter_h_u8_to_i16` — H-filter for sRGB i16 path
- `filter_h_i16_i16` — H-filter for linear i16 path
- `filter_v_row_f16` / `filter_h_row_f32_to_f16` — f32 path

## aarch64 / NEON (2026-07-28 sweep)

**NEON is BASELINE on aarch64.** `#[target_feature(enable="neon")]` is a no-op,
so the "scalar" tier is autovectorized too. A 1.00x NEON-vs-forced-scalar
result therefore means "LLVM already did it", NOT "the SIMD path is missing".
Hand-written intrinsics only pay where the portable form *structurally* cannot
vectorize. Two proven cases, and one proven non-case:

- **PAYS — runtime-variable divisor.** `unpremultiply_u8_row` divides by the
  pixel's own alpha; there is no SIMD integer divide, so it sat at 1.00x /
  2.4 GiB/s while `premultiply_u8_row` (divisor = the constant 255, which LLVM
  turns into a multiply-shift) ran at 14.5 GiB/s. `vld4q_u8` + f32 divide:
  **2.7x**.
- **PAYS — interleaved layout.** `unpremultiply_alpha_row`: `vld4q_f32`
  deinterleaves RGBA so alpha is already a vector. **1.22x**.
- **DOES NOT PAY — `premultiply_alpha_row`.** A `vld4q_f32` version was written
  and measured at 1.00x both before and after; it was reverted. Don't re-attempt
  it; the function carries a comment saying so.

A portable f32-divide rewrite of `unpremultiply_u8_row` inside `wide_kernels`
was tried FIRST and was **worse** (4.1us vs 3.0us): marshalling one pixel at a
time through a vector costs more than the integer divides it removes. The win
came from `vld4q`, i.e. from the deinterleave, not from the divide.

### benches/kernel_tiers.rs — read this before trusting a kernel number

End-to-end resize numbers (1.56x sRGB / 4.29x linear) **cannot** show a kernel
that is slower than its own scalar fallback. `benches/kernel_tiers.rs` compares
each kernel against its forced-scalar tier; `simd::__bench_kernels` is the
`#[doc(hidden)]` forwarder module that lets it reach `pub(crate)` kernels (25
covered — it was 6, and the 19 added were the H/V convolutions that actually
dominate a resize).

Two traps, both hit during that sweep:

1. **In-place kernels must build their buffer in `with_input`, not in the timed
   body.** A 30 KB clone dominates a sub-microsecond kernel. With the clone
   timed, `premultiply_alpha_row` looked like a 0.92x regression (CI [-4.3%,
   -2.8%]) — it is actually 1.00x. That false regression was reported in a
   commit message before it was caught.
2. **Only the IN-RUN paired ratio is trustworthy.** Absolute timings on this
   machine drift up to 2x between runs (the same unchanged code measured 551ns
   and 1.2us). Never claim a cross-run absolute speedup; quote the ratio and its
   CI from a single run, and reproduce it.

### H kernels over-read; callers must pad

The 4-channel H filters loop to `weights.max_taps` and rely on zero-padded
weights, so they read up to `groups4 * 16` elements past the row end. This is a
CONTRACT, not a bug — `streaming.rs` allocates `in_row_len + h_padding_i16` and
zeroes the pad. Any new caller (or bench) must do the same or it will panic.

### filter_h_row_f32_to_f16 was scalar [FIXED]

Its convolution ran fully scalar — a `/` and a `%` per output element — while
its sibling `filter_h_4ch` vectorized over the RGBA pixel, so the NEON tier did
strictly more work than its own scalar fallback (0.94x). Now vectorized the same
way: **4.55x**. For `channels == 4` this is exact, because a flat chunk of 4 IS
one output pixel; other channel counts still take the original path.

## Investigation Notes

### i16 accuracy gap [FIXED]
Intermediate clamping removed from all i16 kernels. Max diff 52→1.
**Measured max diff (u8 output):**
- sRGB-i16 path: 2-6 (downscale), 25 (upscale)
- linear-i16 path: 6-43 (downscale), 49-52 (heavy downscale/upscale)
- f32 path: 1 (near-identical)

**Root cause:** Intermediate clamping after the first filter step. Lanczos ringing pushes intermediates outside [0, 4095] (linear) or [0, 255] (sRGB). Clamping these destroys overshoot information the second filter step needs to cancel ringing. H-first and V-first clamp at different structural positions, yielding divergent results. The f32 path doesn't clamp intermediates, so it's unaffected (max diff 1).

**Fix options (in order of increasing correctness):**
1. **Don't clamp intermediates** — remove `.clamp(0, 4095)` / `.clamp(0, 255)` after the first filter step. Store full i16 range. Ringing values like [-200, 4300] (i12) or [-50, 300] (u8) fit in i16. Only clamp at final output. Reduces diff from 43-52 to ±1-2 (rounding only). Cheapest change. For sRGB path: intermediate changes from u8 to i16 (doubles memory); AVX2 kernel changes `packus_epi16` (unsigned sat) to `packs_epi32` (signed i16 sat). For linear path: just remove the `.clamp(0, 4095)`.
2. **i32 intermediate, single rounding** — accumulate first filter step into i32, store in ring buffer as i32. Second filter reads i32, accumulates in i64 or i32, rounds once at output. H-first and V-first become **bit-identical** (integer arithmetic is associative, single rounding point). Ring buffer cost: 4K Lanczos3 goes from 225KB (u8) to 900KB (i32). Still tiny vs fullframe's 31MB.
3. **f32 everywhere** — already implemented, max diff 1. But slower than i16 for 4ch.

## Known Bugs

(none currently)
