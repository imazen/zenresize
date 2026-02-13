# Context Handoff: hoisted-bounds → as_chunks Migration

## What Was Done

Branch `hoisted-bounds` replaces all 36 `hoisted_bounds` guard usages in `src/simd/x86.rs`
with `as_chunks`/`as_chunks_mut` + `safe_unaligned_simd`, then removes the `hoisted-bounds`
dependency entirely.

### Commits (oldest first)

1. `c8cba29` — Conversion kernels (u8↔f32, premul/unpremul): 6 functions rewritten with
   `as_chunks` iterators + safe_unaligned_simd load/store
2. `63580f5` — V f32 kernel (`filter_v_row_f32_v3`): `as_chunks_mut::<32>()` for 4x-unrolled
   blocks
3. `e38f903` — V i16 kernels: pre-chunk rows into `&[[u8; 16]]` stack arrays / Vec
4. `2b35abc` — H kernels: weight table via `as_chunks::<16>()`, input via `try_from` slice
5. `eb9b4ca` — Add `unsafe_kernels` feature flag with `load_si128_at()` helper using
   `#[archmage::rite]`, remove `hoisted-bounds` dep, bump MSRV to 1.88

### Key Design Decisions

- **`load_si128_at()`** uses `#[archmage::rite]` (not manual `#[target_feature]`) so it
  gets proper `#[target_feature(enable = "...")]` + `#[inline]` from the archmage token
  system. Takes `_token: X64V3Token` as first arg.

- **`unsafe_kernels` feature flag** only controls the 6 H kernel inner-loop loads where
  `left` comes from the weight table (data-dependent, not chunk-aligned). All other SIMD
  operations are fully safe with zero overhead.

- **`#![cfg_attr(feature = "unsafe_kernels", allow(unsafe_code))]`** — the module-level
  allow is now conditional. Without the feature, the only `#[allow(unsafe_code)]` is on
  `load_si128_at` itself (which is dead code in that cfg path but the attribute is needed
  for the cfg'd unsafe block).

- **MSRV bumped to 1.88** because `as_chunks` was stabilized in 1.88.0.

## Performance Summary

| Benchmark (zenresize_srgb) | Baseline | Safe mode | unsafe_kernels |
|----------------------------|----------|-----------|----------------|
| downscale 50% clic 1024    | 1.98ms   | 2.42ms    | 2.18ms         |
| downscale 50% gb82 576     | 700µs    | 820µs     | 744µs          |
| downscale 25% clic 1024    | 1.44ms   | 1.83ms    | 1.59ms         |
| downscale 25% gb82 576     | 583µs    | 636µs     | 589µs          |
| upscale 200% gb82 576      | 3.52ms   | 3.86ms    | 3.33ms         |

- Safe mode: ~15-27% sRGB regression from H kernel bounds checks
- unsafe_kernels: recovers most of it, ~1-10% residual from V/conversion changes
- Linear path: no significant change (uses f32 kernels, not i16)

## What's Left

1. **Residual ~10% gap in some sRGB downscale benchmarks** — likely from V i16 kernel
   or conversion kernel changes. Could investigate if these are real or benchmark noise
   by running with more iterations. The upscale case actually improved, suggesting noise.

2. **Merge strategy** — this branch has 11 commits. Consider squash-merging to main or
   rebasing into fewer logical commits.

3. **Consider raising MSRV further** — currently 1.88 for `as_chunks`. If bumping to 1.93
   is acceptable, could use `as_mut_array()` and other newer APIs.

4. **Dead code warnings** — `filter_v_u8_i16` and `filter_v_u8_i16_scalar` are unused.
   Pre-existing, not related to this change.
