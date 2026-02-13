# zenresize TODO

## Replace guards with as_chunks + safe_unaligned_simd — DONE

All 36 hoisted-bounds guards in x86.rs have been replaced:

- **25 guards (Pattern A/B4/C/E):** Replaced with `as_chunks`/`as_chunks_mut` + `safe_unaligned_simd`.
  Bounds proven at compile time by Rust's slice chunking. Zero overhead.

- **6 sites (H kernel inner loops):** Data-dependent offsets from weight table. Use
  `load_si128_at()` with runtime bounds checking. `unsafe_kernels` switches to unchecked
  pointer loads.

- **V kernel inner loops:** Use `load_v_chunk()`/iterator patterns. `unsafe_kernels` switches
  to `get_unchecked` for both row and chunk indexing.

The `hoisted-bounds` dependency has been removed entirely.

### Performance summary

| Mode | sRGB downscale 50% 1024sq | vs baseline |
|------|--------------------------|-------------|
| Safe (default) | ~2.37ms | +20% |
| unsafe_kernels | ~2.05ms | +3% |
| Baseline (main) | ~1.98ms | — |

H kernel bounds checks are well-predicted branches; restructuring attempts (window
sub-slicing, assert hints) added more overhead than they saved. The ~15% H kernel
overhead is inherent to safe bounds checking.

## Pending

### NEON and WASM128 via `wide` crate — DONE

All 12 kernel entry points implemented using `wide` crate types (f32x4, i16x8,
i32x4, u8x16) that compile to NEON on AArch64 and WASM SIMD128 on wasm32:

- `wide_kernels.rs`: Shared portable implementations, `#[inline(always)]`
- `neon.rs`: NeonToken wrappers → wide_kernels (replaces raw NEON intrinsics)
- `wasm128.rs`: Wasm128Token wrappers → wide_kernels (new)
- archmage `incant!` dispatch updated for wasm128 tier

Key SIMD patterns: f32x4 FMA for float path, i16x8.mul_widen→i32x8 for
integer V kernel, u8x16.narrow_i16x8 for pack-back. Cross-compiled clean
for aarch64-unknown-linux-gnu and wasm32-unknown-unknown.

### archmage 0.6.1 testing helpers — DONE

Added `tests/dispatch_tiers.rs` using `archmage::testing::for_each_token_permutation`
to verify all dispatch tiers produce consistent results (constant color, gradient,
upscale). Run with `--test-threads=1` for accurate token disabling.

### bytemuck chunks
Replace manual `as_chunks` patterns with bytemuck's safe transmute/cast where
applicable. May simplify some of the chunk-based SIMD load patterns.

### AVX-512 support
Add AVX-512 kernel variants behind the existing `avx512` feature flag.
Requires archmage AVX-512 token support.
