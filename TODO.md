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

Paired bench: sRGB downscale 50% 1024sq → 512sq:

| Library | Time (ms) | vs zenresize |
|---------|-----------|-------------|
| **zenresize (safe)** | **1.31** | — |
| pic_scale | 1.31 | tied |
| fast_image_resize | 1.55+ | 18% slower |

Window-slicing eliminates ALL inner-loop bounds checks without unsafe.
Pre-slice input per output pixel, `as_chunks::<16>()` gives exact chunk count,
zip iterator proves bounds at compile time. `unsafe_kernels` feature removed.

## V-first streaming pipeline — DONE

Switched `StreamingResize` from H-first to V-first pipeline order:
- **Before**: `push_row` ran H-filter on every input row, cached `out_width`-wide results
- **After**: `push_row` caches `in_width`-wide linearized/premultiplied rows; output production runs V-filter → H-filter → composite → unpremul

For 2x downscale, H-filter runs 512 times (output rows) instead of 1024 times (input rows).
Streaming linear path: 30% faster than fullframe (4.53ms vs 5.90ms, 1024→512 Lanczos RGBA).

## AVX2 256-bit horizontal f32 filter — DONE

Widened `filter_h_4ch` from SSE 128-bit (4 taps/iteration) to AVX2 256-bit (8 taps/iteration):
- Each 256-bit accumulator processes 2 taps: lower 128 = pixel[t] * w[t], upper 128 = pixel[t+1] * w[t+1]
- Uses `vpermps` (`_mm256_permutevar8x32_ps`) for weight broadcasting: loads 8 weights at once, permutes to create per-tap lane broadcasts (4 port-5 uops vs 12 with individual vbroadcastss+vinsertf128)
- SSE 128-bit remainder loop handles 0-7 leftover taps
- `zenresize_f32` fullframe: 4.86ms → 4.35ms (12% faster, now beats `fir_f32`)

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

### bytemuck chunks — NOT NEEDED

Evaluated: `bytemuck::cast_slice` can't replace `as_chunks` + `wide::type::new()`
because wide SIMD types may have stricter alignment than input slice data.
The current pattern (`as_chunks` for splitting, `::new()` for loading) is zero-overhead
and alignment-safe. bytemuck dependency removed.

### AVX-512 support
Add AVX-512 kernel variants behind the existing `avx512` feature flag.
Requires archmage AVX-512 token support.
