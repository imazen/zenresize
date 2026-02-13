# zenresize TODO

## Replace guards with as_chunks + safe_unaligned_simd — DONE

All 36 hoisted-bounds guards in x86.rs have been replaced:

- **25 guards (Pattern A/B4/C/E):** Replaced with `as_chunks`/`as_chunks_mut` + `safe_unaligned_simd`.
  Bounds proven at compile time by Rust's slice chunking. Zero overhead.

- **11 guards (Pattern B/D/F):** H kernel inner loops where `left` comes from the weight table
  (data-dependent offsets). These use `load_si128_at()` which does runtime slice bounds checking
  by default. The `unsafe_kernels` feature flag switches to unchecked pointer access for these
  hot loops, recovering ~15% of the sRGB path performance.

The `hoisted-bounds` dependency has been removed entirely.

### Performance notes

With default features (safe mode), the sRGB downscale path shows ~10-15% overhead from
the H kernel bounds checks. With `--features unsafe_kernels`, performance matches or
slightly exceeds the original hoisted-bounds baseline.

The `unsafe_kernels` flag only affects 6 load sites in the H kernel inner loops. All other
SIMD operations are fully safe with zero overhead.
