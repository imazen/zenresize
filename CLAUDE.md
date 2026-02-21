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
- `wide_kernels.rs` — Portable SIMD via `wide` crate (NEON, WASM128)
- `neon.rs`, `wasm128.rs` — Token wrappers for wide_kernels
- `mod.rs` — Dispatch via `archmage::incant!`

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

## Investigation Notes

(none currently)

## Known Bugs

(none currently)
