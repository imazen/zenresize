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

## Build Rules

**NEVER compile with `-Ctarget-cpu=native` except for diagnostics (e.g., `cargo asm`).** We will never deploy "native" binaries. All production and benchmark builds must use dynamic dispatch via `incant!`/`arcane`/`rite`. Performance must be verified on every token tier using `dangerously_disable_token_process_wide(true)` — not just the highest tier the machine supports. Use `testable_dispatch` feature in dev-dependencies if needed to override compile-time feature guarantees.

## TODO

### Compositing composability [DONE]
Blend math extracted into `~/work/zen/zenblend/` (MIT OR Apache-2.0).
zenresize and zenpipe both delegate to zenblend for all blend operations.
`BlendMode` enum re-exported from zenresize, `with_blend_mode()` builder on both
`StreamingResize` and `Resizer`. SrcOver has AVX2+FMA SIMD kernel (2 pixels/iter).
Remaining: per-row i16→f32→composite→u8 in i16 paths (avoids full f32 pipeline).

### Native AVX-512 for remaining hot-path kernels
`filter_v_row_i16` has a native AVX-512 kernel (32 i16/iter). Still delegating to AVX2:
- `filter_h_u8_to_i16` — H-filter for sRGB i16 path
- `filter_h_i16_i16` — H-filter for linear i16 path
- `filter_v_row_f16` / `filter_h_row_f32_to_f16` — f32 path

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
