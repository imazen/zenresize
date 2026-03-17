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

### H-first streaming for i16 paths
The streaming pipeline is V-first (V-filter on `in_width`-wide rows, then H-filter). For the i16 paths (sRGB, linear), H-first is 1.5-3.7× faster than both fullframe and V-first streaming because the V-filter operates on `out_width`-wide rows after H-filtering narrows them. The f32 path should stay V-first — the per-row decode+premul+f16-encode overhead cancels the V-filter savings.

**Benchmark results (worktree-tiled-v-filter branch):**
- sRGB 4ch 4K 2×: HF 12.6ms vs FF 16.3ms vs VF 14.8ms (HF wins 0.78×)
- linear 4ch 4K 2×: HF 12.7ms vs FF 28.6ms vs VF 25.7ms (HF wins 0.44×)
- linear 4ch 4K 10×: HF 10.3ms vs FF 21.8ms vs VF 27.0ms (HF wins 0.47×)
- alpha 4ch 4K 2×: HF 12.7ms vs FF 47.1ms vs VF 43.4ms (HF wins 0.27×)
- f32 3ch 4K 10×: VF 24.0ms vs FF 63.7ms vs HF 62.4ms (VF still best for f32)

Implementation: `StreamingResize` should auto-select H-first for i16 paths. The prototype lives in `resize_hfirst_streaming()` on the worktree-tiled-v-filter branch. Needs: proper streaming API (push/pull), batch support, compositing integration.

### Compositing in H-first streaming
Current compositing only works in the f32 path (V-first). For H-first, compositing could happen after the V-filter produces the final output row. Need to investigate:
- Can we composite on i16/u8 output directly (integer composite)?
- Or convert the V-filter output to f32, composite, convert back?
- Consider making compositing a separate post-processing step that's composable with any pipeline order.

### Compositing composability
The current compositing is deeply wired into the streaming pipeline (`composite_dispatch` called between V-filter and unpremul). Consider:
- Extract compositing into a standalone row processor trait/fn
- Let callers compose: resize → composite → encode, or resize → encode (no composite)
- Support compositing on u8 rows directly (integer source-over) for the i16 paths
- This would also allow compositing with fullframe output without forcing f32 path

### Integration test coverage for fullframe vs streaming parity
Current parity tests use tiny images (30×30→15×15) that don't expose the i16 accuracy gap. Need tests at:
- 1024→512 and 4000→400 for both sRGB and linear paths
- Upscale 512→1024
- Document the expected max diff per path combination

## Investigation Notes

### i16 accuracy gap between fullframe (H-first) and streaming (V-first)
**Measured max diff (u8 output):**
- sRGB-i16 path: 2-6 (downscale), 25 (upscale)
- linear-i16 path: 6-43 (downscale), 49-52 (heavy downscale/upscale)
- f32 path: 1 (near-identical)

**Root cause:** Intermediate clamping at different pipeline stages. Both paths do 2 quantization steps (H-filter round+clamp, V-filter round+clamp), but clamping after H (fullframe) vs after V (streaming) discards different Lanczos overshoot information. Negative-lobe ringing causes intermediate values outside [0, 4095] (linear) or [0, 255] (sRGB); clamping these at different structural positions yields divergent final results. The error grows with tap count (more negative lobes at high downscale ratios) and is worst at image edges where clamped boundary pixels amplify the asymmetry.

**Not an overflow issue**: weights sum to 16384, max absolute accumulation is ~80M (well within i32). The f32 path doesn't clamp intermediates, so it doesn't have this problem.

**Implication for H-first streaming**: H-first will match fullframe output (same filter order), not V-first. The quality difference is inherent to filter order with integer clamping, not a bug to fix. Both are valid — neither is "wrong" — but they're not identical.

## Known Bugs

(none currently — the i16 accuracy gap is a known limitation, not a bug)
