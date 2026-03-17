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

### Fullframe as streaming wrapper [DONE on worktree-tiled-v-filter]
`Resizer` now delegates to a cached `StreamingResize` internally. Weight tables computed once, reused via `stream.reset()`. Eliminated ~500 lines of dead fullframe code. 308 tests pass. Needs merge to main.

### Paired output row batching in streaming V-filter [DONE on worktree-tiled-v-filter]
Both i16 paths detect consecutive output rows sharing the same V-filter window and V-filter both back-to-back (L1-hot data). Second row buffered for next `next_output_row()`. Needs merge to main.

### Streaming compositing without extra buffering
The i16 H-first paths don't support compositing (force f32 path fallback). To enable: after V-filter produces i16 output row, convert i16→f32 in temp buffer, composite in f32 (source-over), convert f32→u8. All per-row, no image-sized buffer. The temp buffers already exist in StreamingResize.

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

### AVX-512 kernel expansion
Only `filter_v_row_f32` has an AVX-512 kernel (commit e04e888). Need x86v4 variants for:
- `filter_h_u8_to_i16` (H-filter for sRGB i16 path)
- `filter_v_row_i16` (V-filter for i16 paths)
- `filter_h_i16_i16` (H-filter for linear i16 path)
- `filter_v_row_f16` / `filter_h_row_f32_to_f16` (f32 path)
AVX-512 gives 512-bit registers (32 i16 per register) — potentially 2× throughput over AVX2 for i16 kernels. Use archmage `X64V4Token`.

### Integration test coverage for fullframe vs streaming parity
Current parity tests use tiny images (30×30→15×15). Need tests at 1024+ for sRGB and linear. With the i16 clamping fix, max diff is now 1 — tests should enforce this.

## Investigation Notes

### i16 accuracy gap between fullframe (H-first) and streaming (V-first)
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

### Intermediate clamping reduces i16 path accuracy
The i16 filter kernels clamp intermediate values to [0, 4095] (linear) or [0, 255] (sRGB) after each filter step. This destroys Lanczos overshoot/undershoot information, causing up to 52 LSB difference between H-first and V-first filter order. Fix: remove intermediate clamping (option 1 above). See investigation notes.
