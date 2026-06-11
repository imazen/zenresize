# Public-API Ablation Report: zenresize

**Date:** 2026-06-11  
**Snapshot commit:** 8b41430 (main@origin)  
**Crate version:** 0.3.1  
**Snapshot items (default/all-features):** 886 / 886  
**Governance:** 0.3.x; breaking changes need minor bump.

**Grep template (as of this scan):**
```
grep -r --include="*.rs" --include="*.py" "SYMBOL" /home/lilith/work/zen/ \
  2>/dev/null | grep -v "/zenresize/" | grep -v "/target/" | grep -v "/.jj/"
```

---

## Summary

| Class | Count | % of 886 |
|-------|------:|--------:|
| A — doc(hidden) / deprecated candidate | 1 | 0.1% |
| B — pub(crate)/remove, queued breaking | 1 | 0.1% |
| **Total flagged** | **2** | **0.2%** |

The 886-item count reflects 5 monomorphised transfer-curve types (Srgb, Pq, Hlg, Bt709,
NoTransfer) each generating ~60 items from their trait impls plus 226 items in the composite
module. The real distinct-concept count is much smaller. No feature-flag gating (default=all-features).

---

## Surface overview by module

| Module | Items | External callers confirmed |
|--------|:-----:|--------------------------|
| `composite` | 226 | `Background` trait, `CompositeError`, `NoBackground`, `SolidBackground`, `SliceBackground` — used by zenpipe |
| `StreamingResize` | 62 | zenjpeg (`StreamingResize::with_batch_hint`, `StreamingError`), zenpipe |
| `Srgb` | 60 | 0 external callers |
| `Pq` | 60 | 0 external callers |
| `NoTransfer` | 60 | 0 external callers |
| `Hlg` | 60 | 0 external callers |
| `Bt709` | 60 | 0 external callers |
| `Resizer` | 46 | zenjpeg, zenpipe, sims (`Resizer::new`, `resize`, `resize_into`) |
| `ResizeConfig` | 40 | zenjpeg, zenpipe, zenmetrics (`builder`, field access) |
| `ResizeConfigBuilder` | 38 | zenjpeg, zenpipe |
| `TransferCurve` | 28 | 0 external callers |
| `plane` | 28 | 0 external callers (PlaneResizer) |
| `fit` | 26 | zenjpeg (`FitMode`, `fit_dims`, `fit_cover_source_crop`) |
| `OrientOutput` | 22 | zenjpeg layout, zenpipe |
| `Padding` | 18 | zenpipe |
| `LobeRatio` | 12 | zenpipe |
| `SourceRegion` | 10 | zenjpeg |
| `Background` (trait) | 10 | zenpipe |
| `WorkingFormat` | 8 | 0 external callers |
| `StreamingError` | 8 | zenjpeg |

---

## Flagged items

### A-class: `#[doc(hidden)]` candidate

| Item | Module | External hits | Rationale |
|------|--------|:-------------:|-----------|
| `pub enum WorkingFormat` (variants: `F32`, `I16Linear`, `I16Srgb`) | `streaming.rs` | 0 external callers | `WorkingFormat` describes the internal wire format between streaming pipeline stages. No external caller should need to branch on it — the pipeline chooses the working format automatically based on `ResizeConfig`. Grep finds zero external imports across the org (93 zenresize consumers checked). Adding `#[doc(hidden)]` signals "internal plumbing" without a breaking change. |

**Governance cost:** Non-breaking (adding `#[doc(hidden)]`). Zero callers.

### B-class: pub(crate) or move to crate-private, queued breaking

| Item | Module | External hits | Rationale |
|------|--------|:-------------:|-----------|
| `pub trait TransferCurve` + 5 zero-sized implementors (`Srgb`, `Pq`, `Hlg`, `Bt709`, `NoTransfer`) | `transfer.rs` | 0 external callers | These are SIMD-dispatch ZST marker types implementing an internal rendering trait. `Resizer` and `StreamingResize` are NOT generic over `TransferCurve` — the transfer function is determined at runtime from `ResizeConfig.input`/`output` (the `PixelDescriptor` encodes the transfer function). No caller outside the crate needs to name these types or implement `TransferCurve`. Making the trait and its 5 implementors `pub(crate)` removes ~330 items from the public surface (60 × 5 types + 28 trait items) in one breaking change. |

**Governance cost:** Breaking — requires minor bump (0.3 → 0.4). Batch with any other pending breaks. Zero callers means zero migration burden; purely cosmetic cleanup preventing future accidental external trait impl attempts. Cite in `CHANGELOG.md` under `### QUEUED BREAKING CHANGES`.

---

## Aggregate: TransferCurve + WorkingFormat

The 5 `TransferCurve` implementors + `WorkingFormat` account for **~330 of 886 items** (37%).
Making them `pub(crate)` + `#[doc(hidden)]` respectively would shrink the documented surface
to roughly 556 items with zero caller impact.

---

## Not flagged (with rationale)

- **`composite` module** — `Background`, `NoBackground`, `SolidBackground`, `SliceBackground`, `CompositeError` all have external callers (zenpipe, zenjpeg). KEEP.
- **`composite_over_premul`, `composite_over_solid_*` free functions** — Zero external callers BUT these are standalone Porter-Duff primitives that advanced callers could legitimately use. Conservative default: KEEP. (If no caller appears in 2 minor versions, revisit as B-class.)
- **`PlaneResizer`** — Zero external callers but it is the documented low-level i16 plane API (`//! use zenresize::{PlaneResizer, Filter};` in crate doc comment). KEEP as deliberate low-level escape hatch.
- **`filter` module** — `pub mod filter` with `Filter` + `InterpolationDetails`. `Filter` is re-exported at root (`pub use filter::Filter`); `InterpolationDetails` appears only in zenresize's own benchmarks (not external callers). `InterpolationDetails` is a B-class candidate but the snapshot confirms it is effectively hidden (does not appear in the public-api snapshot under the `zenresize::` path). Not flagging.
- **`ResizeConfig::needs_linearization`, `needs_premultiply`, `effective_in_stride`, `effective_out_stride`, `input_row_len`** — Helper methods with zero external callers. However they are useful introspection for advanced pipeline users; `ResizeConfig` is `#[non_exhaustive]`. Conservative default: KEEP. These are B-class candidates if the surface ever needs cleanup.
- **`Resizer`, `StreamingResize`, `Filter`, `ResizeConfig`, `ResizeConfigBuilder`** — Core API with many external callers. KEEP.
- **`fit` module** — `FitMode`, `fit_cover_source_crop`, `fit_dims` have external callers (zenjpeg, zenpipe). KEEP.
- **`OrientOutput`, `StreamingError`** — Have external callers (zenjpeg layout, zenpipe). KEEP.
- **Mask types** (`LinearGradientMask`, `RadialGradientMask`, `RoundedRectMask`, `MaskSpan`, `MaskFill`, `MaskSource`, `SpanKind`, `mask_pixel_align`) — Re-exported from zenblend; used by zenpipe. KEEP.
