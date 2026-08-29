# Changelog

## [Unreleased]

### QUEUED BREAKING CHANGES
<!-- Breaking changes that will ship together in the next major (or minor for 0.x) release. -->
- `resize_hfirst_streaming` and `resize_hfirst_streaming_f32` now return
  `Result<Vec<u8>, &'static str>` instead of `Vec<u8>`. They previously
  panicked on adversarial inputs; they now validate and surface errors.

### Added
- Split README: `README.md` (GitHub, full badges + benchmarks) and a generated
  `README.crates.md` (crates.io, CI badge only) via `readme = "README.crates.md"`;
  `benchmarks/README.md` documents the fair-comparison methodology and pinned-commit
  reproduction. Crosslink footer refreshed (fixes the stale `heic` link, adds the
  current zen* crates).
- `benches/mask_e2e.rs` extended from 3 points to a 9-point 64²→4K size ladder,
  locating the `with_mask()` f32-fallback crossover for #3: on aarch64 I16Srgb is
  ahead only at ≤128×128 (≤4%), 256²–1440×1080 is a wash, and F32 is 15-30%
  faster at 4K — so an i16 mask path would be a regression there. Two runs plus
  the write-up in `benchmarks/mask_e2e_ladder_aarch64_2026-08-29.meta`; no
  library behaviour change.
- Versioned public-API surface snapshot at `docs/public-api/zenresize.txt`, regenerated on every `cargo test` by `tests/public_api_doc.rs` (`ZEN_API_DOC=check` verifies in the CI clippy job, `=off` skips); `justfile` recipes `fmt` / `api-doc` / `api-doc-check`. Dev-only — not part of the published package (include-whitelist already excludes it).

### Changed
- Exclude `tests/` (405 KB of weights fixtures) and `benches/` from published package tarball; local targets unaffected (declarations kept, `benches/` dir present → `cargo bench`/`cargo test` work as before).

### Fixed
- x86-64 `unpremultiply_alpha_row` (the AVX2 tier) now leaves a pixel whose
  alpha is at or below the `1/1024` threshold untouched, like the scalar, NEON
  and wasm128 tiers. It previously AND-masked the reciprocal, multiplying such
  RGB lanes by 0 (zeroing them, and turning ±inf into NaN); the cross-tier
  bit-identity test `tests/alpha_f32_exact.rs` failed on every x86-64 CI lane.
- x86-64 `filter_h_row_f32_to_f16` (the f16-intermediate H filter used by the
  fullframe `Resizer`) now accumulates each output pixel non-fused, one actual
  tap at a time, so it is bit-identical to the scalar formulation and to the
  NEON/wasm128 kernel (`tests/f16_hfilter_exact.rs`). The previous fused
  four-accumulator form differed from them by 1 f16 ULP on some elements.
- x86-64 `unpremultiply_u8_row` now computes exactly the integer formula
  `min(255, (c*255 + a/2) / a)` of the scalar and NEON tiers (an IEEE divide on
  exactly-representable operands), verified over the whole 256x256 domain by
  `tests/unpremul_u8_exhaustive.rs`. The previous `_mm_rcp_ps` + Newton-step
  approximation with `+0.5` truncation was one below the reference on some
  pairs (e.g. c=1, a=2 gave 127 instead of 128).
- `ResizeConfig::validate()` now bounds the full **padded** canvas: the
  `max_output_pixels` cap applies to `total_output_width * total_output_height`
  (not just `out_width * out_height`), and the canvas byte size
  (`pixels * channels * elem_size`) must fit `usize` on the target. Previously a
  1×1 resize padded to 2^31 × 2^31 passed validation and the allocating
  `Resizer::resize*()` methods computed `total_output_height as usize *
  total_output_row_len` unchecked — a wrapped, undersized allocation followed by
  an out-of-bounds row copy (on 64-bit too, not only i686/wasm32). Those nine
  allocation sites now use a checked multiply (#10, part 3; parts 1–2 shipped
  via `try_new` and `max_output_pixels`).
- The allocating `Resizer::resize()` / `resize_into()` (and every `resize_*`
  type and cross-format variant) now honor canvas padding (`.padding()` /
  `.padding_color()`): the output buffer is sized to the full padded canvas
  (`total_output_height() * total_output_row_len()`) and rows are assembled at
  total dims, matching `StreamingResize`. Previously the buffer was sized at the
  inner resize dims (`out_height * output_row_len()`), so a padded config
  panicked with an out-of-bounds row copy. Post-resize sharpen/blur now run over
  the full canvas. No behavior change when padding is unset (`total_output_*`
  equals the inner dims). Regression test: `tests/allocating_padding.rs`.
- Bound weight-table allocation in `ResizeConfig::validate()` so adversarial
  `in_size`/`out_size` ratios cannot trigger multi-GB allocations from a
  few-byte container header. Cap is ~256 MB worth of f32 entries per axis.
- Reject NaN, infinity, and out-of-range numeric resize-config fields:
  `post_sharpen`, `post_blur_sigma`, `kernel_width_scale`, `LobeRatio::Exact`,
  `LobeRatio::SharpenPercent`. Previously these flowed into weight
  computation and produced NaN-poisoned outputs or infinite loops.
- `StreamingResize::push_row` and `push_row_u16` now require `row.len() >=
  source_row_len` strictly. The prior `min(stride, source_row_len)` check
  let any `in_stride` smaller than the row length (e.g., `in_stride=1`)
  bypass the check and panic on the subsequent slice.
- `StreamingResize::push_rows` uses checked arithmetic throughout to avoid
  panicking on adversarial `stride`/`count` combinations.
- `resize_hfirst_streaming` and `_f32` now run `validate()` and use checked
  arithmetic for all buffer allocations. Tap-reference buffers are
  heap-allocated, removing the prior 128-tap stack ceiling that would
  panic on large downscale ratios.

## [0.3.1] - 2026-04-23

### Added
- `FitMode` enum + `fit_dims()` / `fit_cover_source_crop()` free functions —
  aspect-ratio constraint solver so callers don't re-import `zenlayout` just
  for fit/within/cover math. Ported from zenlayout's `fit_inside` /
  `proportional` / `crop_to_aspect` including the snap-to-target rounding
  logic; brute-force parity verified (`tests/vs_zenlayout.rs`, ~6.25M cases
  per mode, bit-identical output) (7bc5555).
- `ResizeConfigBuilder::fit(mode, max_w, max_h)` — shorthand that sets
  `out_width`/`out_height` (and, for `FitMode::Cover`, a center-anchored
  source region) in one call (7bc5555).
- `From<zenpixels::Orientation> for OrientOutput` — variant-wise 1:1
  conversion so callers holding a `zenpixels::Orientation` can feed it
  straight to `StreamingResize::with_orientation(...)` without manual
  matching (7bc5555).
- Re-export `zenpixels::Orientation` at `zenresize::Orientation` for
  convenience (7bc5555).

## [0.3.0] - 2026-04-17

### Changed
- **BREAKING:** Remove `zenlayout` dependency and `layout` feature; all layout/execute functions removed (bcc3911)
- **BREAKING:** Remove `SolidBackground::from_canvas_color` method (bcc3911)
- **BREAKING:** Remove `zenresize::layout` module (bcc3911)
- Switch Cargo.toml from `exclude` to `include` whitelist (3c287f4)

### Added
- Fused decode+premultiply and unpremultiply+encode passes for reduced memory traffic (e462b44)

### Fixed
- Remove redundant import, fix doc comment continuation, clippy cleanup (ee33412)

## [0.2.2] - 2026-04-08

### Fixed
- Remove unsound `uninit_vec` in `alloc_output` (3d0d0bb)
- Strip target-specific dev-deps on Windows ARM64 CI (c0fbe5c)

## [0.2.1] - 2026-04-06

Initial published release on crates.io.
