# Changelog

## [Unreleased]

### QUEUED BREAKING CHANGES
<!-- Breaking changes that will ship together in the next major (or minor for 0.x) release. -->
- `resize_hfirst_streaming` and `resize_hfirst_streaming_f32` now return
  `Result<Vec<u8>, &'static str>` instead of `Vec<u8>`. They previously
  panicked on adversarial inputs; they now validate and surface errors.

### Added
- One-shot convenience functions `resize_rgba8(img: ImgRef<RGBA8>, out_w, out_h)
  -> Result<ImgVec<RGBA8>, At<ConfigError>>` and `resize_rgba8_to_fit(img:
  ImgRef<RGBA8>, max_w, max_h) -> Result<ImgVec<RGBA8>, At<ConfigError>>` — the
  shortest path for the two most common jobs (exact resize; aspect-fit
  thumbnail), with Lanczos + correct sRGB linear-light defaults. The source
  dimensions and row stride ride with the pixels via `imgref::ImgRef`, so there
  is no redundant width/height argument and no buffer-length-mismatch class of
  bug. Fallible: target dimensions are validated (the 120 MP cap, NaN/degenerate
  rejection) so an untrusted target size returns a `ConfigError` rather than
  panicking. Re-exports `ImgRef`, `ImgVec`, and `RGBA8` at the crate root.
  Additive; the builder + `Resizer` path is unchanged.
- Split README: `README.md` (GitHub, full badges + benchmarks) and a generated
  `README.crates.md` (crates.io, CI badge only) via `readme = "README.crates.md"`;
  `benchmarks/README.md` documents the fair-comparison methodology and pinned-commit
  reproduction. Crosslink footer refreshed (fixes the stale `heic` link, adds the
  current zen* crates).
- Versioned public-API surface snapshot at `docs/public-api/zenresize.txt`, regenerated on every `cargo test` by `tests/public_api_doc.rs` (`ZEN_API_DOC=check` verifies in the CI clippy job, `=off` skips); `justfile` recipes `fmt` / `api-doc` / `api-doc-check`. Dev-only — not part of the published package (include-whitelist already excludes it).

### Changed
- Exclude `tests/` (405 KB of weights fixtures) and `benches/` from published package tarball; local targets unaffected (declarations kept, `benches/` dir present → `cargo bench`/`cargo test` work as before).

### Fixed
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
