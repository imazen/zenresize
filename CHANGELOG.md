# Changelog

## [Unreleased]

### Added
- `FitMode` enum + `fit_dims()` / `fit_cover_source_crop()` free functions —
  aspect-ratio constraint solver so callers don't re-import `zenlayout` just
  for fit/within/cover math. Ported from zenlayout's `fit_inside` /
  `proportional` / `crop_to_aspect` including the snap-to-target rounding
  logic; brute-force parity verified (`tests/vs_zenlayout.rs`, ~6.25M cases
  per mode, bit-identical output).
- `ResizeConfigBuilder::fit(mode, max_w, max_h)` — shorthand that sets
  `out_width`/`out_height` (and, for `FitMode::Cover`, a center-anchored
  source region) in one call.
- `From<zenpixels::Orientation> for OrientOutput` — variant-wise 1:1
  conversion so callers holding a `zenpixels::Orientation` can feed it
  straight to `StreamingResize::with_orientation(...)` without manual
  matching.
- Re-export `zenpixels::Orientation` at `zenresize::Orientation` for
  convenience.

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
