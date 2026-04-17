# Changelog

## [Unreleased]

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
