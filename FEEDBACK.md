# Feedback Log

## 2026-02-21
- User requested V-first streaming pipeline + AVX2 H-filter widening. Implemented both changes with benchmarking after each step.

## 2026-02-14
- User requested API migration in test files: free `resize()`/`resize_f32()` functions replaced with `Resizer::new(&config).resize()`/`.resize_f32()` pattern across 4 test files. Removed `resizer_matches_oneshot_linear_no_alpha` test (no longer applicable). No `ColorSpace` usage found in tests.
