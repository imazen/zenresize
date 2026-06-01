# zenresize f16↔f32: NEON-fp16 hardware vs software (Neoverse-N1)

Measured 2026-06-01 on `arm-big` (Hetzner CAX31, Ampere Altra / Neoverse-N1,
aarch64, 8 cores, rustc 1.96.0). Runtime SIMD dispatch, **`RUSTFLAGS=` (no
`target-cpu=native`)** per the project rule. Best-of-9 (3 warmup) per cell.

This quantifies the win from routing the crate's f16↔f32 row converters
(`simd::{f16_to_f32_row, f32_to_f16_row}`) through **magetypes 0.9.25's
`F16Convert`** — which adds the native NEON-fp16 path (`vcvt_f32_f16` /
`vcvt_f16_f32`, rustversion-gated to Rust ≥ 1.94, MSRV 1.89 unaffected) — vs
the previous element-by-element IEEE software conversion. A/B times both paths
on identical data on the same machine; software baseline is the standard
branchy IEEE reference (matches the former `scalar::*_soft`).

The fused filter kernels (`filter_h_row_f32_to_f16`, `filter_v_row_f16`,
`filter_v_all_f16`) were also migrated to the register-level `i32x4::f16_to_f32`
/ `f32x4::to_f16` (4-lane), verified bit-identical via the golden-output suite
(341/341 pass on both x86 and aarch64).

| n (elements) | f16→f32 HW (ns) | f16→f32 soft (ns) | speedup | f32→f16 HW (ns) | f32→f16 soft (ns) | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 16        | 40      | 40        | 1.00× | 40      | 40        | 1.00× |
| 256       | 80      | 360       | 4.50× | 80      | 560       | 7.00× |
| 4096      | 720     | 5 801     | 8.06× | 720     | 8 280     | 11.50× |
| 65 536    | 10 960  | 93 081    | 8.49× | 11 360  | 132 202   | 11.64× |
| 1 048 576 | 240 684 | 1 506 181 | 6.26× | 361 405 | 2 135 511 | 5.91× |

- Tiny n=16 is dispatch-noise-bound (within `summon()` overhead) — no win, no
  regression. The win turns on by n=256 and peaks in the compute-bound
  mid-range (~8.5× decode, ~11.6× encode), settling to ~6× at 1 MP where the
  kernel is memory-bandwidth-bound.
- Consistent with magetypes' own Neoverse-N1 figure (2.6–4.7× over its NEON
  *software* fallback); the larger ratios here are vs the *scalar* branchy
  reference the crate previously used.
- x86-64 (local, F16C `_v3` via `x86.rs`) was already hardware-accelerated and
  is unchanged by this work; the gain is on aarch64 / wasm.

Bench harness: a one-off `examples/f16_ab_bench.rs` (not committed) timing the
crate's row converters vs an inline IEEE reference. Reproduce by re-adding a
similar A/B example and `cargo run --release --example … ` on a Neoverse-N1 box.
