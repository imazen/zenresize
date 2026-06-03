# f16 V-filter on Neoverse-N1: archmage `F16Convert` (main) vs PR#9 magic-multiply

**Date:** 2026-06-03
**Box:** Hetzner CAX21, Ampere Altra Neoverse-N1, 8 vCPU (aarch64)
**Toolchain:** rustc 1.96.0 (≥1.94, so magetypes NEON-fp16 hardware `vcvt` path is active)
**Harness:** `cargo bench --bench competitors_zen` (zenbench, interleaved round-robin)
**Build flags:** default `RUSTFLAGS` — runtime SIMD dispatch (what users get), **not** `target-cpu=neoverse-n1`. Both sides built identically.

## What was compared

The f32 resize path caches its V-filter intermediates as f16. The f16→f32
decode inside `filter_v_row_f16` was ~73% of the f32 pipeline on N1 (per PR#9's
profiling). Two ways to vectorize that decode on aarch64:

- **main** (`9425d0f3`): magetypes 0.9.26 `F16Convert` register methods
  (`i32x4::f16_to_f32`) — resolves to **hardware NEON-fp16 `vcvt`** on rustc ≥1.94,
  branchless software fallback otherwise. Same kernel, conversion delegated to the
  published archmage/magetypes API.
- **PR#9** (`imazen/zenresize#9`, head `2ed6f2a`): a hand-rolled branchless
  **software** magic-multiply (Fabian Giesen) expressed in `i32x4`/`f32x4`.

Both are bit-identical to the scalar reference; the only question is throughput.

## Result (zenbench `zenresize_f32`, input MiB/s, higher = better)

| scenario      | main (HW NEON-fp16) | PR#9 (sw magic-multiply) | main advantage |
|---------------|--------------------:|-------------------------:|---------------:|
| f32 1024→512  |          **391**    |                 332      |     **+17.8%** |
| f32 4k→2k     |          **379**    |                 321      |     **+18.1%** |

`fir_f32` (fast_image_resize, the x86-tuned competitor) ran identically on both
sides (685/675 and 434/445 MiB/s) — confirming the delta is the f16 decode, not
run-to-run noise.

## Conclusion

The archmage/magetypes `F16Convert` hardware NEON-fp16 path on main is **~18%
faster** than PR#9's software magic-multiply on real N1 hardware. PR#9's goal
(vectorize the aarch64 f16 decode, which it measured at +40–59% over the old
scalar-soft path) is fully achieved on main and then exceeded. **PR#9 is
superseded — closing it drops no improvement; main is strictly faster.**

Raw zenbench output: `arm_n1_f16_main_competitors_zen_2026-06-03.log`,
`arm_n1_f16_pr9_competitors_zen_2026-06-03.log` (this directory).
