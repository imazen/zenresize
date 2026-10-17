# Default-configuration A/B: resampling-conventions cleanup — 2026-10-17

Does the conventions change (PR #18 after cleanup) cost anything when no
convention is used? Every case below runs the default configuration (Lanczos)
through `main` at `e595ef8f` and through this branch at `ef7186ee`.

Host `dev`: AMD Ryzen 9 9950X3D (Zen 5), Linux 7.0, rustc 1.99.0, release
profile, no `target-cpu=native`, runtime SIMD dispatch. Shared box with other
agents' load (peak load 6–13).

## Retired instructions (deterministic)

`perf stat -e instructions:u` over a fixed loop of `Resizer::new(&cfg)` +
`resize` (or `resize_f32`), divided by the iteration count
([`count.sh`](count.sh), [`examples/count.rs`](examples/count.rs)).

| path | size | main | PR18 | delta |
|---|---|---|---|---|
| srgb_i16 | 32x32→16x16 | 290,420 | 290,473 | +0.018% |
| srgb_i16 | 48x48→64x64 | 1,033,702 | 1,035,136 | +0.139% |
| srgb_i16 | 256x256→128x128 | 5,960,780 | 5,959,829 | -0.016% |
| srgb_i16 | 1024x1024→512x512 | 77,360,380 | 77,355,622 | -0.006% |
| linear_i16 | 32x32→16x16 | 297,186 | 297,190 | +0.001% |
| linear_i16 | 48x48→64x64 | 1,059,068 | 1,060,625 | +0.147% |
| linear_i16 | 256x256→128x128 | 6,434,980 | 6,433,787 | -0.019% |
| linear_i16 | 1024x1024→512x512 | 85,010,233 | 85,001,945 | -0.010% |
| linear_f32 | 32x32→16x16 | 209,599 | 209,281 | -0.152% |
| linear_f32 | 48x48→64x64 | 858,948 | 858,960 | +0.001% |
| linear_f32 | 256x256→128x128 | 5,894,191 | 5,893,111 | -0.018% |
| linear_f32 | 1024x1024→512x512 | 83,523,751 | 83,524,035 | +0.000% |
| f32_io | 32x32→16x16 | 187,814 | 187,503 | -0.166% |
| f32_io | 48x48→64x64 | 758,034 | 758,054 | +0.003% |
| f32_io | 256x256→128x128 | 4,574,189 | 4,573,097 | -0.024% |
| f32_io | 1024x1024→512x512 | 62,423,315 | 62,423,638 | +0.001% |

Paths: `srgb_i16` = RGBA8 `.srgb()` (I16Srgb), `linear_i16` = RGBX8 linear
(I16Linear), `linear_f32` = RGBA8 linear with straight alpha (F32 path, f16 row
cache), `f32_io` = RGBA f32 in and out (F32 path).

Construction alone (`count.sh new-only`, `Resizer::new` only):

| path | size | main | PR18 | delta |
|---|---|---|---|---|
| srgb_i16 | 32x32→16x16 | 194,257 | 194,320 | +0.032% |
| srgb_i16 | 48x48→64x64 | 457,097 | 458,427 | +0.291% |
| srgb_i16 | 1024x1024→512x512 | 9,481,812 | 9,477,072 | -0.050% |
| linear_i16 | 32x32→16x16 | 194,224 | 194,199 | -0.013% |
| linear_i16 | 48x48→64x64 | 455,746 | 457,588 | +0.404% |
| linear_i16 | 1024x1024→512x512 | 9,481,909 | 9,477,116 | -0.051% |
| linear_f32 | 32x32→16x16 | 102,792 | 102,269 | -0.509% |
| linear_f32 | 48x48→64x64 | 270,058 | 270,540 | +0.178% |
| linear_f32 | 1024x1024→512x512 | 7,069,202 | 7,063,140 | -0.086% |
| f32_io | 32x32→16x16 | 102,779 | 102,263 | -0.502% |
| f32_io | 48x48→64x64 | 270,046 | 270,534 | +0.181% |
| f32_io | 1024x1024→512x512 | 7,069,234 | 7,063,116 | -0.087% |

The residue is in weight-table construction: about 2 instructions per weight row
for the default-geometry check, and glibc `malloc` taking different paths
because tap counts are now u32 (the >65,535-tap fix). Callgrind attributes
identical kernel evaluations (`sin`, `ceil`, `floor`, filter calls) to both.

## Wall time (interleaved, with an A/A control)

[`benches/ab.rs`](benches/ab.rs) puts three copies of the crate in one zenbench
binary and interleaves them round by round: `main` (git rev, full sha), `main`
again from a short rev (the **A/A control**: identical code, different
addresses) and PR18. Columns are 95% CIs of the paired difference vs `main`.
Placement is unverified (`--cpu-policy=unrestricted`, taskset to CCD0) because
this box does not delegate the cpuset controller to user scopes.

| case | main | A/A, run 1 | PR18, run 1 | A/A, run 2 | PR18, run 2 |
|---|---|---|---|---|---|
| srgb_i16/tiny_down | 13.5 ±0.1µs | -0.1%–+0.3% | -0.1%–+0.5% | +0.3%–+0.5% | +0.3%–+0.6% |
| linear_i16/tiny_down | 13.9 ±0.1µs | -0.2%–+0.1% | +0.2%–+0.5% | +0.1%–+1.1% | -0.4%–+0.3% |
| linear_f32/tiny_down | 10.0 ±0.0µs | -0.9%–-0.6% | -1.7%–-1.1% | +0.6%–+1.0% | -1.4%–-1.0% |
| f32_io/tiny_down | 9.4 ±0.0µs | -0.2%–+0.4% | -0.6%–-0.2% | +0.2%–+0.5% | -1.4%–-1.0% |
| srgb_i16/tiny_up | 42.4 ±0.2µs | +0.8%–+1.1% | +0.9%–+1.2% | -0.2%–+0.7% | +0.2%–+0.7% |
| linear_i16/tiny_up | 45.7 ±0.2µs | -0.5%–-0.1% | -0.0%–+0.3% | +0.2%–+0.4% | +0.2%–+0.5% |
| linear_f32/tiny_up | 35.3 ±0.2µs | -0.1%–+0.3% | -0.8%–-0.3% | -0.2%–+0.2% | -0.2%–+0.2% |
| f32_io/tiny_up | 32.0 ±0.2µs | -0.1%–+0.4% | -1.0%–-0.5% | +0.2%–+0.9% | -1.3%–-0.5% |
| srgb_i16/small_down | 208.4 ±1.0µs | +0.2%–+0.6% | -0.1%–+0.2% | -0.0%–+0.5% | -1.2%–-0.7% |
| linear_i16/small_down | 241.4 ±1.5µs | -0.5%–-0.1% | +1.3%–+1.6% | -0.2%–+0.1% | +1.8%–+2.1% |
| linear_f32/small_down | 239.9 ±0.8µs | -0.1%–+0.2% | -0.7%–-0.5% | -0.3%–+0.0% | -0.1%–+0.2% |
| f32_io/small_down | 212.5 ±0.6µs | -0.5%–-0.1% | -0.3%–+0.3% | -0.5%–+0.5% | -0.6%–+0.2% |
| srgb_i16/medium_down | 2.2 ±0.0ms | -0.8%–-0.2% | -0.4%–+0.3% | -0.5%–+0.1% | -0.9%–-0.3% |
| linear_i16/medium_down | 2.7 ±0.0ms | -0.0%–+0.4% | -0.1%–+0.3% | -0.1%–+0.2% | -0.4%–+0.0% |
| linear_f32/medium_down | 2.9 ±0.0ms | +0.4%–+0.7% | +0.2%–+0.6% | -0.2%–+2.2% | -0.5%–+0.7% |
| f32_io/medium_down | 2.6 ±0.0ms | -0.1%–+0.5% | -0.4%–+0.6% | -0.4%–+0.6% | +0.0%–+1.2% |
| srgb_i16/medium_up | 3.3 ±0.0ms | +3.9%–+4.3% | +2.8%–+3.0% | +2.3%–+3.6% | +1.4%–+2.6% |
| linear_i16/medium_up | 3.9 ±0.0ms | -0.1%–+0.4% | -1.1%–-0.8% | -13.2%–-0.6% | -5.4%–+2.1% |
| linear_f32/medium_up | 4.1 ±0.0ms | -0.2%–-0.1% | -0.1%–+0.0% | -0.9%–+1.2% | -1.1%–+0.1% |
| f32_io/medium_up | 3.6 ±0.0ms | -0.2%–+0.1% | -0.5%–-0.1% | -0.1%–+0.0% | -0.3%–-0.0% |
| srgb_i16/large_down | 15.2 ±0.1ms | -0.7%–+0.2% | -1.2%–-0.6% | -7.2%–+5.9% | -6.3%–+5.1% |
| linear_i16/large_down | 18.7 ±0.1ms | -0.5%–-0.1% | -0.8%–-0.3% | -0.4%–-0.0% | -1.0%–-0.6% |
| linear_f32/large_down | 24.8 ±0.2ms | -0.6%–+0.3% | -0.6%–+0.2% | -0.2%–+0.0% | -0.6%–-0.1% |
| f32_io/large_down | 21.8 ±0.4ms | -0.5%–+0.2% | -0.4%–+0.3% | -0.4%–+0.2% | +0.8%–+1.3% |
| construct/tiny | 5.9 ±0.0µs | -0.1%–+0.3% | -2.4%–-1.9% | -0.4%–-0.1% | -2.3%–-1.8% |
| construct/large_down | 566.7 ±3.4µs | +0.2%–+0.6% | -1.7%–-1.3% | -0.2%–+0.1% | -1.2%–-1.0% |

Read the PR18 columns against the A/A columns: identical code moves by up to
±4% here (and +15–29% in one isolated run of `srgb_i16/medium_down`), from code
and heap layout. Run 2 shows load spikes in a few `main` means. No PR18 shift
outside the A/A band reproduces across both runs except
`linear_i16/small_down` (+1.3–2.1%), where PR18 retires 0.019% *fewer*
instructions than `main`, so that is layout too.

## Regressions found and fixed on the way

- The `Fixed22` branch inlined into `produce_next_i16_srgb` and `push_row`
  measured +3.9–4.2% on `srgb_i16/medium_up` (A/A −0.2–+0.1%), twice. Moving it
  into `#[cold] #[inline(never)]` helpers removed it (`1b3a7eb4`).
- `compute_pixel_weights` evaluated the mapping per tap and f32 setup per row:
  +0.9–1.2% instructions per tiny resize. The default-geometry path and the
  `Axis` struct brought it to the table above (`ef7186ee`).

## Reproduce

```sh
just pr18-ab
```

Raw logs, including the exploratory runs, the five-process confirm run and
the oracle-panel `results.json` before and after: `/mnt/v/output/zenresize/pr18_ab_2026-10-17/`
on dev (`SHA256SUMS` alongside).
