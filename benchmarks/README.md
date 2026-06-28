# zenresize benchmarks — methodology & reproduction

How to run zenresize's comparisons fairly, and how to read the committed result
files. The headline cross-library comparison is `benches/paired_bench.rs`; the
others are listed in the root README.

## Fairness guarantees

`paired_bench` is built so the numbers mean something:

- **Interleaved (paired) measurement.** It runs A,B,A,B… not "all of A, then all
  of B." Both libraries see the same thermal state, turbo residency, and OS
  scheduling, and the paired difference `Aᵢ − Bᵢ` cancels systematic drift —
  giving a tight CI on *relative* speed. (This is the same idea as
  [zenbench](https://github.com/imazen/zenbench); see "Charts" below for porting.)
- **No I/O in the timed region.** The test image is synthesized into a `Vec<u8>`
  (and a `Vec<f32>`) once, before timing starts. The timed closure only calls the
  resize. No file open/read/decode/write is measured. Output is fed to
  `std::hint::black_box` so it isn't optimized away.
- **Single-thread vs single-thread.** Every contender is pinned to one thread
  (`pic-scale` via `ThreadingPolicy::Single`). zenresize is single-threaded per
  call by design, so this is apples-to-apples. Do **not** compare a thread-pooled
  resize against a single-threaded one.
- **Equal SIMD opportunity.** By default `pic-scale` builds scalar-only, which
  would flatter zenresize. Enable `--features bench-simd-competitors` for an
  honest SIMD-vs-SIMD comparison.
- **No `-C target-cpu=native`.** Builds use runtime SIMD dispatch (archmage
  `incant!`), which is what ships. Native builds bake in ISA extensions and give
  misleading numbers.

## Reproduce

```sh
# zenresize
git clone https://github.com/imazen/zenresize && cd zenresize
git checkout <commit>          # the commit named in the result file you're reproducing

# fair SIMD-vs-SIMD, interleaved, single-thread, no I/O:
cargo bench --bench paired_bench --features bench-simd-competitors

# self-regression (zenresize vs its own prior code), and the full sweep:
cargo bench --bench tango_bench
cargo bench --bench sweep_bench        # CSV across sizes 64–7680 px × ratios 12.5–300%
```

Competitors are dev-dependencies, so `cargo` pins them for you. The versions used
for committed results (pin these if reproducing elsewhere):

| Competitor | Version | Notes |
|-----------|---------|-------|
| [`pic-scale`](https://crates.io/crates/pic-scale) | 0.7 | SIMD enabled via `bench-simd-competitors` |
| [`pic-scale-safe`](https://crates.io/crates/pic-scale-safe) | 0.1 | safe-only variant |
| [`fast_image_resize`](https://crates.io/crates/fast_image_resize) | 6 | single-thread |
| [`resize`](https://crates.io/crates/resize) | 0.8 | scalar reference |

## Result files

Each committed run lands as `benchmarks/<topic>_<YYYY-MM-DD>.{md,csv,log}` and
**must** state, in its header: the git commit, the CPU/RAM/OS, `rustc -V`, the exact
command, the threading mode, and whether `bench-simd-competitors` was on. Current files:

- `arm_n1_f16_archmage_vs_pr9_2026-06-03.md` — ARM Neoverse-N1 f16 path A/B.
- `f16_convert_neon_fp16_2026-06-01.md` — NEON-fp16 f16↔f32 conversion.
- `transfer_bench_2026-03-04.csv` — transfer-function throughput.

Do not commit numbers you didn't generate, and don't extrapolate one size to
another — measure each size. Memory claims need heaptrack / `time -v`, not
estimates.

## Charts (what to plot for which decision)

| Question | Chart |
|----------|-------|
| "Which library is fastest?" | horizontal **bar**, sorted by throughput (MP/s); separate bars for 1-thread vs N-thread if relevant |
| "How does speed scale with image size / ratio?" | **line** (or grouped bar), x = pixels or ratio (log); fit `total = α + β·pixels` and report both the fixed overhead and the per-pixel slope |
| "Is the A/B delta real / how noisy?" | **violin** or PDF of per-call times, or the paired 95% CI `paired_bench` already prints |

For new comparison charts, prefer [zenbench](https://github.com/imazen/zenbench)
— it does the interleaving for you and emits a sorted throughput **bar chart**, a
self-contained **SVG** report (`--format=html`), and violin/PDF/regression plots
(plotters.rs). `paired_bench` predates that; porting it to zenbench is the way to
get publishable SVG charts without hand-rolling stats.
