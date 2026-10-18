# hfilter-parity

Old-vs-new differential for zenresize's RGB (3-channel) f32 horizontal filter,
`filter_h_row_f32(.., 3)`. It links two verified `git archive` builds of
zenresize into one process and compares them under every archmage dispatch
permutation (AVX-512 delegate, AVX2+FMA, scalar fallback).

- **`kernel_diff`**: per-call three-way comparison of candidate, baseline and a
  transcription of the baseline's scalar body. Exhaustive small geometries,
  filters × weight variants (blur, lobe ratio) × fixed ratios, and random
  geometries. Inputs include ±0, subnormals, overflow, ±inf and NaN (quiet,
  signalling, payloads), plus physical input tails and sentinel-filled output
  tails.
- **`e2e_diff`**: whole public resizes (`Resizer`, `StreamingResize`; crop,
  padding, strides, `post_blur`, `post_sharpen`, `LobeRatio`,
  `kernel_width_scale`; RGB8 linear, RGB8 sRGB, RGB16, RGBF32) compared exactly.

**Pass** means:

- every non-NaN output is bit-identical;
- NaN appears only where the other build has NaN;
- every output sentinel survives;
- both builds produce identical weight tables.

NaN payloads are counted, not required: Rust does not specify them for
arithmetic.

## Run

```sh
tools/hfilter-parity/run.sh --old e595ef8f --new <candidate> \
  --out ~/tmp/hfp-<date>-<n> --features none --scope quick
# or: just hfilter-parity e595ef8f <candidate> ~/tmp/hfp-<date>-<n> none quick
```

On a shared host, wrap the command in that host's heavy-job wrapper. Every run
builds in its own `<out>/target`: a Cargo target directory shared between run
directories can treat moved sources as fresh and return a stale executable,
so there is no cache option, and retained binaries must postdate the run.

| option | required | values |
|---|---|---|
| `--old`, `--new` | yes | any revision; resolved to full commits |
| `--out` | yes | fresh, disk-backed directory; refuses existing paths and `/tmp`, `/run`, `/dev/shm`, `/var/tmp` |
| `--features` | yes | `none`, `avx512`, `pretty-safe`, `avx512,pretty-safe` (forwarded to both builds) |
| `--scope` | yes | `quick` or `full` |

## Scopes

Counts are per dispatch permutation. Each binary prints its scope line, and a
run records its totals in `identity/result.tsv`.

| | `quick` | `full` |
|---|---|---|
| `kernel_diff` | 1..=16 in/out × 5 filters; 5 filters × 5 variants × 15 ratios × 5 input classes × 5 tail layouts; 60 random geometries | 1..=40 × 5 filters; all filters × 5 variants × 15 ratios × 5 × 5; 600 random |
| `e2e_diff` | 200 cases ≤ 96 px + 12 cases ≤ 1400 px | 1500 + 120 |

## What `run.sh` guarantees, or it stops

- The tool is committed. `tools/hfilter-parity` and `tools/stage-lib.sh` must
  equal `HEAD`, and they are staged from `HEAD`, not the working tree.
- Every exported file matches its committed blob (`git hash-object` against
  `ls-tree`). The only edit is the `[package]` version suffix both builds need
  to share one lockfile, and it is verified to change exactly one line.
- `Cargo.lock` is seeded from the candidate. The tool's additions are resolved
  once, and every later call is `--locked`.
- Each build's normal dependency graph (versions and enabled features) equals a
  minimal production consumer's. The one declared exception is
  `archmage/testable_dispatch`, which is added to both.
- The build uses production defaults: no `RUSTFLAGS`-family variables, no
  `CARGO_PROFILE_*`, and no Cargo config that sets rustflags, target-cpu,
  target-feature or profiles.

## Output (`<out>/`)

| path | contents |
|---|---|
| `identity/` | `run.tsv`, `sources.tsv`, per-export `*.files.sha256` and `*.ls-tree`, alias records, `Cargo.lock.{seed,resolved,diff}`, `graph-{old,new}.{reference,tool,diff}`, `rustc-vV.txt`, `cargo-vV.txt`, `env-rust-cargo.txt`, `cargo-config-files.tsv`, `binaries.tsv`, `result.tsv` |
| `logs/` | complete output of every step |
| `bin/` | the two binaries, retained outside the Cargo target directory |

Recorded runs: `benchmarks/hfilter_parity_*.md`.
