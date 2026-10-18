# hfilter-parity and resize-bench validation runs (2026-10-18)

Correctness and tooling validation only. **No benchmark timing was run.**

- Host: r7900x (Ryzen 9 7900X, Zen 4), Linux. Toolchain: rustc 1.99.1.
- Tools: `0ba13871b3779c3e83e731864b5fe4b9eda0e83f`, staged from `HEAD` by each
  `run.sh`.
- Baseline: `e595ef8f1dd929aebc01dd139936fca127557aad`. Its `src/` is
  byte-identical to the tooling parent `5a3e1319`.
- Candidate: `c0aa16a99246f5567e75e2ef9a5a3830404c5535`.
- Raw run directories (`identity/`, `logs/`, `bin/`, `artifacts/`) are mirrored
  with a verified manifest; see `resize_tooling_2026-10-18.pointer.md`.
- The candidate is an unpublished experiment commit. The tools under `tools/`
  on `main` are byte-identical to those at `b1109347`, the last tool revision
  validated here; they take `--old`/`--new` revisions as arguments.

## hfilter-parity

Every row runs all 10 archmage dispatch permutations, with `CompileTimePolicy::Fail`.

| run | command | result |
|---|---|---|
| quick, no features | `run.sh --old e595ef8f --new c0aa16a9 --features none --scope quick` | PASS: kernel 144,880,530 elements (119,950 calls, 1,089,230 zero-tap and 473,500 padded-last-tap pixels); e2e 34,398,870 elements (2,120 cases) |
| full, no features | `... --features none --scope full` | PASS: kernel 912,895,350 elements (747,250 calls); e2e 321,681,060 elements (16,200 cases, 35,260 streams) |
| quick, avx512 + pretty-safe | `... --features avx512,pretty-safe --scope quick` | PASS, same counts as quick |
| fault-detection check: candidate + FMA in the head loop | `--new 920ff089…` (unreferenced throwaway commit) | detected: both binaries report a mismatch, exit 1 |
| fault-detection check: candidate + 4-lane store bounded by slice length | `--new 096d7176…` (unreferenced throwaway commit) | detected by `kernel_diff` (output tail clobbered), exit 1; `e2e_diff` passes as expected, because production destinations are exact-length |

In every row, every non-NaN output was bit-identical and NaN appeared only
where NaN appeared on the other side; NaN payloads are not compared. Both
normal dependency graphs equal a production consumer's, with
`archmage/testable_dispatch` as the declared addition.

## resize-bench `prepare`

`run.sh prepare --old e595ef8f --new c0aa16a9 --corpus <codec-corpus 28205bbc>`
passed; it never times anything.

- Graphs: identical to a production consumer's except
  `num-traits v0.2.19 +default`. That feature expands to `["std"]`, which
  production already enables, and the source never tests it: proven inert.
- Lock: 15 harness-only additions; nothing in zenresize's production graph
  changed.
- Dry run: Confirm, 5 processes × 30 fixed rounds, margin 1 %, confidence
  0.95, 21 groups × 3 arms (`old`, `new`, `old-aa`). Parity before timing:
  all 21 groups bit-equal, plus 18 raw-f32 twins.
- Selftest: 31/31 controlled failures behaved as required, covering CLI and
  environment plan overrides, child routing, placement, paths, and corrupt or
  missing inputs.
- Fault-detection check: `prepare` with the FMA-mutant candidate stops at the
  dry run (exit 3). Exactly the 14 target groups mismatch, all 7 controls
  stay equal, and no zenbench data is written.
- 4k input: missing in this workload (v1). No CID22 or CLIC source reaches 4k;
  the largest cell is the native 2048×1358 CLIC image. Workload v2 below adds a
  native 12 MP input.

## resize-bench `prepare`, workload v2 (native 12 MP input)

Tools at `b1109347358693677819a71eb3c2d5aebb38c746`; same baseline and candidate.
Workload `zenresize-rgb-hfilter-whole-resize-v2` adds
`imazen26-1001-4000x3000`, an imazen-26 `variant/png-v3` SDR render at
4000×3000, used as a full-frame crop (never resampled). Its four new groups are
three RGB8-linear targets (stream and fullframe to 2000×1500, stream to
1024×768) and an RGBA8-sRGB control (stream to 2000×1500). The acceptance
rules, margin (1 %) and sampling plan are unchanged.

- Source sha256 `15cd2f7c…`, raw RGB8 sha256 `df7a242e…`. The source is
  committed imazen-26 content at `b1faae55` (Git LFS oid proof in the pointer
  file), and an independent decode reproduces the raw hash.
- Dry run: Confirm, 5 processes × 30 fixed rounds, margin 1 %, confidence 0.95,
  25 groups × 3 arms. Parity before timing: all 25 groups bit-equal, including
  each 12 MP group and its raw-f32 twin. Selftest 31/31.
- `prep` without `--imazen26`, or with a missing, empty or repeated value,
  exits 2.
- Re-run from the tooling as landed on `main` (inputs re-staged from the stored
  CAS objects, no corpus checkout read): PASS, with parity rows identical to the
  `b1109347` run. hfilter-parity quick from the same tooling: PASS, with
  identical totals.
- Budgets at 12 MP are unmeasured: no timing was run. A group that cannot
  complete its 30 fixed rounds inside `GROUP_MAX_WALL_SECS` fails acceptance.
