# resize-bench

A native [zenbench](https://lib.rs/crates/zenbench) Confirm comparison of
whole public zenresize resizes. The harness is zenbench at the reviewed commit
`98c25fe8` (`ZR_ZENBENCH_COMMIT` in `run.sh`; version field 0.1.11), staged
from a verified `git archive` like the two zenresize builds. One executable links a baseline build
(`old`) and a candidate build (`new`) of zenresize, each from a verified
`git archive` export with production dependencies, features and build
settings. Every group times three arms, interleaved: `old`, `new`, and
`old-aa`, which runs the baseline again as an A/A control.

`prepare` never times anything. Only `confirm` measures, and it is meant for a
quiet host.

## Frozen plan

Defined in `src/plan.rs` and `src/inputs.rs`, under workload id
`zenresize-rgb-hfilter-whole-resize-v3`.

**Sampling** (predeclared before any timing of this workload):

- 15 fresh processes, run one after another, 100 fixed rounds per group, margin
  1 %, family confidence 0.95. There is no pilot stage, no data-dependent
  stopping and no role change after data.
- Inference (zenbench `inference.rs`): per process, the mean per-iteration time
  of each arm; across processes, a paired log-ratio t interval (14 degrees of
  freedom) with a Bonferroni family of 50. Groups are baseline-only, so the
  analyses are exactly `old → new` and `old → old-aa` in each of the 25 groups.
- Safety caps, not runtime predictions (the runtime of this plan is
  unmeasured): a supervisor deadline of 86400 s for all fifteen processes and a
  3600 s wall limit per group. The 120 s per-group benchmark-time budget cannot
  shorten the fixed rounds (Confirm sets `min_rounds = max_rounds`, and
  `max_time` is checked only after `min_rounds`). The wall limit can; then the
  evaluator rejects the run.

**Timed call.** Each callback is one whole public call:

- `StreamingResize`: push every row and collect every output row;
- `Resizer::new(..).resize*`;
- `resize_hfirst_streaming[_f32]`.

The call's construction, output allocation and drop are inside the clock.
zenbench black-boxes the entire returned buffer. Inputs and configuration are
built once, outside the clock.

**Inputs.** Real codec-corpus images plus one imazen-26 `variant/png-v3`
SDR render, cropped, never resampled:

| input | source | used size |
|---|---|---|
| `cid22-1025469-c64` | CID22-512 validation | 64×64 crop |
| `cid22-1044329-c256` | CID22-512 validation | 256×256 crop |
| `cid22-1189261-512` | CID22-512 validation | native 512×512 |
| `clic-097cb4-c1024` | CLIC2025 | 1024×1024 crop (~1 MP) |
| `clic-0c49a5-2048x1358` | CLIC2025 | native 2048×1358 (2.8 MP) |
| `imazen26-1001-4000x3000` | imazen-26 png-v3 SDR (S21 Ultra photo) | native 4000×3000 (12 MP) |

No CID22 or CLIC source reaches 4k; the 12 MP cell comes from the imazen-26
png-v3 checkout instead (a real camera photograph at native dimensions).

**Groups** (25; every one runs `old`, `new` and `old-aa`):

- **Targets** reach the RGB f32 path and its 3-channel horizontal filter:
  - RGB8 linear: stream and fullframe at 256 and ~1 MP, and 2.8 MP to 1024 px
    wide (plus a stream-only 4× downscale to 512 px); also strided fullframe
    input and a 256→512 upscale;
  - RGB8 sRGB-space, RGB16 and RGBF32, at ~1 MP.
  - RGB8 linear at 12 MP: stream and fullframe to 2000×1500, and a
    stream-only ~4× downscale to 1024×768.
  - The 64×64 groups are startup-boundary targets: there, fixed per-call cost
    dominates.
- **Controls** don't use that filter: h-first f32 (512→1024 upscale and
  1 MP), RGBA8 sRGB (I16, including 12 MP→2000×1500), RGBX8 linear (I16) and
  RGBA8 linear (f32 premultiplied).

**Acceptance.** zenbench's exit 0 is never acceptance on its own. The run is
accepted only if all of the following hold:

- zenbench reports no confirmation issues;
- the resolved plan equals the frozen plan, with mode Confirm;
- all 15 replicates hold every group and arm with 100 completed fixed rounds;
- every `old → old-aa` is Equivalent within ±1 %;
- every control's `old → new` is Equivalent within ±1 %;
- every target's `old → new` is an Improvement (the whole interval below −1 %);
- every startup target's `old → new` is non-inferior (upper bound ≤ +1 %);
- old/new outputs are bit-identical before and after timing in every process.
  That includes raw f32 twins of each u8/u16 case; h-first has no f32 entry
  point.

Exit codes:

| code | meaning |
|---|---|
| 0 | accepted |
| 1 | confirmed regression |
| 2 | invalid, inconclusive, not accepted, or a usage/plan/environment error |
| 3 | input or parity verification failed before timing |

**Command line.** The executable accepts only `--artifacts=DIR`, `--dry-run`,
`--cpu-set`, `--cpu-policy=auto|performance|efficiency` and
`--format=llm|json`. Any other argument, and any `ZENBENCH_*` variable that
would change the plan, exits 2 before anything is written. Exceptions are the
placement variables, `ZENBENCH_COORDINATION_DIR` and the required
`ZENBENCH_DATA_DIR=<artifacts>/zenbench-data`. A child process must carry
exactly the variables zenbench's supervisor and admission lease set for it,
with values equal to the frozen plan.

**CPU dispatch readback.** Every process record (`artifacts/records/*.json`,
`identity.cpu_dispatch`) carries what archmage's runtime detection reports on
that CPU (`available_readback`: X64V3, X64V4, X64V4x) and, separately, the tier
`filter_h_row_f32` takes, which is *inferred*, not observed. Its `incant!` uses
the default tiers, `v4` is gated on zenresize's `avx512` feature, and both
builds use default features, so on x86_64 only the `v3` and `scalar` arms are
compiled and `v3` runs exactly when X64V3 is available. The readback uses a
tool-only `archmage` dependency at the version and feature (`std`) the
production graph already resolves, so the feature union does not change;
`prepare`'s graph check proves it. `prepare` prints the dry run's readback to
`identity/cpu-dispatch.txt`.

## Run

```sh
# Stage, check, build, prepare inputs, dry-run, selftest. No timing.
tools/resize-bench/run.sh prepare --old e595ef8f --new <candidate> \
  --corpus <codec-corpus checkout> --imazen26 <imazen-26 png-v3 checkout> \
  --zenbench <zenbench repository> --out <fresh run dir>
# On a quiet host, with the run directory's own frozen copy of the tool:
<run dir>/tool/tools/resize-bench/run.sh confirm --run <run dir> \
  --scope-cpus 2,3 --pin 2 [--scope-mem 6G] [--dry-run]
```

**`prepare`** checks everything `tools/hfilter-parity` checks:

- the tool is committed and staged from `HEAD`;
- every exported file is a verified blob, and the only edit is the alias
  version line;
- the lock is seeded from the candidate, then everything runs `--locked`;
- no rustflags or profile overrides are present;
- the normal dependency graph of both zenresize copies equals a production
  consumer's exactly.

For the harness, `prepare` also:

- exports zenbench `ZR_ZENBENCH_COMMIT` from the `--zenbench` repository, with
  every file verified against its blob, and checks its package name and version;
- resolves the same workspace a second time with the crates.io zenbench of that
  version, and requires the two locks to differ only in zenbench's `source` and
  `checksum` lines (`identity/Cargo.lock.provenance-check.txt`);
- after the build, reads rustc's dep-info for every compiled source file and
  hashes it (`identity/compiled-sources.tsv`). Every zenresize, zenbench and
  tool source must come from its staged export and match the export manifest
  byte for byte, every other source must come from the Cargo registry, and no
  registry copy of zenresize or zenbench may be compiled.

On top of that, in the benchmark workspace, resolution may not change any
package in zenresize's production graph. PNG decoding for input preparation
builds in a separate workspace.

`prepare` then:

1. decodes and crops the inputs, verifying the frozen source and crop sha256
   values;
2. runs a dry run whose resolved plan (Confirm, 15 × 100, margin, confidence,
   workload id, groups, arms, per-group rounds and time budgets) is checked
   against the frozen plan;
3. runs a selftest that drives the real executable through controlled
   failures: plan overrides on the CLI and in the environment, unrestricted
   placement, group filtering, a missing or temp artifact directory, and
   corrupt or missing input objects.

**`confirm`** verifies the prepared run and the retained executable's sha256,
then runs the executable with the run's artifact directory. It never reuses an
attempt directory.

**Placement.** `--scope-cpus IDS` runs the executable in a transient *system*
scope, `sudo -n systemd-run --scope -p AllowedCPUs=IDS -p MemoryMax=6G
-p MemorySwapMax=0 -- sudo -u $USER env …`, with `--cpu-set=IDS`. A user scope
is not used, because user managers on these hosts are not delegated the cpuset
controller, so `AllowedCPUs` there is silently ignored. `--pin IDS` sets
zenbench's measured-thread pin `ZENBENCH_PIN`, and it must be a proper subset of
the scope, leaving a CPU for zenbench's resource sampler. Nothing in this
workload runs with `ZENBENCH_GATE_TRACE`: the executable refuses it.

**Placement receipt.** With `--scope-cpus`, `proc-receipt.py watch` runs
outside the scope (on the lowest online CPU outside it, at nice 19) and records
from `/proc` every state change of the benchmark processes:
- each process, identified by its executable;
- its threads, by PID, TID and kernel comm, with each one's `Cpus_allowed_list`;
- its cgroup and `cpuset.cpus.effective`.

It reads nothing from inside the processes. `proc-receipt.py check` then
requires all of the following:
- one top-level process, plus the planned number of children (none in a dry run);
- every thread mask inside the scope;
- the cgroup cpuset equal to the scope;
- zenbench's placement line showing a verified single-class placement on a real
  boundary, with `cpu_effective_set` equal to the scope;
- in a timed run, a `resource-sample` thread in every child, with no mask that
  overlaps the pin.

If the receipt fails, the attempt exits 2 even when the executable exited 0
(`attempts/<…>/placement-check.txt`).

## Artifacts (`<run>/`)

| path | contents |
|---|---|
| `identity/` | as in hfilter-parity, plus `dryrun-plan.json`, `dryrun-check.txt`, `prep-Cargo.lock.*`, `*.packages.added-classified`, `zenbench.*` (harness export), `zenbench-registry-Cargo.lock.*` and `Cargo.lock.provenance-check.txt` (lock provenance), `compiled-sources.{tsv,check.txt}` |
| `artifacts/cas/sha256/` | content-addressed, write-once objects: encoded source PNGs, raw crops, derived inputs, every distinct old/new output and f32 twin, each stored once and byte-verified when seen again |
| `artifacts/inputs/manifest-*.json` | input manifests |
| `artifacts/records/{dryrun,parent,child}-{before,after}-*.json` | per-process identity and parity records |
| `artifacts/zenbench-data/` | zenbench's raw packets (all processes and rounds) |
| `artifacts/acceptance/` | the evaluator's decision |
| `attempts/<utc>-<pid>/` | each `confirm` invocation: command, resources, stdout, stderr, exit code, `cpu-dispatch.txt`, and with `--scope-cpus` `proc-receipt.jsonl` and `placement-check.txt` |
| `selftest/results.tsv` | the selftest's cases |

Run directories hold binaries and raw data: keep them out of Git and back them
up as a unit.
