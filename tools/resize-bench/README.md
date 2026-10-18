# resize-bench

A native [zenbench](https://lib.rs/crates/zenbench) 0.1.11 Confirm comparison
of whole public zenresize resizes. One executable links a baseline build
(`old`) and a candidate build (`new`) of zenresize, each from a verified
`git archive` export with production dependencies, features and build
settings. Every group times three arms, interleaved: `old`, `new`, and
`old-aa`, which runs the baseline again as an A/A control.

`prepare` never times anything. Only `confirm` measures, and it is meant for a
quiet host.

## Frozen plan

Defined in `src/plan.rs` and `src/inputs.rs`, under workload id
`zenresize-rgb-hfilter-whole-resize-v2`.

**Sampling**

- 5 fresh processes, 30 fixed rounds per group, margin 1 %, family confidence
  0.95.
- Supervisor deadline 7200 s for all five processes. Per group: 120 s of
  benchmark time and a 600 s wall limit. These are budgets, not predictions.

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
- all 5 replicates hold every group and arm with 30 completed fixed rounds;
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

## Run

```sh
# Stage, check, build, prepare inputs, dry-run, selftest. No timing.
tools/resize-bench/run.sh prepare --old e595ef8f --new <candidate> \
  --corpus <codec-corpus checkout> --imazen26 <imazen-26 png-v3 checkout> \
  --out <fresh run dir>
# On a quiet host, from the same run directory:
tools/resize-bench/run.sh confirm --run <run dir> [--cpu-set IDS] [--cpu-policy performance]
```

**`prepare`** checks everything `tools/hfilter-parity` checks:

- the tool is committed and staged from `HEAD`;
- every exported file is a verified blob, and the only edit is the alias
  version line;
- the lock is seeded from the candidate, then everything runs `--locked`;
- no rustflags or profile overrides are present;
- the normal dependency graph of both zenresize copies equals a production
  consumer's exactly.

On top of that, in the benchmark workspace, resolution may not change any
package in zenresize's production graph. PNG decoding for input preparation
builds in a separate workspace.

`prepare` then:

1. decodes and crops the inputs, verifying the frozen source and crop sha256
   values;
2. runs a dry run whose resolved plan (Confirm, 5 × 30, margin, confidence,
   groups, arms, per-group rounds and time budgets) is checked against the
   frozen plan;
3. runs a selftest that drives the real executable through controlled
   failures: plan overrides on the CLI and in the environment, unrestricted
   placement, group filtering, a missing or temp artifact directory, and
   corrupt or missing input objects.

**`confirm`** verifies the prepared run and the retained executable's sha256,
then runs the executable with the run's artifact directory. It never reuses an
attempt directory.

## Artifacts (`<run>/`)

| path | contents |
|---|---|
| `identity/` | as in hfilter-parity, plus `dryrun-plan.json`, `dryrun-check.txt`, `prep-Cargo.lock.*`, `*.packages.added-classified` |
| `artifacts/cas/sha256/` | content-addressed, write-once objects: encoded source PNGs, raw crops, derived inputs, every distinct old/new output and f32 twin, each stored once and byte-verified when seen again |
| `artifacts/inputs/manifest-*.json` | input manifests |
| `artifacts/records/{dryrun,parent,child}-{before,after}-*.json` | per-process identity and parity records |
| `artifacts/zenbench-data/` | zenbench's raw packets (all processes and rounds) |
| `artifacts/acceptance/` | the evaluator's decision |
| `attempts/<utc>-<pid>/` | each `confirm` invocation: command, resources, stdout, stderr, exit code |
| `selftest/results.tsv` | the selftest's cases |

Run directories hold binaries and raw data: keep them out of Git and back them
up as a unit.
