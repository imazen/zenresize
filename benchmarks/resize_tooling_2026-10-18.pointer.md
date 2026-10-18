# Pointer: resize tooling validation evidence (2026-10-18)

The raw run directories behind `hfilter_parity_2026-10-18.md`, generated data well over
30 KB, live outside git.

- **Location (checked on dev):** `dev:/mnt/v/output/zenresize/tooling-2026-10-18-r7900x/`
  (7.4 GB). It was copied from r7900x `~/tmp` with `rsync -aH`, without deletions, and the
  originals stay in place.
- **Not copied:** only the four Cargo `target/` directories. Each was verified as Cargo output
  (a `CACHEDIR.TAG` with Cargo's signature, plus `.rustc_info.json`).
- **No tower mirror yet.**
- **`MANIFEST.sha256`** lists every mirrored file, with paths relative to the location.
  - 91,959 entries; its own sha256 is
    `14cbaca72b1a5ea75fab92e71be6b810862c98b2fbaf9574b4e2fc90bda8eed6`.
  - `sha256sum -c` passed on dev for every entry.
  - Check it with `cd <location> && sha256sum -c --quiet MANIFEST.sha256`.

| path under the location | what | sha256 |
|---|---|---|
| `zr-tooling-2026-10-18/final/` | hfilter-parity quick/full/avx512+pretty-safe, both fault-detection mutants, and the first `resize-bench prepare` (workload v1, 21 groups) | `summary.tsv` `b3c4f596…` |
| `zr-tooling-2026-10-18/largeinput-prepare-2/` | `resize-bench prepare` for workload v2 (25 groups, native 12 MP input), tools at `b1109347` | input manifest `a8280e74…` |
| `…/largeinput-prepare-2/artifacts/cas/sha256/15/15cd2f7c…` | the 12 MP source PNG (4000×3000, 8-bit RGB, `cICP` BT.709/sRGB) | `15cd2f7c2648e3238f41456ceedc8330f6c2bf53a61a1a2f560e61b96d1e7ee6` |
| `…/largeinput-prepare-2/artifacts/cas/sha256/df/df7a242e…` | its full-frame raw RGB8 crop, 36,000,000 B; an independent stdlib-zlib decode gives the same hash | `df7a242ea5728cc9c5024f83aaecb9db91dc50c835909d2a4e3f2a4ab9b42777` |
| `…/largeinput-prepare-2/bin/resize-bench`, `bin/resize-bench-prep` | the retained bench and prep binaries | `e4e620a7…`, `27e267ed…` |
| `zenresize-promo/rb-prepare-main/` | the same `prepare` re-run from the tooling as landed on `main` (inputs re-staged from the CAS above). PASS; parity rows identical to `largeinput-prepare-2` | bench binary `f332bdd4…` (it embeds its run directory path, so the hash differs) |
| `zenresize-promo/hfp-quick-none-main/` | hfilter-parity quick from the landed tooling. PASS; totals identical to `final/hfp-quick-none` | — |
| `zenresize-promo/bundle/zenresize-c0-b110-experiment.bundle` | git bundle of the unpublished experiment stack (`e595ef8f..b1109347`, head `b1109347`) that the tools were validated against | `64e0fd63a0df8852d1619b5de837ffa1b3525c954f2d9938ffaecb9e4b0c1f14` |
| `zenresize-promo/bundle/mutant-{920ff089…,096d7176…}.patch`, `handoff-zenresize-independent-review/artifacts/mutants/` | the two fault-detection mutants (unreferenced commits) | `babc3daf…`, `999247d6…`; `d91af608…`, `ab849559…` |
| `zenresize-provenance-2026-10-18/` | proof that the 12 MP source is committed imazen-26 content (below) | `verified.json` `d302512e…` |

## Provenance of the 12 MP source

The source was read from an `imazen-26` `variant/png-v3` checkout whose working copy was dirty
(in `MANIFEST.tsv` files). The file is nonetheless committed content, as the
`zenresize-provenance-2026-10-18/` records show. They were fetched from GitHub's Git objects,
without reading the local checkout.

- Commit `b1faae550eba4d29efee1de1a32954ef11a0e1ed` (`tree.json`) holds the path
  `png-v3/1000-lilith-photos-general/1001_general_oceanfront-resort-view_sheraton-waikiki-east-honolulu_s21u_iso32-f1p8_20210602-055442_4000x3000.sdr.png`
  as blob `bed1dcfc6d619a2ffa486b650d9554963b0f3776`.
- That blob is the Git LFS pointer `oid sha256:15cd2f7c2648e3238f41456ceedc8330f6c2bf53a61a1a2f560e61b96d1e7ee6`,
  `size 15358868` (`blob.json`, `pointer.txt`; `git hash-object pointer.txt` = `bed1dcfc…`).
- That oid and size equal the source the benchmark froze (`tools/resize-bench/src/inputs.rs`) and
  the CAS object above.

The file carries no camera metadata, so "native camera size, never resampled" rests on the
corpus. One pixel heuristic agrees with it: the ratio of lag-1 to lag-2 mean |second difference|
is 0.580, against 0.578 for the native CLIC 2048×1358 input and 0.35–0.37 for both after a
synthetic 2× linear upscale. That heuristic cannot detect fractional resampling.
