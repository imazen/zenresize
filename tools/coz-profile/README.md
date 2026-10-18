# coz-profile

Stages two source-exact arms of the coz harness `benches/profile_coz.rs`
(baseline and post) and builds them. It does not profile anything; running
`coz` is a separate step.

```sh
tools/coz-profile/stage.sh --base e595ef8f --post <candidate> --out <fresh dir>
```

- **One harness for both arms.** Each arm, `<out>/src/<arm>/zenresize/`, is
  every file of the tool commit (`HEAD`) except `src/`, plus `src/` from the
  arm's revision. Both are verified `git archive` exports. The two arms share
  `Cargo.toml`, `Cargo.lock`, the justfile and `benches/profile_coz.rs` byte for
  byte and differ only in `src/`; `stage.sh` checks this. The directory is named
  `zenresize` so the recipe's scope `%/zenresize/src/%` matches.
- **Build.** Each arm is built with the committed `just coz-build` recipe in its
  own target directory: `CARGO_PROFILE_RELEASE_DEBUG=1`, `--locked`, and the
  executable path taken from Cargo's JSON artifact message.
- **Recorded in `identity/`:**
  - the binary's sha256 and the presence of `.debug_line`;
  - the lock's sha256;
  - every compiled source, hashed and checked against the arm's export
    (`compiled-sources-<arm>.tsv`);
  - that `rgb3` is accepted and an unknown mode refused (0 rounds, so no resize
    runs).
- **Profiling commands.** `identity/coz-commands.txt` holds the exact `just
  coz-run <new profile> <rounds> rgb3` command per arm. Each rebuilds as a no-op
  and prints the binary's sha256, which must equal `binaries.tsv`.

The harness's pixel content is synthetic (`make_row`). Mode `rgb3` runs only
the RGB8-linear streaming resize, which reaches the 3-channel f32 horizontal
filter. Real photographs are measured by `tools/resize-bench`.
