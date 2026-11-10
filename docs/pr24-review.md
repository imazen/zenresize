# PR 24 investigation (2026-11-10)

Reviewed [PR 24](https://github.com/imazen/zenresize/pull/24), head
`8712e96e0740841dce5604e3e38d89118b76dfbc`, against main
`01e66aace1b6ab90d45914fc4342cb3d0d453a4e`.

## Straight-color regression blocks merging

`apply_mask_i16` rounds all four premultiplied channels to integer levels
before unpremultiplication. At low coverage, this destroys straight color.
For an identity Box resize in sRGB, an opaque `[128, 64, 32, 255]` pixel
under a constant `1/255` mask produces:

| implementation | straight RGBA8 |
|---|---|
| current main | `[128, 64, 32, 1]` |
| PR head | `[255, 0, 0, 1]` |
| PR automatically merged with current main | `[255, 0, 0, 1]` |

The coverage multiplier quantizes to Q16 value 257; rounded premultiplied
RGBA becomes `[1, 0, 0, 1]`. Unpremultiplication expands red to 255.
Re-premultiplying either final pixel to RGBA8 gives `[1, 0, 0, 1]`, so
the PR's `mask_i16_matches_f32` comparison cannot detect this example.
That existing test passes on the merged source while the new raw-output
regression fails.
The public method's claim that output differs by only one additional level
does not describe straight RGBA output.

`just mask-low-alpha-test` preserves current main's behavior. It covers
widths 1/17/33, single and paired output rows, and four coverages including
`1/255` and `2/255`. The same test fails on both PR source trees above.
The automatic merge retains `encode_i16_output` from PR 29; losing that
encoder is not the cause of this failure.

Keep color precision through mask application and output encoding, rather
than accepting this failure by weakening the regression test. One possible
design is to cancel positive mask coverage from the straight-color ratio
and apply it separately to output alpha, with explicit handling of zero and
the f32 unpremultiply threshold. That design has not been implemented or
validated here.

## Dependency and integration

The PR temporarily patches zenblend to git revision
`ee999939524052221f1f5674c2eaec76c6299310`, still labeled version 0.1.3.
Its own merge instructions require a zenblend release containing the span
and no_std fixes, then replacing this git patch with the released dependency.
On the review date, `cargo read zenblend --path-only` resolves published
0.1.3 and the [release list](https://github.com/imazen/zenblend/releases)
contains no 0.1.4. That prerequisite remains outstanding.

The dependency bug is also reproducible on current main. `just mask-span-probe`
compares `RoundedRectMask` spans with the same mask wrapped to use only
`fill_mask_row`. A 17×17 identity Box resize, radius 5.5, produces transparent
black at (5, 0) via spans and `[128, 64, 32, 255]` via full-row fill. Ten
pixels disagree, with maximum alpha difference 255. The other two probe
cases (33×21/radius 7.5 and 64×32/radius 9) have 14 and 36 disagreements,
also with alpha difference 255. In published zenblend 0.1.3,
`apply_mask_spans` expands partial spans into neighboring opaque pixels;
`RoundedRectMask::mask_spans` does not write those mask-buffer positions.
This is an existing dependency bug, separate from the PR's integer precision
regression. The diagnostic records differences; it is not a passing parity
test and does not fix the dependency.

`git merge-tree --write-tree origin/main 8712e96e` finds one conflict, in
CHANGELOG.md; the Rust source merges automatically. The resulting source
was compiled and used for the regression reproduction above. Historical
PR-head CI was green, but it does not cover the newly added regression.

## Performance evidence needs refreshing

The PR's committed October 20 measurements use baseline `93317e51` on a
Ryzen 9 7900X, with AVX-512 disabled and only four rounds per group. They
compare masked-versus-unmasked ratios within two separate executable runs.
They are evidence for that revision and machine, not a directly paired
before/after comparison against today's main. Both f32 and integer kernels
have changed since then. Measure updated candidate versus current main
in one paired experiment, including rounded and full-row partial masks,
before asserting a current speedup or absence of regressions.

## Reproduction

Run the new test on main and copy it into source exports of the PR head and
the merge-tree result. Each export is a separate source directory, with
its own manifest/lockfile and no tracked checkout changes. Commands:

```sh
just mask-low-alpha-test
cargo test --manifest-path <pr-export>/Cargo.toml --test mask_low_alpha -- --nocapture
```

The test executions used runtime dispatch, without target-cpu=native.
This review makes no new performance claim.
