# PR 23 investigation (2026-11-10)

Reviewed [PR 23](https://github.com/imazen/zenresize/pull/23), head
`403a1eeacae0940143e49394397149efe941b724`, after completing
[the PR 24 review](pr24-review.md). Integration baseline is main
`cf96080c4affeddf15dcb01355307b1de2532f3f`, whose production code
matches `01e66aac`.

## Assessment

The patch fixes a real omission. On main,
`next_output_row_premul_linear_f32` and `next_output_row_over` call
`produce_next_f32_raw` and bypass the mask. PR 23 factors the existing
mask operation into `apply_mask_f32` and calls it from those two outputs,
before external compositing. The ordinary composited output still applies
the mask once, in the same location as before. `produce_next_f32_raw`
increments the content-row counter, so `output_rows_produced - 1` selects
the correct mask row. No allocation, second mask pass, or public signature
change is introduced.

No blocker was found in this patch. It does not depend on the integer-mask
optimization in PR 24, and should be considered independently. The existing
zenblend 0.1.3 span bug remains a separate correctness issue; this patch
does not repair it.

## Independent verification

`just mask-output-probe` uses a custom full-row mask, independent of
rounded-rectangle span hints. Coverage cycles through 0, 1/4, 1/2, 3/4,
and 1, shifts with output row, and is applied to varying input RGB and
alpha. The probe compares:

- every premultiplied f32 component against unmasked raw output multiplied
  by that pixel's coverage, bit for bit;
- caller-side SrcOver output against ordinary owned-background output,
  byte for byte, for transparent and opaque white backgrounds.

The matrix uses Box, MagicKernelSharp, and MagicKernelSharp2021, and
20×20→13×17, 17×1→33×4, and 33×17→17×9 shapes. Each of the nine
configurations runs through eight runtime token permutations available in
the x86-64 build, including the disabled-token fallback. On current main,
all nine configurations disagree. For example, the first Box case has
712 incorrect premultiplied components and 844 differing output bytes
summed over the two backgrounds. On PR 23 merged with current main,
every comparison is exact in all nine configurations and eight permutations.
The probe is diagnostic and reports differences; it does not treat its
own successful process exit as proof of parity.

The PR's original regression test also passes on the combined source.
Removing just the two newly added mask calls makes that same test fail;
the diagnostic source export was restored afterward.
The combined library suite, golden output suite, and low-alpha mask
regression pass. Clippy passes for the new probe. Runtime dispatch was
used without target-cpu=native. No new performance measurement was made.

The PR's test uses a rounded mask and compares outputs that share the same
mask implementation, so it does not expose the dependency span bug found
in PR 24. It checks alpha, rather than RGB, for the premultiplied API.
The independent probe fills those two gaps. Promote that coverage to
asserting integration tests when landing the fix.

## Integration

`git merge-tree --write-tree origin/main 403a1eea` reports one conflict in
CHANGELOG.md; streaming.rs merges automatically. The review compiled and
tested that merged source export, without changing the PR branch.
Historical PR CI was green on all reported jobs, but predates current main.

The existing `with_mask` doc comment said the mask runs after compositing;
the source runs it before compositing. That factual comment is corrected
on main in this review. Both PR 23's new calls use the intended order.
Neither PR was merged as part of this investigation.
