# zenresize development recipes

# Format + regenerate the public-API surface snapshots (docs/public-api/).
# The snapshot runner lives in the standalone apidoc/ package, so plain
# `cargo test` never compiles its dependency tree or runs rustdoc.
fmt:
    cargo fmt --all
    cargo test --manifest-path apidoc/Cargo.toml

# Regenerate the public-API surface snapshots only
api-doc:
    cargo test --manifest-path apidoc/Cargo.toml

# Verify the committed snapshots are current
api-doc-check:
    ZEN_API_DOC=check cargo test --manifest-path apidoc/Cargo.toml

# Compare production ARM row kernels against the forced scalar tier on macOS.
arm-kernel-tiers-macos:
    mkdir -p "$HOME/tmp"
    CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4 OMP_NUM_THREADS=4 TMPDIR="$HOME/tmp" nice -n 19 cargo bench --bench kernel_tiers -- --cpu-policy=unrestricted --format=llm > "$HOME/tmp/zenresize-kernel-tiers.log" 2>&1

# Old-vs-new differential of the RGB f32 H-filter (kernel + whole resizes, every
# dispatch tier) from verified git-archive builds. See tools/hfilter-parity/README.md.
# Example: just hfilter-parity e595ef8f HEAD ~/tmp/hfp-run1 none quick
[positional-arguments]
hfilter-parity old new out features scope:
    tools/hfilter-parity/run.sh --old "$1" --new "$2" --out "$3" --features "$4" --scope "$5"

# Native zenbench Confirm comparison of whole public resizes (old / new / old A/A).
# prepare never times anything; run confirm only on a quiet host. The frozen plan
# reads one input from an imazen-26 variant/png-v3 checkout. See tools/resize-bench/README.md.
# Example: just resize-bench-prepare e595ef8f HEAD ~/work/codec-corpus ~/work/zen/imazen-26-png-v3 ~/tmp/rb-run1
[positional-arguments]
resize-bench-prepare old new corpus imazen26 out:
    tools/resize-bench/run.sh prepare --old "$1" --new "$2" --corpus "$3" --imazen26 "$4" --out "$5"

# Extra arguments pass through: --cpu-set IDS, --cpu-policy auto|performance|efficiency, --dry-run.
[positional-arguments]
resize-bench-confirm run *args:
    run="$1"; shift; tools/resize-bench/run.sh confirm --run "$run" "$@"
