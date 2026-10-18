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

# Coz causal profile of benches/profile_coz.rs (Linux; needs `coz` on PATH or
# COZ_BIN, and jq). CARGO_PROFILE_RELEASE_DEBUG=1 gives the build line tables; the
# executable is the path in Cargo's JSON artifact message, never a deps/ glob.
# PROFILE must be a new path: an existing file is refused, never overwritten.
# Example: just coz-run ~/tmp/zenresize-coz-$(date -u +%Y%m%dT%H%M%SZ).jsonl 1000
[positional-arguments]
coz-run profile rounds="1000":
    #!/usr/bin/env bash
    set -euo pipefail
    profile=$1 rounds=$2
    if [ -e "$profile" ]; then echo "error: $profile exists; coz-run only writes a new profile" >&2; exit 2; fi
    command -v jq >/dev/null || { echo "error: coz-run needs jq to read Cargo's artifact messages" >&2; exit 2; }
    bin=$(CARGO_PROFILE_RELEASE_DEBUG=1 cargo build --release --locked --bench profile_coz --message-format=json-render-diagnostics | jq -r 'select(.reason == "compiler-artifact" and .target.name == "profile_coz" and .executable != null) | .executable')
    if [ -z "$bin" ] || [ "$(printf '%s\n' "$bin" | wc -l)" -ne 1 ] || [ ! -x "$bin" ]; then echo "error: Cargo reported no single profile_coz executable: '$bin'" >&2; exit 2; fi
    echo "coz-run: $bin sha256=$(sha256sum "$bin" | cut -d' ' -f1) rounds=$rounds -> $profile" >&2
    "${COZ_BIN:-coz}" run --source-scope '%/zenresize/src/%' --output "$profile" --- "$bin" "$rounds"
