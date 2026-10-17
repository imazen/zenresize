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

# Resampling-convention oracles (benchmarks/conventions_2026-10-15). `python`
# is a venv with that directory's requirements.txt; labels must be new.
# Build the driver, then compare it with the pinned frameworks on the dense panel.
conventions-compare python label:
    cargo build --locked --example conventions
    {{python}} benchmarks/conventions_2026-10-15/compare.py "${CARGO_TARGET_DIR:-target}/debug/examples/conventions" {{label}} --dense

# Re-run the public-API probes behind the pinned constants and unit-test pins.
conventions-probes python label:
    {{python}} benchmarks/conventions_2026-10-15/probe_gaussian.py
    {{python}} benchmarks/conventions_2026-10-15/pin_gaussian_support.py
    {{python}} benchmarks/conventions_2026-10-15/probe_coordinates.py
    {{python}} benchmarks/conventions_2026-10-15/probe_torch_layout.py
    {{python}} benchmarks/conventions_2026-10-15/probe_cv_fixed.py {{label}}
    {{python}} benchmarks/conventions_2026-10-15/fixed_point.py {{label}}
    {{python}} benchmarks/conventions_2026-10-15/pins.py

# Write the all-filter default-output byte capture to `out`.
conventions-defaults out:
    cargo run --locked --example conventions_defaults -- {{out}}

# Default-configuration A/B vs main (benchmarks/pr18_ab_2026-10-17): timed, then
# retired instructions (needs perf). The timed run pins one CCD on asymmetric-L3 hosts.
pr18-ab:
    cd benchmarks/pr18_ab_2026-10-17 && taskset -c 0-7,16-23 cargo bench -- --format=llm --cpu-policy=unrestricted
    benchmarks/pr18_ab_2026-10-17/count.sh
    benchmarks/pr18_ab_2026-10-17/count.sh new-only
