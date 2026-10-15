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

# Caller serializes heavy work with flock/run-heavy.
conventions-defaults out:
    cargo run --locked --example conventions_defaults -- {{out}}

conventions-ci:
    cargo test --locked
    cargo test --locked --no-default-features --lib
    cargo test --locked --features pretty-safe
    cargo build --locked --no-default-features
    cargo clippy --locked --all-targets -- -D warnings
    cargo fmt --all -- --check

conventions-gaussian-probe python:
    {{python}} benchmarks/conventions_2026-10-15/probe_gaussian.py

conventions-build:
    cargo build --locked --examples

conventions-test:
    cargo test --locked --test conventions

conventions-compare python label *flags:
    {{python}} benchmarks/conventions_2026-10-15/compare.py "$CARGO_TARGET_DIR/debug/examples/conventions" {{label}} {{flags}}

conventions-fixed-study python label:
    {{python}} benchmarks/conventions_2026-10-15/fixed_point.py {{label}}

conventions-layout-probe python:
    {{python}} benchmarks/conventions_2026-10-15/probe_torch_layout.py

conventions-coordinate-probe python:
    {{python}} benchmarks/conventions_2026-10-15/probe_coordinates.py

conventions-cv-fixed-probe python label:
    {{python}} benchmarks/conventions_2026-10-15/probe_cv_fixed.py {{label}}

conventions-pins python:
    {{python}} benchmarks/conventions_2026-10-15/pins.py

conventions-gaussian-support-pins python:
    {{python}} benchmarks/conventions_2026-10-15/pin_gaussian_support.py

# Round-8 review regressions, including the deterministic pre-change output hash.
conventions-review-tests:
    cargo test --locked --test conventions --test default_conventions_hash
