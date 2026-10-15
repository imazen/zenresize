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
