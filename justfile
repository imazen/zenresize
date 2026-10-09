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

# Exercise the u8 progress integration and its ordinary cancellation contract.
check-progress:
    cargo fmt -p zenresize -- --check
    cargo test -p zenresize --test how_far_progress
    cargo run --example progress
    cargo clippy -p zenresize --all-targets -- -D warnings

# Full regression suite for changes shared with the existing resize path.
check-progress-full: check-progress
    cargo test -p zenresize
    cargo test -p zenresize --no-default-features --lib
    cargo test -p zenresize --features pretty-safe --lib
