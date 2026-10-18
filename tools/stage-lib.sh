# shellcheck shell=bash
# Shared fail-closed staging helpers for zenresize's generated dev tools
# (tools/hfilter-parity, tools/resize-bench). Sourced by their run.sh, never
# executed directly. Requires bash, git, cargo, rustc, tar, sha256sum, realpath.
#
# A run builds in a fresh caller-chosen directory outside the repository:
#   <run>/src/<side>/   git-archive export of a revision (old = baseline, new = candidate)
#   <run>/tool/         git-archive export of the tool at the repository's HEAD
#   <run>/ws/           generated Cargo workspace: aliased path deps, seeded lock
#   <run>/refws-<side>/ minimal production consumer, for the dependency-graph check
#   <run>/identity/     source, toolchain, lock, graph and binary identity records
#   <run>/logs/         complete command logs (never truncated)
# Every helper stops the run on the first failure.

zr_die() {
	echo "error: $*" >&2
	exit 1
}

zr_log() {
	echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >&2
}

zr_require_cmds() {
	local c
	for c in "$@"; do
		command -v "$c" >/dev/null 2>&1 || zr_die "required command not found: $c"
	done
}

zr_sha256() {
	sha256sum -- "$1" | cut -d' ' -f1
}

# Run a command with complete stdout+stderr in <run>/logs/<name>.log and stop on
# a nonzero exit. The log path is printed so a failure can be read in full.
zr_run_logged() {
	local name=$1
	shift
	local log="$ZR_RUN/logs/$name.log"
	[ ! -e "$log" ] || zr_die "log $log already exists"
	zr_log "$name: $*"
	local rc=0
	"$@" >"$log" 2>&1 || rc=$?
	if [ "$rc" -ne 0 ]; then
		echo "error: step '$name' failed with exit code $rc; full log: $log" >&2
		exit "$rc"
	fi
}

# Create a fresh run directory on disk-backed storage. Refuses /tmp, /run,
# /dev/shm and /var/tmp (wiped or RAM-backed on many hosts) and any existing path.
zr_new_run_dir() {
	local dir parent fstype
	dir=$(realpath -m -- "$1")
	case "$dir/" in
	/tmp/* | /run/* | /dev/shm/* | /var/tmp/*)
		zr_die "run dir $dir is under a temporary or RAM-backed path; choose a disk-backed directory"
		;;
	esac
	[ ! -e "$dir" ] || zr_die "run dir $dir already exists; every run needs a fresh directory"
	parent=$(dirname -- "$dir")
	[ -d "$parent" ] || zr_die "parent directory $parent does not exist"
	if command -v findmnt >/dev/null 2>&1; then
		fstype=$(findmnt -no FSTYPE -T "$parent" || true)
		case "$fstype" in
		tmpfs | ramfs) zr_die "run dir parent $parent is on $fstype" ;;
		esac
	fi
	mkdir -- "$dir"
	mkdir -- "$dir/src" "$dir/identity" "$dir/logs" "$dir/bin"
	ZR_RUN=$dir
}

# Resolve a revision to a full commit id.
zr_resolve_commit() {
	git -C "$ZR_REPO" rev-parse --verify --quiet "$1^{commit}" || zr_die "cannot resolve commit '$1'"
}

# The tool runs exactly as committed: its working-tree files must equal HEAD,
# and it is staged from HEAD (git archive), never from the working tree.
zr_require_committed() {
	local path
	for path in "$@"; do
		git -C "$ZR_REPO" diff --quiet HEAD -- "$path" ||
			zr_die "$path differs from HEAD; commit the tool before running it"
		[ -z "$(git -C "$ZR_REPO" ls-files --others --exclude-standard -- "$path")" ] ||
			zr_die "$path has untracked files; commit the tool before running it"
	done
}

# Export <commit> (optionally limited to pathspecs) into <dest> with git
# archive, then prove every exported file is byte-identical to its committed
# blob and that nothing else was written. Records a sha256 manifest.
# Usage: zr_export_tree <commit> <dest> <identity-name> [pathspec...]
zr_export_tree() {
	local sha=$1 dest=$2 name=$3
	shift 3
	[ ! -e "$dest" ] || zr_die "export destination $dest already exists"
	mkdir -p -- "$dest"
	git -C "$ZR_REPO" archive --format=tar "$sha" -- "$@" | tar -x -C "$dest" ||
		zr_die "git archive of $sha failed"
	local listing="$ZR_RUN/identity/$name.ls-tree"
	git -C "$ZR_REPO" ls-tree -r --full-tree "$sha" -- "$@" >"$listing" ||
		zr_die "git ls-tree of $sha failed"
	[ -s "$listing" ] || zr_die "nothing to export from $sha ${*:-}"
	if grep -qv '^100644 blob \|^100755 blob ' "$listing"; then
		zr_die "$sha contains non-regular entries (symlinks/submodules); export verification does not support them"
	fi
	local expected="$ZR_RUN/identity/$name.expected-blobs" actual="$ZR_RUN/identity/$name.actual-blobs"
	awk -F'\t' '{split($1, m, " "); print m[3] "\t" $2}' "$listing" | LC_ALL=C sort -k2 >"$expected"
	cut -f2 "$expected" | sed "s|^|$dest/|" >"$ZR_RUN/identity/$name.paths"
	git -C "$ZR_REPO" hash-object --no-filters --stdin-paths <"$ZR_RUN/identity/$name.paths" >"$actual.hashes" ||
		zr_die "hashing exported files failed"
	paste "$actual.hashes" <(cut -f2 "$expected") >"$actual"
	rm -f -- "$actual.hashes" "$ZR_RUN/identity/$name.paths"
	cmp -s "$expected" "$actual" || zr_die "exported $sha does not match its committed blobs (see $expected vs $actual)"
	local files_on_disk files_expected
	files_on_disk=$(find "$dest" -type f | wc -l)
	files_expected=$(wc -l <"$expected")
	[ "$files_on_disk" -eq "$files_expected" ] ||
		zr_die "export of $sha has $files_on_disk files, expected $files_expected"
	(cd "$dest" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) >"$ZR_RUN/identity/$name.files.sha256"
	printf '%s\tcommit=%s\tfiles=%s\tmanifest_sha256=%s\n' "$name" "$sha" "$files_expected" \
		"$(zr_sha256 "$ZR_RUN/identity/$name.files.sha256")" >>"$ZR_RUN/identity/sources.tsv"
}

# Two path packages of one name need distinct versions to share a lockfile.
# Change only the [package] version line and prove nothing else moved.
zr_alias_version() {
	local dir=$1 suffix=$2 name=$3 cur changed
	cp -- "$dir/Cargo.toml" "$ZR_RUN/identity/$name.Cargo.toml.archived"
	cur=$(awk '/^\[package\]/{p=1; next} /^\[/{p=0} p && /^version = "/{gsub(/^version = "|"$/, ""); print; exit}' "$dir/Cargo.toml")
	[ -n "$cur" ] || zr_die "no [package] version in $dir/Cargo.toml"
	sed -i "0,/^version = \"$cur\"$/s//version = \"$cur-$suffix\"/" "$dir/Cargo.toml"
	changed=$(diff "$ZR_RUN/identity/$name.Cargo.toml.archived" "$dir/Cargo.toml" | grep -c '^[<>]' || true)
	[ "$changed" -eq 2 ] || zr_die "alias edit of $dir/Cargo.toml changed $changed lines, expected exactly one"
	grep -qx "version = \"$cur-$suffix\"" "$dir/Cargo.toml" || zr_die "alias edit of $dir/Cargo.toml failed"
	printf '%s\talias_version=%s-%s\n' "$name" "$cur" "$suffix" >>"$ZR_RUN/identity/sources.tsv"
	echo "$cur-$suffix"
}

# Reject build inputs that would make this differ from a production build:
# CPU-specific code generation, profile overrides and rustflags of any kind
# (environment or any Cargo config file that applies under <start-dir>). Writes
# nothing, so it runs before the run directory is created.
zr_check_build_env() {
	local start=$1 var d f
	for var in RUSTFLAGS CARGO_ENCODED_RUSTFLAGS CARGO_BUILD_RUSTFLAGS RUSTC CARGO_BUILD_TARGET; do
		[ -z "${!var:-}" ] || zr_die "$var is set ('${!var}'); unset it so the build uses production defaults"
	done
	if env | grep -qE '^CARGO_TARGET_[A-Z0-9_]+_RUSTFLAGS=|^CARGO_PROFILE_'; then
		zr_die "CARGO_TARGET_*_RUSTFLAGS or CARGO_PROFILE_* is set; unset it so the build uses production defaults"
	fi
	ZR_CARGO_CONFIG_FILES=()
	d=$(realpath -m -- "$start")
	while :; do
		for f in "$d/.cargo/config.toml" "$d/.cargo/config"; do
			[ -f "$f" ] && ZR_CARGO_CONFIG_FILES+=("$f")
		done
		[ "$d" = / ] && break
		d=$(dirname -- "$d")
	done
	for f in "${CARGO_HOME:-$HOME/.cargo}/config.toml" "${CARGO_HOME:-$HOME/.cargo}/config"; do
		[ -f "$f" ] && ZR_CARGO_CONFIG_FILES+=("$f")
	done
	for f in "${ZR_CARGO_CONFIG_FILES[@]}"; do
		if grep -qE 'rustflags|target-cpu|target-feature|^\[profile' "$f"; then
			zr_die "Cargo config $f sets rustflags/target-cpu/target-feature/profile; the build must use production defaults"
		fi
	done
}

# Record the toolchain, Rust/Cargo environment and applicable Cargo config
# files (re-running the checks above from the workspace directory).
zr_record_toolchain() {
	local id="$ZR_RUN/identity" f
	zr_check_build_env "$1"
	rustc -vV >"$id/rustc-vV.txt" || zr_die "rustc -vV failed"
	cargo -vV >"$id/cargo-vV.txt" || zr_die "cargo -vV failed"
	env | grep -E '^(RUST|CARGO)' | LC_ALL=C sort >"$id/env-rust-cargo.txt" || true
	: >"$id/cargo-config-files.tsv"
	for f in "${ZR_CARGO_CONFIG_FILES[@]}"; do
		printf '%s\t%s\n' "$f" "$(zr_sha256 "$f")" >>"$id/cargo-config-files.tsv"
	done
}

# Seed a generated workspace's lockfile from <commit>:Cargo.lock, resolve only
# what the tool adds (no --locked yet), record the result, and from then on use
# --locked. Writes identity/<name>.{seed,resolved,diff}; <name> defaults to
# Cargo.lock (the main tool workspace).
# Usage: zr_seed_and_resolve_lock <commit> <ws> [<name>]
zr_seed_and_resolve_lock() {
	local commit=$1 ws=$2 name=${3:-Cargo.lock}
	local id="$ZR_RUN/identity"
	git -C "$ZR_REPO" show "$commit:Cargo.lock" >"$ws/Cargo.lock" || zr_die "no Cargo.lock at $commit"
	cp -- "$ws/Cargo.lock" "$id/$name.seed"
	(cd "$ws" && zr_run_logged "lock-resolve-$name" cargo metadata --format-version 1)
	cp -- "$ws/Cargo.lock" "$id/$name.resolved"
	printf '%s\tseed\t%s\t%s\n%s\tresolved\t%s\n' "$name" "$commit" "$(zr_sha256 "$id/$name.seed")" \
		"$name" "$(zr_sha256 "$id/$name.resolved")" >>"$id/lock.tsv"
	diff "$id/$name.seed" "$id/$name.resolved" >"$id/$name.diff" || true
}

# "name version" per [[package]] entry of a Cargo.lock.
zr_lock_packages() {
	awk '/^\[\[package\]\]/{n=""} /^name = "/{gsub(/^name = "|"$/, ""); n=$0} /^version = "/{gsub(/^version = "|"$/, ""); if (n != "") print n " " $0}' "$1" | LC_ALL=C sort -u
}

# Resolution may only remove packages, add the aliased zenresize copies and the
# tool's own package, and add packages the tool brings in for itself. A name
# already locked by the seed may not gain a version when it is protected:
# with <protected-graph> = "-" every seed name is protected (no additions at
# all); with a graph file (a zr_normal_tree output), names in zenresize's
# production normal graph are protected and the harness's own closure may
# resolve freely. Every addition is recorded.
# Usage: zr_check_lock_subset <name> <tool-package-name> <protected-graph|->
zr_check_lock_subset() {
	local name=$1 tool_pkg=$2 protected=$3
	local id="$ZR_RUN/identity" pkg version bad=0 tag
	zr_lock_packages "$id/$name.seed" >"$id/$name.packages.seed"
	zr_lock_packages "$id/$name.resolved" >"$id/$name.packages.resolved"
	LC_ALL=C comm -13 "$id/$name.packages.seed" "$id/$name.packages.resolved" >"$id/$name.packages.added"
	LC_ALL=C comm -23 "$id/$name.packages.seed" "$id/$name.packages.resolved" >"$id/$name.packages.removed"
	: >"$id/$name.packages.added-classified"
	while read -r pkg version; do
		case "$pkg" in
		zenresize | "$tool_pkg") tag="alias-or-tool" ;;
		*)
			if [ "$protected" = - ]; then
				tag="UNDECLARED"
			elif sed -E 's/^[0-9]+//; s/ v[^ ]+.*$//' "$protected" | grep -qxF -- "$pkg"; then
				tag="PROTECTED-CHANGED"
			else
				tag="harness-only"
			fi
			;;
		esac
		printf '%s %s\t%s\n' "$pkg" "$version" "$tag" >>"$id/$name.packages.added-classified"
		case "$tag" in
		UNDECLARED | PROTECTED-CHANGED)
			echo "error: lock resolution added $pkg $version ($tag)" >&2
			bad=1
			;;
		esac
	done <"$id/$name.packages.added"
	[ "$bad" -eq 0 ] || zr_die "$name: resolution changed packages it may not (see $id/$name.packages.added-classified)"
	printf '%s\tpackages_seed=%s\tresolved=%s\tadded=%s\tremoved=%s\n' "$name" \
		"$(wc -l <"$id/$name.packages.seed")" "$(wc -l <"$id/$name.packages.resolved")" \
		"$(wc -l <"$id/$name.packages.added")" "$(wc -l <"$id/$name.packages.removed")" >>"$id/lock.tsv"
}

# Normalized, complete (`--no-dedupe`) `cargo tree -e normal` subtree of the
# depth-1 dependency `zenresize v<version>` under the workspace root built with
# <features>: packages plus enabled features, paths and the alias suffix
# stripped, depths rebased to the zenresize node.
# Usage: zr_normal_tree <ws> <version> <out> <alias-suffix> [<features-csv>]
zr_normal_tree() {
	local ws=$1 version=$2 out=$3 suffix=$4 features=${5:-}
	local feature_args=()
	[ -z "$features" ] || feature_args=(--features "$features")
	(cd "$ws" && cargo tree --locked -e normal --no-dedupe "${feature_args[@]}" \
		--prefix depth -f '{p} [{f}]' --color never) >"$out.raw" ||
		zr_die "cargo tree in $ws failed"
	awk -v want="zenresize v$version " '
		{ match($0, /^[0-9]+/); depth = substr($0, 1, RLENGTH) + 0; rest = substr($0, RLENGTH + 1) }
		!found && depth == 1 && index(rest, want) == 1 { found = 1; print depth - 1 rest; next }
		found && depth <= 1 { exit }
		found { print depth - 1 rest }
		END { if (!found) exit 3 }
	' "$out.raw" | sed -E 's/ \([^)]*\)//; s/(zenresize v[0-9.]+)-'"$suffix"' /\1 /' >"$out" ||
		zr_die "zenresize v$version is not a depth-1 dependency in $out.raw"
}

# Prove the aliased <side> package sees exactly the normal dependency graph
# (versions and enabled features) that a minimal production consumer gets from
# the same revision, lock seed and declared zenresize features. <extra-deps> is
# TOML for dependencies the tool deliberately adds on top of production (empty
# for benchmarks); it must be declared, not discovered.
# Usage: zr_check_normal_graph <side> <alias-version> <suffix> <features-csv> <extra-deps-toml>
zr_check_normal_graph() {
	local side=$1 alias_version=$2 suffix=$3 features=$4 extra_deps=$5
	local ref="$ZR_RUN/refws-$side" feature_toml=""
	if [ -n "$features" ]; then
		feature_toml=", features = [\"${features//,/\", \"}\"]"
	fi
	mkdir -- "$ref" "$ref/src"
	cat >"$ref/Cargo.toml" <<EOF
# Generated by tools/stage-lib.sh: a minimal production consumer of the $side
# zenresize export, used only as the dependency-graph reference.
[package]
name = "zr-graph-reference-$side"
version = "0.0.0"
edition = "2024"
publish = false

[workspace]

[dependencies]
zenresize = { path = "../src/$side"$feature_toml }
$extra_deps
EOF
	echo 'fn main() {}' >"$ref/src/main.rs"
	cp -- "$ZR_RUN/identity/Cargo.lock.seed" "$ref/Cargo.lock"
	(cd "$ref" && zr_run_logged "graph-reference-resolve-$side" cargo metadata --format-version 1)
	zr_normal_tree "$ref" "$alias_version" "$ZR_RUN/identity/graph-$side.reference" "$suffix"
	# The tool side is queried with the tool's own --features (ZR_TOOL_FEATURES),
	# exactly as it is built.
	zr_normal_tree "$ZR_WS" "$alias_version" "$ZR_RUN/identity/graph-$side.tool" "$suffix" "${ZR_TOOL_FEATURES:-}"
	diff -u "$ZR_RUN/identity/graph-$side.reference" "$ZR_RUN/identity/graph-$side.tool" \
		>"$ZR_RUN/identity/graph-$side.diff" ||
		zr_die "the $side zenresize dependency graph in the tool workspace differs from a production consumer's (see $ZR_RUN/identity/graph-$side.diff)"
	printf 'graph-%s\tidentical\tlines=%s\tsha256=%s\n' "$side" "$(wc -l <"$ZR_RUN/identity/graph-$side.tool")" \
		"$(zr_sha256 "$ZR_RUN/identity/graph-$side.tool")" >>"$ZR_RUN/identity/graphs.tsv"
}

# Copy built binaries out of Cargo's replaceable target directory into
# <run>/bin and record their sha256. Each must have been written during this
# run (after <run>/identity was created): a target directory shared between
# run directories can leave Cargo treating moved sources as fresh and hand back
# a stale executable, so runs always build in their own <run>/target.
zr_retain_binaries() {
	local target_dir=$1 profile=$2
	shift 2
	local b
	for b in "$@"; do
		[ -f "$target_dir/$profile/$b" ] || zr_die "built binary $target_dir/$profile/$b is missing"
		[ "$target_dir/$profile/$b" -nt "$ZR_RUN/identity/run.tsv" ] ||
			zr_die "binary $target_dir/$profile/$b predates this run; refusing a possibly stale build"
		cp -- "$target_dir/$profile/$b" "$ZR_RUN/bin/$b"
		printf '%s\t%s\n' "$b" "$(zr_sha256 "$ZR_RUN/bin/$b")" >>"$ZR_RUN/identity/binaries.tsv"
	done
}
