#!/usr/bin/env bash
# Stage two source-exact coz arms of benches/profile_coz.rs and build them, without
# profiling anything. See README.md.
#
# usage: tools/coz-profile/stage.sh --base REV --post REV --out NEW_DIR
#
# Each arm is <out>/src/<arm>/zenresize/: every file of the tool commit (HEAD),
# except src/, which comes from the arm's revision. Both arms therefore share one
# harness, manifest, lock and justfile byte for byte and differ only in src/. The
# directory is named zenresize so the recipe's coz scope '%/zenresize/src/%'
# matches. Each arm is built with the committed `just coz-build` recipe in its own
# target directory; the binary, its sha256, its DWARF line table, the lock and
# the compiled sources are recorded, and `identity/coz-commands.txt` holds the
# exact `just coz-run` commands for a later, separately authorized profile.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../stage-lib.sh
source "$script_dir/../stage-lib.sh"

usage() {
	sed -n '5p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
	exit 2
}

base_rev="" post_rev="" out=""
while [ $# -gt 0 ]; do
	case "$1" in
	--base) base_rev=${2:?}; shift 2 ;;
	--post) post_rev=${2:?}; shift 2 ;;
	--out) out=${2:?}; shift 2 ;;
	*) echo "error: unknown argument '$1'" >&2; usage ;;
	esac
done
[ -n "$base_rev" ] && [ -n "$post_rev" ] && [ -n "$out" ] || usage

zr_require_cmds git cargo rustc just jq tar sha256sum realpath awk sed diff cmp find xargs sort readelf python3
ZR_REPO=$(git -C "$script_dir" rev-parse --show-toplevel)
zr_require_committed benches/profile_coz.rs Cargo.toml Cargo.lock justfile tools/coz-profile tools/stage-lib.sh
tool_commit=$(zr_resolve_commit HEAD)
base_commit=$(zr_resolve_commit "$base_rev")
post_commit=$(zr_resolve_commit "$post_rev")
zr_check_build_env "$(dirname -- "$(realpath -m -- "$out")")"
zr_new_run_dir "$out"
{
	printf 'tool\tcoz-profile\n'
	printf 'tool_commit\t%s\n' "$tool_commit"
	printf 'base\t%s\t%s\tsrc_tree=%s\n' "$base_rev" "$base_commit" "$(git -C "$ZR_REPO" rev-parse "$base_commit:src")"
	printf 'post\t%s\t%s\tsrc_tree=%s\n' "$post_rev" "$post_commit" "$(git -C "$ZR_REPO" rev-parse "$post_commit:src")"
	printf 'started_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
	printf 'host\t%s\t%s\n' "$(uname -srm)" "$(hostname)"
} >"$ZR_RUN/identity/run.tsv"

# Top-level entries of the tool commit other than src/: the shared overlay.
mapfile -t overlay < <(git -C "$ZR_REPO" ls-tree --name-only "$tool_commit" | grep -vx src)
[ "${#overlay[@]}" -gt 0 ] || zr_die "tool commit $tool_commit has nothing outside src/"

for arm in base post; do
	commit_var="${arm}_commit"
	dest="$ZR_RUN/src/$arm/zenresize"
	zr_export_tree "$tool_commit" "$dest" "$arm-overlay" "${overlay[@]}"
	zr_export_tree "${!commit_var}" "$ZR_RUN/src/$arm-src-export" "$arm-src" src
	mv -- "$ZR_RUN/src/$arm-src-export/src" "$dest/src"
	rmdir -- "$ZR_RUN/src/$arm-src-export"
	# The assembled tree is exactly the two verified exports.
	(cd "$dest" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) >"$ZR_RUN/identity/$arm.files.sha256"
	LC_ALL=C sort -k2 "$ZR_RUN/identity/$arm-overlay.files.sha256" "$ZR_RUN/identity/$arm-src.files.sha256" |
		cmp -s - <(LC_ALL=C sort -k2 "$ZR_RUN/identity/$arm.files.sha256") ||
		zr_die "assembled $arm tree is not exactly its overlay and src exports"
done
cmp -s "$ZR_RUN/identity/base-overlay.files.sha256" "$ZR_RUN/identity/post-overlay.files.sha256" ||
	zr_die "the two arms' overlays differ"
diff <(grep -v '  ./src/' "$ZR_RUN/identity/base.files.sha256") <(grep -v '  ./src/' "$ZR_RUN/identity/post.files.sha256") >/dev/null ||
	zr_die "the arms differ outside src/"
printf 'arms\tidentical outside src/ (overlay manifest sha256 %s); src/ files base=%s post=%s\n' \
	"$(zr_sha256 "$ZR_RUN/identity/base-overlay.files.sha256")" \
	"$(grep -c '  ./src/' "$ZR_RUN/identity/base.files.sha256")" "$(grep -c '  ./src/' "$ZR_RUN/identity/post.files.sha256")" |
	tee -a "$ZR_RUN/identity/run.tsv"
printf 'lock\t%s\n' "$(zr_sha256 "$ZR_RUN/src/base/zenresize/Cargo.lock")" >>"$ZR_RUN/identity/run.tsv"
zr_record_toolchain "$ZR_RUN/src/base/zenresize"

for arm in base post; do
	dest="$ZR_RUN/src/$arm/zenresize"
	target="$ZR_RUN/target-$arm"
	(cd "$dest" && zr_run_logged "coz-build-$arm" env CARGO_TARGET_DIR="$target" just coz-build)
	bin=$(tail -n 1 "$ZR_RUN/logs/coz-build-$arm.log")
	case "$bin" in "$target"/release/deps/profile_coz-*) ;; *) zr_die "coz-build for $arm reported '$bin', not an executable under $target" ;; esac
	[ -x "$bin" ] && [ "$bin" -nt "$ZR_RUN/identity/run.tsv" ] || zr_die "$bin is missing or predates this run"
	readelf -S "$bin" | grep -q '\.debug_line' || zr_die "$bin has no .debug_line: coz needs line tables"
	cp -- "$bin" "$ZR_RUN/bin/profile_coz-$arm"
	printf 'profile_coz-%s\t%s\tbuilt=%s\tdebug_line=yes\n' "$arm" "$(zr_sha256 "$ZR_RUN/bin/profile_coz-$arm")" "$bin" \
		>>"$ZR_RUN/identity/binaries.tsv"
	ZR_COMPILED_TAG="-$arm" zr_record_compiled_sources "$target" release "$dest" zenresize "$arm=$dest"
	# Argument handling only (0 rounds: no resize runs): rgb3 is accepted, an
	# unknown mode is refused.
	"$ZR_RUN/bin/profile_coz-$arm" 0 rgb3 >"$ZR_RUN/logs/mode-rgb3-$arm.log" 2>&1 ||
		zr_die "profile_coz-$arm refused mode rgb3 (see $ZR_RUN/logs/mode-rgb3-$arm.log)"
	if "$ZR_RUN/bin/profile_coz-$arm" 0 bogus >"$ZR_RUN/logs/mode-bogus-$arm.log" 2>&1; then
		zr_die "profile_coz-$arm accepted an unknown mode"
	fi
done
cmp -s "$ZR_RUN/bin/profile_coz-base" "$ZR_RUN/bin/profile_coz-post" &&
	zr_die "the two arms built byte-identical binaries; their src/ trees should differ"

{
	echo "# Not run by stage.sh. Each command rebuilds (a no-op) through the committed"
	echo "# recipe and must print the sha256 recorded in binaries.tsv; PROFILE must be new."
	for arm in base post; do
		printf 'cd %q && CARGO_TARGET_DIR=%q just coz-run <new-%s-profile.jsonl> <rounds> rgb3\n' \
			"$ZR_RUN/src/$arm/zenresize" "$ZR_RUN/target-$arm" "$arm"
	done
} >"$ZR_RUN/identity/coz-commands.txt"
printf 'stage_status\tPASS\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$ZR_RUN/identity/run.tsv"
cat "$ZR_RUN/identity/binaries.tsv"
echo "coz arms staged and built (nothing profiled): $ZR_RUN; commands in $ZR_RUN/identity/coz-commands.txt"
