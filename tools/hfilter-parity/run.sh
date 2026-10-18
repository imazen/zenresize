#!/usr/bin/env bash
# Old-vs-new differential of zenresize's RGB f32 horizontal filter, at kernel
# level (kernel_diff) and through whole public resizes (e2e_diff), under every
# archmage dispatch permutation. See README.md.
#
# usage: tools/hfilter-parity/run.sh --old REV --new REV --out NEW_RUN_DIR \
#          --features none|avx512|pretty-safe|avx512,pretty-safe --scope quick|full
#
# Builds from verified git-archive exports in a fresh run directory; never
# touches the working tree. Exit 0 only if both binaries pass.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../stage-lib.sh
source "$script_dir/../stage-lib.sh"

usage() {
	sed -n '6,7p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
	exit 2
}

old_rev="" new_rev="" out="" features="" scope="" target_dir=""
while [ $# -gt 0 ]; do
	case "$1" in
	--old) old_rev=${2:?}; shift 2 ;;
	--new) new_rev=${2:?}; shift 2 ;;
	--out) out=${2:?}; shift 2 ;;
	--features) features=${2:?}; shift 2 ;;
	--scope) scope=${2:?}; shift 2 ;;
	*) echo "error: unknown argument '$1'" >&2; usage ;;
	esac
done
[ -n "$old_rev" ] && [ -n "$new_rev" ] && [ -n "$out" ] && [ -n "$features" ] && [ -n "$scope" ] || usage
case "$features" in
none) cargo_features=() feature_csv="" ;;
avx512 | pretty-safe | avx512,pretty-safe) cargo_features=(--features "$features") feature_csv=$features ;;
*) echo "error: --features must be none, avx512, pretty-safe or avx512,pretty-safe" >&2; usage ;;
esac
case "$scope" in quick | full) ;; *) echo "error: --scope must be quick or full" >&2; usage ;; esac

zr_require_cmds git cargo rustc tar sha256sum realpath awk sed diff cmp find xargs sort
ZR_REPO=$(git -C "$script_dir" rev-parse --show-toplevel)
zr_require_committed tools/hfilter-parity tools/stage-lib.sh
tool_commit=$(zr_resolve_commit HEAD)
old_commit=$(zr_resolve_commit "$old_rev")
new_commit=$(zr_resolve_commit "$new_rev")

zr_check_build_env "$(dirname -- "$(realpath -m -- "$out")")"
zr_new_run_dir "$out"
ZR_WS="$ZR_RUN/ws"
target_dir="$ZR_RUN/target"
{
	printf 'tool\thfilter-parity\n'
	printf 'tool_commit\t%s\n' "$tool_commit"
	printf 'old\t%s\t%s\n' "$old_rev" "$old_commit"
	printf 'new\t%s\t%s\n' "$new_rev" "$new_commit"
	printf 'features\t%s\n' "$features"
	printf 'scope\t%s\n' "$scope"
	printf 'target_dir\t%s\n' "$target_dir"
	printf 'started_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
	printf 'host\t%s\t%s\n' "$(uname -srm)" "$(hostname)"
} >"$ZR_RUN/identity/run.tsv"

zr_export_tree "$old_commit" "$ZR_RUN/src/old" old
zr_export_tree "$new_commit" "$ZR_RUN/src/new" new
zr_export_tree "$tool_commit" "$ZR_RUN/tool" tool tools/hfilter-parity tools/stage-lib.sh
old_version=$(zr_alias_version "$ZR_RUN/src/old" old old)
new_version=$(zr_alias_version "$ZR_RUN/src/new" new new)

mkdir -- "$ZR_WS"
cp -- "$ZR_RUN/tool/tools/hfilter-parity/Cargo.toml.in" "$ZR_WS/Cargo.toml"
zr_record_toolchain "$ZR_WS"
zr_seed_and_resolve_lock "$new_commit" "$ZR_WS"
zr_check_lock_subset Cargo.lock zenresize-hfilter-parity -
ZR_TOOL_FEATURES=$feature_csv
archmage_hook='archmage = { version = "0.9.26", features = ["std", "macros", "testable_dispatch"] }'
zr_check_normal_graph old "$old_version" old "$feature_csv" "$archmage_hook"
zr_check_normal_graph new "$new_version" new "$feature_csv" "$archmage_hook"

(cd "$ZR_WS" && zr_run_logged build \
	env CARGO_TARGET_DIR="$target_dir" cargo build --locked --release --bins "${cargo_features[@]}")
zr_retain_binaries "$target_dir" release kernel_diff e2e_diff

status=0
for bin in kernel_diff e2e_diff; do
	rc=0
	(cd "$ZR_RUN" && "$ZR_RUN/bin/$bin" --scope="$scope") >"$ZR_RUN/logs/$bin.log" 2>&1 || rc=$?
	printf '%s\texit=%s\t%s\n' "$bin" "$rc" "$(grep -E '^(kernel_diff|e2e_diff) OK|^totals:' "$ZR_RUN/logs/$bin.log" | tr '\n' ' ')" \
		>>"$ZR_RUN/identity/result.tsv"
	if [ "$rc" -ne 0 ]; then
		echo "error: $bin failed with exit code $rc; full log: $ZR_RUN/logs/$bin.log" >&2
		status=1
	fi
done
printf 'finished_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$ZR_RUN/identity/run.tsv"
cat "$ZR_RUN/identity/result.tsv"
[ "$status" -eq 0 ] || exit 1
echo "hfilter-parity PASS ($scope, features=$features): $ZR_RUN"
