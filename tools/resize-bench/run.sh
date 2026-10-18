#!/usr/bin/env bash
# Native zenbench Confirm comparison of whole public zenresize resizes, old vs
# new vs old A/A, in one executable built from verified git-archive exports.
# See README.md.
#
# usage: tools/resize-bench/run.sh prepare --old REV --new REV --corpus CODEC_CORPUS_DIR \
#          [--imazen26 IMAZEN26_PNG_V3_DIR] --out NEW_RUN_DIR
#        tools/resize-bench/run.sh confirm --run PREPARED_RUN_DIR [--cpu-set IDS] \
#          [--cpu-policy auto|performance|efficiency] [--dry-run]
#
# prepare: stage, check, build, prepare inputs, dry-run and selftest. Never times anything.
# confirm: verify a prepared run, then run the retained executable (five fresh
#          processes) into a fresh attempt directory. Its exit code is the decision.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../stage-lib.sh
source "$script_dir/../stage-lib.sh"

usage() {
	sed -n '6,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
	exit 2
}

# Disk and memory headroom before any heavy step (Linux; refuses elsewhere).
check_resources() {
	local dir=$1 need_disk_mib=$2 need_mem_mib=$3 avail_kib mem_kib
	avail_kib=$(df -Pk -- "$dir" | awk 'NR == 2 {print $4}')
	[ -n "$avail_kib" ] || zr_die "cannot read free disk space for $dir"
	[ "$((avail_kib / 1024))" -ge "$need_disk_mib" ] ||
		zr_die "only $((avail_kib / 1024)) MiB free under $dir; need $need_disk_mib MiB"
	[ -r /proc/meminfo ] || zr_die "cannot read /proc/meminfo; this benchmark runs on Linux only"
	mem_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
	[ "$((mem_kib / 1024))" -ge "$need_mem_mib" ] ||
		zr_die "only $((mem_kib / 1024)) MiB memory available; need $need_mem_mib MiB"
	printf 'resources\t%s\tdisk_free_mib=%s\tmem_available_mib=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
		"$((avail_kib / 1024))" "$((mem_kib / 1024))"
}

# Validate the dry-run's resolved plan against the frozen plan the binary
# recorded, using only the Python standard library.
check_dry_run() {
	local plan_json=$1 record=$2 artifacts=$3 binary_sha=$4
	python3 -I - "$plan_json" "$record" "$artifacts" "$binary_sha" <<'PY'
import json, os, sys
plan_path, record_path, artifacts, binary_sha = sys.argv[1:5]
dry = json.load(open(plan_path))
rec = json.load(open(record_path))
ident = rec["identity"]
p, frozen = dry["plan"], ident["plan"]
problems = []
def need(cond, msg):
    if not cond:
        problems.append(msg)
need(p["mode"] == "confirm", f"mode {p['mode']}")
need(p["processes"] == frozen["processes"] == 5, "processes")
need(p["rounds"] == frozen["rounds"] == 30, "rounds")
need(p["margin_pct"] == frozen["margin_pct"] == 1.0, "margin")
need(p["confidence"] == frozen["confidence"] == 0.95, "confidence")
need(p["workload_id"] == ident["workload_id"], "workload id")
need(p["scope"] is None, "scope")
need(p["timeout"]["secs"] == frozen["timeout_secs"] and p["timeout"]["nanos"] == 0, "timeout")
need(dry["binary_sha256"] == binary_sha == ident["binary_sha256"], "binary sha256")
need(os.path.realpath(dry["data_directory"]) == os.path.realpath(os.path.join(artifacts, "zenbench-data")), "data dir")
want = [g["id"] for g in ident["groups"]]
got = [g["name"] for g in dry["groups"]]
need(sorted(got) == sorted(want) and len(got) == len(set(got)), f"groups {got} != {want}")
for g in dry["groups"]:
    c = g["config"]
    need(g["benchmarks"] == frozen["arms"], f"{g['name']} arms {g['benchmarks']}")
    need(c["max_rounds"] == 30 and c["min_rounds"] == 30 and c["auto_rounds"] is False, f"{g['name']} rounds {c}")
    need(c["baseline_only"] is True, f"{g['name']} baseline_only")
    need(c["max_time"]["secs"] == frozen["group_max_time_secs"], f"{g['name']} max_time")
    need(c["max_wall_time"]["secs"] == frozen["group_max_wall_secs"], f"{g['name']} max_wall_time")
need(rec["all_equal"] is True and len(rec["parity"]) == len(want), "dry-run parity record")
if problems:
    print("dry-run plan check FAILED:\n  " + "\n  ".join(problems), file=sys.stderr)
    sys.exit(1)
print(f"dry-run plan check OK: confirm, 5 processes x 30 fixed rounds, margin 1, confidence 0.95, "
      f"{len(got)} groups x {len(frozen['arms'])} arms, workload {p['workload_id']}")
PY
}

cmd=${1:-}
[ $# -gt 0 ] && shift || usage

prepare() {
	local old_rev="" new_rev="" out="" corpus="" imazen26="" target_dir=""
	while [ $# -gt 0 ]; do
		case "$1" in
		--old) old_rev=${2:?}; shift 2 ;;
		--new) new_rev=${2:?}; shift 2 ;;
		--out) out=${2:?}; shift 2 ;;
		--corpus) corpus=${2:?}; shift 2 ;;
		--imazen26) imazen26=${2:?}; shift 2 ;;
			*) echo "error: unknown argument '$1'" >&2; usage ;;
		esac
	done
	[ -n "$old_rev" ] && [ -n "$new_rev" ] && [ -n "$out" ] && [ -n "$corpus" ] || usage
	[ -d "$corpus" ] || zr_die "corpus dir $corpus does not exist"
	corpus=$(realpath -- "$corpus")
	local imazen26_arg=()
	if [ -n "$imazen26" ]; then
		[ -d "$imazen26" ] || zr_die "imazen26 dir $imazen26 does not exist"
		imazen26=$(realpath -- "$imazen26")
		imazen26_arg=("--imazen26=$imazen26")
	fi

	zr_require_cmds git cargo rustc tar sha256sum realpath awk sed diff cmp find xargs sort comm df python3 cp
	ZR_REPO=$(git -C "$script_dir" rev-parse --show-toplevel)
	zr_require_committed tools/resize-bench tools/stage-lib.sh
	local tool_commit old_commit new_commit
	tool_commit=$(zr_resolve_commit HEAD)
	old_commit=$(zr_resolve_commit "$old_rev")
	new_commit=$(zr_resolve_commit "$new_rev")
	zr_check_build_env "$(dirname -- "$(realpath -m -- "$out")")"
	zr_new_run_dir "$out"
	ZR_WS="$ZR_RUN/ws"
	target_dir="$ZR_RUN/target"
	{
		printf 'tool\tresize-bench\n'
		printf 'tool_commit\t%s\n' "$tool_commit"
		printf 'old\t%s\t%s\n' "$old_rev" "$old_commit"
		printf 'new\t%s\t%s\n' "$new_rev" "$new_commit"
		printf 'corpus\t%s\tcommit=%s\n' "$corpus" "$(git -C "$corpus" rev-parse HEAD 2>/dev/null || echo not-a-git-checkout)"
		if [ -n "$imazen26" ]; then
			printf 'imazen26\t%s\tcommit=%s\n' "$imazen26" "$(git -C "$imazen26" rev-parse HEAD 2>/dev/null || echo not-a-git-checkout)"
		fi
		printf 'target_dir\t%s\n' "$target_dir"
		printf 'started_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
		printf 'host\t%s\t%s\n' "$(uname -srm)" "$(hostname)"
	} >"$ZR_RUN/identity/run.tsv"
	check_resources "$ZR_RUN" 4096 4096 >>"$ZR_RUN/identity/run.tsv"

	zr_export_tree "$old_commit" "$ZR_RUN/src/old" old
	zr_export_tree "$new_commit" "$ZR_RUN/src/new" new
	zr_export_tree "$tool_commit" "$ZR_RUN/tool" tool tools/resize-bench tools/stage-lib.sh
	local old_version new_version
	old_version=$(zr_alias_version "$ZR_RUN/src/old" old old)
	new_version=$(zr_alias_version "$ZR_RUN/src/new" new new)

	# Benchmark workspace: production graph for both zenresize copies.
	mkdir -- "$ZR_WS"
	cp -- "$ZR_RUN/tool/tools/resize-bench/Cargo.toml.in" "$ZR_WS/Cargo.toml"
	zr_record_toolchain "$ZR_WS"
	zr_seed_and_resolve_lock "$new_commit" "$ZR_WS"
	ZR_TOOL_FEATURES=""
	zr_check_normal_graph old "$old_version" old "" ""
	zr_check_normal_graph new "$new_version" new "" ""
	zr_check_lock_subset Cargo.lock zenresize-resize-bench "$ZR_RUN/identity/graph-new.reference"

	# Input-preparation workspace (PNG decoding stays out of the benchmark graph).
	local ws_prep="$ZR_RUN/ws-prep"
	mkdir -- "$ws_prep"
	cp -- "$ZR_RUN/tool/tools/resize-bench/prep.Cargo.toml.in" "$ws_prep/Cargo.toml"
	zr_seed_and_resolve_lock "$new_commit" "$ws_prep" prep-Cargo.lock
	zr_check_lock_subset prep-Cargo.lock zenresize-resize-bench-prep "$ZR_RUN/identity/graph-new.reference"

	(cd "$ZR_WS" && zr_run_logged build-bench \
		env CARGO_TARGET_DIR="$target_dir" cargo build --locked --release --bin resize-bench)
	(cd "$ws_prep" && zr_run_logged build-prep \
		env CARGO_TARGET_DIR="$target_dir" cargo build --locked --release --bin resize-bench-prep)
	zr_retain_binaries "$target_dir" release resize-bench resize-bench-prep
	local bin="$ZR_RUN/bin/resize-bench" binary_sha
	binary_sha=$(zr_sha256 "$bin")

	local artifacts="$ZR_RUN/artifacts"
	mkdir -- "$artifacts" "$artifacts/zenbench-data"
	(cd "$ZR_RUN" && zr_run_logged prep "$ZR_RUN/bin/resize-bench-prep" --corpus="$corpus" "${imazen26_arg[@]}" --artifacts="$artifacts")

	# Dry run: registration and parity run; no benchmark callback is timed.
	local rc=0
	(cd "$ZR_RUN" && env ZENBENCH_DATA_DIR="$artifacts/zenbench-data" "$bin" --artifacts="$artifacts" --dry-run \
		>"$ZR_RUN/identity/dryrun-plan.json" 2>"$ZR_RUN/logs/dryrun.stderr.log") || rc=$?
	[ "$rc" -eq 0 ] || zr_die "dry run failed with exit code $rc; see $ZR_RUN/logs/dryrun.stderr.log"
	local record
	record=$(find "$artifacts/records" -name 'dryrun-before-*.json' | head -n 1)
	[ -n "$record" ] || zr_die "dry run wrote no parity record"
	check_dry_run "$ZR_RUN/identity/dryrun-plan.json" "$record" "$artifacts" "$binary_sha" | tee "$ZR_RUN/identity/dryrun-check.txt"

	selftest "$bin" "$artifacts"
	printf 'prepare_status\tPASS\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$ZR_RUN/identity/run.tsv"
	echo "resize-bench prepared (no timing): $ZR_RUN"
	echo "confirm on a quiet host with: tools/resize-bench/run.sh confirm --run $ZR_RUN [--cpu-set IDS]"
}

# Controlled failures through the real executable's main routing. Each case
# must exit with the stated code; rejected-before-mutation cases must leave the
# artifacts directory untouched.
selftest() {
	local bin=$1 artifacts=$2
	local st="$ZR_RUN/selftest" log="$ZR_RUN/logs/selftest.log"
	mkdir -- "$st"
	: >"$log"
	local snapshot="$st/artifacts-listing"
	(cd "$artifacts" && find . -type f | LC_ALL=C sort) >"$snapshot"
	local data="$artifacts/zenbench-data" fails=0 n=0
	ln -s /tmp "$st/link-to-tmp"
	mkdir -- "$st/elsewhere"

	expect() {
		local code=$1 untouched=$2 name=$3
		shift 3
		local rc=0
		n=$((n + 1))
		(cd "$ZR_RUN" && "$@") >>"$log" 2>&1 </dev/null || rc=$?
		local ok=1
		[ "$rc" -eq "$code" ] || ok=0
		if [ "$untouched" = untouched ] &&
			! (cd "$artifacts" && find . -type f | LC_ALL=C sort | cmp -s - "$snapshot"); then
			ok=0
		fi
		printf '%s\t%s\texit=%s\texpected=%s\t%s\n' "$([ $ok = 1 ] && echo PASS || echo FAIL)" "$name" "$rc" "$code" "$untouched" |
			tee -a "$st/results.tsv"
		[ "$ok" = 1 ] || fails=$((fails + 1))
	}
	local A="--artifacts=$artifacts"
	local E=(env ZENBENCH_DATA_DIR="$data")
	expect 2 untouched cli-mode "${E[@]}" "$bin" "$A" --dry-run --mode=iterate
	expect 2 untouched cli-rounds "${E[@]}" "$bin" "$A" --dry-run --rounds=3
	expect 2 untouched cli-processes "${E[@]}" "$bin" "$A" --dry-run --processes=3
	expect 2 untouched cli-margin "${E[@]}" "$bin" "$A" --dry-run --margin=5
	expect 2 untouched cli-confidence "${E[@]}" "$bin" "$A" --dry-run --confidence=0.5
	expect 2 untouched cli-workload-id "${E[@]}" "$bin" "$A" --dry-run --workload-id=other
	expect 2 untouched cli-timeout "${E[@]}" "$bin" "$A" --dry-run --timeout=10
	expect 2 untouched cli-save-baseline "${E[@]}" "$bin" "$A" --save-baseline=x
	expect 2 untouched cli-baseline "${E[@]}" "$bin" "$A" --baseline=x
	expect 2 untouched cli-update-on-pass "${E[@]}" "$bin" "$A" --update-on-pass
	expect 2 untouched cli-no-busy-gate "${E[@]}" "$bin" "$A" --dry-run --no-busy-gate
	expect 2 untouched cli-unrestricted "${E[@]}" "$bin" "$A" --dry-run --cpu-policy=unrestricted
	expect 2 untouched cli-group-filter "${E[@]}" "$bin" "$A" --dry-run --group=t-rgb8lin
	expect 2 untouched cli-legacy-repetition "${E[@]}" "$bin" "$A" --mean-of-processes=5
	expect 2 untouched cli-format-csv "${E[@]}" "$bin" "$A" --dry-run --format=csv
	expect 2 untouched cli-unknown "${E[@]}" "$bin" "$A" --dry-run --bogus
	expect 2 untouched cli-duplicate "${E[@]}" "$bin" "$A" --dry-run --dry-run
	expect 2 untouched cli-missing-artifacts "${E[@]}" "$bin" --dry-run
	expect 2 untouched env-rounds env ZENBENCH_DATA_DIR="$data" ZENBENCH_ROUNDS=3 "$bin" "$A" --dry-run
	expect 2 untouched env-mode env ZENBENCH_DATA_DIR="$data" ZENBENCH_MODE=iterate "$bin" "$A" --dry-run
	expect 2 untouched env-no-save env ZENBENCH_DATA_DIR="$data" ZENBENCH_NO_SAVE=1 "$bin" "$A" --dry-run
	expect 2 untouched env-no-calibrate env ZENBENCH_DATA_DIR="$data" ZENBENCH_NO_CALIBRATE=1 "$bin" "$A" --dry-run
	expect 2 untouched env-unrestricted env ZENBENCH_DATA_DIR="$data" ZENBENCH_CPU_POLICY=unrestricted "$bin" "$A" --dry-run
	expect 2 untouched env-fake-child env ZENBENCH_DATA_DIR="$data" ZENBENCH_SUBPROCESS=1 "$bin" "$A" --dry-run
	expect 2 untouched env-data-dir-missing env -u ZENBENCH_DATA_DIR "$bin" "$A" --dry-run
	expect 2 untouched env-data-dir-elsewhere env ZENBENCH_DATA_DIR="$st/elsewhere" "$bin" "$A" --dry-run
	expect 2 untouched artifacts-via-tmp env ZENBENCH_DATA_DIR="$data" "$bin" --artifacts="$st/link-to-tmp" --dry-run

	# Corrupt and missing inputs in a copy of the artifacts: refused (exit 3)
	# before any parity or timing, and the real artifacts stay untouched.
	local raw_c64 obj
	raw_c64=$(python3 -I -c 'import json,sys; m=json.load(open(sys.argv[1])); print([i for i in m["inputs"] if i["id"]=="cid22-1025469-c64"][0]["raw_rgb8_sha256"])' \
		"$(find "$artifacts/inputs" -name 'manifest-*.json' | head -n 1)")
	local t
	for t in corrupt missing; do
		mkdir -- "$st/$t" "$st/$t/zenbench-data"
		cp -a -- "$artifacts/cas" "$st/$t/cas"
		obj="$st/$t/cas/sha256/${raw_c64:0:2}/$raw_c64"
		[ -f "$obj" ] || zr_die "selftest: input object $obj not found"
		chmod u+w -- "$obj"
		if [ "$t" = corrupt ]; then
			printf '\x00' | dd of="$obj" bs=1 seek=100 conv=notrunc status=none
		else
			rm -- "$obj"
		fi
		expect 3 untouched "input-$t" env ZENBENCH_DATA_DIR="$st/$t/zenbench-data" "$bin" --artifacts="$st/$t" --dry-run
		[ -z "$(find "$st/$t/zenbench-data" "$st/$t" -maxdepth 1 -name records)" ] ||
			{ echo "FAIL input-$t wrote parity records" | tee -a "$st/results.tsv"; fails=$((fails + 1)); }
	done

	# Child routing without timing: the exact environment zenbench's supervisor
	# and admission reservation give a child must pass the environment check,
	# stopping at the missing input object (exit 3) before any timing; a child
	# whose plan differs must not pass it.
	local child_env=(env ZENBENCH_DATA_DIR="$st/missing/zenbench-data" ZENBENCH_SUBPROCESS=1
		ZENBENCH_RESULT_PATH="$st/missing/child-result.json.gz" ZENBENCH_LAUNCHER_PIDS=1
		ZENBENCH_MODE=confirm ZENBENCH_PROCESSES=5 ZENBENCH_ROUNDS=30 ZENBENCH_MARGIN=1
		ZENBENCH_CONFIDENCE=0.95 ZENBENCH_TIMEOUT=7200
		ZENBENCH_WORKLOAD_ID=zenresize-rgb-hfilter-whole-resize-v2
		ZENBENCH_COORDINATION_DIR="$st" ZENBENCH_COORDINATION_LEASE=1)
	expect 3 untouched child-env-accepted "${child_env[@]}" "$bin" --artifacts="$st/missing"
	expect 2 untouched child-env-wrong-rounds "${child_env[@]}" ZENBENCH_ROUNDS=3 "$bin" --artifacts="$st/missing"
	[ ! -e "$st/missing/child-result.json.gz" ] ||
		{ echo "FAIL child routing produced a result packet" | tee -a "$st/results.tsv"; fails=$((fails + 1)); }

	printf 'selftest\tcases=%s\tfailures=%s\n' "$n" "$fails" | tee -a "$st/results.tsv"
	[ "$fails" -eq 0 ] || zr_die "selftest: $fails of $n controlled-failure cases did not fail as required (see $st/results.tsv, $log)"
}

confirm() {
	local run="" dry_run=0 passthrough=()
	while [ $# -gt 0 ]; do
		case "$1" in
		--run) run=${2:?}; shift 2 ;;
		--cpu-set) passthrough+=("--cpu-set=${2:?}"); shift 2 ;;
		--cpu-policy)
			case "${2:-}" in auto | performance | efficiency) ;; *) zr_die "--cpu-policy must be auto, performance or efficiency" ;; esac
			passthrough+=("--cpu-policy=$2")
			shift 2
			;;
		--dry-run) dry_run=1; shift ;;
		*) echo "error: unknown argument '$1'" >&2; usage ;;
		esac
	done
	[ -n "$run" ] || usage
	ZR_RUN=$(realpath -- "$run")
	grep -q $'^prepare_status\tPASS' "$ZR_RUN/identity/run.tsv" ||
		zr_die "$ZR_RUN was not prepared successfully (no prepare_status PASS)"
	local bin="$ZR_RUN/bin/resize-bench" want got
	want=$(awk -F'\t' '$1 == "resize-bench" {print $2}' "$ZR_RUN/identity/binaries.tsv")
	got=$(zr_sha256 "$bin")
	[ -n "$want" ] && [ "$want" = "$got" ] || zr_die "retained binary sha256 $got does not match the prepared $want"
	if env | grep -q '^ZENBENCH_'; then
		local v
		for v in $(env | grep '^ZENBENCH_' | cut -d= -f1); do
			case "$v" in ZENBENCH_CPU_POLICY | ZENBENCH_CPU_SET | ZENBENCH_COORDINATION_DIR) ;; *) zr_die "$v is set; the plan is frozen" ;; esac
		done
	fi
	local artifacts="$ZR_RUN/artifacts" attempt
	attempt="$ZR_RUN/attempts/$(date -u +%Y%m%dT%H%M%SZ)-$$"
	mkdir -p -- "$ZR_RUN/attempts"
	mkdir -- "$attempt"
	check_resources "$ZR_RUN" 2048 4096 >"$attempt/resources.tsv"
	local args=(--artifacts="$artifacts" "${passthrough[@]}")
	[ "$dry_run" -eq 0 ] || args+=(--dry-run)
	printf '%s\n' "$bin" "${args[@]}" >"$attempt/command.txt"
	echo "binary_sha256 $got" >"$attempt/binary.txt"
	local rc=0
	(cd "$ZR_RUN" && env ZENBENCH_DATA_DIR="$artifacts/zenbench-data" "$bin" "${args[@]}" \
		>"$attempt/stdout.txt" 2>"$attempt/stderr.txt") || rc=$?
	echo "$rc" >"$attempt/exit_code"
	if [ "$dry_run" -eq 1 ]; then
		echo "resize-bench dry run exit code $rc (no benchmark callback was timed; this is not a decision; see $attempt)"
	else
		echo "resize-bench exit code $rc (0 = every declared rule passed; see $attempt and $artifacts/acceptance)"
	fi
	exit "$rc"
}

case "$cmd" in
prepare) prepare "$@" ;;
confirm) confirm "$@" ;;
*) usage ;;
esac
