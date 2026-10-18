//! Native zenbench Confirm comparison of whole public zenresize resizes:
//! baseline (`old`), candidate (`new`) and an A/A arm (`old-aa`) in one
//! executable, five fresh processes, 30 fixed rounds, margin 1 %, family
//! confidence 0.95, all frozen in `plan.rs`. See README.md.
//!
//! Exit codes: 0 = every declared acceptance rule passed; 1 = confirmed
//! regression; 2 = usage/plan/environment error, or invalid, inconclusive or
//! not accepted; 3 = input or old/new parity verification failed (before any
//! timing in that process). zenbench's own exit 0 is never treated as acceptance.

mod acceptance;
mod cas;
mod inputs;
mod plan;
mod resize_ops;

use cas::{Cas, sha256_hex, unique_suffix, write_new};
use inputs::{INPUTS, WORKLOAD_ID};
use plan::{ARM_AA, ARM_NEW, ARM_OLD, GROUPS, GroupSpec};
use resize_ops::{Buf, derive_f32_twin, derive_input, new_ops, old_ops};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use zenbench::prelude::*;

const USAGE: &str = "usage: resize-bench --artifacts=DIR [--dry-run] [--cpu-set=IDS] \
[--cpu-policy=auto|performance|efficiency] [--format=llm|json]
Mode, rounds, processes, margin, confidence, workload id and timeout are frozen in
plan.rs; any attempt to override them (CLI or ZENBENCH_* environment) exits 2.
ZENBENCH_DATA_DIR must be set to DIR/zenbench-data.";

fn fail(code: i32, msg: &str) -> ! {
    eprintln!("resize-bench: error: {msg}");
    std::process::exit(code);
}

struct Cli {
    artifacts: PathBuf,
    dry_run: bool,
    args: Vec<String>,
}

/// Strict whitelist; validated completely before anything is written.
fn parse_cli() -> Cli {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.iter().any(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        std::process::exit(0);
    }
    let mut artifacts = None;
    let mut dry_run = false;
    let mut seen = std::collections::HashSet::new();
    let mut i = 0;
    while i < raw.len() {
        let arg = &raw[i];
        let (name, value) = match arg.split_once('=') {
            Some((n, v)) => (n.to_string(), Some(v.to_string())),
            None if arg == "--cpu-set" || arg == "--cpu-policy" => {
                i += 1;
                (arg.clone(), raw.get(i).cloned())
            }
            None => (arg.clone(), None),
        };
        if !seen.insert(name.clone()) {
            fail(2, &format!("duplicate argument {name}\n{USAGE}"));
        }
        match (name.as_str(), value.as_deref()) {
            ("--artifacts", Some(v)) if !v.is_empty() => artifacts = Some(PathBuf::from(v)),
            ("--dry-run", None) => dry_run = true,
            ("--cpu-set", Some(v)) if !v.is_empty() && !v.starts_with("--") => {}
            ("--cpu-policy", Some("auto" | "performance" | "efficiency")) => {}
            ("--format", Some("llm" | "json")) => {}
            _ => fail(
                2,
                &format!(
                    "argument {arg:?} is not allowed: only placement, --dry-run, --format and --artifacts\n{USAGE}"
                ),
            ),
        }
        i += 1;
    }
    let Some(artifacts) = artifacts else {
        fail(2, &format!("--artifacts=DIR is required\n{USAGE}"))
    };
    Cli {
        artifacts,
        dry_run,
        args: raw,
    }
}

/// Top-level runs may carry only placement/coordination and the data dir; a
/// child must carry exactly the frozen plan its parent passes down.
fn check_env(child: bool) {
    let mut allowed: Vec<&str> = vec![
        "ZENBENCH_DATA_DIR",
        "ZENBENCH_CPU_POLICY",
        "ZENBENCH_CPU_SET",
        "ZENBENCH_COORDINATION_DIR",
    ];
    if child {
        // Exactly what zenbench's process supervisor and its admission
        // reservation set on each child (run_processes_planned, coordination).
        allowed.extend([
            "ZENBENCH_COORDINATION_LEASE",
            "ZENBENCH_SUBPROCESS",
            "ZENBENCH_RESULT_PATH",
            "ZENBENCH_LAUNCHER_PIDS",
            "ZENBENCH_MODE",
            "ZENBENCH_PROCESSES",
            "ZENBENCH_ROUNDS",
            "ZENBENCH_MARGIN",
            "ZENBENCH_CONFIDENCE",
            "ZENBENCH_TIMEOUT",
            "ZENBENCH_WORKLOAD_ID",
        ]);
    }
    for (k, v) in std::env::vars() {
        if k.starts_with("ZENBENCH_") && !allowed.contains(&k.as_str()) {
            fail(
                2,
                &format!("environment variable {k}={v:?} is not allowed (the plan is frozen)"),
            );
        }
    }
    if std::env::var("ZENBENCH_CPU_POLICY")
        .is_ok_and(|p| !matches!(p.as_str(), "auto" | "performance" | "efficiency"))
    {
        fail(
            2,
            "ZENBENCH_CPU_POLICY must be auto, performance or efficiency (never unrestricted)",
        );
    }
    if child {
        let num = |k: &str| -> f64 {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| fail(2, &format!("child is missing {k}")))
        };
        let ok = std::env::var("ZENBENCH_MODE").as_deref() == Ok("confirm")
            && num("ZENBENCH_PROCESSES") == plan::PROCESSES as f64
            && num("ZENBENCH_ROUNDS") == plan::ROUNDS as f64
            && num("ZENBENCH_MARGIN") == plan::MARGIN_PCT
            && num("ZENBENCH_CONFIDENCE") == plan::CONFIDENCE
            && num("ZENBENCH_TIMEOUT") == plan::TIMEOUT_SECS as f64
            && std::env::var("ZENBENCH_WORKLOAD_ID").as_deref() == Ok(WORKLOAD_ID)
            && std::env::var_os("ZENBENCH_RESULT_PATH").is_some();
        if !ok {
            fail(
                2,
                "child process environment does not match the frozen plan",
            );
        }
    }
}

/// Disk-backed, caller-chosen artifact directory with the zenbench data dir
/// inside it. Nothing is created here.
fn check_paths(cli: &Cli) -> PathBuf {
    let artifacts = std::fs::canonicalize(&cli.artifacts).unwrap_or_else(|e| {
        fail(
            2,
            &format!("artifacts dir {}: {e}", cli.artifacts.display()),
        )
    });
    if !artifacts.is_dir() {
        fail(2, &format!("{} is not a directory", artifacts.display()));
    }
    for banned in ["/tmp", "/run", "/dev/shm", "/var/tmp"] {
        if artifacts.starts_with(banned) {
            fail(
                2,
                &format!(
                    "artifacts dir {} is under {banned}; use disk-backed storage",
                    artifacts.display()
                ),
            );
        }
    }
    let data_dir = std::env::var_os("ZENBENCH_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            fail(
                2,
                "ZENBENCH_DATA_DIR must be set to <artifacts>/zenbench-data",
            )
        });
    let data_dir = std::fs::canonicalize(&data_dir)
        .unwrap_or_else(|e| fail(2, &format!("ZENBENCH_DATA_DIR {}: {e}", data_dir.display())));
    if data_dir != artifacts.join("zenbench-data") {
        fail(
            2,
            &format!(
                "ZENBENCH_DATA_DIR {} is not {}/zenbench-data",
                data_dir.display(),
                artifacts.display()
            ),
        );
    }
    artifacts
}

/// Refuse to start without headroom (Linux /proc/meminfo; other platforms
/// cannot be checked and are refused rather than skipped).
fn check_memory() -> u64 {
    let meminfo = std::fs::read_to_string("/proc/meminfo").unwrap_or_else(|e| {
        fail(
            2,
            &format!("cannot read /proc/meminfo ({e}); this benchmark runs on Linux only"),
        )
    });
    let kib: u64 = meminfo
        .lines()
        .find_map(|l| l.strip_prefix("MemAvailable:"))
        .and_then(|v| v.trim().trim_end_matches("kB").trim().parse().ok())
        .unwrap_or_else(|| fail(2, "MemAvailable missing from /proc/meminfo"));
    if kib < 2 * 1024 * 1024 {
        fail(
            2,
            &format!("only {} MiB available; need at least 2048 MiB", kib / 1024),
        );
    }
    kib
}

struct Group {
    spec: &'static GroupSpec,
    input: Arc<Buf>,
    input_sha: String,
    old_cfg: zr_old::ResizeConfig,
    new_cfg: zr_new::ResizeConfig,
    twin: Option<(Buf, zr_old::ResizeConfig, zr_new::ResizeConfig)>,
}

fn load_groups(cas: &Cas) -> (Vec<Group>, Vec<Value>) {
    let mut raw = std::collections::HashMap::new();
    let mut inputs_json = Vec::new();
    for i in INPUTS {
        if i.raw_rgb8_sha256.len() != 64 || i.source_sha256.len() != 64 {
            fail(
                2,
                &format!("plan input {} is not frozen (missing sha256)", i.id),
            );
        }
        let bytes = cas
            .get_verified(i.raw_rgb8_sha256)
            .unwrap_or_else(|e| fail(3, &format!("input {}: {e} (run prep first)", i.id)));
        let (w, h) = (i.crop.2 as usize, i.crop.3 as usize);
        if bytes.len() != w * h * 3 {
            fail(
                3,
                &format!(
                    "input {} has {} bytes, expected {w}x{h}x3",
                    i.id,
                    bytes.len()
                ),
            );
        }
        inputs_json.push(json!({"id": i.id, "corpus": format!("{:?}", i.corpus), "corpus_path": i.corpus_path,
            "source_sha256": i.source_sha256,
            "source_size": [i.source_w, i.source_h],
            "crop": [i.crop.0, i.crop.1, i.crop.2, i.crop.3], "raw_rgb8_sha256": i.raw_rgb8_sha256}));
        raw.insert(i.id, bytes);
    }
    let mut groups = Vec::new();
    for spec in GROUPS {
        let inp = plan::input(spec.input);
        let (w, h) = (inp.crop.2, inp.crop.3);
        let rgb8 = &raw[spec.input];
        let input = derive_input(
            spec.format,
            w as usize,
            h as usize,
            spec.in_stride_pad,
            rgb8,
        );
        let input_sha = cas
            .put(&input.le_bytes())
            .unwrap_or_else(|e| fail(3, &format!("storing derived input for {}: {e}", spec.id)));
        let twin =
            derive_f32_twin(spec.format, spec.engine, w as usize, h as usize, rgb8).map(|buf| {
                let ch = spec.format.channels();
                (
                    buf,
                    old_ops::twin_config(ch, w, h, spec.out_w, spec.out_h),
                    new_ops::twin_config(ch, w, h, spec.out_w, spec.out_h),
                )
            });
        groups.push(Group {
            spec,
            input: Arc::new(input),
            input_sha,
            old_cfg: old_ops::config(
                spec.format,
                w,
                h,
                spec.out_w,
                spec.out_h,
                spec.in_stride_pad,
            ),
            new_cfg: new_ops::config(
                spec.format,
                w,
                h,
                spec.out_w,
                spec.out_h,
                spec.in_stride_pad,
            ),
            twin,
        });
    }
    (groups, inputs_json)
}

/// Run every group through the old and new builds (and the raw-f32 twin) and
/// require bit-identical outputs; every distinct output goes to the CAS.
fn parity(cas: &Cas, groups: &[Group]) -> (Vec<Value>, bool) {
    let mut rows = Vec::new();
    let mut all = true;
    for g in groups {
        let e = g.spec.engine;
        let o = old_ops::run(e, &g.old_cfg, &g.input);
        let n = new_ops::run(e, &g.new_cfg, &g.input);
        let equal = o.bit_equal(&n);
        let put = |b: &Buf| {
            cas.put(&b.le_bytes())
                .unwrap_or_else(|err| fail(3, &format!("storing output: {err}")))
        };
        let (os, ns) = (put(&o), put(&n));
        drop((o, n));
        let out_kind = match g.spec.format {
            plan::Format::Rgb16Linear => "u16-le",
            plan::Format::RgbF32Linear => "f32-le",
            _ => "u8",
        };
        let mut row = json!({"group": g.spec.id, "input_sha256": g.input_sha, "out_kind": out_kind,
            "old_sha256": os, "new_sha256": ns, "equal": equal});
        let mut twin_equal = true;
        if let Some((buf, oc, nc)) = &g.twin {
            let (to, tn) = (old_ops::run(e, oc, buf), new_ops::run(e, nc, buf));
            twin_equal = to.bit_equal(&tn);
            row["twin_f32_input_sha256"] = json!(sha256_hex(&buf.le_bytes()));
            row["twin_f32_old_sha256"] = json!(put(&to));
            row["twin_f32_new_sha256"] = json!(put(&tn));
            row["twin_f32_equal"] = json!(twin_equal);
        } else {
            row["twin_f32"] = json!("n/a (output already f32, or h-first has no f32 entry point)");
        }
        all &= equal && twin_equal;
        rows.push(row);
    }
    (rows, all)
}

fn binary_sha256() -> String {
    let exe = std::env::current_exe().unwrap_or_else(|e| fail(2, &format!("current_exe: {e}")));
    sha256_hex(&cas::read_all(&exe).unwrap_or_else(|e| fail(2, &e)))
}

fn zenbench_env() -> Value {
    let mut m = serde_json::Map::new();
    for (k, v) in std::env::vars() {
        if k.starts_with("ZENBENCH_") {
            m.insert(k, json!(v));
        }
    }
    Value::Object(m)
}

fn write_record(artifacts: &Path, role: &str, phase: &str, body: Value) -> PathBuf {
    let dir = artifacts.join("records");
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| fail(3, &format!("records dir: {e}")));
    let path = dir.join(format!("{role}-{phase}-{}.json", unique_suffix()));
    let bytes = serde_json::to_vec_pretty(&body).expect("record serializes");
    write_new(&path, &bytes).unwrap_or_else(|e| fail(3, &e));
    path
}

fn register(suite: &mut Suite, groups: &[Group]) {
    let p = suite.plan();
    p.mode = zenbench::BenchmarkMode::Confirm;
    p.processes = plan::PROCESSES;
    p.rounds = Some(plan::ROUNDS);
    p.margin_pct = plan::MARGIN_PCT;
    p.confidence = plan::CONFIDENCE;
    p.workload_id = Some(WORKLOAD_ID.to_string());
    p.scope = None;
    p.timeout = Duration::from_secs(plan::TIMEOUT_SECS);
    for g in groups {
        let e = g.spec.engine;
        let (input, old_cfg, new_cfg) = (g.input.clone(), g.old_cfg.clone(), g.new_cfg.clone());
        suite.group(g.spec.id, |bg| {
            bg.baseline(ARM_OLD);
            bg.config()
                .max_rounds(plan::ROUNDS)
                .min_rounds(plan::ROUNDS)
                .auto_rounds(false)
                .baseline_only(true)
                .max_time(Duration::from_secs(plan::GROUP_MAX_TIME_SECS))
                .max_wall_time(Duration::from_secs(plan::GROUP_MAX_WALL_SECS));
            // Each callback times one whole public resize: construction, every
            // row, the output allocation and (via `iter`) the output's drop.
            // zenbench black-boxes the whole returned buffer.
            let (i, c) = (input.clone(), old_cfg.clone());
            bg.bench(ARM_OLD, move |b| b.iter(|| old_ops::run(e, &c, &i)));
            let (i, c) = (input.clone(), new_cfg.clone());
            bg.bench(ARM_NEW, move |b| b.iter(|| new_ops::run(e, &c, &i)));
            let (i, c) = (input.clone(), old_cfg.clone());
            bg.bench(ARM_AA, move |b| b.iter(|| old_ops::run(e, &c, &i)));
        });
    }
}

fn main() {
    let start = SystemTime::now();
    let cli = parse_cli();
    let child = std::env::var("ZENBENCH_SUBPROCESS").as_deref() == Ok("1");
    check_env(child);
    let artifacts = check_paths(&cli);
    let mem_kib = check_memory();
    // Validation is complete; from here on the run may write artifacts.
    let cas = Cas::open(&artifacts).unwrap_or_else(|e| fail(3, &e));
    let binary = binary_sha256();
    let role = if cli.dry_run {
        "dryrun"
    } else if child {
        "child"
    } else {
        "parent"
    };
    let (groups, inputs_json) = load_groups(&cas);
    let identity = json!({
        "workload_id": WORKLOAD_ID, "binary_sha256": binary, "pid": std::process::id(),
        "args": cli.args, "zenbench_env": zenbench_env(), "mem_available_kib": mem_kib,
        "plan": {"processes": plan::PROCESSES, "rounds": plan::ROUNDS, "margin_pct": plan::MARGIN_PCT,
            "confidence": plan::CONFIDENCE, "timeout_secs": plan::TIMEOUT_SECS,
            "group_max_time_secs": plan::GROUP_MAX_TIME_SECS, "group_max_wall_secs": plan::GROUP_MAX_WALL_SECS,
            "arms": [ARM_OLD, ARM_NEW, ARM_AA], "baseline": ARM_OLD},
        "groups": GROUPS.iter().map(|g| json!({"id": g.id, "role": g.role.name(), "engine": g.engine.name(),
            "format": g.format.name(), "input": g.input, "out": [g.out_w, g.out_h], "in_stride_pad": g.in_stride_pad,
            "acceptance": acceptance::rule_text(g.role)})).collect::<Vec<_>>(),
        "inputs": inputs_json,
    });

    let (rows, ok) = parity(&cas, &groups);
    let before = write_record(
        &artifacts,
        role,
        "before",
        json!({"identity": identity, "parity": rows, "all_equal": ok}),
    );
    eprintln!(
        "resize-bench: {role} parity before timing: {} -> {}",
        if ok { "all equal" } else { "MISMATCH" },
        before.display()
    );
    if !ok {
        fail(3, "old/new outputs differ; refusing to time");
    }

    let result = zenbench::run(|suite| register(suite, &groups));
    // Dry runs exit inside zenbench::run after printing the resolved plan.

    if child {
        zenbench::postprocess_result(&result);
        let (rows, ok) = parity(&cas, &groups);
        let after = write_record(
            &artifacts,
            role,
            "after",
            json!({"identity": identity, "parity": rows, "all_equal": ok}),
        );
        eprintln!(
            "resize-bench: child parity after timing -> {}",
            after.display()
        );
        std::process::exit(if ok { 0 } else { 3 });
    }

    let (rows, ok) = parity(&cas, &groups);
    write_record(
        &artifacts,
        role,
        "after",
        json!({"identity": identity, "parity": rows, "all_equal": ok}),
    );
    let (verdict, code) = acceptance::evaluate(&result, &artifacts, start, &binary, ok);
    let dir = artifacts.join("acceptance");
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| fail(2, &format!("acceptance dir: {e}")));
    let path = dir.join(format!("acceptance-{}.json", unique_suffix()));
    write_new(
        &path,
        &serde_json::to_vec_pretty(&verdict).expect("verdict serializes"),
    )
    .unwrap_or_else(|e| fail(2, &e));
    eprintln!("resize-bench: acceptance exit {code} -> {}", path.display());
    zenbench::postprocess_result(&result);
    std::process::exit(code);
}
