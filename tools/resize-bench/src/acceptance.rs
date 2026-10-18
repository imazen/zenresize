//! Declared acceptance rules, evaluated on the aggregated Confirm result.
//! zenbench's own exit code only says "nothing regressed and the evidence is
//! valid"; acceptance here additionally requires every A/A and control to be
//! Equivalent within the margin and every target to improve by more than it.

use crate::inputs::WORKLOAD_ID;
use crate::plan::{self, ARM_AA, ARM_NEW, ARM_OLD, GROUPS, Role};
use serde_json::{Value, json};
use std::path::Path;
use std::time::SystemTime;
use zenbench::{BenchmarkMode, SuiteResult, Verdict};

pub(crate) fn rule_text(role: Role) -> &'static str {
    match role {
        Role::Target => "old->new Improvement (upper < -margin); old->old-aa Equivalent",
        Role::StartupTarget => "old->new non-inferior (upper <= +margin); old->old-aa Equivalent",
        Role::Control => "old->new Equivalent (within +-margin); old->old-aa Equivalent",
    }
}

fn new_ok(role: Role, v: Verdict) -> bool {
    match role {
        Role::Target => v == Verdict::Improvement,
        Role::StartupTarget => matches!(
            v,
            Verdict::Improvement | Verdict::Equivalent | Verdict::NonInferior
        ),
        Role::Control => v == Verdict::Equivalent,
    }
}

/// Count this run's child parity records for one phase; all must be equal and
/// from the same executable.
fn child_records(
    artifacts: &Path,
    start: SystemTime,
    phase: &str,
    binary: &str,
) -> (usize, Vec<String>) {
    let mut ok = 0;
    let mut problems = Vec::new();
    let dir = artifacts.join("records");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return (0, vec![format!("cannot read {}", dir.display())]);
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.starts_with(&format!("child-{phase}-")) {
            continue;
        }
        let fresh = entry
            .metadata()
            .and_then(|m| m.modified())
            .is_ok_and(|t| t >= start);
        if !fresh {
            continue;
        }
        let parsed: Option<Value> = std::fs::read(entry.path())
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok());
        match parsed {
            Some(v)
                if v["all_equal"] == json!(true)
                    && v["identity"]["binary_sha256"] == json!(binary)
                    && v["identity"]["workload_id"] == json!(WORKLOAD_ID) =>
            {
                ok += 1
            }
            _ => problems.push(format!(
                "child record {name} is not an all-equal record from this binary"
            )),
        }
    }
    (ok, problems)
}

pub(crate) fn evaluate(
    result: &SuiteResult,
    artifacts: &Path,
    start: SystemTime,
    binary: &str,
    parent_after_equal: bool,
) -> (Value, i32) {
    let mut problems: Vec<String> = Vec::new();
    let issues = result.confirmation_issues();
    problems.extend(issues.iter().map(|i| format!("zenbench: {i}")));

    match &result.execution {
        Some(ex) => {
            let p = &ex.plan;
            if p.mode != BenchmarkMode::Confirm {
                problems.push(format!("resolved mode is {}, not confirm", p.mode));
            }
            if p.processes != plan::PROCESSES
                || p.rounds != Some(plan::ROUNDS)
                || p.margin_pct != plan::MARGIN_PCT
                || p.confidence != plan::CONFIDENCE
                || p.workload_id.as_deref() != Some(WORKLOAD_ID)
                || p.scope.is_some()
                || p.timeout.as_secs() != plan::TIMEOUT_SECS
            {
                problems.push(format!("resolved plan differs from the frozen plan: {p:?}"));
            }
            if !ex.independent_processes {
                problems.push("aggregate is not from independent processes".into());
            }
            if ex.binary_sha256 != binary {
                problems.push(format!(
                    "executed binary {} is not {binary}",
                    ex.binary_sha256
                ));
            }
        }
        None => problems.push("no execution record".into()),
    }

    if result.replicates.len() != plan::PROCESSES {
        problems.push(format!(
            "{} replicates, expected {}",
            result.replicates.len(),
            plan::PROCESSES
        ));
    }
    for (r, rep) in result.replicates.iter().enumerate() {
        if rep.comparisons.len() != GROUPS.len() {
            problems.push(format!(
                "replicate {r}: {} groups, expected {}",
                rep.comparisons.len(),
                GROUPS.len()
            ));
        }
        for g in GROUPS {
            let Some(c) = rep.comparisons.iter().find(|c| c.group_name == g.id) else {
                problems.push(format!("replicate {r}: group {} missing", g.id));
                continue;
            };
            let mut names: Vec<&str> = c.benchmarks.iter().map(|b| b.name.as_str()).collect();
            names.sort_unstable();
            let mut want = vec![ARM_OLD, ARM_NEW, ARM_AA];
            want.sort_unstable();
            if names != want {
                problems.push(format!("replicate {r}: group {} arms {names:?}", g.id));
            }
            let fixed = c
                .config
                .as_ref()
                .is_some_and(|cfg| cfg.max_rounds == plan::ROUNDS && !cfg.auto_rounds);
            if !fixed || c.completed_rounds != plan::ROUNDS {
                problems.push(format!(
                    "replicate {r}: group {} completed {} of {} fixed rounds",
                    g.id,
                    c.completed_rounds,
                    plan::ROUNDS
                ));
            }
        }
    }

    let mut regression = false;
    let mut rows = Vec::new();
    for g in GROUPS {
        for (cand, is_aa) in [(ARM_NEW, false), (ARM_AA, true)] {
            let found: Vec<_> = result
                .process_comparisons
                .iter()
                .filter(|c| c.group == g.id && c.baseline == ARM_OLD && c.candidate == cand)
                .collect();
            let [c] = found.as_slice() else {
                problems.push(format!(
                    "group {}: {} comparisons old->{cand}, expected 1",
                    g.id,
                    found.len()
                ));
                continue;
            };
            let pass = if is_aa {
                c.verdict == Verdict::Equivalent
                    && c.lower_pct >= -plan::MARGIN_PCT
                    && c.upper_pct <= plan::MARGIN_PCT
            } else {
                new_ok(g.role, c.verdict)
            };
            if !is_aa && c.verdict == Verdict::Regression {
                regression = true;
            }
            if c.family_confidence != plan::CONFIDENCE || c.margin_pct != plan::MARGIN_PCT {
                problems.push(format!(
                    "group {}: comparison used confidence {} margin {}",
                    g.id, c.family_confidence, c.margin_pct
                ));
            }
            if !pass {
                problems.push(format!(
                    "group {} ({}) old->{cand}: {} [{:.2}%, {:.2}%] does not meet the rule",
                    g.id,
                    g.role.name(),
                    c.verdict,
                    c.lower_pct,
                    c.upper_pct
                ));
            }
            rows.push(json!({"group": g.id, "role": g.role.name(), "baseline": c.baseline, "candidate": c.candidate,
                "estimate_pct": c.estimate_pct, "lower_pct": c.lower_pct, "upper_pct": c.upper_pct,
                "verdict": c.verdict.to_string(), "comparisons_in_family": c.comparisons_in_family,
                "rule": if is_aa { "A/A Equivalent within +-margin" } else { rule_text(g.role) }, "pass": pass}));
        }
    }

    for phase in ["before", "after"] {
        let (n, p) = child_records(artifacts, start, phase, binary);
        problems.extend(p);
        if n != plan::PROCESSES {
            problems.push(format!(
                "{n} child parity records ({phase}), expected {}",
                plan::PROCESSES
            ));
        }
    }
    if !parent_after_equal {
        problems.push("parent parity after timing failed".into());
    }

    let accepted = problems.is_empty();
    let code = if accepted {
        0
    } else if regression {
        1
    } else {
        2
    };
    for p in &problems {
        eprintln!("resize-bench: NOT ACCEPTED: {p}");
    }
    eprintln!(
        "resize-bench: {}",
        if accepted {
            "ACCEPTED: every declared rule passed"
        } else {
            "NOT ACCEPTED"
        }
    );
    (
        json!({"workload_id": WORKLOAD_ID, "binary_sha256": binary, "accepted": accepted, "exit_code": code,
            "problems": problems, "comparisons": rows, "zenbench_confirmation_issues": issues}),
        code,
    )
}
