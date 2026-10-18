#!/usr/bin/env python3
"""External CPU-placement receipt for resize-bench processes (Linux /proc only).

`watch` runs outside the benchmark (run.sh pins it to a CPU outside the
measurement scope). It walks the process tree under --root-pid and, for every
process whose executable is --exe, records each thread's TID, kernel comm and
Cpus_allowed_list, plus the process's cgroup and that cgroup's
cpuset.cpus.effective. It writes a JSON line only when that state changes, and
exits when the root process is gone. Nothing is read from or written into the
benchmark processes themselves.

`check` judges a receipt against the requested placement and prints one
PASS/FAIL line per rule; it exits 1 if any rule fails. Threads are identified by
PID, TID and comm only: zenbench names its resource sampler thread
"resource-sampler", which the kernel truncates to the 15-byte comm
"resource-sample" (zenbench 98c25fe8 src/monitor.rs:44).

usage: proc-receipt.py watch --root-pid PID --exe PATH --out NEW_FILE [--interval SECONDS]
       proc-receipt.py check --receipt FILE --scope CPUS [--pin CPUS] [--children N]
                             [--placement-stderr FILE] [--require-cgroup-cpuset]
"""

import argparse
import json
import os
import re
import sys
import time

SAMPLER_COMM = "resource-sample"


def cpu_list(spec):
    out = set()
    for part in spec.strip().split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def read(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def status_field(text, name):
    m = re.search(rf"(?m)^{name}:\s*(.*)$", text or "")
    return m.group(1).strip() if m else None


def ppid_map():
    out = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        stat = read(f"/proc/{d}/stat")
        if not stat:
            continue
        # comm may contain spaces and parentheses: the ppid follows the last ')'.
        rest = stat[stat.rfind(")") + 2:].split()
        if len(rest) >= 2:
            out[int(d)] = int(rest[1])
    return out


def descendants(root):
    parents = ppid_map()
    children = {}
    for pid, ppid in parents.items():
        children.setdefault(ppid, []).append(pid)
    seen, stack = [], [root]
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.append(pid)
        stack.extend(children.get(pid, []))
    return seen, parents


def snapshot(root, exe):
    pids, parents = descendants(root)
    procs = {}
    for pid in pids:
        try:
            if os.path.realpath(f"/proc/{pid}/exe") != exe:
                continue
        except OSError:
            continue
        cgroup = (read(f"/proc/{pid}/cgroup") or "").strip()
        cg_path = cgroup.split("::", 1)[1] if cgroup.startswith("0::") else None
        effective = read(f"/sys/fs/cgroup{cg_path}/cpuset.cpus.effective") if cg_path else None
        threads = {}
        try:
            tids = os.listdir(f"/proc/{pid}/task")
        except OSError:
            continue
        for tid in tids:
            st = read(f"/proc/{pid}/task/{tid}/status")
            comm = (read(f"/proc/{pid}/task/{tid}/comm") or "").strip()
            mask = status_field(st, "Cpus_allowed_list")
            if st is None or mask is None:
                continue
            threads[tid] = {"comm": comm, "cpus_allowed": mask}
        procs[str(pid)] = {
            "ppid": parents.get(pid),
            "cgroup": cg_path,
            "cpuset_effective": effective.strip() if effective else None,
            "threads": threads,
        }
    return procs


def watch(args):
    exe = os.path.realpath(args.exe)
    if os.path.exists(args.out):
        sys.exit(f"proc-receipt: {args.out} exists; a receipt is never overwritten")
    start = time.monotonic()
    last = None
    lines = 0
    with open(args.out, "x", encoding="utf-8") as out:
        out.write(json.dumps({"receipt": "resize-bench proc placement", "root_pid": args.root_pid,
                              "exe": exe, "interval_s": args.interval,
                              "observer_pid": os.getpid(),
                              "observer_cpus": sorted(os.sched_getaffinity(0))}) + "\n")
        while True:
            alive = os.path.exists(f"/proc/{args.root_pid}")
            procs = snapshot(args.root_pid, exe) if alive else {}
            if procs != last:
                out.write(json.dumps({"t_s": round(time.monotonic() - start, 3),
                                      "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                      "processes": procs}, sort_keys=True) + "\n")
                out.flush()
                lines += 1
                last = procs
            if not alive:
                break
            time.sleep(args.interval)
    print(f"proc-receipt: {lines} state changes recorded in {args.out}")


def check(args):
    scope = cpu_list(args.scope)
    pin = cpu_list(args.pin) if args.pin else set()
    rules = []

    def rule(ok, text):
        rules.append((bool(ok), text))

    states = []
    with open(args.receipt, encoding="utf-8") as f:
        header = json.loads(f.readline())
        for line in f:
            states.append(json.loads(line)["processes"])
    observer = set(header.get("observer_cpus", []))
    rule(observer and not (observer & scope), f"observer ran on CPUs {sorted(observer)}, outside the scope {sorted(scope)}")
    seen = {}
    for procs in states:
        for pid, p in procs.items():
            seen.setdefault(pid, []).append(p)
    rule(seen, f"{len(seen)} benchmark process(es) observed")
    bench_pids = set(seen)
    roots = [pid for pid, ps in seen.items() if str(ps[0]["ppid"]) not in bench_pids]
    children = [pid for pid in seen if pid not in roots]
    rule(len(roots) == 1, f"exactly one top-level benchmark process (found {sorted(roots, key=int)})")
    if args.children is not None:
        rule(len(children) == args.children, f"{len(children)} child benchmark processes observed, expected {args.children}")
    outside, cpusets, sampler_obs, sampler_bad, main_pinned = [], set(), {}, [], {}
    for pid, ps in seen.items():
        for p in ps:
            if p["cpuset_effective"] is not None:
                cpusets.add(p["cpuset_effective"])
            for tid, t in p["threads"].items():
                mask = cpu_list(t["cpus_allowed"])
                if not mask <= scope:
                    outside.append(f"pid {pid} tid {tid} ({t['comm']}): {t['cpus_allowed']}")
                if t["comm"] == SAMPLER_COMM:
                    sampler_obs.setdefault(pid, set()).add(t["cpus_allowed"])
                    if pin and mask & pin:
                        sampler_bad.append(f"pid {pid} tid {tid}: {t['cpus_allowed']} overlaps pin {sorted(pin)}")
                if tid == pid and pin and mask == pin:
                    main_pinned[pid] = True
    rule(not outside, "every observed thread mask lies inside the scope" + ("" if not outside else ": " + "; ".join(outside[:5])))
    if args.require_cgroup_cpuset:
        rule(cpusets and all(cpu_list(c) == scope for c in cpusets),
             f"cgroup cpuset.cpus.effective {sorted(cpusets) or 'not readable'} equals the scope")
    if pin and children:
        missing = [pid for pid in children if pid not in sampler_obs]
        rule(not missing, f"a '{SAMPLER_COMM}' thread was observed in every child (missing in {missing})")
        rule(not sampler_bad, f"every '{SAMPLER_COMM}' thread mask is disjoint from the pin" + ("" if not sampler_bad else ": " + "; ".join(sampler_bad[:5])))
        print("info: child main threads observed pinned to exactly the pin: "
              f"{sum(1 for pid in children if main_pinned.get(pid))} of {len(children)} (snapshot-based, not a rule)")
    if args.placement_stderr:
        text = read(args.placement_stderr) or ""
        m = re.search(r"\[zenbench\] cpu_policy=\S+ cpu_placement=(\S+) .*?cpu_enforcement=(\S+) cpu_effective_set=\[([0-9, ]*)\]", text)
        if not m:
            rule(False, "zenbench placement line present in stderr")
        else:
            placement, enforcement, eff = m.group(1), m.group(2), cpu_list(m.group(3).replace(" ", ""))
            rule(placement in ("uniform_hardware", "enforced_uniform"), f"zenbench cpu_placement={placement} is verified single-class")
            rule(enforcement in ("cpuset", "process_mask", "job_affinity"), f"zenbench cpu_enforcement={enforcement} is a real OS boundary")
            rule(eff == scope, f"zenbench cpu_effective_set={sorted(eff)} equals the scope {sorted(scope)}")
    for ok, text in rules:
        print(("PASS " if ok else "FAIL ") + text)
    failed = sum(1 for ok, _ in rules if not ok)
    print(f"placement receipt: {len(rules) - failed} of {len(rules)} rules passed")
    sys.exit(1 if failed else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("watch")
    w.add_argument("--root-pid", type=int, required=True)
    w.add_argument("--exe", required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--interval", type=float, default=0.5)
    c = sub.add_parser("check")
    c.add_argument("--receipt", required=True)
    c.add_argument("--scope", required=True)
    c.add_argument("--pin")
    c.add_argument("--children", type=int)
    c.add_argument("--placement-stderr")
    c.add_argument("--require-cgroup-cpuset", action="store_true")
    args = ap.parse_args()
    watch(args) if args.cmd == "watch" else check(args)


if __name__ == "__main__":
    main()
