"""End-of-circuit fault distance of an outer-check gadget design on every place where an
outer-check part can end a circuit (scripts/job_distance.py --method restricted; one case
per task: GPURUN_TASK).

Cases: non-adaptive extraction (rounds 0-1, every part) in the X basis with the Z checks
last (the protocol's order: their cats spread Z errors, seen then only by the final
measurement) and in the Z basis with either order; the rotating rules' single outer
checks (L3r_j of R8) ending the circuit in both bases. Logical classes 0, 1 and 0+1.

Usage: python scripts/distance_suite.py <design> list|run [--task-offset k]
       python scripts/distance_suite.py <design> pool [n] [all|design|suite]   (missing cases, n workers)
"""
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def cases(design):
    out = []
    for basis, order in (("X", "XZ"), ("Z", "ZX"), ("Z", "XZ")):
        for combo in ("0", "1", "01"):
            out.append((f"g3={design},g3order={order}", basis, "full", combo,
                        f"results/distance/end__{design}__{order}__{basis}__c{combo}.json"))
    # a quiet last round (level 1 only) after a full one, as in adaptive extraction: the
    # restriction also admits the last level-1 comparisons (--method restricted_l1)
    for basis in "XZ":
        for combo in ("0", "1", "01"):
            out.append((f"g3={design}", basis, "0:L1_a+L2_all+L3/1:L1_b", combo,
                        f"results/distance/end__{design}__quiet__{basis}__c{combo}.json"))
    rot = f"l2=on_event,l3=ambiguous,rot=8,g3={design}"
    for j in range(8):
        for basis in "XZ":
            for combo in ("0", "1", "01"):
                path = f"0:L1_a+L2_all+L3/1:L1_b+L3r_{j}"
                out.append((rot, basis, path, combo, f"results/distance/end__{design}__R8_{j}__{basis}__c{combo}.json"))
    return out


def design_cases():
    """The outer-check gadget designs compared (X basis, non-adaptive round)."""
    out = []
    for design in ("latin4", "k4cyc_e", "k5pre_e", "k6_e", "k5pre", "k6"):
        for combo in ("0", "1", "01"):
            out.append((f"g3={design}", "X", "full", combo, f"results/distance/design__{design}__X__c{combo}.json"))
    return out


def method(path):
    return "restricted_l1" if path.endswith("/1:L1_b") else "restricted"


def run_case(c):
    rule, basis, path, combo, out = c
    if os.path.exists(os.path.join(ROOT, out)):
        return out, "exists"
    r = subprocess.run([sys.executable, "scripts/job_distance.py", "--rule", rule, "--basis", basis, "--rounds", "1",
                        "--path", path, "--W", "10", "--combo", combo, "--method", method(path),
                        "--time-limit", "150000" if method(path) == "restricted_l1" else "40000", "--out", out],
                       cwd=ROOT, capture_output=True, text=True)
    return out, (r.stdout.strip().splitlines() or [""])[-1] + r.stderr[-300:]


def main():
    design, what = sys.argv[1], sys.argv[2]
    cs = cases(design)
    if what == "pool":
        # every missing case of the suite and of the design comparison, n workers
        from multiprocessing import Pool
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
        which = sys.argv[4] if len(sys.argv) > 4 else "all"     # all | design | suite
        pick = {"all": cs + design_cases(), "design": design_cases(), "suite": cs}[which]
        todo = [c for c in pick if not os.path.exists(os.path.join(ROOT, c[4]))]
        with Pool(n) as pool:
            for out, msg in pool.imap_unordered(run_case, todo):
                print(out, msg, flush=True)
        return
    if what == "list":
        for i, c in enumerate(cs):
            print(i, *c)
        return
    off = int(sys.argv[sys.argv.index("--task-offset") + 1]) if "--task-offset" in sys.argv else 0
    i = int(os.environ.get("GPURUN_TASK", "0")) + off
    rule, basis, path, combo, out = cs[i]
    subprocess.run([sys.executable, "scripts/job_distance.py", "--rule", rule, "--basis", basis, "--rounds", "1",
                    "--path", path, "--W", "10", "--combo", combo, "--method", method(path),
                    "--time-limit", "150000" if method(path) == "restricted_l1" else "40000", "--out", out],
                   cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
