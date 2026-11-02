"""Sampling and distance configurations with fixed counts and reproducible seeds.

Print local commands for all configurations or only those missing their result files.
Use the pinned dependencies to reproduce numerical samples; execution times can vary.
The c4_repro campaign samples full-extraction C4 under uniform noise in X and Z
for the authors' Fig. 11a comparison; scripts/c4_baseline.py refits the samples.
Usage:
    python scripts/campaigns.py list
    python scripts/campaigns.py commands <name,...|all> [--missing]
Commands write to results/. Use a separate copy when preserving the supplied evidence.
"""
import glob
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def strat(camp, tag, proto, rule, noise, pdec, kappa, basis, rounds, shots, tasks=1, procs=4, seed=1,
          decoder=None):
    out = f"results/final/{camp}/{tag}/{basis}/k{kappa}/s{seed}_t{{task}}.json"
    # Carbon-Helix decoder: Frontier K = 512, Delta = 12, forward + backward, re-decoding
    # stages (1024, 16) and (4096, 24) on disagreement / no path / gap < 8
    dec = decoder if decoder is not None else (["--K", "512", "--Delta", "12"] if proto == "carbonhelix" else [])
    # Memory allowance for the larger Carbon-Helix decoder models.
    mem = 4 if proto == "carbonhelix" else (procs + 1 if procs > 1 else 2)
    return dict(campaign=camp, tasks=tasks, procs=procs, mem=mem,
                cmd=["scripts/sample_strata.py", "--proto", proto, "--rule", rule, "--noise", noise,
                     "--p-dec", str(pdec), "--kappa", str(kappa), "--basis", basis,
                     "--rounds", str(rounds), "--shots", str(shots), "--procs", str(procs),
                     "--seed", str(seed)] + dec + ["--out", out])


def campaigns():
    C = {}
    runs = []
    for b in "XZ":
        for kappa, seed in ((1.5, 41), (2.5, 42)):
            runs.append(strat("c4_repro", "full", "c4helix", "full", "uniform", 9e-4, kappa, b, 12,
                              shots=10000, tasks=5, procs=1, seed=seed))
            runs[-1]["cmd"] += ["--K", "512", "--Delta", "12", "--gap-stages", "1"]
            runs[-1]["mem"] = 2
    C["c4_repro"] = runs
    # ---------------------------------------------------------------- Carbon-Helix
    # main comparison: 4 rounds after the preparation round; X basis (the weaker one under
    # dephasing-dominated noise) with two odds scales, Z basis with one
    # Each Carbon configuration uses one process and its recorded seed.
    main_rules = {"full": "full", "C": "l2=on_event,l3=any", "B4": "l2=on_event,l3=ambiguous,T3=4",
                  "R4": "l2=on_event,l3=ambiguous,rot=4", "R8": "l2=on_event,l3=ambiguous,rot=8",
                  "R8F": "l2=on_event,l3=ambiguous,rot=8,l2fast=1"}
    runs = []
    for tag, rule in main_rules.items():
        # the Z basis (fewer failures under dephasing-dominated noise) only as a check of the
        # basis asymmetry, for non-adaptive extraction
        zs = (("Z", 2.5, 12, 2000, 123),) if tag == "full" else ()
        # Pool every high-odds sample file; base_tasks does not restrict the analysis.
        for b, kappa, tasks, shots, seed in (("X", 2.0, 24, 2000, 121), ("X", 3.0, 32, 500, 122)) + zs:
            runs.append(strat("ch_main", tag, "carbonhelix", rule, "uniform", 1e-3, kappa, b, 4,
                              shots=shots, tasks=tasks, procs=1, seed=seed))
            if kappa == 3.0:
                runs[-1]["base_tasks"] = 16
        zs = (("Z", 2.0, 12, 2000, 126),) if tag == "full" else ()
        for b, kappa, tasks, shots, seed in (("X", 1.5, 24, 2000, 124), ("X", 2.2, 32, 500, 125)) + zs:
            runs.append(strat("ch_helios", tag, "carbonhelix", rule, "helios", 7.9e-4, kappa, b, 4,
                              shots=shots, tasks=tasks, procs=1, seed=seed))
            if kappa == 2.2:
                runs[-1]["base_tasks"] = 16
    C["carbon"] = runs
    # unpublished Helios quantities: memory error per unit time (x0.5, x2) and the
    # measurement/reset duration (4 ms instead of 2 ms per batch)
    runs = []
    for noise in ("helios_mem0.5", "helios_mem2", "helios_tm4"):
        for tag, rule in (("full", "full"), ("R4", "l2=on_event,l3=ambiguous,rot=4")):
            runs.append(strat("ch_helios_sens", f"{noise}/{tag}", "carbonhelix", rule, noise, 7.9e-4, 1.8,
                              "X", 4, shots=1000, tasks=16, procs=1, seed=151))
            if noise == "helios_mem0.5":
                # with half the memory error the strata at kappa 1.8 hold almost no failures:
                # a second set of strata at more faults per shot
                runs.append(strat("ch_helios_sens", f"{noise}/{tag}", "carbonhelix", rule, noise, 7.9e-4, 3.0,
                                  "X", 4, shots=500, tasks=24, procs=1, seed=153))
        for tag, rule in (("full", "full"), ("ase", "mode=ase")):
            j = strat("c4_helios_sens", f"{noise}/{tag}", "c4helix", rule, noise, 7.9e-4, 1.5, "X", 4,
                      shots=40000, tasks=2, procs=1, seed=152)
            j["cmd"] += ["--K", "512", "--Delta", "12", "--gap-stages", "1"]
            runs.append(j)
    C["helios_sens"] = runs
    # direct sampling of Carbon-Helix at a high p: an independent check of the stratified
    # estimate and of the extrapolation model (a held-out point)
    runs = []
    for p, tasks in ((2e-3, 12), (3e-3, 6)):
        runs.append(dict(campaign="ch_direct", tasks=tasks, procs=1, mem=2,
                         cmd=["scripts/sample_direct.py", "--proto", "carbonhelix", "--rule", "full",
                              "--noise", "uniform", "--p", str(p), "--basis", "X", "--rounds", "4",
                              "--shots", "2000", "--K", "512", "--Delta", "12", "--Delta2", "24",
                              "--seed", "161", "--out", f"results/final/ch_direct/p{p}/X/task{{task}}.json"]))
    C["ch_direct"] = runs
    # Interval comparison: 8 rounds with a forced full measurement every T rounds.
    # Both odds scales contribute to the pooled fault-count histograms.
    runs = []
    cfgs = {"full": "full"}
    for T in (2, 4, 8):
        cfgs[f"B_T{T}"] = f"l2=on_event,l3=ambiguous,T2={T},T3={T}"
    cfgs["B_Tinf"] = "l2=on_event,l3=ambiguous"
    cfgs["C_T4"] = "l2=on_event,l3=any,T2=4,T3=4"
    cfgs["C_Tinf"] = "l2=on_event,l3=any"
    cfgs["R4"] = "l2=on_event,l3=ambiguous,rot=4"
    cfgs["R8"] = "l2=on_event,l3=ambiguous,rot=8"
    for tag, rule in cfgs.items():
        for kappa, tasks, shots, seed in ((2.0, 16, 250, 132), (3.3, 24, 500, 133)):
            runs.append(strat("ch_interval", tag, "carbonhelix", rule, "uniform", 1e-3, kappa, "X", 8,
                              shots=shots, tasks=tasks, procs=1, seed=seed))
    C["interval"] = runs
    # windows that always reach the forced interval: no escalation, the outer checks only
    # every T rounds (weight-6 checks on events as in rule B). Under rule B escalations end
    # most windows early at the error rates where estimates are data-driven, so the forced
    # interval barely matters there; without them the window penalty is fully exposed and
    # its p-dependence measurable (same strata, rounds and decoder as the interval campaign)
    runs = []
    for T in (2, 4, 8):
        for kappa, tasks, shots, seed in ((2.0, 16, 250, 134), (3.3, 24, 500, 135)):
            runs.append(strat("ch_interval", f"P_T{T}", "carbonhelix", f"l2=on_event,l3=never,T2={T},T3={T}",
                              "uniform", 1e-3, kappa, "X", 8, shots=shots, tasks=tasks, procs=1, seed=seed))
    C["interval_periodic"] = runs
    # Extend the periodic comparison to a full-check interval of 16 rounds.
    # Keep the same total sample count at each odds scale as for the 8-round memories.
    runs = []
    for kappa, tasks, shots, seed in ((2.0, 32, 125, 136), (3.3, 48, 250, 137)):
        runs.append(strat("ch_interval", "P_T16", "carbonhelix", "l2=on_event,l3=never,T2=16,T3=16",
                          "uniform", 1e-3, kappa, "X", 16, shots=shots, tasks=tasks, procs=1, seed=seed))
    C["interval_periodic16"] = runs
    runs = []
    # C4-Helix under the same configuration as the Carbon-Helix main comparison (4 rounds,
    # X basis, same noise models and decoder family)
    for noise, pdec, ks in (("uniform", 1e-3, (0.5, 1.0, 2.0)), ("helios", 7.9e-4, (0.5, 1.0, 1.5))):
        for tag, rule in (("full", "full"), ("ase", "mode=ase"), ("aseT4", "mode=ase,T=4")):
            for b in "X":
                for k in ks:
                    j = strat(f"c4_strat_{noise}", tag, "c4helix", rule, noise, pdec, k, b, 4,
                              shots=40000, tasks=4, procs=1, seed=109)
                    j["cmd"] += ["--K", "512", "--Delta", "12", "--gap-stages", "1"]
                    runs.append(j)
    C["c4_strat"] = runs
    # circuit fault distance (scripts/path_distance.py --method restricted: integer programming
    # over the mechanisms seen only by the final measurement and the flags, i.e. faults at the
    # end of the circuit, where data errors left by an outer-check gadget meet no later check)
    # at every place an outer-check part can end a circuit (scripts/distance_suite.py), and
    # the same for the gadget designs compared (results/distance/design__*.json)
    runs = []
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from distance_suite import cases as dist_cases, method as dist_method
    for rule, basis, path, combo, out in dist_cases("k4cyc"):
        # a quiet last round admits four times as many mechanisms: hours and up to 16 GB
        quiet = dist_method(path) == "restricted_l1"
        runs.append(dict(campaign="distance", tasks=1, procs=1, mem=16 if quiet else 3, time="2d" if quiet else "12h",
                         cmd=["scripts/path_distance.py", "--rule", rule, "--basis", basis, "--rounds", "1",
                              "--path", path, "--W", "10", "--combo", combo, "--method", dist_method(path),
                              "--time-limit", "150000" if quiet else "40000", "--out", out]))
    # the whole two-round circuit, unrestricted (integer programming for 11 hours: the proven
    # lower bound and the best logical error found are reported, converged or not)
    for basis, order in (("X", "XZ"), ("Z", "ZX")):
        runs.append(dict(campaign="distance", tasks=1, procs=1, mem=16, time="14h",
                         cmd=["scripts/path_distance.py", "--rule", f"g3=k4cyc,g3order={order}", "--basis", basis,
                              "--rounds", "1", "--path", "full", "--W", "10", "--combo", "1", "--method", "ilp",
                              "--time-limit", "40000",
                              "--out", f"results/distance/whole__k4cyc__{order}__{basis}__c1.json"]))
    from distance_suite import design_cases
    for rule, basis, path, combo, out in design_cases():
        runs.append(dict(campaign="distance", tasks=1, procs=1, mem=3, time="8h",
                         cmd=["scripts/path_distance.py", "--rule", rule, "--basis", basis, "--rounds", "1",
                              "--path", path, "--W", "10", "--combo", combo, "--method", "restricted",
                              "--time-limit", "20000", "--out", out]))
    C["distance"] = runs
    # one Carbon block inside a window between full measurements (scripts/inner_window_distance.py):
    # exact circuit distance of its inner logicals, every placement of the weight-6 checks
    C["inner_window"] = [dict(campaign="inner_window", tasks=1, procs=1, mem=2, time="8h" if T == 8 else "2h",
                              cmd=["scripts/inner_window_distance.py", "--T", str(T), "--basis", basis,
                                   "--out", f"results/inner_window/T{T}_{basis}.json"])
                         for T in (2, 4, 8) for basis in "XZ"]
    C["inner_window"].append(dict(campaign="inner_window", tasks=1, procs=1, mem=2, time="2h",
                                  cmd=["scripts/inner_window_distance.py", "--structure",
                                       "--out", "results/inner_window/structure.json"]))
    # the whole circuit's fault distance by decomposition (scripts/relaxed_distance.py): every
    # subset of active outer-check instances of each path kind, shared out over the tasks
    runs = []
    for basis, order, kind, tasks in (("Z", "ZX", "full1", 1), ("Z", "XZ", "full1", 1), ("X", "XZ", "full1", 1),
                                      ("X", "XZ", "P8x1", 1), ("X", "XZ", "P4x2", 1), ("X", "XZ", "P2x4", 6),
                                      ("X", "XZ", "P16x1", 1),
                                      ("X", "XZ", "full8", 32), ("Z", "XZ", "full8", 32), ("X", "XZ", "any8", 48),
                                      ("X", "XZ", "any4", 8), ("Z", "XZ", "any4", 8)):
        runs.append(dict(campaign="relaxed_distance", tasks=tasks, procs=1, mem=4, time="8h",
                         cmd=["scripts/relaxed_distance.py", "--basis", basis, "--order", order, "--kind", kind,
                              "--ntasks", str(tasks),
                              "--out", f"results/relaxed_distance/{basis}_{order}_{kind}/task{{task}}.json"]))
    C["relaxed_distance"] = runs
    return C


def commands(run):
    """Local shell commands for each sample or solver partition of a configuration."""
    import shlex
    out = []
    for task in range(run["tasks"]):
        cmd = [sys.executable] + run["cmd"]
        if "--task" not in cmd and run["cmd"][0] != "scripts/inner_window_distance.py":
            cmd += ["--task", str(task)]
        if run["cmd"][0] == "scripts/relaxed_distance.py":
            cmd += ["--ntasks", str(run["tasks"])]
        out.append(" ".join(shlex.quote(x) for x in cmd))
    return out


def result_path(run, task):
    out = run["cmd"][run["cmd"].index("--out") + 1]
    return os.path.join(ROOT, out.replace("{task}", str(task)))


def main():
    C = campaigns()
    what = sys.argv[1] if len(sys.argv) > 1 else "list"
    if what == "list":
        for k, v in C.items():
            have = sum(os.path.exists(result_path(j, t)) for j in v for t in range(j["tasks"]))
            total = sum(j["tasks"] for j in v)
            print(f"{k:22s} {len(v):3d} runs {total:5d} tasks {have:5d} result files present")
        return
    if what == "commands":
        names = list(C) if len(sys.argv) < 3 or sys.argv[2] == "all" else sys.argv[2].split(",")
        missing_only = "--missing" in sys.argv
        for nm in names:
            for run in C[nm]:
                for task, cmd in enumerate(commands(run)):
                    if missing_only and os.path.exists(result_path(run, task)):
                        continue
                    print(cmd)
        return
    sys.exit("usage: python scripts/campaigns.py [list | commands <name,...|all> [--missing]]")


if __name__ == "__main__":
    main()
