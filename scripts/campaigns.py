"""Registry of every simulation campaign behind the reported numbers.

Each campaign is a list of jobs (script + arguments, number of tasks, processes per
task). Shot counts are fixed and seeds deterministic, so rerunning a campaign at the same
commit reproduces its output files exactly. Usage:
    python scripts/campaigns.py list
    python scripts/campaigns.py submit <campaign|all> [--only-missing] [--match TEXT]
    python scripts/campaigns.py status
Submission goes through xpu (results come back by themselves); `status` counts
returned result files per campaign.
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
    # memory: Carbon-Helix tasks use up to 3.4 GB (decoder models of the larger circuits)
    mem = 4 if proto == "carbonhelix" else (procs + 1 if procs > 1 else 2)
    return dict(campaign=camp, tasks=tasks, procs=procs, mem=mem,
                cmd=["scripts/job_strat.py", "--proto", proto, "--rule", rule, "--noise", noise,
                     "--p-dec", str(pdec), "--kappa", str(kappa), "--basis", basis,
                     "--rounds", str(rounds), "--shots", str(shots), "--procs", str(procs),
                     "--seed", str(seed)] + dec + ["--out", out])


def campaigns():
    C = {}
    # ---------------------------------------------------------------- C4-Helix reproduction
    # uniform-noise memory of the paper's Fig. 11a by stratified sampling (12 rounds, decoder
    # priors at p = 9e-4, evaluated at 7e-4, 9e-4 and 1.1e-3)
    jobs = []
    for b in "XZ":
        for kappa, seed in ((1.5, 41), (2.5, 42)):
            jobs.append(strat("c4_repro", "full", "c4helix", "full", "uniform", 9e-4, kappa, b, 12,
                              shots=10000, tasks=5, procs=1, seed=seed))
            jobs[-1]["cmd"] += ["--K", "512", "--Delta", "12", "--gap-stages", "1"]
            jobs[-1]["mem"] = 2
    C["c4_repro"] = jobs
    # structural checks incl. exact circuit distances (integer programming, hours)
    C["checks"] = [dict(campaign="checks", tasks=1, procs=1, mem=3, time="1d",
                        cmd=["scripts/checks.py", "--out", "results/checks.json"])]
    jobs = []
    for name, noise, b, tasks, shots in (("c4helix", "helios", "X", 20, 4000), ("c4helix", "helios", "Z", 8, 4000),
                                        ("c4helix", "helios_mem0.5", "X", 8, 4000),
                                        ("c4helix", "helios_mem2", "X", 8, 4000),
                                        ("toric10", "helios", "X", 6, 20000)):
        camp = "c4_helios_mem" if name == "c4helix" else "t10_helios_mem"
        jobs.append(dict(campaign=camp, tasks=tasks, procs=1, mem=2,
                         cmd=["scripts/job_protocol.py", "--proto", name, "--rule", "full", "--noise", noise,
                              "--p", "7.9e-4", "--basis", b, "--rounds", "20", "--shots", str(shots),
                              "--K", "512", "--Delta", "12", "--seed", "31",
                              "--out", f"results/final/{camp}/{noise}/{b}/task{{task}}.json"]))
    C["helios_mem"] = jobs
    # the paper's Fig. 11 by direct sampling with the decoder's confidence gaps: memory, 12
    # rounds, both bases, at its three points; error and rejection per round under
    # forced-gap postselection (gamma up to 4 nats); gamma = 0 checks the stratified c4_repro
    jobs = []
    for b in "XZ":
        for p, tasks in ((7e-4, 40), (9e-4, 40), (1.1e-3, 20)):
            jobs.append(dict(campaign="c4_fig11", tasks=tasks, procs=1, mem=2, time="12h",
                             cmd=["scripts/job_protocol.py", "--proto", "c4helix", "--rule", "full", "--noise", "uniform",
                                  "--p", str(p), "--basis", b, "--rounds", "12", "--shots", "100000",
                                  "--K", "512", "--Delta", "12", "--seed", "33",
                                  "--out", f"results/final/c4_fig11/uniform_p{p}/{b}/task{{task}}.json"]))
    C["c4_fig11"] = jobs
    # decoder sensitivity of the forced-gap comparison for memory: the same shots (X basis,
    # p = 7e-4, tasks 0-9) decoded with an 8x wider beam
    C["c4_fig11_beam"] = [dict(campaign="c4_fig11_beam", tasks=10, procs=1, mem=6, time="24h",
                               cmd=["scripts/job_protocol.py", "--proto", "c4helix", "--rule", "full", "--noise", "uniform",
                                    "--p", "0.0007", "--basis", "X", "--rounds", "12", "--shots", "100000",
                                    "--K", "4096", "--Delta", "24", "--K2", "16384", "--Delta2", "32", "--seed", "33",
                                    "--out", "results/final/c4_fig11_beam/uniform_p0.0007/X/task{task}.json"])]
    # the paper's Fig. 14 ('C4-Helix transv.'): two blocks, 6 transversal CNOTs interleaved
    # with full rounds on both blocks, both bases, at its three points
    jobs = []
    for b in "XZ":
        for p, tasks in ((7e-4, 30), (9e-4, 14), (1.1e-3, 8)):
            jobs.append(dict(campaign="c4_fig14", tasks=tasks, procs=1, mem=3, time="12h",
                             cmd=["scripts/job_cnot2.py", "--n-cnot", "6", "--p", str(p), "--basis", b,
                                  "--shots", "50000", "--seed", "35",
                                  "--out", f"results/final/c4_fig14/uniform_p{p}/{b}/task{{task}}.json"]))
    C["c4_fig14"] = jobs
    jobs = []
    for noise, p in (("uniform", 1e-3), ("uniform", 1e-4), ("uniform", 5e-4), ("helios", 7.9e-4)):
        for ad in (0, 1):
            for m in (5, 7, 13):
                jobs.append(dict(campaign="tqrb_counts2", tasks=1, procs=1, mem=2,
                                 cmd=["scripts/job_tqrb.py", "--m", str(m), "--p", str(p), "--noise", noise,
                                      "--adaptive", str(ad), "--seqs", "400", "--shots-per-seq", "20",
                                      "--decode", "0", "--seed", "54",
                                      "--out", f"results/final/tqrb_counts2/{noise}_p{p}/a{ad}/m{m}/task{{task}}.json"]))
    C["tqrb_counts"] = jobs
    # logical error per Clifford by importance sampling (site odds x kappa); one run at p
    # gives every p' = s p of job_tqrb.IS_SCALES (uniform: 5e-4 is s = 0.5)
    jobs = []
    for noise, p, ms, kappa in (("uniform", 1e-3, (3, 5, 7), 2.5), ("helios", 7.9e-4, (5, 7, 13), 2.5)):
        for ad in (1, 0):
            for m in ms:
                jobs.append(dict(campaign="tqrb_is", tasks=2, procs=1, mem=8 if m >= 13 else 2,
                                 cmd=["scripts/job_tqrb.py", "--m", str(m), "--p", str(p), "--noise", noise,
                                      "--adaptive", str(ad), "--seqs", "50", "--shots-per-seq", "100",
                                      "--kappa", str(kappa), "--seed", "93",
                                      "--out", f"results/final/tqrb_is/{noise}_p{p}/a{ad}/m{m}/task{{task}}.json"]))
    C["tqrb_eps"] = jobs
    # logical error per Clifford by direct sampling at the paper's points (the importance-
    # sampled estimates above are too broad: TQRB sequences have ~1500 two-qubit gates)
    jobs = []
    for noise, p, ms in (("uniform", 1e-3, (3, 5, 7)), ("helios", 7.9e-4, (5, 7, 13))):
        for ad in (1, 0):
            for m in ms:
                jobs.append(dict(campaign="tqrb_direct", tasks=4, procs=1, mem=8 if m >= 13 else 3, time="24h",
                                 cmd=["scripts/job_tqrb.py", "--m", str(m), "--p", str(p), "--noise", noise,
                                      "--adaptive", str(ad), "--seqs", "100", "--shots-per-seq", "100",
                                      "--seed", "95",
                                      "--out", f"results/final/tqrb_direct/{noise}_p{p}/a{ad}/m{m}/task{{task}}.json"]))
    # the same at p = 5e-4, the paper's second point (App. Fig. 13: adaptive extraction 1.8x
    # lower per Clifford); failures are ten times rarer, so 1e6 shots per point (2e6 at m = 7)
    for ad in (1, 0):
        for m, tasks in ((3, 50), (5, 50), (7, 100)):
            jobs.append(dict(campaign="tqrb_direct", tasks=tasks, procs=1, mem=3, time="24h",
                             cmd=["scripts/job_tqrb.py", "--m", str(m), "--p", "0.0005", "--noise", "uniform",
                                  "--adaptive", str(ad), "--seqs", "200", "--shots-per-seq", "100",
                                  "--seed", "97",
                                  "--out", f"results/final/tqrb_direct/uniform_p0.0005/a{ad}/m{m}/task{{task}}.json"]))
    C["tqrb_direct"] = jobs
    # the paper's other TQRB simulation (App. Fig. 12: non-adaptive, the same uniform noise,
    # 6 random Cliffords and the inverse, i.e. m = 7 here, error per Clifford
    # 1 - (1 - p_L)^(1/7)) at its three points, with the decoder's confidence gaps for its
    # forced-gap postselection
    jobs = []
    for p in (7e-4, 9e-4, 1.1e-3):
        jobs.append(dict(campaign="tqrb_fig12", tasks=20, procs=1, mem=3, time="24h",
                         cmd=["scripts/job_tqrb.py", "--m", "7", "--p", str(p), "--noise", "uniform",
                              "--adaptive", "0", "--seqs", "100", "--shots-per-seq", "100", "--gap-hist", "1",
                              "--seed", "96",
                              "--out", f"results/final/tqrb_fig12/uniform_p{p}/a0/m7/task{{task}}.json"]))
    C["tqrb_fig12"] = jobs
    # TQRB circuit checks: exact fault distance of complete sequences (non-adaptive and both
    # adaptive branches), and the failures decoded with a gap >= 12 nats reduced to minimal
    # failing cores with their exact most-likely-error classification
    jobs = []
    for m in (3, 7):
        jobs.append(dict(campaign="tqrb_diag", tasks=3, procs=1, mem=4, time="24h",
                         cmd=["scripts/tqrb_distance.py", "--m", str(m), "--seed", "1", "--time-limit", "3600",
                              "--out", f"results/tqrb_distance/m{m}/task{{task}}.json"]))
    for p in (7e-4, 1.1e-3):
        jobs.append(dict(campaign="tqrb_diag", tasks=10, procs=1, mem=4, time="24h",
                         cmd=["scripts/tqrb_confident.py", "--p", str(p), "--nseq", "4", "--shots", "2500",
                              "--out", f"results/tqrb_confident/p{p}/task{{task}}.json"]))
    C["tqrb_diag"] = jobs
    # Fig. 13 under the Helios-like model scaled to two-qubit error p = 1e-3 (does Fig. 13's
    # "p" stand for a device-like model rather than the uniform one?)
    jobs = []
    for ad in (1, 0):
        for m in (3, 5, 7):
            jobs.append(dict(campaign="tqrb_fig13_helios", tasks=20, procs=1, mem=3, time="24h",
                             cmd=["scripts/job_tqrb.py", "--m", str(m), "--p", "0.001", "--noise", "helios",
                                  "--adaptive", str(ad), "--seqs", "100", "--shots-per-seq", "100",
                                  "--seed", "98",
                                  "--out", f"results/final/tqrb_fig13_helios/helios_p0.001/a{ad}/m{m}/task{{task}}.json"]))
    # and at Fig. 13's second point, p = 5e-4 (failures rarer: 2e5 shots per point)
    for ad in (1, 0):
        for m in (3, 5, 7):
            jobs.append(dict(campaign="tqrb_fig13_helios", tasks=20, procs=1, mem=3, time="24h",
                             cmd=["scripts/job_tqrb.py", "--m", str(m), "--p", "0.0005", "--noise", "helios",
                                  "--adaptive", str(ad), "--seqs", "100", "--shots-per-seq", "100",
                                  "--seed", "99",
                                  "--out", f"results/final/tqrb_fig13_helios/helios_p0.0005/a{ad}/m{m}/task{{task}}.json"]))
    C["tqrb_fig13_helios"] = jobs
    # the adaptive gain with a wider pruning window: the same shots (same seeds) at p = 1e-3
    # decoded with Delta = 24 (campaign decoder: Delta = 12)
    jobs = []
    for ad in (1, 0):
        for m in (3, 5, 7):
            jobs.append(dict(campaign="tqrb_fig13_helios_D24", tasks=20, procs=1, mem=4, time="24h",
                             cmd=["scripts/job_tqrb.py", "--m", str(m), "--p", "0.001", "--noise", "helios",
                                  "--adaptive", str(ad), "--seqs", "100", "--shots-per-seq", "100",
                                  "--seed", "98", "--K", "512", "--Delta", "24", "--K2", "4096", "--Delta2", "32",
                                  "--out", f"results/final/tqrb_fig13_helios/helios_p0.001_D24/a{ad}/m{m}/task{{task}}.json"]))
    C["tqrb_fig13_helios_D24"] = jobs
    # the adaptive gain's dependence on Helios's unpublished memory error per unit time:
    # the same sequences under the Helios-like model with the memory error doubled
    jobs = []
    for ad in (1, 0):
        for m in (3, 5, 7):
            jobs.append(dict(campaign="tqrb_fig13_helios_mem2", tasks=20, procs=1, mem=3, time="24h",
                             cmd=["scripts/job_tqrb.py", "--m", str(m), "--p", "0.001", "--noise", "helios_mem2",
                                  "--adaptive", str(ad), "--seqs", "100", "--shots-per-seq", "100", "--seed", "98",
                                  "--out", f"results/final/tqrb_fig13_helios/helios_mem2_p0.001/a{ad}/m{m}/task{{task}}.json"]))
    C["tqrb_fig13_helios_mem2"] = jobs
    # decoder sensitivity of the forced-gap comparison: the same shots (same seeds, tasks
    # 0-9 at p = 1.1e-3) decoded with an 8x wider beam
    C["tqrb_fig12_beam"] = [dict(campaign="tqrb_fig12_beam", tasks=10, procs=1, mem=8, time="24h",
                                 cmd=["scripts/job_tqrb.py", "--m", "7", "--p", "0.0011", "--noise", "uniform",
                                      "--adaptive", "0", "--seqs", "100", "--shots-per-seq", "100", "--gap-hist", "1",
                                      "--seed", "96", "--K", "4096", "--Delta", "24", "--K2", "16384", "--Delta2", "32",
                                      "--out", "results/final/tqrb_fig12_beam/uniform_p0.0011/a0/m7/task{task}.json"]),
                                 dict(campaign="tqrb_fig12_beam", tasks=10, procs=1, mem=3, time="24h",
                                 cmd=["scripts/job_tqrb.py", "--m", "7", "--p", "0.0011", "--noise", "uniform",
                                      "--adaptive", "0", "--seqs", "100", "--shots-per-seq", "100", "--gap-hist", "1",
                                      "--seed", "96", "--K", "64", "--Delta", "12", "--K2", "512", "--Delta2", "24",
                                      "--out", "results/final/tqrb_fig12_beam/uniform_p0.0011_K64/a0/m7/task{task}.json"]),
                                 dict(campaign="tqrb_fig12_beam", tasks=10, procs=1, mem=4, time="24h",
                                 cmd=["scripts/job_tqrb.py", "--m", "7", "--p", "0.0011", "--noise", "uniform",
                                      "--adaptive", "0", "--seqs", "100", "--shots-per-seq", "100", "--gap-hist", "1",
                                      "--seed", "96", "--K", "512", "--Delta", "24", "--K2", "4096", "--Delta2", "32",
                                      "--out", "results/final/tqrb_fig12_beam/uniform_p0.0011_D24/a0/m7/task{task}.json"])]
    # ---------------------------------------------------------------- Carbon-Helix
    # main comparison: 4 rounds after the preparation round; X basis (the weaker one under
    # dephasing-dominated noise) with two odds scales, Z basis with one
    # single-process tasks (they fit into single free cores anywhere); seeds 1xx
    main_rules = {"full": "full", "C": "l2=on_event,l3=any", "B4": "l2=on_event,l3=ambiguous,T3=4",
                  "R4": "l2=on_event,l3=ambiguous,rot=4", "R8": "l2=on_event,l3=ambiguous,rot=8",
                  "R8F": "l2=on_event,l3=ambiguous,rot=8,l2fast=1"}
    jobs = []
    for tag, rule in main_rules.items():
        # the Z basis (fewer failures under dephasing-dominated noise) only as a check of the
        # basis asymmetry, for non-adaptive extraction
        zs = (("Z", 2.5, 12, 2000, 123),) if tag == "full" else ()
        # high-odds sets extended from 16 to 32 tasks once cx3 was available ("base_tasks"
        # marks the first stage; the analysis uses all tasks)
        for b, kappa, tasks, shots, seed in (("X", 2.0, 24, 2000, 121), ("X", 3.0, 32, 500, 122)) + zs:
            jobs.append(strat("ch_main", tag, "carbonhelix", rule, "uniform", 1e-3, kappa, b, 4,
                              shots=shots, tasks=tasks, procs=1, seed=seed))
            if kappa == 3.0:
                jobs[-1]["base_tasks"] = 16
        zs = (("Z", 2.0, 12, 2000, 126),) if tag == "full" else ()
        for b, kappa, tasks, shots, seed in (("X", 1.5, 24, 2000, 124), ("X", 2.2, 32, 500, 125)) + zs:
            jobs.append(strat("ch_helios", tag, "carbonhelix", rule, "helios", 7.9e-4, kappa, b, 4,
                              shots=shots, tasks=tasks, procs=1, seed=seed))
            if kappa == 2.2:
                jobs[-1]["base_tasks"] = 16
    C["carbon"] = jobs
    # unpublished Helios quantities: memory error per unit time (x0.5, x2) and the
    # measurement/reset duration (4 ms instead of 2 ms per batch)
    jobs = []
    for noise in ("helios_mem0.5", "helios_mem2", "helios_tm4"):
        for tag, rule in (("full", "full"), ("R4", "l2=on_event,l3=ambiguous,rot=4")):
            jobs.append(strat("ch_helios_sens", f"{noise}/{tag}", "carbonhelix", rule, noise, 7.9e-4, 1.8,
                              "X", 4, shots=1000, tasks=16, procs=1, seed=151))
            if noise == "helios_mem0.5":
                # with half the memory error the strata at kappa 1.8 hold almost no failures:
                # a second set of strata at more faults per shot
                jobs.append(strat("ch_helios_sens", f"{noise}/{tag}", "carbonhelix", rule, noise, 7.9e-4, 3.0,
                                  "X", 4, shots=500, tasks=24, procs=1, seed=153))
        for tag, rule in (("full", "full"), ("ase", "mode=ase")):
            j = strat("c4_helios_sens", f"{noise}/{tag}", "c4helix", rule, noise, 7.9e-4, 1.5, "X", 4,
                      shots=40000, tasks=2, procs=1, seed=152)
            j["cmd"] += ["--K", "512", "--Delta", "12", "--gap-stages", "1"]
            jobs.append(j)
    C["helios_sens"] = jobs
    # direct sampling of Carbon-Helix at a high p: an independent check of the stratified
    # estimate and of the extrapolation model (a held-out point)
    jobs = []
    for p, tasks in ((2e-3, 12), (3e-3, 6)):
        jobs.append(dict(campaign="ch_direct", tasks=tasks, procs=1, mem=2,
                         cmd=["scripts/job_protocol.py", "--proto", "carbonhelix", "--rule", "full",
                              "--noise", "uniform", "--p", str(p), "--basis", "X", "--rounds", "4",
                              "--shots", "2000", "--K", "512", "--Delta", "12", "--Delta2", "24",
                              "--seed", "161", "--out", f"results/final/ch_direct/p{p}/X/task{{task}}.json"]))
    C["ch_direct"] = jobs
    # frontier push on time per round: one outer check every other round (R16F) against R8F,
    # both over 16 rounds after preparation (every outer check measured at least once, the
    # rotation's windows not cut by the end of the experiment), Helios model
    jobs = []
    for tag, rule in (("R8F", "l2=on_event,l3=ambiguous,rot=8,l2fast=1"),
                      ("R16F", "l2=on_event,l3=ambiguous,rot=16,l2fast=1")):
        # 16 rounds: about 25 s per shot and 3.4 GB; 75 shots per task keep each task within
        # a few hours on the lab machines (64 x 75 = 4800 shots per rule)
        jobs.append(strat("ch_helios_long", tag, "carbonhelix", rule, "helios", 7.9e-4, 3.5, "X", 16,
                          shots=75, tasks=64, procs=1, seed=171))
        jobs[-1].update(time="12h", mem=4)
    C["long"] = jobs
    # the next push on time per round after R16F's success: one outer check every fourth
    # round (R32F, each outer check once per 32 rounds) against R16F, both over 32 rounds
    # (every outer check measured once at least). At odds scale 3.5 a 32-round shot holds
    # ~250 faults and often needs every re-decoding stage (about an hour for the first shot
    # of a task, model build included), and a quarter or more of the shots fail: 50 tasks x
    # 16 shots per rule give hundreds of failures
    jobs = []
    for tag, rule in (("R16F", "l2=on_event,l3=ambiguous,rot=16,l2fast=1"),
                      ("R32F", "l2=on_event,l3=ambiguous,rot=32,l2fast=1")):
        jobs.append(strat("ch_helios_long32", tag, "carbonhelix", rule, "helios", 7.9e-4, 3.5, "X", 32,
                          shots=16, tasks=50, procs=1, seed=181))
        # about 7100 detectors: the 128-limb build of the decoder (scripts/job_strat_wide.py)
        jobs[-1]["cmd"][0] = "scripts/job_strat_wide.py"
        jobs[-1].update(time="24h", mem=8)
    C["long32"] = jobs
    # push 7 after R32F's success, on its furthest criterion (time per round): a block whose
    # inner events are of one type measures only its weight-6 checks of that type (73 instead
    # of 100 ms; src/hase/variants.py, l2type=1), over the same 32 rounds and strata
    jobs = []
    jobs.append(strat("ch_helios_long32", "R32FT", "carbonhelix", "l2=on_event,l3=ambiguous,rot=32,l2fast=1,l2type=1",
                      "helios", 7.9e-4, 3.5, "X", 32, shots=16, tasks=50, procs=1, seed=181))
    # about 9000 detectors in the superset model (three alternative weight-6 parts per block
    # and round): the 192-limb build (scripts/job_strat_xwide.py)
    jobs[-1]["cmd"][0] = "scripts/job_strat_xwide.py"
    jobs[-1].update(time="24h", mem=10)
    C["push7"] = jobs
    # interval study: 8 rounds, forced full measurement every T rounds. Odds scale 3.3 gives
    # the same fault density per round as 3.0 at 4 rounds (2.0 gives too few failures at 8
    # rounds); the 2.0 tasks run first are kept as additional strata.
    jobs = []
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
            jobs.append(strat("ch_interval", tag, "carbonhelix", rule, "uniform", 1e-3, kappa, "X", 8,
                              shots=shots, tasks=tasks, procs=1, seed=seed))
    C["interval"] = jobs
    # windows that always reach the forced interval: no escalation, the outer checks only
    # every T rounds (weight-6 checks on events as in rule B). Under rule B escalations end
    # most windows early at the error rates where estimates are data-driven, so the forced
    # interval barely matters there; without them the window penalty is fully exposed and
    # its p-dependence measurable (same strata, rounds and decoder as the interval campaign)
    jobs = []
    for T in (2, 4, 8):
        for kappa, tasks, shots, seed in ((2.0, 16, 250, 134), (3.3, 24, 500, 135)):
            jobs.append(strat("ch_interval", f"P_T{T}", "carbonhelix", f"l2=on_event,l3=never,T2={T},T3={T}",
                              "uniform", 1e-3, kappa, "X", 8, shots=shots, tasks=tasks, procs=1, seed=seed))
    C["interval_periodic"] = jobs
    # frontier push 8 (time per round, the criterion furthest from its limit for the periodic
    # rule, non-dominated at uniform p = 1e-3): outer checks every 16 instead of 8 rounds (the
    # periodic share of the time halves); 16 rounds, the same strata totals as the 8-round runs
    jobs = []
    for kappa, tasks, shots, seed in ((2.0, 32, 125, 136), (3.3, 48, 250, 137)):
        jobs.append(strat("ch_interval", "P_T16", "carbonhelix", "l2=on_event,l3=never,T2=16,T3=16",
                          "uniform", 1e-3, kappa, "X", 16, shots=shots, tasks=tasks, procs=1, seed=seed))
    C["push8"] = jobs
    # frontier push 9 (the error, the criterion furthest from its limit for both periodic
    # configurations after push 8): the weight-6 checks of a block measured with X and Z in
    # parallel when at most three blocks need them (as R8F for the rotations); outer checks
    # every 8 rounds, the interval campaign's strata and seeds' sizes
    jobs = []
    for kappa, tasks, shots, seed in ((2.0, 16, 250, 138), (3.3, 24, 500, 139)):
        jobs.append(strat("ch_periodic_fast", "P_T8F", "carbonhelix", "l2=on_event,l3=never,T2=8,T3=8,l2fast=1",
                          "uniform", 1e-3, kappa, "X", 8, shots=shots, tasks=tasks, procs=1, seed=seed))
    C["push9"] = jobs
    jobs = []
    # C4-Helix under the same configuration as the Carbon-Helix main comparison (4 rounds,
    # X basis, same noise models and decoder family)
    for noise, pdec, ks in (("uniform", 1e-3, (0.5, 1.0, 2.0)), ("helios", 7.9e-4, (0.5, 1.0, 1.5))):
        for tag, rule in (("full", "full"), ("ase", "mode=ase"), ("aseT4", "mode=ase,T=4")):
            for b in "X":
                for k in ks:
                    j = strat(f"c4_strat_{noise}", tag, "c4helix", rule, noise, pdec, k, b, 4,
                              shots=40000, tasks=4, procs=1, seed=109)
                    j["cmd"] += ["--K", "512", "--Delta", "12", "--gap-stages", "1"]
                    jobs.append(j)
    C["c4_strat"] = jobs
    # circuit fault distance (scripts/job_distance.py --method restricted: integer programming
    # over the mechanisms seen only by the final measurement and the flags, i.e. faults at the
    # end of the circuit, where data errors left by an outer-check gadget meet no later check)
    # at every place an outer-check part can end a circuit (scripts/distance_suite.py), and
    # the same for the gadget designs compared (results/distance/design__*.json)
    jobs = []
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from distance_suite import cases as dist_cases, method as dist_method
    for rule, basis, path, combo, out in dist_cases("k4cyc"):
        # a quiet last round admits four times as many mechanisms: hours and up to 16 GB
        quiet = dist_method(path) == "restricted_l1"
        jobs.append(dict(campaign="distance", tasks=1, procs=1, mem=16 if quiet else 3, time="2d" if quiet else "12h",
                         cmd=["scripts/job_distance.py", "--rule", rule, "--basis", basis, "--rounds", "1",
                              "--path", path, "--W", "10", "--combo", combo, "--method", dist_method(path),
                              "--time-limit", "150000" if quiet else "40000", "--out", out]))
    # the whole two-round circuit, unrestricted (integer programming for 11 hours: the proven
    # lower bound and the best logical error found are reported, converged or not)
    for basis, order in (("X", "XZ"), ("Z", "ZX")):
        jobs.append(dict(campaign="distance", tasks=1, procs=1, mem=16, time="14h",
                         cmd=["scripts/job_distance.py", "--rule", f"g3=k4cyc,g3order={order}", "--basis", basis,
                              "--rounds", "1", "--path", "full", "--W", "10", "--combo", "1", "--method", "ilp",
                              "--time-limit", "40000",
                              "--out", f"results/distance/whole__k4cyc__{order}__{basis}__c1.json"]))
    from distance_suite import design_cases
    for rule, basis, path, combo, out in design_cases():
        jobs.append(dict(campaign="distance", tasks=1, procs=1, mem=3, time="8h",
                         cmd=["scripts/job_distance.py", "--rule", rule, "--basis", basis, "--rounds", "1",
                              "--path", path, "--W", "10", "--combo", combo, "--method", "restricted",
                              "--time-limit", "20000", "--out", out]))
    C["distance"] = jobs
    # one Carbon block inside a window between full measurements (scripts/inner_window_distance.py):
    # exact circuit distance of its inner logicals, every placement of the weight-6 checks
    C["inner_window"] = [dict(campaign="inner_window", tasks=1, procs=1, mem=2, time="8h" if T == 8 else "2h",
                              cmd=["scripts/inner_window_distance.py", "--T", str(T), "--basis", basis,
                                   "--out", f"results/inner_window/T{T}_{basis}.json"])
                         for T in (2, 4, 8) for basis in "XZ"]
    C["inner_window"].append(dict(campaign="inner_window", tasks=1, procs=1, mem=2, time="2h",
                                  cmd=["scripts/inner_window_distance.py", "--structure",
                                       "--out", "results/inner_window/structure.json"]))
    # chain-map CNOT from the distance-5 surface code to C4-Helix (src/hase/chainmap.py): the
    # paper's App. Fig. 14 'Surface-C4-Helix' (6 CNOTs, Z basis, uniform noise) and the
    # circuit distance of one chain-map CNOT (14 logical classes, exact integer programming)
    jobs = [dict(campaign="chainmap", tasks=14, procs=1, mem=4, time="12h",
                 cmd=["scripts/chainmap_search.py", "distance", "--out", "results/chainmap/distance_{task}.json"])]
    for p, tasks, shots in ((7e-4, 60, 30000), (9e-4, 40, 25000), (1.1e-3, 30, 20000)):
        jobs.append(dict(campaign="chainmap", tasks=tasks, procs=1, mem=2, time="4h",
                         cmd=["scripts/job_chainmap.py", "--n-cnot", "6", "--p", str(p), "--basis", "Z",
                              "--shots", str(shots), "--seed", "41",
                              "--out", f"results/final/chainmap_fig14/uniform_p{p:g}/Z/task{{task}}.json"]))
    # the X basis as well (the paper's transversal series is the mean of the two bases)
    for p, tasks, shots in ((7e-4, 30, 20000), (9e-4, 20, 15000), (1.1e-3, 20, 10000)):
        jobs.append(dict(campaign="chainmap", tasks=tasks, procs=1, mem=2, time="4h",
                         cmd=["scripts/job_chainmap.py", "--n-cnot", "6", "--p", str(p), "--basis", "X",
                              "--shots", str(shots), "--seed", "42",
                              "--out", f"results/final/chainmap_fig14/uniform_p{p:g}/X/task{{task}}.json"]))
    # the inter-code GHZ experiment (paper Sec. 4) under the Helios model, measured in Z and in X
    for meas in "ZX":
        jobs.append(dict(campaign="chainmap", tasks=20, procs=1, mem=2, time="4h",
                         cmd=["scripts/job_chainmap.py", "--ghz", meas, "--noise", "helios", "--p", "0.00079",
                              "--shots", "250000", "--batch", "2000", "--seed", "43",
                              "--out", f"results/final/chainmap_ghz/helios_p0.00079/{meas}/task{{task}}.json"]))
    # the same with Helios's memory error per unit time doubled (the unpublished input)
    for meas in "ZX":
        jobs.append(dict(campaign="chainmap", tasks=10, procs=1, mem=2, time="4h",
                         cmd=["scripts/job_chainmap.py", "--ghz", meas, "--noise", "helios_mem2", "--p", "0.00079",
                              "--shots", "250000", "--batch", "2000", "--seed", "44",
                              "--out", f"results/final/chainmap_ghz/helios_mem2_p0.00079/{meas}/task{{task}}.json"]))
    # the X basis without idle noise (what the surface code's X-basis failures come from)
    jobs.append(dict(campaign="chainmap", tasks=10, procs=1, mem=2, time="4h",
                     cmd=["scripts/job_chainmap.py", "--n-cnot", "6", "--p", "0.0011", "--basis", "X", "--idle-off",
                          "--shots", "10000", "--seed", "49",
                          "--out", "results/final/chainmap_fig14_noidle/uniform_p0.0011/X/task{task}.json"]))
    # other depth-2 chain maps (scripts/chainmap_search.py alternatives): Fig. 14's top point, Z basis
    for i in (1, 2, 3):
        jobs.append(dict(campaign="chainmap", tasks=10, procs=1, mem=2, time="4h",
                         cmd=["scripts/job_chainmap.py", "--n-cnot", "6", "--p", "0.0011", "--basis", "Z",
                              "--shots", "20000", "--seed", str(45 + i), "--chain-map", f"results/chainmap/F_alt{i}.json",
                              "--out", f"results/final/chainmap_fig14_alt{i}/uniform_p0.0011/Z/task{{task}}.json"]))
    C["chainmap"] = jobs
    # the whole circuit's fault distance by decomposition (scripts/relaxed_distance.py): every
    # subset of active outer-check instances of each path kind, shared out over the tasks
    jobs = []
    for basis, order, kind, tasks in (("Z", "ZX", "full1", 1), ("Z", "XZ", "full1", 1), ("X", "XZ", "full1", 1),
                                      ("X", "XZ", "P8x1", 1), ("X", "XZ", "P4x2", 1), ("X", "XZ", "P2x4", 6),
                                      ("X", "XZ", "P16x1", 1),
                                      ("X", "XZ", "full8", 32), ("Z", "XZ", "full8", 32), ("X", "XZ", "any8", 48),
                                      ("X", "XZ", "any4", 8), ("Z", "XZ", "any4", 8)):
        jobs.append(dict(campaign="relaxed_distance", tasks=tasks, procs=1, mem=4, time="8h",
                         cmd=["scripts/relaxed_distance.py", "--basis", basis, "--order", order, "--kind", kind,
                              "--ntasks", str(tasks),
                              "--out", f"results/relaxed_distance/{basis}_{order}_{kind}/task{{task}}.json"]))
    C["relaxed_distance"] = jobs
    return C


def hours(t):
    """'8h', '1d', '2d', '14h' -> hours."""
    return int(t[:-1]) * (24 if t.endswith("d") else 1)


def xpu_lines(job, cluster=None):
    """xpu command for a job: its N tasks in one submission (xpu decides where they run; the
    `cluster` argument of the older tooling is ignored). xpu numbers the tasks in XPU_TASK;
    the job scripts read their index as GPURUN_TASK, the name of the earlier tool."""
    import shlex
    inner = 'GPURUN_TASK="$XPU_TASK" python ' + " ".join(shlex.quote(x) for x in job["cmd"])
    return [["xpu", "--cpus", str(max(job["procs"], 1)), "--mem", str(job["mem"]),
             "--time", str(hours(job.get("time", "8h"))), "--tasks", str(job["tasks"]), "bash", "-c", inner]]


gpurun_lines = xpu_lines        # the name used by the older queue tooling (scripts/build_queues.py)


REGISTRY = os.path.join(ROOT, "logs", "submitted.json")


def _registry():
    import json
    try:
        return json.load(open(REGISTRY))
    except (OSError, ValueError):
        return {}


def in_flight(out):
    """True when the xpu job last submitted for this output pattern is still queued,
    running or held (its results are not back yet)."""
    jid = _registry().get(out)
    if not jid:
        return False
    r = subprocess.run(["xpu", "jobs", jid], cwd=ROOT, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    words = (r.stdout.strip().split() + ["", ""])
    return words[0] == jid and words[1] in ("queued", "running", "held")


def submit(job, cluster=None):
    import json
    outs = []
    for cmd in xpu_lines(job):
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, stdin=subprocess.DEVNULL)
        o = (r.stdout + r.stderr).strip()
        outs.append(o.splitlines()[-1] if o else "")
    jid = outs[-1].split()[0] if outs and outs[-1] else ""
    if jid:
        reg = _registry()
        reg[job["cmd"][job["cmd"].index("--out") + 1]] = jid
        os.makedirs(os.path.dirname(REGISTRY), exist_ok=True)
        json.dump(reg, open(REGISTRY, "w"), indent=1)
    return "; ".join(outs)


def main():
    C = campaigns()
    if len(sys.argv) < 2 or sys.argv[1] == "list":
        for k, v in C.items():
            print(f"{k:12s} {len(v):3d} jobs, {sum(j['tasks'] for j in v):4d} tasks")
        return
    if sys.argv[1] == "status":
        for k, v in C.items():
            camps = {j["campaign"] for j in v}
            n = sum(len(glob.glob(os.path.join(ROOT, f"results/final/{c}/**/*.json"), recursive=True)) for c in camps)
            print(f"{k:12s} result files: {n}")
        return
    if sys.argv[1] == "audits":
        # post-processing of every Carbon-Helix configuration (scripts/submit_audits.py):
        # decoder audit and minimal failing sets, here run locally one after the other
        subprocess.run([sys.executable, "scripts/submit_audits.py", "local", "--only-missing"], cwd=ROOT, check=True)
        return
    if sys.argv[1] == "complete":
        # exit status 0 when every task of every campaign has its result file
        missing = 0
        for nm, jobs in C.items():
            for job in jobs:
                out = job["cmd"][job["cmd"].index("--out") + 1]
                for t in range(job["tasks"]):
                    if not os.path.exists(os.path.join(ROOT, out.replace("{task}", str(t)))):
                        missing += 1
        print(f"{missing} result files missing")
        sys.exit(1 if missing else 0)
    if sys.argv[1] == "queue":
        # write gpurun lines (missing results only) to a queue file for submit_queue.py
        which, cluster, qfile = sys.argv[2], sys.argv[3], sys.argv[4]
        names = list(C) if which == "all" else which.split(",")
        import shlex
        n = 0
        # interleave tasks across jobs so that every configuration progresses evenly
        per_job = [gpurun_lines(job, cluster) for nm in names for job in C[nm]]
        order = [lines[t] for t in range(max(len(x) for x in per_job)) for lines in per_job if t < len(lines)]
        with open(qfile, "a") as fh:
            for cmd in order:
                out = cmd[cmd.index("--out") + 1]
                t = int(cmd[cmd.index("--task-offset") + 1]) if "--task-offset" in cmd else None
                if t is not None and os.path.exists(os.path.join(ROOT, out.replace("{task}", str(t)))):
                    continue
                fh.write(" ".join(shlex.quote(x) for x in cmd) + "\n")
                n += 1
        print(f"queued {n} submissions to {qfile}")
        return
    if sys.argv[1] == "submit":
        # submit <campaign|all> [--only-missing] [--match TEXT]: --only-missing skips jobs whose
        # result files all exist or whose last submission (logs/submitted.json) is still in
        # flight (a job with some results missing is submitted whole: its tasks find their own
        # results and stop); --match keeps jobs whose output path contains TEXT
        which = sys.argv[2]
        cluster = "condor"
        if "--cluster" in sys.argv:
            cluster = sys.argv[sys.argv.index("--cluster") + 1]
        match = sys.argv[sys.argv.index("--match") + 1] if "--match" in sys.argv else None
        names = list(C) if which == "all" else [which]
        for nm in names:
            for job in C[nm]:
                out = job["cmd"][job["cmd"].index("--out") + 1]
                if match and match not in out:
                    continue
                if "--only-missing" in sys.argv and (all(
                        os.path.exists(os.path.join(ROOT, out.replace("{task}", str(t)))) for t in range(job["tasks"]))
                        or in_flight(out)):
                    continue
                print(nm, out, submit(job, cluster), flush=True)


if __name__ == "__main__":
    main()
