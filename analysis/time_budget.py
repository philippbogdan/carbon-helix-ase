"""Where the time per round goes (frontier: time per round is the criterion furthest from its
limit). Executed paths sampled directly at Helios's p (no decoding needed: the path depends
on the syndromes only), every part's Helios-model duration summed by kind: inner checks
(L1, the limit), weight-6 checks after inner events (both types, or one type with
l2type=1), full outer measurements after escalations (L3) and the rotating outer checks
(L3r); for the periodic rule (uniform p = 1e-3, where it is non-dominated) the periodic
weight-6 and outer measurements. Writes results/time_budget.json.

Usage: python analysis/time_budget.py [shots]
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hase.protocols import make_protocol  # noqa: E402
from hase.engine import simulate  # noqa: E402
from hase.timing import path_time_ms  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RULES = {"R32F": ("l2=on_event,l3=ambiguous,rot=32,l2fast=1", 32, "helios", 7.9e-4),
         "R32FT": ("l2=on_event,l3=ambiguous,rot=32,l2fast=1,l2type=1", 32, "helios", 7.9e-4),
         "R16F": ("l2=on_event,l3=ambiguous,rot=16,l2fast=1", 32, "helios", 7.9e-4),
         # the periodic rule, non-dominated at uniform p = 1e-3, and its push 8
         "P8": ("l2=on_event,l3=never,T2=8,T3=8", 8, "uniform", 1e-3),
         "P16": ("l2=on_event,l3=never,T2=16,T3=16", 16, "uniform", 1e-3),
         "P8F": ("l2=on_event,l3=never,T2=8,T3=8,l2fast=1", 8, "uniform", 1e-3)}


def kind(pt, periodic=False):
    if pt.startswith("L1"):
        return "inner checks (L1)"
    if pt.startswith("L2fX") or pt.startswith("L2fZ"):
        return "weight-6 checks, one type"
    if periodic and pt == "L2_all":
        return "weight-6 checks, periodic (all blocks)"
    if pt.startswith("L2"):
        return "weight-6 checks, both types"
    if pt.startswith("L3r"):
        return "rotating outer checks"
    if pt == "L3":
        return "full outer measurement (periodic)" if periodic else "full outer measurement (escalation)"
    return "waits"


def main(shots=400):
    out = {}
    for tag, (rule, rounds, noise, p) in RULES.items():
        periodic = "l3=never" in rule
        proto = make_protocol("carbonhelix", p, "X", rounds, rule, noise)
        st = simulate(proto, shots, np.random.default_rng(1))
        tot, cnt = defaultdict(float), defaultdict(float)
        for path in st.paths:
            for r, pt in path:
                if r == 0:
                    continue
                tot[kind(pt, periodic)] += path_time_ms(proto, [(r, pt)])
                cnt[kind(pt, periodic)] += 1
        n = shots * rounds
        out[tag] = dict(rule=rule, rounds=rounds, shots=shots, noise=noise, p=p,
                        ms_per_round={k: v / n for k, v in sorted(tot.items())},
                        instances_per_round={k: v / n for k, v in sorted(cnt.items())},
                        total_ms_per_round=sum(tot.values()) / n)
        print(tag, round(out[tag]["total_ms_per_round"], 1),
              {k: round(v, 1) for k, v in out[tag]["ms_per_round"].items()}, flush=True)
    json.dump(out, open(os.path.join(ROOT, "results", "time_budget.json"), "w"), indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 400)
