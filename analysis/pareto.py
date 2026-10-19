"""Non-dominated set (logical error per round, two-qubit gates per round, Helios time per
round, peak qubits) over every configuration tried, at given noise points, and the
distance of each criterion from its limit:
  gates  -> level-1-only round (cheapest round that still checks every inner block),
  time   -> level-1-only round duration (Helios timing),
  qubits -> 60 data + 2 ancillas (fully serialised extraction),
  error  -> best error per round achieved by any configuration at that point.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from strat import load, StratModel  # noqa: E402


def collect(patterns):
    rows = []
    for pat in patterns:
        for key, d in load(pat).items():
            rows.append((key, StratModel(key, d)))
    return rows


def metrics(sm, p):
    e, s, lo, hi = sm.eps(p)
    g, cov = sm.twoq_per_round(p)
    t = sm.time_per_round(p)
    q = sm.peak_qubits()
    return dict(eps=e, eps_sigma=s, eps_upper=hi, twoq=g, time_ms=t, qubits=q,
                measured=bool(hi < 1.3 * max(e, 1e-300)))


def pareto(points, keys=("eps", "twoq", "time_ms", "qubits")):
    nd = []
    for i, a in enumerate(points):
        dominated = False
        for j, b in enumerate(points):
            if i == j:
                continue
            if all(b[k] is not None and a[k] is not None and b[k] <= a[k] for k in keys) and \
               any(b[k] < a[k] for k in keys):
                dominated = True
                break
        if not dominated:
            nd.append(a)
    return nd


def limits(proto_name):
    from hase.protocols import make_protocol
    from hase.timing import apart_duration
    pr = make_protocol(proto_name, 1e-3, "X", 2, "full")
    l1 = "L1_a" if proto_name == "carbonhelix" else "inner_a"
    return dict(twoq=pr.parts[l1].two_q, time_ms=apart_duration(pr.aparts[l1]),
                qubits=pr.n_data + 2)


if __name__ == "__main__":
    rows = collect(["results/raw/ch_strat/*/*/*/*.json", "results/raw/ch_strat_helios/*/*/*/*.json",
                    "results/raw/ch_interval/*/*/*/*.json"])
    for noise, ps in (("uniform", (2e-3, 1e-3, 5e-4)), ("helios", (7.9e-4,))):
        for p in ps:
            pts = []
            for key, sm in rows:
                if key[2] != noise or key[3] != "X":
                    continue
                m = metrics(sm, p)
                m["config"] = f"{key[0]}|{key[1]}|r{key[4]}"
                pts.append(m)
            if not pts:
                continue
            nd = pareto([x for x in pts if x["time_ms"] is not None])
            print(f"{noise} p={p:.2e}: {len(pts)} configurations, non-dominated:")
            for x in sorted(nd, key=lambda x: x["eps"]):
                print("   ", x["config"], f"eps={x['eps']:.2e} 2q={x['twoq']:.0f} t={x['time_ms']:.0f}ms q={x['qubits']}")
