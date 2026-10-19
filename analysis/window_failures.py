"""Constructed failures that test the fault-counting picture of docs/interval_theory.md.

1. Six faults: Z errors at preparation on 6 of the 12 qubits of the weight-12 logical
   Zbar_0 have the same syndrome as the other 6 (tie for a most-likely-error decoder), so
   at least one of every complementary pair fails; 5 of the 12 are always corrected.
   -> the asymptotic exponent is at most 6 (and equals 6 if nothing smaller fails).
2. Window failures: two "silent" weight-3 errors (3 of the 4 qubits of the inner logical
   that Zbar_0 restricts to on a block; syndrome identical to a single-qubit error, so no
   escalation under the ambiguous-event rule) in two blocks, injected as idle Z faults at
   the end of the inner-check part of rounds rA and rB. Full extraction resolves each one
   in its own round, so they fail only when simultaneous; with outer checks measured only
   every T rounds they also fail when in the same window.
Writes results/minimal/constructed.json.
"""
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hase.protocols import make_protocol  # noqa: E402
from hase.runner import SupersetDecoder, fails_with  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RULES = {"full": "full", "B4": "l2=on_event,l3=ambiguous,T3=4", "B8": "l2=on_event,l3=ambiguous,T2=8,T3=8",
         "C": "l2=on_event,l3=any", "R4": "l2=on_event,l3=ambiguous,rot=4"}


def z_idle_terms(cp, n_data):
    """data qubit -> term index of an idle fault that leaves exactly Z on that qubit and
    flips no measurement of the part (i.e. it occurs after the part's measurements)."""
    out = {}
    for s, d in enumerate(cp.site_desc):
        if d[0] != "idle":
            continue
        for t in cp.site_terms[s]:
            if cp.term_meas[t].nnz:
                continue
            row = cp.term_out[t].toarray().ravel()
            xs, zs = np.flatnonzero(row[:n_data]), np.flatnonzero(row[n_data:])
            if len(xs) == 0 and len(zs) == 1:
                out.setdefault(int(zs[0]), int(t))
    return out


def main(rounds=8):
    res = {}
    p0 = make_protocol("carbonhelix", 1e-3, "X", rounds, "full")
    supp = [int(q) for q in np.flatnonzero(p0.code.lz[0])]
    blocks = [[q for q in supp if q in set(b)] for b in p0.code.blocks]
    blocks = [b for b in blocks if b]
    # (1) six-fault ties at preparation
    dec = SupersetDecoder(p0, K=512, Delta=12, committee=True, K2=4096, Delta2=24)
    rng = np.random.default_rng(1)
    pairs = []
    seen = set()
    while len(pairs) < 20:
        h = tuple(sorted(int(x) for x in rng.choice(supp, 6, replace=False)))
        if h in seen:
            continue
        seen.add(h)
        c = [q for q in supp if q not in h]
        pairs.append((fails_with(p0, dec, {"init": list(h)})[0], fails_with(p0, dec, {"init": c})[0]))
    five = [fails_with(p0, dec, {"init": [int(x) for x in rng.choice(supp, 5, replace=False)]})[0]
            for _ in range(20)]
    res["six_fault_ties"] = dict(pairs=len(pairs), at_least_one_fails=sum(a or b for a, b in pairs),
                                 both_fail=sum(a and b for a, b in pairs), five_fault_fails=sum(five),
                                 five_fault_trials=len(five))
    print(res["six_fault_ties"], flush=True)
    # (2) window failures
    A, B = blocks[0][:3], blocks[1][:3]
    combos = [(1, 1), (2, 2), (1, 2), (1, 3), (2, 3), (5, 7), (2, 6), (4, 5)]
    res["window"] = {}
    for tag, rule in RULES.items():
        proto = make_protocol("carbonhelix", 1e-3, "X", rounds, rule)
        d = SupersetDecoder(proto, K=512, Delta=12, committee=True, K2=4096, Delta2=24)
        row = {}
        for rA, rB in combos:
            fl = {}
            for r, qs in ((rA, A), (rB, B)):
                pt = "L1_a" if r % 2 == 0 else "L1_b"
                terms = z_idle_terms(proto.parts[pt], proto.n_data)
                fl.setdefault((r, pt), []).extend(terms[q] for q in qs)
            f, path = fails_with(proto, d, fl)
            outer = [r for r, pt in path if pt in ("L3",) or pt.startswith("L3r")]
            row[f"{rA},{rB}"] = dict(fails=bool(f), outer_rounds=outer)
        res["window"][tag] = row
        print(tag, {k: v["fails"] for k, v in row.items()}, flush=True)
    os.makedirs(os.path.join(ROOT, "results", "minimal"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "results", "minimal", "constructed.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    main()
