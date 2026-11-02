"""Check code structure and full-extraction detector models; write results/checks.json.

Checks code parameters and distances of C4, Carbon, C4-Helix and Carbon-Helix,
engine/Stim detector signatures and probabilities, and small extraction-circuit distances.
Usage: python scripts/structure_checks.py [--quick] [--out results/checks.json]
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from hase import codes  # noqa: E402
from hase.version import versions  # noqa: E402


def code_checks():
    out = {}
    out["C4"] = dict(n=4, k=2, d=list(codes.css_distance_bruteforce(codes.c4_code())))
    cb = codes.carbon_code()
    out["Carbon"] = dict(n=12, k=2, d=list(codes.css_distance_bruteforce(cb)))
    ch = codes.c4_helix()
    pt = codes.c4_helix_paper_table()
    from hase.gf2 import rank
    same = bool(rank(np.vstack([ch.hx, pt.hx])) == rank(ch.hx) == rank(pt.hx))
    out["C4-Helix"] = dict(n=ch.n, k=ch.k, d=list(codes.css_distance_bruteforce(ch)),
                           equals_paper_table=same)
    dx, dz, det = codes.concatenated_distance(codes.twisted_toric_10(), cb)
    chh = codes.carbon_helix()
    out["Carbon-Helix"] = dict(n=chh.n, k=chh.k, d=[dx, dz],
                               outer_check_weights=[int(r.sum()) for r, l in zip(chh.hx, chh.hx_level) if l == "outer"],
                               inner_check_weights=sorted({int(r.sum()) for r, l in zip(chh.hx, chh.hx_level) if l == "inner"}))
    return out


def dem_equality_checks():
    from hase.protocols import make_protocol
    from hase.engine import PathDEM, emit_stim
    from hase.distance import dem_to_matrices
    out = {}
    for name, rounds in (("c4helix", 3), ("carbonhelix", 1)):
        for basis in "XZ":
            proto = make_protocol(name, 1e-3, basis, rounds, "full")
            path = [(r, pt) for r in range(proto.n_rounds) for pt, _ in proto.plan(r)]
            c, pd = emit_stim(proto, path)
            H2, L2, p2 = dem_to_matrices(c.detector_error_model(decompose_errors=False))

            def colset(H, L, P):
                H = H.tocsc(); L = L.tocsc(); o = {}
                for j in range(H.shape[1]):
                    k = (tuple(sorted(H.indices[H.indptr[j]:H.indptr[j + 1]].tolist())),
                         tuple(L.indices[L.indptr[j]:L.indptr[j + 1]].tolist()))
                    o[k] = o.get(k, 0) + P[j]
                return o
            a, b = colset(pd.H, pd.L, pd.priors), colset(H2, L2, p2)
            common = set(a) & set(b)
            out[f"{name}_{basis}"] = dict(engine_columns=len(a), stim_columns=len(b),
                                          common=len(common),
                                          max_rel_prob_diff=float(max(abs(a[k] - b[k]) / b[k] for k in common)))
    return out


def circuit_distance_checks(quick=False, sink=None):
    from hase.protocols import make_protocol
    from hase.engine import PathDEM, _merge_columns
    from hase.distance import min_weight_logical
    from hase.carbonhelix import CarbonBlockProtocol
    from hase.ir import NoiseModel
    out = {}

    def dist(H, L, tl):
        res = []
        for combo in ((0,), (1,), (0, 1)):
            w, x, r = min_weight_logical(H, L, combo, time_limit=tl)
            res.append(w)
        return min(res)
    cases = [("carbon_block", 3)]
    if not quick:
        # C4-Helix: the cat-qubit assignment used ('xpair') and the two other natural ones
        cases += [("c4helix", 1), ("c4helix_paired", 1), ("c4helix_shifted", 1)]
    for name, rounds in cases:
        for basis in "XZ":
            t0 = time.time()
            if name == "carbon_block":
                proto = CarbonBlockProtocol(rounds + 1, basis, NoiseModel.uniform(1e-3))
            elif name.startswith("c4helix_"):
                from hase.c4helix import C4HelixProtocol
                proto = C4HelixProtocol(rounds + 1, basis, NoiseModel.uniform(1e-3),
                                        ordering=name.split("_")[1])
            else:
                proto = make_protocol(name, 1e-3, basis, rounds, "full")
            path = [(r, pt) for r in range(proto.n_rounds) for pt, _ in proto.plan(r)]
            pd = PathDEM(proto, path)
            out[f"{name}_{basis}_{proto.n_rounds}rounds"] = dict(distance=dist(pd.H, pd.L, 3600),
                                                                  seconds=time.time() - t0)
            if sink:
                sink(f"{name}_{basis}_{proto.n_rounds}rounds", out[f"{name}_{basis}_{proto.n_rounds}rounds"])
    return out


def main():
    quick = "--quick" in sys.argv
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "results/checks.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    res = dict(quick=quick)

    def save():
        res["versions"] = versions()
        with open(out, "w") as f:
            json.dump(res, f, indent=1, default=str)
    # each section is written as soon as it is done (the circuit distances take hours)
    for name, fn in (("codes", code_checks), ("dem_equality", dem_equality_checks)):
        res[name] = fn()
        save()
    res["circuit_distance"] = {}
    circuit_distance_checks(quick, sink=lambda k, v: (res["circuit_distance"].__setitem__(k, v), save()))
    res["complete"] = True
    save()
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
