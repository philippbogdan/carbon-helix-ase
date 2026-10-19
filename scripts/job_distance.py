"""Cluster job: is there an undetectable logical error of at most W fault mechanisms on one
executed path of a Carbon-Helix protocol? (SAT: CryptoMiniSat with native XOR constraints
for the detectors and the observable, a cardinality bound on the mechanisms;
src/hase/distance.py:sat_has_logical_within.) 'unsat' proves the path's circuit fault
distance exceeds W; with W = 10 a most-likely-error decoder then needs at least six faults
to fail on that path.

Paths: 'full' (every part every round: non-adaptive extraction) or 'quiet' (the path of a
shot without faults: the parts an adaptive rule executes when nothing happens - the fewest
detectors, where silent errors would hide).

Usage: python scripts/job_distance.py --rule R --basis B --rounds N --path full|quiet --W W
       --combo 0|1|01 --time-limit S --method sat|ilp|restricted --out results/distance/<name>.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

from hase.version import commit, provenance  # noqa: E402
from hase.protocols import make_protocol  # noqa: E402
from hase.engine import PathDEM, simulate_with_faults  # noqa: E402
from hase.distance import sat_has_logical_within  # noqa: E402


def quiet_path(proto):
    n = proto.n_data
    faults = {"init": np.zeros((1, n), dtype=bool), "final": np.zeros((1, n), dtype=bool)}
    st = simulate_with_faults(proto, faults, 1)
    return list(st.paths[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", default="full")
    ap.add_argument("--basis", default="X")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--path", default="full")
    ap.add_argument("--W", type=int, required=True)
    ap.add_argument("--combo", default="0")
    ap.add_argument("--time-limit", type=float, default=72000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--task-offset", type=int, default=0, help="ignored (queue tooling)")
    ap.add_argument("--method", default="sat", help="sat (decide W), ilp (minimise, HiGHS, proven bound) or "
                    "restricted (ilp over the end-of-circuit mechanisms: an upper bound)")
    a = ap.parse_args()
    a.out = a.out.replace("{task}", str(int(os.environ.get("GPURUN_TASK", "0")) + a.task_offset))
    if os.path.exists(a.out) and not os.environ.get("HASE_OVERWRITE"):
        print("result exists:", a.out)
        return
    t0 = time.time()
    proto = make_protocol("carbonhelix", 1e-3, a.basis, a.rounds, a.rule, "uniform")
    if a.path == "full":
        path = [(r, pt) for r in range(proto.n_rounds) for pt, _ in proto.plan(r)]
    elif a.path == "quiet":
        path = quiet_path(proto)
    else:
        # explicit path "0:L1_a+L2_all+L3/1:L1_b+L3r_1" (round:parts in order)
        path = [(int(rr), pt) for seg in a.path.split("/") for rr, pts in [seg.split(":")]
                for pt in pts.split("+")]
    pd = PathDEM(proto, path)
    combo = tuple(int(c) for c in a.combo)
    extra = {}
    if a.method in ("restricted", "restricted_l1", "restricted_last"):
        # integer programming over the mechanisms whose detectors are all final-data or flag
        # detectors (faults at the end of the circuit, seen only by the perfect final
        # measurement and the flags): its optimum is an upper bound on the path's distance,
        # found in minutes where the unrestricted problem does not converge in hours
        from hase.distance import min_weight_logical
        H, L = pd.H.tocsc(), pd.L.tocsc()
        allowed = {i for i, dd in enumerate(pd.dets) if dd[0] in ("final", "flag")}
        if a.method == "restricted_last":
            # every check's last comparison: all faults of the last round (measurement errors
            # included) and those left before it
            last = {}
            for i, dd in enumerate(pd.dets):
                if dd[0] == "chk":
                    last[dd[1]] = max(last.get(dd[1], (-1, None)), (dd[2], i))
            allowed |= {i for _, i in last.values()}
        if a.method == "restricted_l1":
            # also the last comparison of every level-1 (C4) check: faults of the last round's
            # level-1 part and errors left before it, e.g. by an outer check of the round
            # before (a quiet last round, as in adaptive extraction)
            last = {}
            for i, dd in enumerate(pd.dets):
                if dd[0] == "chk" and dd[1][0] == "c4":
                    last[dd[1]] = max(last.get(dd[1], (-1, None)), (dd[2], i))
            allowed |= {i for _, i in last.values()}
        ok = [j for j in range(H.shape[1])
              if all(d in allowed for d in H.indices[H.indptr[j]:H.indptr[j + 1]])]
        ok = np.array(ok, dtype=np.int64)
        w, xs, r = min_weight_logical(H[:, ok].tocsr(), L[:, ok].tocsr(), combo, time_limit=a.time_limit)
        x = None
        if xs is not None:
            x = np.zeros(H.shape[1], dtype=np.uint8)
            x[ok[np.flatnonzero(xs)]] = 1
            assert not ((pd.H.astype(np.int64) @ x) % 2).any()
            lo = (pd.L.astype(np.int64) @ x) % 2
            assert sum(int(lo[o]) for o in combo) % 2 == 1
        bound = getattr(r, "mip_dual_bound", None)
        bound = float(bound) if bound is not None else None
        res = True if (w is not None and w <= a.W) else (False if r.status == 0 else None)
        extra = dict(ilp_best=w, ilp_lower_bound=bound, ilp_status=int(r.status), ilp_message=str(r.message),
                     restricted_mechanisms=int(len(ok)))
    elif a.method == "ilp":
        # minimum weight by integer programming; the solver's proven lower bound counts even
        # when the time limit stops it: 'unsat' for W when that bound exceeds W
        from hase.distance import min_weight_logical
        w, x, r = min_weight_logical(pd.H, pd.L, combo, time_limit=a.time_limit)
        bound = getattr(r, "mip_dual_bound", None)
        bound = float(bound) if bound is not None else None
        res = (True if (w is not None and w <= a.W) else
               False if (bound is not None and bound > a.W + 1e-6) or (w is not None and w > a.W and r.status == 0)
               else None)
        extra = dict(ilp_best=w, ilp_lower_bound=bound, ilp_status=int(r.status), ilp_message=str(r.message))
    else:
        res, x = sat_has_logical_within(pd.H, pd.L, combo, a.W, time_limit=a.time_limit)
    out = dict(rule=a.rule, basis=a.basis, rounds=a.rounds, path_kind=a.path, path=[list(p) for p in path],
               W=a.W, combo=list(combo), method=a.method, detectors=int(pd.H.shape[0]), mechanisms=int(pd.H.shape[1]),
               result={True: "sat", False: "unsat", None: "undecided"}[res],
               weight=int(x.sum()) if x is not None else None,
               solution=[int(i) for i in np.flatnonzero(x)] if x is not None else None,
               seconds=time.time() - t0,
               time_limit=a.time_limit, commit=commit(), provenance=provenance(), **extra)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("rule", "basis", "path_kind", "W", "combo", "result", "weight", "seconds")}))


if __name__ == "__main__":
    main()
