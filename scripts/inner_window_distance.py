"""Circuit fault distance of one Carbon block inside a window between full measurements:
the inner logical errors that the adaptive rules can leave undetected until the next
full (outer) measurement.

A window of T rounds: round 0 measures the C4-level checks (L1) and the weight-6 Carbon
checks (L2), rounds 1 .. T-1 measure L1 only (the path of a block without events) or L1
and L2 in a chosen subset of rounds (the paths of blocks with events), round T measures
L1 and L2 again, and the data are measured at the end. Exact integer programming
(src/hase/distance.py:min_weight_logical) over the path's detector error model, every
nontrivial class of the block's two logical qubits. With every block-local inner logical
needing d_in faults, an outer logical (inner logicals on at least three blocks, outer
distance 3) left inside a window needs 3 d_in faults, so a most-likely-error decoder fails
on such a window only with at least 3 d_in / 2 faults.

The two structural facts this rests on are checked with --structure: (i) on executed paths
without outer checks every fault mechanism flips inner detectors of one block only, (ii)
no logical operator of Carbon-Helix is supported on two blocks or fewer.

Usage: python scripts/inner_window_distance.py --T 4 --basis X [--subsets all|none]
       --out results/inner_window/T4_X.json
       python scripts/inner_window_distance.py --structure --out results/inner_window/structure.json
"""
import argparse
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from hase.version import commit, provenance  # noqa: E402
from hase.ir import NoiseModel  # noqa: E402
from hase.carbonhelix import CarbonBlockProtocol  # noqa: E402
from hase.engine import PathDEM  # noqa: E402
from hase.distance import min_weight_logical  # noqa: E402


def window_path(T, l2_rounds):
    path = []
    for r in range(T + 1):
        path.append((r, "L1_a" if r % 2 == 0 else "L1_b"))
        if r in (0, T) or r in l2_rounds:
            path.append((r, "L2"))
    return path


def _nullspace(A):
    import numpy as np
    A = (np.array(A, dtype=np.uint8) % 2).copy()
    m, k = A.shape
    piv, r = [], 0
    for col in range(k):
        rows = [i for i in range(r, m) if A[i, col]]
        if not rows:
            continue
        A[[r, rows[0]]] = A[[rows[0], r]]
        for i in range(m):
            if i != r and A[i, col]:
                A[i] ^= A[r]
        piv.append(col)
        r += 1
        if r == m:
            break
    out = []
    for f in (j for j in range(k) if j not in piv):
        v = np.zeros(k, dtype=np.uint8)
        v[f] = 1
        for i, pc in enumerate(piv):
            if A[i, f]:
                v[pc] = 1
        out.append(v)
    return np.array(out, dtype=np.uint8).reshape(-1, k)


def structure():
    import numpy as np
    from hase.carbonhelix import carbon_helix
    from hase.protocols import make_protocol
    from hase.gf2 import rank
    c = carbon_helix()
    res = {}
    # (ii) logical operators supported on at most two blocks
    for typ, chk, stab in (("X", c.hz, c.hx), ("Z", c.hx, c.hz)):
        found = []
        r0 = rank(stab)
        for nb in (1, 2):
            for bs in itertools.combinations(range(len(c.blocks)), nb):
                q = sorted(sum((list(c.blocks[b]) for b in bs), []))
                N = _nullspace(chk[:, q])
                full = np.zeros((len(N), c.n), dtype=np.uint8)
                full[:, q] = N
                if any(rank(np.vstack([stab, v])) > r0 for v in full):
                    found.append(list(bs))
        res[f"{typ}_logicals_on_at_most_two_blocks"] = found
    # (i) block-locality of the mechanisms of a path with level-1, level-2 and waiting parts
    proto = make_protocol("carbonhelix", 1e-3, "X", 3, "l2=on_event,l3=ambiguous")
    path = [(0, "L1_a"), (0, "L2_all"), (1, "L1_b"), (2, "L1_a"), (2, "L2_1"), (2, "W2_0"), (2, "W2_2"),
            (3, "L1_b"), (3, "L2_all")]
    pd = PathDEM(proto, path)
    blk = []
    for d in pd.dets:
        if d[0] in ("chk", "final") and d[1][0] in ("c4", "c6"):
            blk.append(d[1][1])
        else:
            blk.append(-1 if d[0] != "flag" else -2)
    blk = np.array(blk)
    H = pd.H.tocsc()
    multi, flag_blocks = 0, {}
    for j in range(H.shape[1]):
        rows = H.indices[H.indptr[j]:H.indptr[j + 1]]
        bs = {int(blk[r]) for r in rows if blk[r] >= 0}
        multi += len(bs) > 1
        for r in rows:
            if blk[r] == -2:
                flag_blocks.setdefault(int(r), set()).update(bs)
    res["path"] = [list(x) for x in path]
    res["mechanisms"] = int(H.shape[1])
    res["mechanisms_on_inner_detectors_of_several_blocks"] = int(multi)
    res["flags_shared_between_blocks"] = int(sum(len(v) > 1 for v in flag_blocks.values()))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--structure", action="store_true")
    ap.add_argument("--T", type=int)
    ap.add_argument("--basis", default="X")
    ap.add_argument("--subsets", default="all", help="all: every subset of rounds 1..T-1 also runs L2; "
                    "none: the quiet path only")
    ap.add_argument("--time-limit", type=float, default=3600)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        print("result exists:", a.out)
        return
    if a.structure:
        res = structure()
        res["commit"], res["provenance"] = commit(), provenance()
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
        print({k: v for k, v in res.items() if k not in ("provenance", "path")})
        return
    proto = CarbonBlockProtocol(a.T + 1, a.basis, NoiseModel.uniform(1e-3))
    inner = list(range(1, a.T))
    subsets = [()] if a.subsets == "none" else [s for k in range(len(inner) + 1)
                                                 for s in itertools.combinations(inner, k)]
    res = dict(T=a.T, basis=a.basis, block="Carbon [[12,2,4]] block, L1 every round, L2 at rounds 0, T and "
               "the listed rounds", paths={})
    for s in subsets:
        t0 = time.time()
        pd = PathDEM(proto, window_path(a.T, set(s)))
        cls = {}
        for combo in ((0,), (1,), (0, 1)):
            w, x, r = min_weight_logical(pd.H, pd.L, combo, time_limit=a.time_limit)
            cls["".join(map(str, combo))] = dict(weight=w, optimal=bool(r.status == 0))
        d = min(v["weight"] for v in cls.values() if v["weight"] is not None)
        res["paths"][",".join(map(str, s)) or "quiet"] = dict(
            l2_rounds=list(s), detectors=int(pd.n_det), mechanisms=int(pd.H.shape[1]), classes=cls, distance=d,
            all_optimal=all(v["optimal"] for v in cls.values()), seconds=time.time() - t0)
        print(a.basis, a.T, s or "quiet", d, {k: (v["weight"], v["optimal"]) for k, v in cls.items()},
              f"{time.time() - t0:.0f}s", flush=True)
    ds = [v["distance"] for v in res["paths"].values()]
    res["min_distance"] = min(ds)
    res["all_optimal"] = all(v["all_optimal"] for v in res["paths"].values())
    res["commit"], res["provenance"] = commit(), provenance()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
