"""Upper bounds on the fault distance of a protocol's circuit by constrained minimum-weight
search on its detector error model (every mechanism has unit weight).

For a mechanism e and a nontrivial logical class c, the lightest error E with the syndrome
of e and logical class (class of e) + c gives an undetectable logical error E + e of weight
<= |E| + 1. E is searched with Frontier in max-log mode (beam search for the single most
likely error) on the model whose observables are appended as detectors, forward and
backward. The minimum over mechanisms is an upper bound on the circuit distance (equal to
it when the search is exact and e runs over a minimum-weight logical's support).
A decoder that finds most-likely errors fails on some configuration of ceil(d/2) faults,
so the asymptotic exponent of the logical error rate is ceil(d/2) for this decoder class.

Usage: python analysis/distance_search.py <proto> <basis> <rounds> [n_mechanisms] [K] [Delta]
       [out.json] [rule]
"""
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hase.protocols import make_protocol  # noqa: E402
from hase.engine import PathDEM  # noqa: E402
from hase.decoding import FrontierDecoder  # noqa: E402

P0 = 1e-3


def full_path(proto):
    return [(r, pt) for r in range(proto.n_rounds) for pt, _ in proto.plan(r)]


class MinWeightSearch:
    def __init__(self, H, L, K=1024, Delta=30.0):
        self.H = H.tocsc()
        self.L = L.tocsc()
        self.nd, self.no = H.shape[0], L.shape[0]
        Haug = sp.vstack([self.H, self.L]).tocsc()
        n = H.shape[1]
        self.n = n
        self.fd = FrontierDecoder(H=Haug, L=sp.csc_matrix((1, n), dtype=np.uint8),
                                  priors=np.full(n, P0), K=K, Delta=Delta, committee=True)
        self.K, self.Delta = K, Delta
        self.unit = np.log(P0 / (1 - P0))
        self.const = n * np.log1p(-P0)

    def lightest(self, targets):
        """targets: bool array [m, nd + no]; returns (weights [m] or inf)."""
        hn = self.fd._hn
        rows = [self.fd._limbs(t) for t in targets]
        best = np.full(len(targets), np.inf)
        for model in (self.fd.native_fwd, self.fd.native_bwd):
            res = hn.decode_many(model, tuple(rows), int(self.K), float(self.Delta),
                                 float(self.fd.alpha), "maxlog_int", 1024)
            for i, r in enumerate(res):
                r = dict(r)
                if str(r.get("status")) != "ok":
                    continue
                w = (float(r["log_evidence"]) - self.const) / self.unit
                best[i] = min(best[i], w)
        return best

    def search(self, cols, batch=64):
        """For each mechanism in cols and each nonzero class c, weight bound |E| + 1."""
        out = []
        classes = [c for c in range(1, 2 ** self.no)]
        for s in range(0, len(cols), batch):
            cs = cols[s:s + batch]
            T, meta = [], []
            for j in cs:
                syn = np.zeros(self.nd + self.no, dtype=bool)
                syn[self.H.indices[self.H.indptr[j]:self.H.indptr[j + 1]]] = True
                lo = np.zeros(self.no, dtype=bool)
                lo[self.L.indices[self.L.indptr[j]:self.L.indptr[j + 1]]] = True
                for c in classes:
                    t = syn.copy()
                    t[self.nd:] = lo ^ np.array([(c >> o) & 1 for o in range(self.no)], dtype=bool)
                    T.append(t)
                    meta.append((int(j), c))
            w = self.lightest(np.array(T))
            for (j, c), x in zip(meta, w):
                out.append((j, c, float(x + 1)))
        return out


def calibrate(ms):
    """The lightest error with the syndrome and class of a single mechanism must be 1."""
    j = 0
    syn = np.zeros(ms.nd + ms.no, dtype=bool)
    syn[ms.H.indices[ms.H.indptr[j]:ms.H.indptr[j + 1]]] = True
    syn[ms.nd + ms.L.indices[ms.L.indptr[j]:ms.L.indptr[j + 1]]] = True
    return float(ms.lightest(syn[None, :])[0])


def main(proto_name, basis, rounds, n_mech=2000, K=1024, Delta=30.0, out=None, rule="full",
         seed=0):
    proto = make_protocol(proto_name, P0, basis, rounds, rule)
    pd = PathDEM(proto, full_path(proto))
    ms = MinWeightSearch(pd.H, pd.L, K=K, Delta=Delta)
    cal = calibrate(ms)
    rng = np.random.default_rng(seed)
    n = pd.H.shape[1]
    cols = np.arange(n) if n_mech >= n else np.sort(rng.choice(n, n_mech, replace=False))
    t0 = time.time()
    res = ms.search(cols)
    ws = np.array([x[2] for x in res])
    fin = ws[np.isfinite(ws)]
    dmin = float(fin.min()) if len(fin) else None
    hist = {int(round(v)): int(c) for v, c in zip(*np.unique(np.round(fin), return_counts=True))}
    summary = dict(proto=proto_name, rule=rule, basis=basis, rounds=rounds, n_mechanisms=int(n),
                   n_detectors=int(pd.H.shape[0]), searched=int(len(cols)), K=K, Delta=Delta,
                   calibration_weight=cal, distance_upper_bound=dmin, bound_histogram=hist,
                   unresolved=int((~np.isfinite(ws)).sum()), seconds=time.time() - t0)
    print(json.dumps(summary))
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        best = [dict(mechanism=j, cls=c, weight=w) for j, c, w in res if dmin is not None and w <= dmin + 0.5]
        json.dump(dict(summary, minimal=best[:200]), open(out, "w"), indent=1)
    return summary


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], int(a[3]), n_mech=int(a[4]) if len(a) > 4 else 2000,
         K=int(a[5]) if len(a) > 5 else 1024, Delta=float(a[6]) if len(a) > 6 else 30.0,
         out=a[7] if len(a) > 7 else None, rule=a[8] if len(a) > 8 else "full")
