"""Audit decoder approximation on saved failing shots: re-decode each with stronger Frontier
settings and with an exact most-likely-error ILP on the shot's own path DEM; classify each
failure as genuine (exact MLE also fails) or decoder-induced."""
import glob, pickle, sys, os, json
import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hase.protocols import make_protocol
from hase.engine import simulate_with_faults, PathDEM
from hase.decoding import FrontierDecoder


def shot_from_faults(proto, fl):
    n = proto.n_data
    faults = {}
    for key, terms in fl.items():
        if key in ("init", "final"):
            continue
        faults[key] = (np.zeros(len(terms), dtype=np.int64), np.array(terms, dtype=np.int64))
    for nm in ("init", "final"):
        arr = np.zeros((1, n), dtype=bool)
        for s in fl.get(nm, []):
            arr[0, s] = True
        faults[nm] = arr
    st = simulate_with_faults(proto, faults, 1)
    path = tuple(st.paths[0])
    pd = PathDEM(proto, path)
    D, O = pd.detectors_from_records(st.records[0], st.final[0])
    return pd, D, O


def exact_mle(pd, D, time_limit=120):
    H = pd.H.tocsr().astype(np.int64)
    w = np.log((1 - pd.priors) / pd.priors)
    m, ncol = H.shape
    A = sp.hstack([H, -2 * sp.identity(m, format="csr", dtype=np.int64)]).tocsr()
    c = np.concatenate([w, np.zeros(m)])
    deg = np.asarray(H.sum(axis=1)).ravel()
    res = milp(c, constraints=LinearConstraint(A, D.astype(float), D.astype(float)),
               integrality=np.ones(ncol + m),
               bounds=Bounds(np.zeros(ncol + m), np.concatenate([np.ones(ncol), deg // 2 + 1])),
               options={"time_limit": time_limit})
    if res.x is None:
        return None
    x = np.round(res.x[:ncol]).astype(int)
    return (pd.L @ x) % 2


def audit(pattern, proto_args, wmax=99, limit=200, settings=((512, 12.0, True), (4096, 24.0, True))):
    proto = make_protocol(*proto_args)
    rows = []
    for f in sorted(glob.glob(pattern)):
        for it in pickle.load(open(f, "rb")):
            if it["W"] > wmax:
                continue
            pd, D, O = shot_from_faults(proto, it["faults"])
            r = {"W": it["W"]}
            for K, Dl, comm in settings:
                fd = FrontierDecoder(H=pd.H, L=pd.L, priors=pd.priors, K=K, Delta=Dl, committee=comm)
                p, g, s = fd.decode_batch(D[None, :])
                r[f"K{K}"] = bool(np.any(p[0] != O))
            mle = exact_mle(pd, D)
            r["mle_fails"] = None if mle is None else bool(np.any(mle != O))
            rows.append(r)
            if len(rows) >= limit:
                return rows
    return rows


if __name__ == "__main__":
    pattern, name, rule, basis, rounds = sys.argv[1:6]
    wmax = int(sys.argv[6]) if len(sys.argv) > 6 else 99
    rows = audit(pattern, (name, 1e-3, basis, int(rounds), rule), wmax=wmax)
    from collections import Counter
    c = Counter((r["W"], r["mle_fails"], tuple(v for k, v in r.items() if k.startswith("K"))) for r in rows)
    for k, v in sorted(c.items(), key=lambda x: x[0][0]):
        print(k, v)
