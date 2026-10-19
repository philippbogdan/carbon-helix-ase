"""Exact circuit-level distance of a detector error model by integer programming.

The circuit distance is the minimum number of error mechanisms whose combined effect
flips no detector but flips some logical observable. We solve
    min sum x_j  s.t.  H x = 2 y,  L_S x = 2 z + 1,  x binary,
with scipy.optimize.milp (HiGHS), for each nonempty combination S of observables.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import scipy.sparse as sp
import stim
from scipy.optimize import Bounds, LinearConstraint, milp


def dem_to_matrices(dem: stim.DetectorErrorModel):
    """Return (H csr [dets x errs], L csr [obs x errs], probs) with errors merged
    by identical symptoms (no decomposition)."""
    rows, cols, lrows, lcols, probs = [], [], [], [], []
    j = 0
    seen = {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        dets = tuple(sorted(t.val for t in inst.targets_copy() if t.is_relative_detector_id()))
        obs = tuple(sorted(t.val for t in inst.targets_copy() if t.is_logical_observable_id()))
        key = (dets, obs)
        if key in seen:
            k = seen[key]
            probs[k] = probs[k] * (1 - p) + p * (1 - probs[k])
            continue
        seen[key] = j
        for d in dets:
            rows.append(d); cols.append(j)
        for o in obs:
            lrows.append(o); lcols.append(j)
        probs.append(p)
        j += 1
    H = sp.csr_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)),
                      shape=(dem.num_detectors, j))
    L = sp.csr_matrix((np.ones(len(lrows), dtype=np.uint8), (lrows, lcols)),
                      shape=(dem.num_observables, j))
    return H, L, np.array(probs)


def min_weight_logical(H, L, obs_combo, time_limit=600, weights=None, verbose=False):
    """Minimum weight x with Hx=0 and (sum of rows obs_combo of L) x = 1 (mod 2)."""
    m, n = H.shape
    Lrow = np.zeros(n, dtype=np.int64)
    for o in obs_combo:
        Lrow ^= L[o].toarray().ravel().astype(np.int64)
    # keep only detectors touched by something
    Hc = H.tocsr().astype(np.int64)
    nv = n + m + 1        # x (n), y (m), z (1)
    c = np.zeros(nv)
    c[:n] = 1.0 if weights is None else weights
    A_det = sp.hstack([Hc, -2 * sp.identity(m, format="csr", dtype=np.int64),
                       sp.csr_matrix((m, 1), dtype=np.int64)])
    A_obs = sp.hstack([sp.csr_matrix(Lrow.reshape(1, -1)), sp.csr_matrix((1, m), dtype=np.int64),
                       sp.csr_matrix(np.array([[-2]]))])
    A = sp.vstack([A_det, A_obs]).tocsr()
    lb = np.concatenate([np.zeros(m), [1]])
    ub = np.concatenate([np.zeros(m), [1]])
    integrality = np.ones(nv)
    deg = np.asarray(Hc.sum(axis=1)).ravel()
    vub = np.concatenate([np.ones(n), np.maximum(deg // 2, 0), [max(1, Lrow.sum() // 2)]])
    res = milp(c, constraints=LinearConstraint(A, lb, ub), integrality=integrality,
               bounds=Bounds(np.zeros(nv), vub),
               options={"time_limit": time_limit, "disp": verbose})
    if res.x is None:
        return None, None, res
    x = np.round(res.x[:n]).astype(np.uint8)
    return int(round(res.fun)) if weights is None else res.fun, x, res


def circuit_distance(dem: stim.DetectorErrorModel, time_limit=600, verbose=False):
    H, L, probs = dem_to_matrices(dem)
    k = L.shape[0]
    best = None
    detail = {}
    for r in range(1, k + 1):
        for combo in combinations(range(k), r):
            w, x, res = min_weight_logical(H, L, combo, time_limit=time_limit, verbose=verbose)
            detail[combo] = (w, res.status, getattr(res, "mip_dual_bound", None))
            if w is not None and (best is None or w < best[0]):
                best = (w, combo, x)
    return best, detail


def sat_has_logical_within(H, L, obs_combo, W, time_limit=None, verbose=False):
    """SAT (CryptoMiniSat with native XORs + cardinality encoding from pysat): is there an
    error set of at most W mechanisms with zero syndrome flipping the observable combo?
    Returns (True, solution) / (False, None) / (None, None) if undecided."""
    import pycryptosat
    from pysat.card import CardEnc, EncType
    H = H.tocsr()
    L = L.tocsr()
    n = H.shape[1]
    s = pycryptosat.Solver(threads=1, time_limit=time_limit) if time_limit else pycryptosat.Solver(threads=1)
    for d in range(H.shape[0]):
        cols = H.indices[H.indptr[d]:H.indptr[d + 1]]
        if len(cols):
            s.add_xor_clause([int(c) + 1 for c in cols], False)
    lrow = np.zeros(n, dtype=np.int64)
    for o in obs_combo:
        lrow ^= L[o].toarray().ravel().astype(np.int64)
    lc = np.flatnonzero(lrow)
    s.add_xor_clause([int(c) + 1 for c in lc], True)
    card = CardEnc.atmost(lits=list(range(1, n + 1)), bound=W, top_id=n, encoding=EncType.seqcounter)
    for cl in card.clauses:
        s.add_clause(cl)
    sat, sol = s.solve()
    if sat is None:
        return None, None
    if sat:
        x = np.array([1 if sol[i + 1] else 0 for i in range(n)], dtype=np.uint8)
        return True, x
    return False, None


def cpsat_min_weight_logical(H, L, obs_combo, time_limit=600, workers=8, upper=None, verbose=False):
    """Minimum weight x with Hx = 0 and (sum of rows obs_combo of L) x = 1 (mod 2) by OR-tools
    CP-SAT (parity constraints as XOR clauses, objective the number of mechanisms).
    Returns (best weight or None, x or None, proven lower bound, status name)."""
    from ortools.sat.python import cp_model
    H = H.tocsr()
    L = L.tocsr()
    n = H.shape[1]
    m = cp_model.CpModel()
    x = [m.NewBoolVar(f"x{j}") for j in range(n)]
    one = m.NewConstant(1)
    for d in range(H.shape[0]):
        cols = H.indices[H.indptr[d]:H.indptr[d + 1]]
        if len(cols):
            m.AddBoolXOr([x[int(c)] for c in cols] + [one])      # even parity
    lrow = np.zeros(n, dtype=np.int64)
    for o in obs_combo:
        lrow ^= L[o].toarray().ravel().astype(np.int64)
    m.AddBoolXOr([x[int(c)] for c in np.flatnonzero(lrow)])      # odd parity
    if upper is not None:
        m.Add(sum(x) <= int(upper))
    m.Minimize(sum(x))
    sv = cp_model.CpSolver()
    sv.parameters.max_time_in_seconds = float(time_limit)
    sv.parameters.num_search_workers = int(workers)
    sv.parameters.log_search_progress = bool(verbose)
    st = sv.Solve(m)
    name = sv.StatusName(st)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        xs = np.array([int(sv.Value(v)) for v in x], dtype=np.uint8)
        return int(round(sv.ObjectiveValue())), xs, float(sv.BestObjectiveBound()), name
    return None, None, float(sv.BestObjectiveBound()), name
