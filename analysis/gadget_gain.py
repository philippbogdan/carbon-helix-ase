"""Worst-case error amplification of a single extraction gadget, by integer programming.

For one gadget (one outer weight-16 check, or one level-1 / level-2 part) and one Pauli type,
consider every set F of fault mechanisms inside the gadget that leaves all of the gadget's
flag (and cat-parity) outcomes unflipped. F leaves a data error E on the gadget's support.
The gain of F is wt_eff(E) - |F|, where wt_eff is the weight of E up to the measured check
(min(wt, w - wt) for a check of weight w). A circuit whose gadgets all have gain <= 0 has
circuit distance equal to the code distance (data errors pass through the other gadgets
without spreading); a gadget with gain-1 sets of at least 6 faults and no gain-2 set keeps
the circuit distance >= 11, i.e. failures still need 6 faults. Syndrome outcome flips are
left free (each could be cancelled elsewhere), so the computed gain is an upper bound.

Usage: python analysis/gadget_gain.py [design ...]   (carbonhelix.OUTER_DESIGNS)
       python analysis/gadget_gain.py blocks           (level-1 and level-2 parts)
       python analysis/gadget_gain.py --out results/gadget_gain.json   (all reported)
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hase.codes import carbon_helix  # noqa: E402
from hase.carbonhelix import (PAIRS_K6, cat4_16_layers, cat4_16_parity_layers,  # noqa: E402
                              outer_design_size, outer_rows, scheduled_layers)
from hase.engine import APart, compile_part  # noqa: E402
from hase.ir import NoiseModel  # noqa: E402

from hase.carbonhelix import OUTER_DESIGNS as DESIGNS  # noqa: E402


def single_outer_part(kind, i, design):
    code = carbon_helix()
    xs, zs = outer_rows(code, 4)
    rows = xs if kind == "X" else zs
    H = code.hx if kind == "X" else code.hz
    groups = []
    for ri in rows:
        supp = set(np.flatnonzero(H[ri]))
        parts = [[q for q in blk if q in supp] for blk in code.blocks]
        groups.append([p for p in parts if p])
    s = scheduled_layers(groups, 4)[i]
    q = [("d", s[l][(r + l) % 4]) for l in range(4) for r in range(4)]
    cat = tuple(("a", j) for j in range(4))
    key = ("out", kind, i)
    if design is None:
        fk = [("f16", kind, i, t) for t in range(3)]
        layers = cat4_16_layers(kind, q, *cat, key, fk)
        npar = 0
    else:
        npar, nflag = outer_design_size(design)
        par = tuple(("a", 4 + j) for j in range(npar))
        fk = [("f16", kind, i, t) for t in range(nflag + 3)]
        layers = cat4_16_parity_layers(kind, q, cat, par, key, fk, design)
    part = APart("G", n_anc=4 + npar)
    for lay in layers:
        part.add(lay)
    part.meas = {key: ("chk", key)}
    for k in fk:
        part.meas[k] = ("flag",)
    return part, sorted({x[1] for x in q})


def block_part(level, variant="a"):
    """Level-1 (C4 circuits on the three sub-blocks) or level-2 (weight-6 checks) part of
    Carbon block 0, with the block's 12 data qubits as support."""
    from hase.carbonhelix import level1_apart, level2_sched_apart
    from hase.codes import carbon_code
    blk = carbon_helix().blocks[0]
    part = level1_apart([blk], variant) if level == 1 else level2_sched_apart([blk], carbon_code())
    return part, list(blk)


def effects(part, supp, pauli, noise, n_data=60, strict=False):
    """strict: every measurement outcome of the part (checks too) must stay unflipped."""
    cp = compile_part(part, noise, n_data)
    cols = list(supp) if pauli == "X" else [n_data + s for s in supp]
    D = cp.term_out[:, cols].toarray().astype(np.uint8)
    flags = [j for j, kd in enumerate(cp.meas_kind) if kd == "flag" or strict]
    Fm = cp.term_meas[:, flags].toarray().astype(np.uint8) if flags else np.zeros((D.shape[0], 0), np.uint8)
    seen = {}
    for t in range(D.shape[0]):
        k = (tuple(np.flatnonzero(D[t])), tuple(np.flatnonzero(Fm[t])))
        if k[0] or k[1]:
            seen.setdefault(k, t)
    keys = list(seen)
    A = np.zeros((len(supp), len(keys)), dtype=np.int64)
    B = np.zeros((len(flags), len(keys)), dtype=np.int64)
    for j, (dq, fl) in enumerate(keys):
        A[list(dq), j] = 1
        B[list(fl), j] = 1
    desc = [cp.site_desc[cp.site_of_term[seen[k]]] for k in keys]
    return A, B, desc, cp


def max_gain(A, B, k, w, time_limit=300, stabs=None):
    """max over |F| = k of the weight of the data error up to stabilizers, flags all zero:
    stabs = list of 0/1 vectors over the support (the error's weight is the minimum over
    E + S); default {0, all-ones} (one check of weight w = len(support))."""
    nq, J = A.shape
    nf = B.shape[0]
    # variables: x (J), e (nq), y (nq), u (nf), z (1)
    nv = J + nq + nq + nf + 1
    ix, ie, iy, iu, iz = 0, J, J + nq, J + 2 * nq, J + 2 * nq + nf
    rows, lb, ub = [], [], []
    M = []
    for q in range(nq):
        r = np.zeros(nv)
        r[ix:ix + J] = A[q]
        r[ie + q] = -1
        r[iy + q] = -2
        M.append(r); lb.append(0); ub.append(0)
    for f in range(nf):
        r = np.zeros(nv)
        r[ix:ix + J] = B[f]
        r[iu + f] = -2
        M.append(r); lb.append(0); ub.append(0)
    if stabs is None:
        stabs = [np.zeros(nq, dtype=int), np.ones(nq, dtype=int)] if w == nq else [np.zeros(nq, dtype=int)]
    for S in stabs:
        # z <= wt(e + S) = sum_q S_q (1 - e_q) + (1 - S_q) e_q
        S = np.asarray(S, dtype=int)
        r = np.zeros(nv); r[iz] = 1; r[ie:ie + nq] = 2 * S - 1
        M.append(r); lb.append(-np.inf); ub.append(int(S.sum()))
    r = np.zeros(nv); r[ix:ix + J] = 1
    M.append(r); lb.append(k); ub.append(k)
    c = np.zeros(nv); c[iz] = -1
    vub = np.concatenate([np.ones(J), np.ones(nq), np.full(nq, J), np.full(nf, J), [w]])
    integ = np.ones(nv); integ[iz] = 0
    res = milp(c, constraints=LinearConstraint(sp.csr_matrix(np.array(M)), lb, ub),
               integrality=integ, bounds=Bounds(np.zeros(nv), vub),
               options={"time_limit": time_limit})
    if res.x is None:
        return None, None
    x = np.round(res.x[ix:ix + J]).astype(int)
    return int(round(-res.fun)), np.flatnonzero(x)


def profile(design_name, kind="Z", pauli=None, kmax=9, noise=None):
    design = DESIGNS[design_name]
    pauli = pauli or kind          # a Z check's cat spreads Z errors, an X check's X errors
    part, supp = single_outer_part(kind, 0, design)
    noise = noise or NoiseModel(p2=1e-3, p1=1e-4, pm=1e-3, pr=1e-3, pidle=1e-4, idle_kind="DEP",
                                name="all-Pauli")
    A, B, desc, cp = effects(part, supp, pauli, noise)
    out = {"design": design_name, "kind": kind, "pauli": pauli, "mechanisms": A.shape[1],
           "two_qubit_gates": part.two_qubit_gates(), "ancillas": part.n_anc,
           "layers": part.n_layers(), "gain": {}}
    for k in range(1, kmax + 1):
        z, sol = max_gain(A, B, k, len(supp))
        out["gain"][k] = None if z is None else z - k
        if z is not None and z - k >= 1:
            out.setdefault("witness", {})[k] = [str(desc[j]) for j in sol]
    return out


def block_profile(level, pauli, kmax=6, variant="a", noise=None, strict=False):
    """Gain of level-1/level-2 parts, the weight counted up to the Carbon block's
    stabilizers; strict: the part's check outcomes must stay unflipped too."""
    part, supp = block_part(level, variant)
    noise = noise or NoiseModel(p2=1e-3, p1=1e-4, pm=1e-3, pr=1e-3, pidle=1e-4, idle_kind="DEP",
                                name="all-Pauli")
    A, B, desc, cp = effects(part, supp, pauli, noise, strict=strict)
    out = {"part": f"L{level}" + (variant if level == 1 else ""), "pauli": pauli, "mechanisms": A.shape[1],
           "strict": strict, "gain": {}}
    from hase.codes import carbon_code
    from itertools import product
    cb = carbon_code()
    H = cb.hx if pauli == "X" else cb.hz      # an X error is reduced by X-type stabilizers
    gens = [np.array(r, dtype=int) % 2 for r in H]
    stabs = []
    for bits in product((0, 1), repeat=len(gens)):
        v = np.zeros(12, dtype=int)
        for b, g in zip(bits, gens):
            if b:
                v ^= g
        stabs.append(v)
    for k in range(1, kmax + 1):
        z, sol = max_gain(A, B, k, len(supp), stabs=stabs)
        out["gain"][k] = None if z is None else z - k
        if z is not None and z - k >= 1:
            out.setdefault("witness", {})[k] = [str(desc[j]) for j in sol]
    return out


def write_all(path):
    """Every profile reported: outer-check designs (both check types) and the level-1/2
    parts of a block; stamped with the code commit."""
    from hase.version import commit, provenance
    out = {"outer": {}, "blocks": {}}
    for nm in ("latin4", "k4cyc", "k5pre", "k6"):
        for kind in ("Z", "X"):
            out["outer"][f"{nm}_{kind}"] = profile(nm, kind)
    for level, var in ((1, "a"), (1, "b"), (2, "a")):
        for pauli in "XZ":
            for strict in (False, True):
                r = block_profile(level, pauli, variant=var, strict=strict)
                out["blocks"][f"{r['part']}_{pauli}" + ("_strict" if strict else "")] = r
    out["commit"] = commit()
    out["provenance"] = provenance()
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    if "--out" in sys.argv:
        write_all(sys.argv[sys.argv.index("--out") + 1])
        sys.exit(0)
    if sys.argv[1:2] == ["blocks"]:
        for level, var in ((1, "a"), (1, "b"), (2, "a")):
            for pauli in "XZ":
                r = block_profile(level, pauli, variant=var)
                wit = r.pop("witness", {})
                print(json.dumps(r), flush=True)
                for k in sorted(wit)[:2]:
                    print("   witness k=%d:" % k, wit[k], flush=True)
        sys.exit(0)
    names = sys.argv[1:] or list(DESIGNS)
    for nm in names:
        for kind in ("Z", "X"):
            r = profile(nm, kind)
            wit = r.pop("witness", {})
            print(json.dumps(r), flush=True)
            for k in sorted(wit)[:2]:
                print("   witness k=%d:" % k, wit[k], flush=True)
