"""Stim export of any executed path of an engine protocol, logical-gate parts included,
with the detectors and observables of GeneralDEM (reference-tracked through the gates):
an independent check of the engine's detector error model for the TQRB and two-block
CNOT circuits (the memory circuits are checked by engine.emit_stim).
"""
from __future__ import annotations

import numpy as np

from .engine import C1_MAPS, PAULI_2Q, _merge_columns

last_positions = None


def _c1_words():
    """An H/S word for every single-qubit Clifford class of engine.C1_MAPS (symplectic
    action on (x, z), the engine's convention: rows are images of X and Z)."""
    Hm = np.array([[0, 1], [1, 0]])
    Sm = np.array([[1, 0], [1, 1]])
    words = {((1, 0), (0, 1)): []}
    frontier = [((1, 0), (0, 1))]
    while frontier:
        nxt = []
        for M in frontier:
            for g, G in (("H", Hm), ("S", Sm)):
                # the engine composes as loc = G @ loc for a gate applied after loc
                N = tuple(map(tuple, (G @ np.array(M)) % 2))
                if N not in words:
                    words[N] = words[M] + [g]
                    nxt.append(N)
        frontier = nxt
    return {k: words[tuple(map(tuple, v))] for k, v in C1_MAPS.items()}


def emit_stim_general(proto, path, noiseless=False):
    """Stim circuit of `path` (list of (r, ptype)) with the GeneralDEM detectors and
    observables. Returns (circuit, GeneralDEM)."""
    import stim
    from .general_dem import GeneralDEM
    words = _c1_words()
    nz = proto.noise
    n = proto.n_data
    gd = GeneralDEM(proto, path)
    c = stim.Circuit()
    phys_of_slot = list(range(n))
    free, nxt = [], [n]

    def get():
        if free:
            return free.pop(0)
        q = nxt[0]
        nxt[0] += 1
        return q
    c.append("RX" if proto.basis == "X" else "R", list(range(n)))
    if nz.pr and not noiseless:
        c.append("Z_ERROR" if proto.basis == "X" else "X_ERROR", list(range(n)), nz.pr)
    c.append("TICK")
    rec_pos = {}            # (k, j) -> position in stim's measurement record
    nrec = 0
    for k, (r, ptype) in enumerate(path):
        ap = proto.aparts[ptype]
        keys = proto.parts[ptype].meas_keys
        anc, name_map = {}, {}

        def phys(q):
            if q in name_map:
                return name_map[q]
            if q[0] == "d":
                return phys_of_slot[q[1]]
            if q not in anc:
                anc[q] = get()
            return anc[q]
        live = set(phys_of_slot)
        for layer in ap.layers:
            touched = set()
            only_virtual = all(op[0] in ("FBX", "FBZ", "RELABEL") for op in layer)
            for op in layer:
                name, qs = op[0], op[1]
                key = op[2] if len(op) > 2 else None
                if name == "RELABEL":
                    a, b = qs
                    pa, pb = phys(a), phys(b)
                    name_map[a], name_map[b] = pb, pa
                    if a[0] == "d":
                        phys_of_slot[a[1]] = pb
                    if b[0] == "d":
                        phys_of_slot[b[1]] = pa
                    continue
                ps = [phys(q) for q in qs]
                touched.update(ps)
                if name in ("R", "RX"):
                    c.append(name, ps)
                    if nz.pr and not noiseless:
                        c.append("X_ERROR" if name == "R" else "Z_ERROR", ps, nz.pr)
                    live.update(ps)
                elif name in ("M", "MX"):
                    c.append(name, ps, nz.pm) if (nz.pm and not noiseless) else c.append(name, ps)
                    rec_pos[(k, keys.index(key))] = nrec
                    nrec += 1
                    live.discard(ps[0])
                elif name in ("H", "S", "SDG") or name.startswith("C1_"):
                    seq = [name] if not name.startswith("C1_") else words[int(name[3:])]
                    for g in seq:
                        c.append({"SDG": "S_DAG"}.get(g, g), ps)
                    if nz.p1 and not noiseless:
                        c.append("DEPOLARIZE1", ps, nz.p1)
                elif name in ("CX", "CZ"):
                    c.append(name, ps)
                    if not noiseless:
                        if getattr(nz, "pauli2", None) is not None:
                            c.append("PAULI_CHANNEL_2", ps, [nz.pauli2.get(P, 0.0) for P in PAULI_2Q])
                        elif nz.p2:
                            c.append("DEPOLARIZE2", ps, nz.p2)
                elif name in ("FBX", "FBZ"):
                    back = rec_pos[(k, keys.index(key))] - nrec
                    c.append("CX" if name == "FBX" else "CZ", [stim.target_rec(back), ps[0]])
                else:
                    raise ValueError(f"op {name} not exported")
            if only_virtual:
                continue
            if nz.pidle and not noiseless:
                idle = sorted(q for q in live if q not in touched)
                if idle:
                    c.append("Z_ERROR" if nz.idle_kind == "Z" else "DEPOLARIZE1", idle, nz.pidle)
            c.append("TICK")
        used = set(anc.values()) | {v for kk, v in name_map.items() if kk[0] == "a"}
        for q in sorted(used):
            if q not in phys_of_slot and q not in free:
                free.append(q)
    fin0 = nrec
    if nz.pm and not noiseless:
        c.append("MX" if proto.basis == "X" else "M", phys_of_slot, nz.pm)
    else:
        c.append("MX" if proto.basis == "X" else "M", phys_of_slot)
    total = fin0 + n
    pos_of_g = {g: rec_pos[kj] for kj, g in gd.meas_index.items()}
    for s in range(n):
        pos_of_g[gd.final_index[s]] = fin0 + s
    global last_positions
    last_positions = pos_of_g        # measurement index g -> position in stim's record

    def rec(g):
        return stim.target_rec(pos_of_g[g] - total)
    for d in gd.dets:
        c.append("DETECTOR", [rec(g) for g in sorted(d)])
    for i, o in enumerate(gd.obs):
        c.append("OBSERVABLE_INCLUDE", [rec(g) for g in sorted(o)], i)
    return c, gd


def compare_with_stim(proto, path):
    """Mechanisms (detector set, observable set) -> probability of the engine's GeneralDEM
    against stim's DEM of the exported circuit."""
    import scipy.sparse as sp
    from .distance import dem_to_matrices
    c, gd = emit_stim_general(proto, path)
    H2, L2, p2 = dem_to_matrices(c.detector_error_model(decompose_errors=False))
    Hm, Lm, pm = _merge_columns(sp.csr_matrix(gd.Hraw), sp.csr_matrix(gd.Lraw), gd.probs)

    def colset(H, L, P):
        H, L, o = H.tocsc(), L.tocsc(), {}
        for j in range(H.shape[1]):
            k = (tuple(sorted(H.indices[H.indptr[j]:H.indptr[j + 1]].tolist())),
                 tuple(sorted(L.indices[L.indptr[j]:L.indptr[j + 1]].tolist())))
            o[k] = o.get(k, 0) + P[j]
        return o
    a, b = colset(Hm, Lm, pm), colset(H2, L2, p2)
    common = set(a) & set(b)
    return dict(engine_columns=len(a), stim_columns=len(b), common=len(common),
                max_rel_prob_diff=float(max(abs(a[k] - b[k]) / b[k] for k in common)) if common else None,
                detectors=int(gd.n_det))
