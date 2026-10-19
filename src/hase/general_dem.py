"""General detector error models for paths containing logical gates.

Detectors and observables are found by tracking, for every stabilizer generator and
logical operator of the code, an affine reference: the set of earlier measurement outcomes
(and unknown random bits from the projective preparation) whose parity equals the
operator's value in the noiseless reference execution. A logical gate U maps references
through U^dag P U. Each fault term's effect on every later measurement is obtained by
propagating its residual data frame through the remaining parts.

Works for any sequence of parts on the engine (memory rounds are the special case without
gates) and supports per-path priors with erasure mechanisms for measurements that were not
executed, as SupersetDEM does for memory protocols.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .logical import conj_pauli, decompose


class RefTracker:
    """Affine references (meas set, unknown-variable set) for named Paulis."""

    def __init__(self, names, vecs, deterministic):
        self.names = list(names)
        self.vecs = np.array(vecs, dtype=np.uint8)
        self.M = {nm: set() for nm in self.names}
        self.V = {}
        self.nvar = 0
        for nm, det in zip(self.names, deterministic):
            if det:
                self.V[nm] = set()
            else:
                self.V[nm] = {self.nvar}
                self.nvar += 1

    def ref(self, nm):
        return self.M[nm], self.V[nm]

    def measure(self, nm, g):
        """Measurement outcome g of operator nm. Returns detector set or None (random)."""
        M, V = self.M[nm], self.V[nm]
        if not V:
            det = set(M) ^ {g}
            self.M[nm] = {g}
            return det
        # resolve one unknown: v = g + M + (V - v)
        v = min(V)
        subM = set(M) ^ {g}
        subV = set(V) - {v}
        for other in self.names:
            if other != nm and v in self.V[other]:
                self.V[other] = self.V[other] ^ {v} ^ subV
                self.M[other] = self.M[other] ^ subM
        self.M[nm] = {g}
        self.V[nm] = set()
        return None

    def apply_gate(self, ops, n, stabs_vecs, logs_vecs, stab_names, log_names):
        """References after the Clifford `ops` on the data: value_after(P) = value_before(U^dag P U).
        U^dag P U is computed by conjugating with the inverse circuit."""
        inv = invert_ops(ops)
        newM, newV = {}, {}
        for nm, vec in zip(self.names, self.vecs):
            img = conj_pauli(vec, inv, n)
            d = decompose(img, stabs_vecs, logs_vecs)
            if d is None:
                raise ValueError("gate does not preserve the code")
            a, b = d
            M, V = set(), set()
            for i in np.flatnonzero(b):
                M ^= self.M[stab_names[i]]
                V ^= self.V[stab_names[i]]
            for i in np.flatnonzero(a):
                M ^= self.M[log_names[i]]
                V ^= self.V[log_names[i]]
            newM[nm], newV[nm] = M, V
        self.M, self.V = newM, newV


def _left_nullspace(A):
    """Basis (rows) of {c : c A = 0} over GF(2), in reduced form."""
    A = (np.array(A, dtype=np.uint8) % 2)
    m = A.shape[0]
    aug = np.concatenate([A.copy(), np.eye(m, dtype=np.uint8)], axis=1)
    r = 0
    for col in range(A.shape[1]):
        piv = [i for i in range(r, m) if aug[i, col]]
        if not piv:
            continue
        aug[[r, piv[0]]] = aug[[piv[0], r]]
        for i in range(m):
            if i != r and aug[i, col]:
                aug[i] ^= aug[r]
        r += 1
    return aug[r:, A.shape[1]:]


def invert_ops(ops):
    out = []
    for op in reversed(ops):
        name = op[0]
        if name == "S":
            out.append(("SDG",) + tuple(op[1:]))
        elif name == "SDG":
            out.append(("S",) + tuple(op[1:]))
        elif name == "PERM":
            perm = op[1]
            invp = [0] * len(perm)
            for i, j in enumerate(perm):
                invp[j] = i
            out.append(("PERM", invp))
        else:
            out.append(op)
    return out


class GeneralDEM:
    """DEM of a path of part instances (all executed) with reference-tracked detectors.

    proto: needs .parts (CompiledPart), .n_data, .basis, .code (stabilizer generators hx,hz
           and logicals), .check_vec(name) -> symplectic vector of a named check,
           .gate_ops(ptype) -> data Clifford of a gate part (None for extraction parts).
    path: list of (r, ptype).
    """

    def __init__(self, proto, path):
        from .logical import code_paulis
        self.proto = proto
        self.path = list(path)
        n = proto.n_data
        stabs_vecs, logs_vecs = code_paulis(proto.code)
        stab_names = [("hx", i) for i in range(proto.code.hx.shape[0])] + \
                     [("hz", i) for i in range(proto.code.hz.shape[0])]
        log_names = [("lx", i) for i in range(proto.code.lx.shape[0])] + \
                    [("lz", i) for i in range(proto.code.lz.shape[0])]
        basis = proto.basis
        det_ops = []
        for nm in stab_names:
            det_ops.append((nm[0] == "hz") if basis == "Z" else (nm[0] == "hx"))
        for nm in log_names:
            det_ops.append((nm[0] == "lz") if basis == "Z" else (nm[0] == "lx"))
        tr = RefTracker(stab_names + log_names, list(stabs_vecs) + list(logs_vecs), det_ops)
        # map check names to code generator names
        gen_of_check = {c: proto.check_generator(c) for c in proto.checks}
        # global measurement index
        self.meas_index = {}
        g = 0
        dets = []
        det_info = []
        for k, (r, ptype) in enumerate(self.path):
            cp = proto.parts[ptype]
            ops = proto.gate_ops(ptype)
            if ops is not None:
                tr.apply_gate(ops, n, stabs_vecs, logs_vecs, stab_names, log_names)
            for j, (kd, c) in enumerate(zip(cp.meas_kind, cp.meas_check)):
                self.meas_index[(k, j)] = g
                if kd == "flag":
                    dets.append({g})
                    det_info.append(("flag", k, j))
                else:
                    d = tr.measure(gen_of_check[c], g)
                    if d is not None:
                        dets.append(d)
                        det_info.append(("chk", c, k, j))
                g += 1
        self.n_meas_rounds = g
        # final data measurement in the basis
        fin = {s: g + s for s in range(n)}
        self.final_index = fin
        self.M_total = g + n
        measured_gens = set(gen_of_check.values())
        for nm, vec in zip(stab_names, stabs_vecs):
            if nm not in measured_gens:
                continue
            if (basis == "X" and nm[0] == "hx") or (basis == "Z" and nm[0] == "hz"):
                supp = np.flatnonzero(vec[:n] if basis == "X" else vec[n:])
                M, V = tr.ref(nm)
                if V:
                    continue
                dets.append(set(M) ^ {fin[int(s)] for s in supp})
                det_info.append(("final", nm))
        obs = []
        finals = []
        for nm, vec in zip(log_names, logs_vecs):
            if (basis == "X" and nm[0] == "lx") or (basis == "Z" and nm[0] == "lz"):
                supp = np.flatnonzero(vec[:n] if basis == "X" else vec[n:])
                M, V = tr.ref(nm)
                if V and not getattr(proto, "logical_products", False):
                    raise ValueError(f"final logical {nm} not deterministic")
                finals.append((nm, set(M) ^ {fin[int(s)] for s in supp}, set(V)))
        self.obs_logicals = [[nm] for nm, _, V in finals if not V]
        if any(V for _, _, V in finals):
            # entangled logicals (e.g. a GHZ state): the observables are a basis of the
            # products of final logicals whose unknown bits cancel
            vars_ = sorted(set().union(*(V for _, _, V in finals)))
            A = np.array([[1 if v in V else 0 for v in vars_] for _, _, V in finals], dtype=np.uint8)
            combos = _left_nullspace(A)
            self.obs_logicals = []
            for c in combos:
                s = set()
                for i in np.flatnonzero(c):
                    s ^= finals[i][1]
                obs.append(s)
                self.obs_logicals.append([finals[i][0] for i in np.flatnonzero(c)])
        else:
            obs = [s for _, s, _ in finals]
        self.dets, self.det_info, self.obs = dets, det_info, obs
        self.n_det = len(dets)
        # membership matrices (measurement -> detectors / observables)
        rows, cols = [], []
        for d, s in enumerate(dets):
            for m in s:
                rows.append(m); cols.append(d)
        self.Dm = sp.csr_matrix((np.ones(len(rows), dtype=np.int64), (rows, cols)),
                                shape=(self.M_total, self.n_det))
        rows, cols = [], []
        for o, s in enumerate(obs):
            for m in s:
                rows.append(m); cols.append(o)
        self.Om = sp.csr_matrix((np.ones(len(rows), dtype=np.int64), (rows, cols)),
                                shape=(self.M_total, len(obs)))
        self._fault_columns()

    def _fault_columns(self):
        proto = self.proto
        n = proto.n_data
        K = len(self.path)
        # G_k: frame after instance k -> flips of all later measurements (global indices)
        final_map = sp.lil_matrix((2 * n, self.M_total), dtype=np.int64)
        for s in range(n):
            if proto.basis == "X":
                final_map[n + s, self.final_index[s]] = 1
            else:
                final_map[s, self.final_index[s]] = 1
        G = [None] * K
        cur = final_map.tocsr()
        for k in range(K - 1, -1, -1):
            G[k] = cur
            if k == 0:
                break
            cp = proto.parts[self.path[k][1]]
            m = len(cp.meas_keys)
            own = sp.lil_matrix((2 * n, self.M_total), dtype=np.int64)
            im = cp.in_meas
            for j in range(m):
                gidx = self.meas_index[(k, j)]
                col = np.flatnonzero(im[:, j])
                for i in col:
                    own[i, gidx] = 1
            nxt = (sp.csr_matrix(cp.in_out.astype(np.int64)) @ cur)
            nxt.data %= 2
            cur = (own.tocsr() + nxt)
            cur.data %= 2
            cur.eliminate_zeros()
        Hcols, Lcols, owner, probs = [], [], [], []
        for k, (r, ptype) in enumerate(self.path):
            cp = proto.parts[ptype]
            m = len(cp.meas_keys)
            gl = np.array([self.meas_index[(k, j)] for j in range(m)], dtype=np.int64)
            own = sp.csr_matrix((np.ones(len(gl), dtype=np.int64), (np.arange(m), gl)),
                                shape=(m, self.M_total))
            F = cp.term_meas.astype(np.int64) @ own + cp.term_out.astype(np.int64) @ G[k]
            F.data %= 2
            F.eliminate_zeros()
            Hk = F @ self.Dm
            Hk.data %= 2
            Hk.eliminate_zeros()
            Lk = F @ self.Om
            Lk.data %= 2
            Lk.eliminate_zeros()
            Hcols.append(Hk.tocsr())
            Lcols.append(Lk.tocsr())
            owner += [k] * Hk.shape[0]
            probs.append(cp.term_prob)
        # data init flips (before instance 0) and final measurement flips
        for kind, q in (("init", proto.noise.pr), ("final", proto.noise.pm)):
            if not q:
                continue
            F = sp.lil_matrix((n, self.M_total), dtype=np.int64)
            for s in range(n):
                e = np.zeros(2 * n, dtype=np.int64)
                if proto.basis == "X":
                    e[n + s] = 1
                else:
                    e[s] = 1
                if kind == "init":
                    row = sp.csr_matrix(e.reshape(1, -1)) @ G0_from_start(self, G)
                else:
                    row = sp.csr_matrix(e.reshape(1, -1)) @ final_map.tocsr()
                for c in row.indices[row.data % 2 == 1]:
                    F[s, c] = 1
            F = F.tocsr()
            Hk = F @ self.Dm
            Hk.data %= 2
            Hk.eliminate_zeros()
            Lk = F @ self.Om
            Lk.data %= 2
            Lk.eliminate_zeros()
            Hcols.append(Hk)
            Lcols.append(Lk)
            owner += [-1] * n
            probs.append(np.full(n, q))
        self.Hraw = sp.vstack(Hcols).tocsr()
        self.Lraw = sp.vstack(Lcols).tocsr()
        self.owner = np.array(owner, dtype=np.int64)
        self.probs = np.concatenate(probs)


def G0_from_start(dem: GeneralDEM, G):
    """Map from the data frame before instance 0 to all measurement flips."""
    proto = dem.proto
    n = proto.n_data
    cp = proto.parts[dem.path[0][1]]
    m = len(cp.meas_keys)
    own = sp.lil_matrix((2 * n, dem.M_total), dtype=np.int64)
    for j in range(m):
        gidx = dem.meas_index[(0, j)]
        for i in np.flatnonzero(cp.in_meas[:, j]):
            own[i, gidx] = 1
    nxt = sp.csr_matrix(cp.in_out.astype(np.int64)) @ G[0]
    out = own.tocsr() + nxt
    out.data %= 2
    out.eliminate_zeros()
    return out


class GeneralSupersetDEM:
    """Superset version of GeneralDEM: every possible part instance present; per path,
    mechanisms of non-executed instances get prior 0 and every measurement of a
    non-executed optional instance becomes an erasure mechanism of prior 1/2."""

    def __init__(self, proto):
        plan_all = [proto.plan(r) for r in range(proto.n_rounds)]
        path = []
        optional = []
        for r, entries in enumerate(plan_all):
            for pt, cond in entries:
                path.append((r, pt))
                optional.append(cond is not None)
        self.proto = proto
        self.instances = path
        self.inst_index = {x: k for k, x in enumerate(path)}
        self.optional = np.array(optional, dtype=bool)
        base = GeneralDEM(proto, path)
        self.base = base
        n_mech = base.Hraw.shape[0]
        # erasure columns for measurements of optional instances
        er_H, er_L, er_inst = [], [], []
        Dm, Om = base.Dm.tocsr(), base.Om.tocsr()
        for (k, j), g in base.meas_index.items():
            if not self.optional[k]:
                continue
            er_H.append(Dm[g])
            er_L.append(Om[g])
            er_inst.append(k)
        if er_H:
            Her = sp.vstack(er_H).tocsr()
            Ler = sp.vstack(er_L).tocsr()
        else:
            Her = sp.csr_matrix((0, base.n_det), dtype=np.int64)
            Ler = sp.csr_matrix((0, base.Om.shape[1]), dtype=np.int64)
        Hall = sp.vstack([base.Hraw, Her]).tocsr()
        Lall = sp.vstack([base.Lraw, Ler]).tocsr()
        keys, group = {}, np.zeros(Hall.shape[0], dtype=np.int64)
        gHr, gHc, gLr, gLc = [], [], [], []
        for i in range(Hall.shape[0]):
            d = tuple(sorted(Hall.indices[Hall.indptr[i]:Hall.indptr[i + 1]].tolist()))
            l = tuple(sorted(Lall.indices[Lall.indptr[i]:Lall.indptr[i + 1]].tolist()))
            key = (d, l)
            if key not in keys:
                gi = len(keys)
                keys[key] = gi
                gHr += list(d); gHc += [gi] * len(d)
                gLr += list(l); gLc += [gi] * len(l)
            group[i] = keys[key]
        ng = len(keys)
        self.n_groups, self.group = ng, group
        self.H = sp.csr_matrix((np.ones(len(gHr), dtype=np.uint8), (gHr, gHc)), shape=(base.n_det, ng))
        self.L = sp.csr_matrix((np.ones(len(gLr), dtype=np.uint8), (gLr, gLc)),
                               shape=(base.Om.shape[1], ng))
        self.mech_inst = np.concatenate([base.owner, np.full(len(er_inst), -2)])
        self.mech_prob = np.concatenate([base.probs, np.full(len(er_inst), 0.5)])
        self.er_index = np.concatenate([np.full(n_mech, -1), np.array(er_inst, dtype=np.int64)])
        self.n_det = base.n_det
        self.null_prior = 1e-30

    def executed_mask(self, path):
        ex = np.zeros(len(self.instances), dtype=bool)
        for x in path:
            ex[self.inst_index[x]] = True
        return ex

    def priors_for(self, executed):
        ex_inst = np.concatenate([executed, [True, False]])
        m = self.mech_inst
        on = np.where(m >= 0, ex_inst[np.clip(m, 0, None)], m == -1)
        er = m == -2
        on = np.where(er, ~ex_inst[np.clip(self.er_index, 0, None)], on)
        p = np.where(on, self.mech_prob, 0.0)
        f = np.log(np.clip(1 - 2 * p, 1e-300, None))
        s = np.zeros(self.n_groups)
        np.add.at(s, self.group, f)
        pg = (1 - np.exp(s)) / 2
        return np.maximum(pg, self.null_prior)

    def syndrome(self, path, records, final):
        b = self.base
        vals = np.zeros(b.M_total, dtype=np.uint8)
        pos = {x: i for i, x in enumerate(path)}
        for (k, j), g in b.meas_index.items():
            inst = self.instances[k]
            if inst in pos:
                vals[g] = records[pos[inst]][j]
        for s, g in b.final_index.items():
            vals[g] = final[s]
        v = (vals.astype(np.int64) @ b.Dm.toarray()) % 2 if b.Dm.shape[1] < 4000 else None
        if v is None:
            v = np.asarray(b.Dm.T @ vals.astype(np.int64)).ravel() % 2
        o = np.asarray(b.Om.T @ vals.astype(np.int64)).ravel() % 2
        return v.astype(np.uint8), o.astype(np.uint8)
