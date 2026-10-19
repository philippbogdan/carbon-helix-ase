"""Chain-map (homomorphic) CNOT from the rotated surface code [[25,1,5]] to the first
logical qubit of C4-Helix [[20,2,6]] (arXiv:2609.03194 Sec. 4 and App. Fig. 14; the
chain map itself is not published, its search method is arXiv:2607.02482).

Physical CNOTs from surface data qubit a (control) to C4-Helix data qubits b with
F[b, a] = 1. Conjugation maps X on a to X on a and on F a, and Z on b to Z on b and on
F^T b, so the circuit is a logical CNOT (surface -> C4-Helix qubit 0) leaving C4-Helix
qubit 1 alone exactly when, with M = [hz; lz] of C4-Helix and N = [hx; lx] of the surface
code,
    M F N^T = T  (mod 2),  T zero except T[lz_0, lx] = 1:
X stabilizers of the surface code land on X stabilizers of C4-Helix, the surface logical X
on C4-Helix logical X_0, Z stabilizers of C4-Helix on Z stabilizers of the surface code,
its Z_0 on the surface logical Z and its Z_1 on a surface stabilizer. Solved by integer
programming with at most two CNOTs per qubit on either side (depth 2: the bipartite
graph of F has degree <= 2 and splits into two matchings) and the fewest CNOTs.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp

from .codes import CSSCode, c4_helix_paper_table
from .engine import APart, CheckDef, Protocol, compile_part
from .surface import rotated_surface_code


def chain_map_conditions(src: CSSCode, dst: CSSCode, logical=0):
    M = np.vstack([dst.hz, dst.lz]).astype(np.int64)
    N = np.vstack([src.hx, src.lx]).astype(np.int64)
    T = np.zeros((M.shape[0], N.shape[0]), dtype=np.int64)
    T[dst.hz.shape[0] + logical, src.hx.shape[0]] = 1
    return M, N, T


def is_chain_map(F, src: CSSCode, dst: CSSCode, logical=0):
    M, N, T = chain_map_conditions(src, dst, logical)
    return bool((((M @ F.astype(np.int64) @ N.T) - T) % 2 == 0).all())


def find_chain_map(src: CSSCode = None, dst: CSSCode = None, logical=0, max_deg=2, time_limit=600,
                   forbid=(), objective="min"):
    """F (dst.n x src.n) solving the chain-map conditions with row and column weights at
    most max_deg; `forbid`: earlier solutions excluded (no-good cuts); objective 'min'
    minimises the number of CNOTs."""
    src = src or rotated_surface_code(5)
    dst = dst or c4_helix_paper_table()
    M, N, T = chain_map_conditions(src, dst, logical)
    nb, na = dst.n, src.n
    nf = nb * na
    neq = M.shape[0] * N.shape[0]
    # variables: F (nf, binary), y (neq, integer) with  sum_{b,a} M[i,b] N[j,a] F[b,a] - 2 y_ij = T_ij
    rows, cols, vals = [], [], []
    rhs = []
    e = 0
    for i in range(M.shape[0]):
        for j in range(N.shape[0]):
            coef = np.outer(M[i], N[j]).ravel()
            for v in np.flatnonzero(coef):
                rows.append(e)
                cols.append(v)
                vals.append(1)
            rows.append(e)
            cols.append(nf + e)
            vals.append(-2)
            rhs.append(T[i, j])
            e += 1
    A_eq = sp.csr_matrix((vals, (rows, cols)), shape=(neq, nf + neq))
    cons = [LinearConstraint(A_eq, np.array(rhs), np.array(rhs))]
    # degree constraints
    rr, cc = [], []
    for b in range(nb):
        for a in range(na):
            rr.append(b)
            cc.append(b * na + a)
    A_row = sp.csr_matrix((np.ones(len(rr)), (rr, cc)), shape=(nb, nf + neq))
    rr, cc = [], []
    for a in range(na):
        for b in range(nb):
            rr.append(a)
            cc.append(b * na + a)
    A_col = sp.csr_matrix((np.ones(len(rr)), (rr, cc)), shape=(na, nf + neq))
    cons += [LinearConstraint(A_row, 0, max_deg), LinearConstraint(A_col, 0, max_deg)]
    for G in forbid:
        g = np.asarray(G, dtype=np.int64).ravel()
        row = np.concatenate([np.where(g == 1, 1.0, -1.0), np.zeros(neq)])
        cons.append(LinearConstraint(row.reshape(1, -1), -np.inf, g.sum() - 1))
    c = np.concatenate([np.ones(nf) if objective == "min" else np.zeros(nf), np.zeros(neq)])
    ub = np.concatenate([np.ones(nf), np.full(neq, max(1, nb * na))])
    res = milp(c, constraints=cons, integrality=np.ones(nf + neq), bounds=Bounds(np.zeros(nf + neq), ub),
               options={"time_limit": time_limit})
    if res.x is None:
        return None, res
    F = np.round(res.x[:nf]).astype(np.uint8).reshape(nb, na)
    assert is_chain_map(F, src, dst, logical)
    return F, res


def find_chain_map_sat(src: CSSCode = None, dst: CSSCode = None, logical=0, max_deg=2, max_total=None,
                       time_limit=None, forbid=(), min_total=None):
    """The same conditions as a SAT problem (CryptoMiniSat with native XOR clauses, pysat
    cardinality encodings): row and column weights <= max_deg, total <= max_total;
    `forbid`: earlier solutions excluded. Returns F or None (unsatisfiable / undecided:
    second value 'unsat' or 'undecided')."""
    import pycryptosat
    from pysat.card import CardEnc, EncType
    src = src or rotated_surface_code(5)
    dst = dst or c4_helix_paper_table()
    M, N, T = chain_map_conditions(src, dst, logical)
    nb, na = dst.n, src.n

    def var(b, a):
        return b * na + a + 1
    top = nb * na
    s = pycryptosat.Solver(threads=1, time_limit=time_limit) if time_limit else pycryptosat.Solver(threads=1)
    for i in range(M.shape[0]):
        for j in range(N.shape[0]):
            coef = np.outer(M[i], N[j]) % 2
            lits = [int(var(b, a)) for b, a in zip(*np.nonzero(coef))]
            if lits:
                s.add_xor_clause(lits, bool(T[i, j]))
            elif T[i, j]:
                return None, "unsat"
    groups = [[var(b, a) for a in range(na)] for b in range(nb)] + [[var(b, a) for b in range(nb)] for a in range(na)]
    for g in groups:
        enc = CardEnc.atmost(lits=g, bound=max_deg, top_id=top, encoding=EncType.seqcounter)
        top = max(top, enc.nv)
        for cl in enc.clauses:
            s.add_clause(cl)
    if max_total is not None:
        enc = CardEnc.atmost(lits=list(range(1, nb * na + 1)), bound=max_total, top_id=top,
                             encoding=EncType.seqcounter)
        top = max(top, enc.nv)
        for cl in enc.clauses:
            s.add_clause(cl)
    if min_total is not None:
        enc = CardEnc.atleast(lits=list(range(1, nb * na + 1)), bound=min_total, top_id=top,
                              encoding=EncType.seqcounter)
        top = max(top, enc.nv)
        for cl in enc.clauses:
            s.add_clause(cl)
    for G in forbid:
        G = np.asarray(G)
        s.add_clause([-var(b, a) if G[b, a] else var(b, a) for b in range(nb) for a in range(na)])
    sat, sol = s.solve()
    if sat is None:
        return None, "undecided"
    if not sat:
        return None, "unsat"
    F = np.array([[1 if sol[var(b, a)] else 0 for a in range(na)] for b in range(nb)], dtype=np.uint8)
    assert is_chain_map(F, src, dst, logical)
    return F, "sat"


def two_layers(F):
    """Split the CNOTs of F (bipartite graph of degree <= 2: paths and even cycles) into two
    matchings: [(a, b), ...] per layer."""
    edges = [(int(a), int(b)) for b, a in zip(*np.nonzero(F))]
    adj = {}
    for a, b in edges:
        adj.setdefault(("s", a), []).append(("h", b))
        adj.setdefault(("h", b), []).append(("s", a))
    color = {}
    for start in sorted(adj, key=lambda v: len(adj[v])):
        for nb_ in adj[start]:
            e0 = frozenset((start, nb_))
            if e0 in color:
                continue
            # walk the path/cycle through this edge, alternating colours
            used = {c for v in (start, nb_) for w in adj[v] for c in [color.get(frozenset((v, w)))] if c is not None}
            col = 0 if 0 not in used else 1
            prev, cur = start, nb_
            color[e0] = col
            while True:
                nxt = [w for w in adj[cur] if w != prev and frozenset((cur, w)) not in color]
                if not nxt:
                    break
                col = 1 - col
                prev, cur = cur, nxt[0]
                color[frozenset((prev, cur))] = col
    layers = [[], []]
    for a, b in edges:
        layers[color[frozenset((("s", a), ("h", b)))]].append((a, b))
    for lay in layers:
        assert len({a for a, _ in lay}) == len(lay) and len({b for _, b in lay}) == len(lay), "not a matching"
    return layers


# ----------------------------------------------------------------------------- protocol
def sum_code(helix: CSSCode, surf: CSSCode) -> CSSCode:
    """C4-Helix on slots 0..19 and the surface code on 20..44; rows and logicals of
    C4-Helix first (lx/lz rows 0, 1), then the surface code's (row 2)."""
    def bd(A, B):
        return np.block([[A, np.zeros((A.shape[0], B.shape[1]), dtype=np.uint8)],
                         [np.zeros((B.shape[0], A.shape[1]), dtype=np.uint8), B]]).astype(np.uint8)
    c = CSSCode("C4-Helix + rotated surface d=5", bd(helix.hx, surf.hx), bd(helix.hz, surf.hz),
                bd(helix.lx, surf.lx), bd(helix.lz, surf.lz), [list(b) for b in helix.blocks],
                list(helix.hx_level) + ["surface"] * surf.hx.shape[0],
                list(helix.hz_level) + ["surface"] * surf.hz.shape[0],
                list(helix.hx_block) + [-1] * surf.hx.shape[0], list(helix.hz_block) + [-1] * surf.hz.shape[0])
    c.check()
    return c


def merge_aligned(a, b, slots, name):
    """Part `a` with part `b`'s layer k placed in parallel with a's layer slots[k] (disjoint
    qubits)."""
    assert len(slots) == len(b.layers)
    m = APart(name, n_anc=max(a.n_anc, b.n_anc))
    extra = {s: lay for s, lay in zip(slots, b.layers)}
    for i, lay in enumerate(a.layers):
        m.add(list(lay) + list(extra.get(i, [])))
    m.meas.update(a.meas)
    m.meas.update(b.meas)
    return m


DEFAULT_F = None        # filled from results/chainmap/F.json by default_chain_map()


def default_chain_map():
    """The chain map used: results/chainmap/F.json if present (written by
    scripts/chainmap_search.py), else the first SAT solution."""
    import json
    import os
    global DEFAULT_F
    if DEFAULT_F is None:
        path = os.path.join(os.path.dirname(__file__), "..", "..", "results", "chainmap", "F.json")
        if os.path.exists(path):
            DEFAULT_F = np.array(json.load(open(path))["F"], dtype=np.uint8)
        else:
            DEFAULT_F, _ = find_chain_map_sat()
    return DEFAULT_F


class ChainMapCNOT(Protocol):
    """Surface code -> C4-Helix qubit 0 chain-map CNOTs interleaved with syndrome
    extraction on both codes (arXiv:2609.03194 App. Fig. 14, 'Surface-C4-Helix'):
    round 0 prepares both codes in the basis (C4-Helix inner part with the surface code's
    extraction in parallel, then the C4-Helix outer part); rounds 1..n_cnot apply the
    chain-map CNOT (two CX layers) and one such round; the data of both codes are then
    measured destructively in the basis."""

    def __init__(self, n_cnot, basis, noise, F=None, outer_rows=(0, 1, 2, 3), ghz=False, surface_order="safe",
                 surface_rounds=1):
        from .c4helix import c4_inner_apart, c4_outer_apart, solve_outer_orientation
        from .surface import surface_apart
        helix = c4_helix_paper_table()
        surf = rotated_surface_code(5)
        self.helix, self.surf = helix, surf
        n1 = helix.n
        self.n1 = n1
        self.code = sum_code(helix, surf)
        self.n_data = self.code.n
        self.basis, self.noise = basis, noise
        self.n_cnot = n_cnot
        self.n_rounds = n_cnot + 1
        self.F = default_chain_map() if F is None else np.asarray(F, dtype=np.uint8)
        self.layers = two_layers(self.F)
        rows = [helix.hx[5 + i] for i in outer_rows]
        orient = solve_outer_orientation(helix, rows)
        self.checks = {}
        for b in range(5):
            supp = np.array(helix.blocks[b])
            self.checks[("in", b, "X")] = CheckDef("X", supp)
            self.checks[("in", b, "Z")] = CheckDef("Z", supp)
        for i in outer_rows:
            self.checks[("out", "X", i)] = CheckDef("X", np.flatnonzero(helix.hx[5 + i]))
            self.checks[("out", "Z", i)] = CheckDef("Z", np.flatnonzero(helix.hz[5 + i]))
        for i in range(surf.hx.shape[0]):
            self.checks[("sx", i)] = CheckDef("X", np.flatnonzero(surf.hx[i]) + n1)
        for i in range(surf.hz.shape[0]):
            self.checks[("sz", i)] = CheckDef("Z", np.flatnonzero(surf.hz[i]) + n1)
        L = self.code.lx if basis == "X" else self.code.lz
        self.logical_supports = [np.flatnonzero(r) for r in L]
        self.aparts = {}
        for v in "ab":
            inner = c4_inner_apart(helix, v)
            from .surface import ORDER_ALIGNED
            sq = surface_apart(5, d_off=n1, a_off=inner.n_anc, name="SQ",
                               order=ORDER_ALIGNED if surface_order == "aligned" else None)
            # inner layers: resets, 4 CX, relabels, measurements, feedback
            self.aparts[f"inner_{v}"] = merge_aligned(inner, sq, [0, 1, 2, 3, 4, 6], f"inner_{v}")
        self.aparts["outer"] = c4_outer_apart(helix, list(outer_rows), orient, "xpair")
        # sensitivity variant: further surface-code rounds after each C4-Helix round
        self.surface_rounds = surface_rounds
        self.aparts["SQ"] = surface_apart(5, d_off=n1, a_off=0, name="SQ")
        cm = APart("CM", n_anc=0)
        for lay in self.layers:
            cm.add([("CX", (("d", n1 + a), ("d", b))) for a, b in lay])
        self.aparts["CM"] = cm
        self._gate_ops = {"CM": [("CX", n1 + a, b) for lay in self.layers for a, b in lay]}
        self.parts = {k: compile_part(v, noise, self.n_data) for k, v in self.aparts.items()}
        self.setup()

    def gate_ops(self, ptype):
        return self._gate_ops.get(ptype)

    def check_generator(self, c):
        if c[0] == "in":
            return ("hx" if c[2] == "X" else "hz", c[1])
        if c[0] == "out":
            return ("hx" if c[1] == "X" else "hz", 5 + c[2])
        return ("hx" if c[0] == "sx" else "hz", 10 + c[1])

    def plan(self, r):
        inner = "inner_a" if r % 2 == 0 else "inner_b"
        extra = [("SQ", None)] * (self.surface_rounds - 1)
        if r == 0:
            return [(inner, None), ("outer", None)] + extra
        return [("CM", None), (inner, None), ("outer", None)] + extra



def perm_to_swaps(pos, offset=0):
    """SWAP ops moving the state at q to pos[q] (cycles via transpositions with the head)."""
    n = len(pos)
    seen = [False] * n
    ops = []
    for q in range(n):
        if seen[q] or pos[q] == q:
            seen[q] = True
            continue
        cyc = [q]
        seen[q] = True
        x = pos[q]
        while x != q:
            cyc.append(x)
            seen[x] = True
            x = pos[x]
        ops += [("SWAP", offset + cyc[0], offset + c) for c in cyc[1:]]
    return ops


def surface_hadamard_ops(n1=20, d=5):
    """Logical Hadamard of the rotated surface code: H on every data qubit, then the
    quarter turn (i, j) -> (j, d-1-i), which maps the X faces of the turned code back onto
    the Z faces (as the SWAP-transversal Hadamard of arXiv:2609.03194 Sec. 4)."""
    pos = [d * j + (d - 1 - i) for i in range(d) for j in range(d)]
    return [("H", n1 + q) for q in range(d * d)] + perm_to_swaps(pos, offset=n1)


class ChainMapGHZ(ChainMapCNOT):
    """The inter-code GHZ experiment of arXiv:2609.03194 Sec. 4: both codes prepared in the
    Z basis (one round of extraction on both), the surface code's logical Hadamard, the
    chain-map CNOT (surface -> C4-Helix qubit 0), the automorphism CNOT 0 -> 1 of C4-Helix
    (a relabelling: aut_H then aut_HS of src/hase/logical.py), and destructive measurement:
    in Z ('Z': the observables Z_S Z_0 and Z_0 Z_1) or in X ('X': transversal H on both
    codes first, as a measurement in X; the observable X_S X_0 X_1)."""

    logical_products = True

    def __init__(self, meas, noise, F=None):
        from .tqrb import automorphism_apart
        from .logical import c4helix_generators
        super().__init__(1, "Z", noise, F=F)
        self.meas = meas
        g = c4helix_generators()
        ops_h = surface_hadamard_ops(self.n1)
        ops_aut = g["aut_H"] + g["aut_HS"]
        ops_hx = ops_h + g["aut_HH"]
        self._gate_ops.update(HS=ops_h, AUT=ops_aut, HX=ops_hx)
        for k, ops in (("HS", ops_h), ("AUT", ops_aut), ("HX", ops_hx)):
            self.aparts[k] = automorphism_apart(k, ops)
            self.parts[k] = compile_part(self.aparts[k], noise, self.n_data)
        self.setup()

    def plan(self, r):
        if r == 0:
            return [("inner_a", None), ("outer", None)]
        return [("HS", None), ("CM", None), ("AUT", None)] + ([("HX", None)] if self.meas == "X" else [])
