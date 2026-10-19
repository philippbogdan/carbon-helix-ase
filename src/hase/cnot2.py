"""Two C4-Helix blocks with transversal CNOTs interleaved with syndrome extraction, as in
arXiv:2609.03194 App. C (Fig. 14, 'C4-Helix transv.'): projective preparation of both
blocks in the basis (one full round on each, in parallel), then n_cnot times the 20
pairwise physical CNOTs (block A controls, block B targets: two logical CNOTs) followed
by one full round on both blocks, and the destructive measurement of both blocks in the
basis. The rounds are the memory circuits of src/hase/c4helix.py (Fig. 8), run on both
blocks at once.
"""
from __future__ import annotations

import numpy as np

from .c4helix import c4_inner_apart, c4_outer_apart, solve_outer_orientation
from .codes import CSSCode, c4_helix_paper_table
from .engine import APart, CheckDef, Protocol, compile_part


def doubled_code(c: CSSCode) -> CSSCode:
    """Two copies of `c` on qubits 0..n-1 (block A) and n..2n-1 (block B); generator and
    logical rows of A first, then those of B."""
    n = c.n

    def two(M):
        Z = np.zeros_like(M)
        return np.block([[M, Z], [Z, M]]).astype(np.uint8)
    d = CSSCode(c.name + " x2", two(c.hx), two(c.hz), two(c.lx), two(c.lz),
                [list(b) for b in c.blocks] + [[q + n for q in b] for b in c.blocks],
                list(c.hx_level) * 2, list(c.hz_level) * 2,
                list(c.hx_block) + [b + len(c.blocks) if b >= 0 else -1 for b in c.hx_block],
                list(c.hz_block) + [b + len(c.blocks) if b >= 0 else -1 for b in c.hz_block])
    d.check()
    return d


def relocate(part: APart, d_off: int, a_off: int, rename, name: str) -> APart:
    """The same part on data slots shifted by d_off and ancillas by a_off, with measurement
    keys and check names renamed."""
    def q(x):
        if x[0] == "d":
            return ("d", x[1] + d_off)
        if x[0] == "a":
            return ("a", x[1] + a_off)
        return x
    new = APart(name, n_anc=part.n_anc)
    for lay in part.layers:
        ops = []
        for op in lay:
            qs = tuple(q(x) for x in op[1])
            ops.append((op[0], qs, rename(op[2])) + tuple(op[3:]) if len(op) > 2 else (op[0], qs))
        new.layers.append(ops)
    for k, v in part.meas.items():
        new.meas[rename(k)] = ("chk", rename(v[1])) if v[0] == "chk" else v
    return new


def merge(a: APart, b: APart, name: str) -> APart:
    """Two parts with the same layer structure on disjoint qubits, run in parallel."""
    assert len(a.layers) == len(b.layers)
    m = APart(name, n_anc=a.n_anc + b.n_anc)
    for la, lb in zip(a.layers, b.layers):
        m.add(list(la) + list(lb))
    m.meas.update(a.meas)
    m.meas.update(b.meas)
    return m


def _block_b(k):
    """Names of block B's checks and flags (block A keeps the single-block names)."""
    if k[0] == "in":
        return ("in", k[1] + 5, k[2])
    if k[0] == "out":
        return ("out", k[1], k[2] + 5)
    if k[0] == "flag":
        return ("flag", k[1], k[2] + 5, k[3])
    return k


class TwoBlockCNOT(Protocol):
    """Rounds: 0 = preparation (full round on both blocks); r = 1..n_cnot: transversal
    CNOT, then a full round on both blocks (inner circuits alternating a/b as in memory)."""

    def __init__(self, n_cnot, basis, noise, outer_rows=(0, 1, 2, 3)):
        c1 = c4_helix_paper_table()
        n1 = c1.n
        self.code = doubled_code(c1)
        self.n_data = 2 * n1
        self.basis = basis
        self.noise = noise
        self.n_cnot = n_cnot
        self.n_rounds = n_cnot + 1
        self.mode = "cnot2"
        rows = [c1.hx[5 + i] for i in outer_rows]
        orient = solve_outer_orientation(c1, rows)
        self.checks = {}
        for blk in (0, 1):
            off = blk * n1
            for b in range(5):
                supp = np.array(c1.blocks[b]) + off
                self.checks[("in", b + 5 * blk, "X")] = CheckDef("X", supp)
                self.checks[("in", b + 5 * blk, "Z")] = CheckDef("Z", supp)
            for i in outer_rows:
                self.checks[("out", "X", i + 5 * blk)] = CheckDef("X", np.flatnonzero(c1.hx[5 + i]) + off)
                self.checks[("out", "Z", i + 5 * blk)] = CheckDef("Z", np.flatnonzero(c1.hz[5 + i]) + off)
        L = self.code.lx if basis == "X" else self.code.lz
        self.logical_supports = [np.flatnonzero(r) for r in L]
        makers = {"inner_a": lambda: c4_inner_apart(c1, "a"), "inner_b": lambda: c4_inner_apart(c1, "b"),
                  "outer": lambda: c4_outer_apart(c1, list(outer_rows), orient, "xpair")}
        self.aparts = {}
        for nm, mk in makers.items():
            pa = mk()
            pb = relocate(mk(), n1, pa.n_anc, _block_b, nm)
            self.aparts[nm] = merge(pa, pb, nm)
        cx = APart("CNOT", n_anc=0)
        cx.add([("CX", (("d", q), ("d", q + n1))) for q in range(n1)])
        self.aparts["CNOT"] = cx
        self._gate_ops = {"CNOT": [("CX", q, q + n1) for q in range(n1)]}
        self.parts = {k: compile_part(v, noise, self.n_data) for k, v in self.aparts.items()}
        self.setup()

    def gate_ops(self, ptype):
        return self._gate_ops.get(ptype)

    def check_generator(self, c):
        """Row of the doubled code's hx/hz that a check measures (A: rows 0-9, B: 10-19)."""
        if c[0] == "in":
            b = c[1]
            row = b if b < 5 else 10 + (b - 5)
            return ("hx" if c[2] == "X" else "hz", row)
        i = c[2]
        row = 5 + i if i < 5 else 15 + (i - 5)
        return ("hx" if c[1] == "X" else "hz", row)

    def plan(self, r):
        inner = "inner_a" if r % 2 == 0 else "inner_b"
        if r == 0:
            return [(inner, None), ("outer", None)]
        return [("CNOT", None), (inner, None), ("outer", None)]
