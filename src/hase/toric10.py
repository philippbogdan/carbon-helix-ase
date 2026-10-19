"""[[10,2,3]] twisted toric code memory (the unconcatenated code of arXiv:2609.03194).

The paper measures its checks with 'the standard N-Z schedule for surface codes'
(App. A) without giving the layout. Here each weight-4 check uses one bare ancilla and the
unique pair order for which a mid-circuit hook (a weight-2 error) is never contained in a
weight-3 logical operator, so the circuit distance stays 3. X checks are measured in
parallel, then Z checks. The four independent checks of each type are measured.
"""
from __future__ import annotations

import itertools

import numpy as np

from .codes import twisted_toric_10
from .engine import APart, CheckDef, Protocol, compile_part
from .gf2 import in_rowspace


def _weight3_logicals(hcomm, hstab, n=10):
    out = []
    for s in itertools.combinations(range(n), 3):
        v = np.zeros(n, dtype=np.uint8)
        v[list(s)] = 1
        if ((hcomm.astype(int) @ v) % 2).any() or in_rowspace(v, hstab):
            continue
        out.append(set(s))
    return out


def safe_order(row, logicals):
    supp = [int(x) for x in np.flatnonzero(row)]
    for i in (1, 2, 3):
        a = [supp[0], supp[i]]
        b = [x for x in supp if x not in a]
        if not any(set(a) <= L or set(b) <= L for L in logicals):
            return a + b
    raise RuntimeError("no hook-safe order")


def toric_round_apart(code, rows=(0, 1, 2, 3)) -> APart:
    LX = _weight3_logicals(code.hz, code.hx)     # X-type logicals: X-check hooks are X errors
    LZ = _weight3_logicals(code.hx, code.hz)
    part = APart("round", n_anc=2 * len(rows))
    lists = []
    a = 0
    for kind, H, Ls in (("X", code.hx, LX), ("Z", code.hz, LZ)):
        for i in rows:
            order = safe_order(H[i], Ls)
            anc = ("a", a)
            a += 1
            key = (kind, i)
            part.meas[key] = ("chk", key)
            L = [[("RX" if kind == "X" else "R", (anc,))]]
            for q in order:
                L.append([("CX", (anc, ("d", q)) if kind == "X" else (("d", q), anc))])
            L.append([("MX" if kind == "X" else "M", (anc,), key)])
            lists.append((kind, L))
    # X checks in parallel (greedy layer packing, ops on a data qubit keep check order),
    # then Z checks
    for kind in ("X", "Z"):
        seqs = [L for k, L in lists if k == kind]
        depth = max(len(L) for L in seqs)
        for t in range(depth):
            pending = [op for L in seqs if t < len(L) for op in L[t]]
            while pending:
                used, now, later = set(), [], []
                for op in pending:
                    qs = set(op[1])
                    if qs & used:
                        later.append(op)
                    else:
                        now.append(op)
                        used |= qs
                part.add(now)
                pending = later
    return part


class ToricProtocol(Protocol):
    def __init__(self, n_rounds, basis, noise, rows=(0, 1, 2, 3)):
        self.code = twisted_toric_10()
        c = self.code
        self.n_data, self.basis, self.noise, self.n_rounds = 10, basis, noise, n_rounds
        self.checks = {}
        for i in rows:
            self.checks[("X", i)] = CheckDef("X", np.flatnonzero(c.hx[i]))
            self.checks[("Z", i)] = CheckDef("Z", np.flatnonzero(c.hz[i]))
        L = c.lx if basis == "X" else c.lz
        self.logical_supports = [np.flatnonzero(r) for r in L]
        self.aparts = {"round": toric_round_apart(c, rows)}
        self.parts = {k: compile_part(v, noise, 10) for k, v in self.aparts.items()}
        self.setup()

    def plan(self, r):
        return [("round", None)]
