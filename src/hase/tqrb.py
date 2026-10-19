"""Two-qubit logical randomized benchmarking (TQRB) on C4-Helix, as in arXiv:2609.03194.

Each random two-qubit Clifford (uniform over Sp(4,2); Paulis are tracked in software and
do not change any detector) is compiled by the lookup table into automorphisms (free:
single-qubit gates and qubit relabelling) and S0S1 gates (Fig. 5c, every physical CZ
replaced by the flag gadget of Fig. 5b). Each S0S1 is followed by one round of syndrome
extraction. Adaptive rule (paper App. A, 'Adaptive syndrome extraction'): if no C4 check
fires in that round, measure only two outer X checks with support on the suspect blocks
coupled by the inter-block CZs; otherwise measure all outer checks.
"""
from __future__ import annotations

import numpy as np

from .c4helix import (C4HelixProtocol, c4_inner_apart, c4_outer_apart,
                      solve_outer_orientation)
from .codes import c4_helix_paper_table
from .engine import APart, CheckDef, Protocol, compile_part
from .logical import (c4helix_generators, logical_action, cayley_costs, conj_pauli,
                      code_paulis, decompose)

CZ_PAIRS = [(0, 1), (4, 5), (8, 19), (9, 17), (10, 18), (11, 16), (12, 14)]
S_QUBITS = [2, 3, 6, 7, 13, 15]
SUSPECT_BLOCKS = (2, 4)


def s0s1_apart(flagged=True) -> APart:
    part = APart("S0S1", n_anc=2 * len(CZ_PAIRS) if flagged else 0)
    if not flagged:
        part.add([("CZ", (("d", a), ("d", b))) for a, b in CZ_PAIRS])
        part.add([("S", (("d", q),)) for q in S_QUBITS])
        return part
    L0, L1, L2, L3, L4, L5 = [], [], [], [], [], []
    for i, (a, b) in enumerate(CZ_PAIRS):
        A, B = ("d", a), ("d", b)
        f0, fp = ("a", 2 * i), ("a", 2 * i + 1)
        L0 += [("R", (f0,)), ("RX", (fp,))]
        L1 += [("CX", (A, f0)), ("CX", (fp, B))]
        L2 += [("CZ", (A, B))]
        L3 += [("CZ", (A, fp))]
        L4 += [("CX", (A, f0)), ("CX", (fp, B))]
        L5 += [("M", (f0,), ("fs", i, 0)), ("MX", (fp,), ("fs", i, 1))]
        part.meas[("fs", i, 0)] = ("flag",)
        part.meas[("fs", i, 1)] = ("flag",)
    for lay in (L0, L1, L2, L3, L4, L5):
        part.add(lay)
    part.add([("S", (("d", q),)) for q in S_QUBITS])
    return part


def _c1_class(M):
    from .engine import C1_MAPS
    for k, v in C1_MAPS.items():
        if tuple(map(tuple, M)) == v:
            return k
    raise ValueError(M)


def merged_automorphism_apart(name, op_lists, n=20) -> APart:
    """Product of automorphisms as one layer of single-qubit Cliffords (one noise location
    per qubit with a non-identity local Clifford) followed by a relabelling."""
    pos = list(range(n))           # current position of the state originally at q
    at = list(range(n))            # which original qubit is at position p
    loc = [np.eye(2, dtype=int) for _ in range(n)]
    Hm = np.array([[0, 1], [1, 0]])
    Sm = np.array([[1, 0], [1, 1]])
    for ops in op_lists:
        for op in ops:
            if op[0] == "SWAP":
                a, b = op[1], op[2]
                qa, qb = at[a], at[b]
                at[a], at[b] = qb, qa
                pos[qa], pos[qb] = b, a
            else:
                q = at[op[1]]
                G = Hm if op[0] == "H" else Sm
                loc[q] = (G @ loc[q]) % 2
    part = APart(name, n_anc=0)
    lay = []
    for q in range(n):
        k = _c1_class(loc[q] % 2)
        if k != 0:
            lay.append((f"C1_{k}", (("d", q),)))
    if lay:
        part.add(lay)
    # permutation: state at q moves to pos[q]; cycles via transpositions with the cycle head
    seen = [False] * n
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
        for c in cyc[1:]:
            part.add([("RELABEL", (("d", cyc[0]), ("d", c)))])
    return part


def automorphism_apart(name, ops, n=20) -> APart:
    """Single-qubit gates as gates; SWAPs as virtual relabellings (no time, no noise)."""
    part = APart(name, n_anc=0)
    one = [op for op in ops if op[0] in ("H", "S", "SDG")]
    if one:
        part.add([(op[0], (("d", op[1]),)) for op in one])
    for op in ops:
        if op[0] == "SWAP":
            part.add([("RELABEL", (("d", op[1]), ("d", op[2])))])
    return part


def suspect_x_rows(code, rows_idx):
    """Smallest set of outer X rows (indices into rows_idx) detecting every Z-type
    inner logical error on the suspect blocks."""
    import itertools
    n = code.n
    errs = []
    for b in SUSPECT_BLOCKS:
        q = code.blocks[b]
        for pair in ((q[2], q[3]), (q[0], q[3]), (q[0], q[2])):    # Z0, Z1, Z0Z1 of C4
            e = np.zeros(n, dtype=np.uint8)
            e[list(pair)] = 1
            errs.append(e)
    rows = [code.hx[5 + i] for i in rows_idx]
    for k in (1, 2, 3, 4):
        for combo in itertools.combinations(range(len(rows)), k):
            if all(any((rows[c].astype(int) @ e) % 2 for c in combo) for e in errs):
                return [rows_idx[c] for c in combo]
    raise RuntimeError


class TQRBProtocol(C4HelixProtocol):
    """One TQRB sequence (fixed list of generator words) as an engine protocol.

    rounds: round 0 = projective preparation (full); then for each S0S1: S0S1 part,
    inner round, outer part (full, or 2 X checks if adaptive and no inner event).
    Automorphisms are folded into the following S0S1 step (they are relabellings and
    single-qubit gates)."""

    def __init__(self, words, noise, adaptive=True, outer_rows=(0, 1, 2, 3), flagged=True,
                 basis="Z"):
        code = c4_helix_paper_table()
        self.code = code
        self.n_data = code.n
        self.basis = basis
        self.noise = noise
        self.adaptive = adaptive
        rows = [code.hx[5 + i] for i in outer_rows]
        orient = solve_outer_orientation(code, rows)
        self.checks = {}
        for b in range(5):
            supp = np.array(code.blocks[b])
            self.checks[("in", b, "X")] = CheckDef("X", supp)
            self.checks[("in", b, "Z")] = CheckDef("Z", supp)
        for i in outer_rows:
            self.checks[("out", "X", i)] = CheckDef("X", np.flatnonzero(code.hx[5 + i]))
            self.checks[("out", "Z", i)] = CheckDef("Z", np.flatnonzero(code.hz[5 + i]))
        L = code.lx if basis == "X" else code.lz
        self.logical_supports = [np.flatnonzero(r) for r in L]
        gens = c4helix_generators()
        self.susp = suspect_x_rows(code, list(outer_rows))
        self.aparts = {"inner_a": c4_inner_apart(code, "a"), "inner_b": c4_inner_apart(code, "b"),
                       "outer": c4_outer_apart(code, list(outer_rows), orient, "xpair"),
                       "outer2": c4_outer_apart(code, list(outer_rows), orient, "xpair",
                                                which=[("X", i) for i in self.susp], name="outer2"),
                       "S0S1": s0s1_apart(flagged)}
        self._gate_ops = {"S0S1": gens["S0S1"]}
        for g in ("aut_H", "aut_HS", "aut_HH", "aut_CZ"):
            self.aparts[g] = automorphism_apart(g, gens[g])
            self._gate_ops[g] = gens[g]
        self.parts = {k: compile_part(v, noise, self.n_data) for k, v in self.aparts.items()}
        # schedule: list of steps; each step = list of (ptype, cond-kind)
        self.steps = [[("inner_a", None), ("outer", None)]]
        pending = []
        rr = 1
        self._merged = {}

        def flush(pend):
            if not pend:
                return []
            key = "aut:" + "-".join(pend)
            if key not in self.aparts:
                self.aparts[key] = merged_automorphism_apart(key, [gens[g] for g in pend])
                self._gate_ops[key] = [op for g in pend for op in gens[g]]
            return [(key, None)]
        for w in words:
            for g in w:
                if g == "S0S1":
                    inner = "inner_a" if rr % 2 == 0 else "inner_b"
                    st = flush(pending) + [("S0S1", None), (inner, None)]
                    if adaptive:
                        st += [("outer2", "quiet"), ("outer", "noisy")]
                    else:
                        st += [("outer", None)]
                    self.steps.append(st)
                    pending = []
                    rr += 1
                else:
                    pending.append(g)
        if pending:
            self.steps.append(flush(pending))
        for k, v in self.aparts.items():
            if k not in self.parts:
                self.parts[k] = compile_part(v, noise, self.n_data)
        self.n_rounds = len(self.steps)
        self.mode = "tqrb"
        self.setup()
        self.inner_idx = [self.check_index[("in", b, t)] for b in range(5) for t in "XZ"]

    def plan(self, r):
        out = []
        for ptype, kind in self.steps[r]:
            if kind is None:
                out.append((ptype, None))
            elif kind == "quiet":
                out.append((ptype, lambda st, idx: ~st.round_events[idx][:, self.inner_idx].any(axis=1)))
            else:
                out.append((ptype, lambda st, idx: st.round_events[idx][:, self.inner_idx].any(axis=1)))
        return out


# ------------------------------------------------------------------ random sequences
class CliffordSampler:
    def __init__(self):
        code = c4_helix_paper_table()
        gens = c4helix_generators()
        self.acts = {n: logical_action(code, ops) for n, ops in gens.items()}
        costs = {n: (1 if n == "S0S1" else 0) for n in self.acts}
        self.best, self.mats = cayley_costs(self.acts, costs)
        self.keys = list(self.best)

    def sequence(self, m, rng):
        """m-1 random Cliffords plus the inverse (m Cliffords in total, as the paper's
        sequence length 'including the final corrective Clifford'). Returns list of
        words (lists of generator names)."""
        words = []
        tot = np.eye(4, dtype=np.uint8)
        for _ in range(m - 1):
            k = self.keys[rng.integers(len(self.keys))]
            words.append(list(self.best[k][1]))
            tot = (tot.astype(int) @ self.mats[k].astype(int)) % 2
        # inverse: find element M with tot @ M = I
        inv = None
        for k in self.keys:
            if np.array_equal((tot.astype(int) @ self.mats[k].astype(int)) % 2, np.eye(4, dtype=int)):
                inv = k
                break
        words.append(list(self.best[inv][1]))
        return words
