"""C4-Helix [[20,2,6]] syndrome-extraction circuits as drawn in arXiv:2609.03194
(Fig. 8 = new_se_circuit.png):

(a)/(b) inner [[4,2,2]] circuit: two fresh ancillas (|+>, |0>) measure XXXX and ZZZZ of a
    block with 8 CNOTs; the last CNOT of each ancilla is reversed and the ancilla is
    swapped (virtually) with a data qubit, so two data qubits are refreshed per round;
    classically-controlled Paulis restore the data. (a) refreshes d2,d3 and (b) d0,d1;
    rounds alternate (a),(b).
(c) outer weight-8 checks: 'modified Shor-style' 4-qubit cat (1 syndrome + 3 flags),
    CNOT depth 6, 14 two-qubit gates per check.
"""
from __future__ import annotations

import numpy as np

from .codes import c4_helix_paper_table, CSSCode
from .ir import Op, Part


class QubitPool:
    def __init__(self, first_free: int):
        self.free = []
        self.next = first_free
        self.peak = first_free

    def get(self):
        if self.free:
            return self.free.pop(0)
        q = self.next
        self.next += 1
        self.peak = max(self.peak, self.next)
        return q

    def put(self, q):
        self.free.append(q)


def inner_c4_ops(p, qa, qb, variant, key_x, key_z):
    """Return (layers, new_phys_for_slots, measured_qubits) for one C4 block.
    p: physical qubits of slots (d0,d1,d2,d3); qa: |+> ancilla, qb: |0> ancilla."""
    d0, d1, d2, d3 = p
    if variant == "a":
        cx = [[("CX", d0, qb), ("CX", qa, d1)],
              [("CX", d1, qb), ("CX", qa, d0)],
              [("CX", d2, qb), ("CX", qa, d3)],
              [("CX", qb, d3), ("CX", d2, qa)]]
        # virtual swaps: slot2 <- qa, slot3 <- qb; old d2 measured in X, old d3 in Z
        new = (d0, d1, qa, qb)
        mx, mz = d2, d3
    else:
        cx = [[("CX", d2, qb), ("CX", qa, d3)],
              [("CX", d3, qb), ("CX", qa, d2)],
              [("CX", d0, qb), ("CX", qa, d1)],
              [("CX", qb, d1), ("CX", d0, qa)]]
        new = (qa, qb, d2, d3)
        mx, mz = d0, d1
    layers = [[Op("RX", (qa,)), Op("R", (qb,))]]
    for lay in cx:
        layers.append([Op(n, (a, b)) for n, a, b in lay])
    layers.append([Op("MX", (mx,), key_x), Op("M", (mz,), key_z)])
    # feedback: Z-syndrome result -> X on the refreshed qubit that took the |0> ancilla,
    #           X-syndrome result -> Z on the refreshed qubit that took the |+> ancilla
    layers.append([Op("FBX", (qb,), key_z), Op("FBZ", (qa,), key_x)])
    return layers, new, (mx, mz)


def outer_check_layers(kind, data, s, f1, f2, f3, key, flag_keys):
    """Modified Shor-style weight-8 check, Fig. 8(c). kind 'X' or 'Z'. data: 8 physical
    qubits in the order q1..q8 of the figure. Returns list of layers (6 CNOT layers plus
    reset and measurement layers), as lists of Op."""
    assert len(data) == 8
    q = data
    if kind == "X":
        L = [[Op("RX", (s,)), Op("R", (f1,)), Op("R", (f2,)), Op("R", (f3,))],
             [Op("CX", (s, f1))],
             [Op("CX", (s, f2)), Op("CX", (f1, f3))],
             [Op("CX", (s, q[0])), Op("CX", (f1, q[1])), Op("CX", (f2, q[2])), Op("CX", (f3, q[3]))],
             [Op("CX", (s, q[4])), Op("CX", (f1, q[5])), Op("CX", (f2, q[6])), Op("CX", (f3, q[7]))],
             [Op("CX", (s, f3)), Op("CX", (f1, f2))],
             [Op("CX", (s, f1))],
             [Op("MX", (s,), key), Op("M", (f1,), flag_keys[0]), Op("M", (f2,), flag_keys[1]),
              Op("M", (f3,), flag_keys[2])]]
    else:  # Hadamard conjugate: reverse CNOTs and swap bases
        L = [[Op("R", (s,)), Op("RX", (f1,)), Op("RX", (f2,)), Op("RX", (f3,))],
             [Op("CX", (f1, s))],
             [Op("CX", (f2, s)), Op("CX", (f3, f1))],
             [Op("CX", (q[0], s)), Op("CX", (q[1], f1)), Op("CX", (q[2], f2)), Op("CX", (q[3], f3))],
             [Op("CX", (q[4], s)), Op("CX", (q[5], f1)), Op("CX", (q[6], f2)), Op("CX", (q[7], f3))],
             [Op("CX", (f3, s)), Op("CX", (f2, f1))],
             [Op("CX", (f1, s))],
             [Op("M", (s,), key), Op("MX", (f1,), flag_keys[0]), Op("MX", (f2,), flag_keys[1]),
              Op("MX", (f3,), flag_keys[2])]]
    return L


def merge_layer_lists(lists, offsets):
    """Merge several layer lists into one, list i starting at layer offsets[i]."""
    n = max(o + len(l) for l, o in zip(lists, offsets))
    out = [[] for _ in range(n)]
    for l, o in zip(lists, offsets):
        for i, lay in enumerate(l):
            out[o + i].extend(lay)
    return [lay for lay in out]


def order_outer_support(code: CSSCode, row: np.ndarray, ordering: str):
    """Split a weight-8 outer check into q1..q8. Each block contributes 2 qubits (x,y).
    'paired': q1..q4 = first qubit of blocks A..D, q5..q8 = second qubits of A..D
              (cat qubit i touches both qubits of one block).
    'shifted': q5..q8 = second qubits of B,C,D,A (cat qubit i touches two blocks)."""
    supp = np.nonzero(row)[0]
    blocks = []
    for blk in code.blocks:
        s = [q for q in blk if q in set(supp)]
        if s:
            assert len(s) == 2, s
            blocks.append(s)
    assert len(blocks) == 4
    first = [b[0] for b in blocks]
    second = [b[1] for b in blocks]
    if ordering == "paired":
        return first + second
    if ordering == "shifted":
        return first + second[1:] + second[:1]
    raise ValueError(ordering)


# ---------------------------------------------------------------------------------
from itertools import product as _product

from .memory import CheckInfo, RoundRecord


def solve_outer_orientation(code: CSSCode, rows):
    """Choose, for each outer check and each of its 4 blocks, which of the block's two
    qubits is touched in data layer 1 vs 2, so that no data qubit is used twice in one
    layer across the parallel checks. Returns list of (first4, second4) per row."""
    per_row = []
    for row in rows:
        supp = set(np.nonzero(row)[0])
        blocks = [[q for q in blk if q in supp] for blk in code.blocks]
        blocks = [b for b in blocks if b]
        assert len(blocks) == 4 and all(len(b) == 2 for b in blocks)
        per_row.append(blocks)
    nvar = 4 * len(rows)
    for bits in _product([0, 1], repeat=nvar):
        used = [set(), set()]
        ok = True
        out = []
        for i, blocks in enumerate(per_row):
            first, second = [], []
            for j, (x, y) in enumerate(blocks):
                a, b = (x, y) if bits[4 * i + j] == 0 else (y, x)
                if a in used[0] or b in used[1]:
                    ok = False
                    break
                used[0].add(a); used[1].add(b)
                first.append(a); second.append(b)
            if not ok:
                break
            out.append((first, second))
        if ok:
            return out
    raise RuntimeError("no 2-layer orientation")


class C4HelixScheme:
    """Round structure of the C4-Helix memory: inner part (Fig. 8a/b, alternating) every
    round, outer part (Fig. 8c, X checks then Z checks staggered by two layers)."""

    def __init__(self, outer_rows=(0, 1, 2, 3), ordering="shifted", code=None):
        self.code = code or c4_helix_paper_table()
        c = self.code
        self.n_data = c.n
        self.outer_x_rows = [c.hx[5 + i] for i in outer_rows]
        self.outer_z_rows = [c.hz[5 + i] for i in outer_rows]
        self.outer_idx = list(outer_rows)
        self.ordering = ordering
        self.orient = solve_outer_orientation(c, self.outer_x_rows)
        self.checks = {}
        for b in range(5):
            supp = tuple(c.blocks[b])
            self.checks[("in", b, "X")] = CheckInfo(("in", b, "X"), "X", supp)
            self.checks[("in", b, "Z")] = CheckInfo(("in", b, "Z"), "Z", supp)
        for i, row in zip(self.outer_idx, self.outer_x_rows):
            self.checks[("out", "X", i)] = CheckInfo(("out", "X", i), "X",
                                                     tuple(np.nonzero(row)[0]))
        for i, row in zip(self.outer_idx, self.outer_z_rows):
            self.checks[("out", "Z", i)] = CheckInfo(("out", "Z", i), "Z",
                                                     tuple(np.nonzero(row)[0]))
        self.reset()

    def reset(self):
        self.phys = list(range(self.n_data))
        self.pool = QubitPool(self.n_data)

    def data_phys(self):
        return list(self.phys)

    def logicals(self, basis):
        L = self.code.lx if basis == "X" else self.code.lz
        return [tuple(np.nonzero(r)[0]) for r in L]

    def inner_part(self, r):
        variant = "a" if r % 2 == 0 else "b"
        lists, rec = [], RoundRecord()
        new_phys = list(self.phys)
        freed = []
        for b in range(5):
            slots = self.code.blocks[b]
            p = [self.phys[s] for s in slots]
            qa, qb = self.pool.get(), self.pool.get()
            kx, kz = ("in", b, "X", r), ("in", b, "Z", r)
            layers, new, measured = inner_c4_ops(p, qa, qb, variant, kx, kz)
            lists.append(layers)
            for s, q in zip(slots, new):
                new_phys[s] = q
            freed.extend(measured)
            rec.checks += [(("in", b, "X"), kx), (("in", b, "Z"), kz)]
        part = Part(name=f"inner{r}")
        for lay in merge_layer_lists(lists, [0] * 5):
            part.add_layer(lay)
        self.phys = new_phys
        for q in freed:
            self.pool.put(q)
        return part, rec

    def outer_part(self, r, which=None):
        """which: optional subset of outer checks [('X', i), ('Z', j)] to measure."""
        rec = RoundRecord()
        lists, offsets, used = [], [], []
        for kind, rows, off in (("X", self.outer_x_rows, 0), ("Z", self.outer_z_rows, 2)):
            for (i, row), (first, second) in zip(zip(self.outer_idx, rows), self.orient):
                if which is not None and (kind, i) not in which:
                    continue
                if self.ordering == "paired":
                    order = first + second
                elif self.ordering == "shifted":
                    order = first + second[1:] + second[:1]
                elif self.ordering == "xpair":
                    # cat qubits whose hook pairs leave a weight-1 flag pattern, (s,f3) and
                    # (f1,f2), together touch all four blocks once each
                    order = first + [second[1], second[0], second[3], second[2]]
                else:
                    raise ValueError(self.ordering)
                data = [self.phys[s] for s in order]
                anc = [self.pool.get() for _ in range(4)]
                used += anc
                key = ("out", kind, i, r)
                fkeys = [("flag", kind, i, j, r) for j in range(3)]
                lists.append(outer_check_layers(kind, data, *anc, key, fkeys))
                offsets.append(off)
                rec.checks.append((("out", kind, i), key))
                rec.flags += fkeys
        part = Part(name=f"outer{r}")
        for lay in merge_layer_lists(lists, offsets):
            part.add_layer(lay)
        for q in used:
            self.pool.put(q)
        return part, rec

    def round_parts(self, r, full=True, which=None):
        out = []
        part, rec = self.inner_part(r)
        out.append((part, rec, "inner"))
        if full:
            part, rec = self.outer_part(r, which)
            out.append((part, rec, "outer"))
        return out


# =================================================================================
# Abstract parts for the adaptive engine (virtual qubits: ('d', slot), ('a', i))
from .engine import APart, CheckDef, Protocol, compile_part


def c4_inner_apart(code: CSSCode, variant: str, name=None) -> APart:
    """All five C4 blocks measured in parallel with circuit Fig. 8(a) or 8(b)."""
    part = APart(name or f"inner_{variant}", n_anc=10)
    resets, cx_layers, relabels, meas, fb = [], [[], [], [], []], [], [], []
    for b, slots in enumerate(code.blocks):
        d = [("d", s) for s in slots]
        qa, qb = ("a", 2 * b), ("a", 2 * b + 1)
        resets += [("RX", (qa,)), ("R", (qb,))]
        if variant == "a":
            lay = [[("CX", (d[0], qb)), ("CX", (qa, d[1]))],
                   [("CX", (d[1], qb)), ("CX", (qa, d[0]))],
                   [("CX", (d[2], qb)), ("CX", (qa, d[3]))],
                   [("CX", (qb, d[3])), ("CX", (d[2], qa))]]
            sx, sz = 2, 3          # slots refreshed: d2 takes qa, d3 takes qb
        else:
            lay = [[("CX", (d[2], qb)), ("CX", (qa, d[3]))],
                   [("CX", (d[3], qb)), ("CX", (qa, d[2]))],
                   [("CX", (d[0], qb)), ("CX", (qa, d[1]))],
                   [("CX", (qb, d[1])), ("CX", (d[0], qa))]]
            sx, sz = 0, 1
        for i in range(4):
            cx_layers[i] += lay[i]
        relabels += [("RELABEL", (d[sx], qa)), ("RELABEL", (d[sz], qb))]
        kx, kz = ("in", b, "X"), ("in", b, "Z")
        # after relabel, qa names the old data qubit d[sx] (measured in X), qb the old d[sz]
        meas += [("MX", (qa,), kx), ("M", (qb,), kz)]
        fb += [("FBX", (d[sz],), kz), ("FBZ", (d[sx],), kx)]
        part.meas[kx] = ("chk", ("in", b, "X"))
        part.meas[kz] = ("chk", ("in", b, "Z"))
    part.add(resets)
    for l in cx_layers:
        part.add(l)
    part.add(relabels)
    part.add(meas)
    part.add(fb)
    return part


def c4_outer_apart(code: CSSCode, rows_idx, orient, ordering="xpair", which=None,
                   name="outer") -> APart:
    """Outer weight-8 checks, X checks then Z checks staggered by two layers (Fig. 8c)."""
    lists, offsets = [], []
    meas = {}
    a = 0
    for kind, off in (("X", 0), ("Z", 2)):
        H = code.hx if kind == "X" else code.hz
        for i, (first, second) in zip(rows_idx, orient):
            if which is not None and (kind, i) not in which:
                continue
            if ordering == "xpair":
                order = first + [second[1], second[0], second[3], second[2]]
            elif ordering == "shifted":
                order = first + second[1:] + second[:1]
            else:
                order = first + second
            q = [("d", s) for s in order]
            s_, f1, f2, f3 = ("a", a), ("a", a + 1), ("a", a + 2), ("a", a + 3)
            a += 4
            key = ("out", kind, i)
            fk = [("flag", kind, i, j) for j in range(3)]
            meas[key] = ("chk", ("out", kind, i))
            for k in fk:
                meas[k] = ("flag",)
            if kind == "X":
                L = [[("RX", (s_,)), ("R", (f1,)), ("R", (f2,)), ("R", (f3,))],
                     [("CX", (s_, f1))],
                     [("CX", (s_, f2)), ("CX", (f1, f3))],
                     [("CX", (s_, q[0])), ("CX", (f1, q[1])), ("CX", (f2, q[2])), ("CX", (f3, q[3]))],
                     [("CX", (s_, q[4])), ("CX", (f1, q[5])), ("CX", (f2, q[6])), ("CX", (f3, q[7]))],
                     [("CX", (s_, f3)), ("CX", (f1, f2))],
                     [("CX", (s_, f1))],
                     [("MX", (s_,), key), ("M", (f1,), fk[0]), ("M", (f2,), fk[1]), ("M", (f3,), fk[2])]]
            else:
                L = [[("R", (s_,)), ("RX", (f1,)), ("RX", (f2,)), ("RX", (f3,))],
                     [("CX", (f1, s_))],
                     [("CX", (f2, s_)), ("CX", (f3, f1))],
                     [("CX", (q[0], s_)), ("CX", (q[1], f1)), ("CX", (q[2], f2)), ("CX", (q[3], f3))],
                     [("CX", (q[4], s_)), ("CX", (q[5], f1)), ("CX", (q[6], f2)), ("CX", (q[7], f3))],
                     [("CX", (f3, s_)), ("CX", (f2, f1))],
                     [("CX", (f1, s_))],
                     [("M", (s_,), key), ("MX", (f1,), fk[0]), ("MX", (f2,), fk[1]), ("MX", (f3,), fk[2])]]
            lists.append(L)
            offsets.append(off)
    part = APart(name, n_anc=a)
    for lay in merge_layer_lists(lists, offsets):
        part.add(lay)
    part.meas = meas
    return part


class C4HelixProtocol(Protocol):
    """C4-Helix memory protocol on the engine.

    mode: 'full' (every round inner+outer), 'ase' (outer when any inner detection event
          in the round, or every `forced` rounds), 'never' (outer only at round 0).
    Round 0 is the projective-preparation round (always full)."""

    def __init__(self, n_rounds, basis, noise, mode="full", forced=None, outer_rows=(0, 1, 2, 3),
                 ordering="xpair", code=None):
        self.code = code or c4_helix_paper_table()
        c = self.code
        self.n_data = c.n
        self.basis = basis
        self.noise = noise
        self.n_rounds = n_rounds
        self.mode = mode
        self.forced = forced
        rows = [c.hx[5 + i] for i in outer_rows]
        orient = solve_outer_orientation(c, rows)
        self.checks = {}
        for b in range(5):
            supp = np.array(c.blocks[b])
            self.checks[("in", b, "X")] = CheckDef("X", supp)
            self.checks[("in", b, "Z")] = CheckDef("Z", supp)
        for i in outer_rows:
            self.checks[("out", "X", i)] = CheckDef("X", np.flatnonzero(c.hx[5 + i]))
            self.checks[("out", "Z", i)] = CheckDef("Z", np.flatnonzero(c.hz[5 + i]))
        L = c.lx if basis == "X" else c.lz
        self.logical_supports = [np.flatnonzero(r) for r in L]
        self.aparts = {"inner_a": c4_inner_apart(c, "a"), "inner_b": c4_inner_apart(c, "b"),
                       "outer": c4_outer_apart(c, list(outer_rows), orient, ordering)}
        self.parts = {k: compile_part(v, noise, self.n_data) for k, v in self.aparts.items()}
        self.setup()

    def check_generator(self, c):
        if c[0] == "in":
            return ("hx" if c[2] == "X" else "hz", c[1])
        return ("hx" if c[1] == "X" else "hz", 5 + c[2])

    def gate_ops(self, ptype):
        return getattr(self, "_gate_ops", {}).get(ptype)

    def plan(self, r):
        inner = "inner_a" if r % 2 == 0 else "inner_b"
        if self.mode == "full" or r == 0:
            return [(inner, None), ("outer", None)]
        if self.mode == "never":
            return [(inner, None)]
        inner_idx = [self.check_index[("in", b, t)] for b in range(5) for t in "XZ"]

        def cond(st, idx):
            trig = st.round_events[idx][:, inner_idx].any(axis=1)
            if self.forced and r % self.forced == 0:
                trig[:] = True
            return trig
        return [(inner, None), ("outer", cond)]
