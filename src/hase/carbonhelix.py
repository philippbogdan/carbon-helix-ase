"""Carbon-Helix [[60,2,12]] extraction circuits (designed here; none was published).

Three levels of checks, matching the code's concatenation C4 -> C6 (= Carbon) -> Helix:
  level 1: the 15 C4 sub-blocks {0-3},{4-7},{8-11} of each Carbon block; XXXX and ZZZZ
           measured together with the C4-Helix circuit of arXiv:2609.03194 Fig. 8(a)/(b)
           (2 ancillas, 8 CNOTs, refreshes two data qubits per sub-block per round).
  level 2: the four weight-6 Carbon checks of each block (products of three C4 logicals),
           each with a 3-qubit cat (Shor-style; 1 syndrome + 2 flag qubits, 10 CNOTs);
           each cat qubit touches two different C4 sub-blocks.
  level 3: the eight weight-16 Helix checks (products of four Carbon logicals), each with a
           4-qubit cat as in Fig. 8(c) but with four data CNOTs per cat qubit arranged as a
           Latin square, so every cat qubit touches each Carbon block exactly once, and the
           cat's parities read halfway through the data CNOTs (OUTER_DESIGNS['k4cyc']: four
           parity ancillas on the cycle s-f1-f2-f3-s, 30 CNOTs per check; the first design
           without them, 'latin4', has circuit fault distance 9). Data order: each half of a
           check's four qubits in a block holds one qubit of each C4 sub-block
           (scheduled_layers_halves).
"""
from __future__ import annotations

import numpy as np

from .codes import carbon_code, carbon_helix, CSSCode
from .engine import APart, CheckDef, Protocol, compile_part
from .c4helix import merge_layer_lists

SUB = [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11)]


def c4_layers(d, qa, qb, variant):
    """C4 circuit Fig. 8(a)/(b) on data names d[0..3] with ancillas qa (|+>), qb (|0>).
    Returns (cx_layers[4], relabels, (sx, sz))."""
    if variant == "a":
        lay = [[("CX", (d[0], qb)), ("CX", (qa, d[1]))],
               [("CX", (d[1], qb)), ("CX", (qa, d[0]))],
               [("CX", (d[2], qb)), ("CX", (qa, d[3]))],
               [("CX", (qb, d[3])), ("CX", (d[2], qa))]]
        sx, sz = 2, 3
    else:
        lay = [[("CX", (d[2], qb)), ("CX", (qa, d[3]))],
               [("CX", (d[3], qb)), ("CX", (qa, d[2]))],
               [("CX", (d[0], qb)), ("CX", (qa, d[1]))],
               [("CX", (qb, d[1])), ("CX", (d[0], qa))]]
        sx, sz = 0, 1
    return lay, [("RELABEL", (d[sx], qa)), ("RELABEL", (d[sz], qb))], (sx, sz)


def level1_apart(blocks, variant, name=None) -> APart:
    """C4-level checks of all sub-blocks of the given Carbon blocks (list of slot lists)."""
    nsub = 3 * len(blocks)
    part = APart(name or f"L1_{variant}", n_anc=2 * nsub)
    resets, cxl, rel, meas, fb = [], [[], [], [], []], [], [], []
    a = 0
    for bi, slots in enumerate(blocks):
        for si, sub in enumerate(SUB):
            d = [("d", slots[j]) for j in sub]
            qa, qb = ("a", a), ("a", a + 1)
            a += 2
            resets += [("RX", (qa,)), ("R", (qb,))]
            lay, rl, (sx, sz) = c4_layers(d, qa, qb, variant)
            for i in range(4):
                cxl[i] += lay[i]
            rel += rl
            blk = slots[0] // 12
            kx, kz = ("c4", blk, si, "X"), ("c4", blk, si, "Z")
            meas += [("MX", (qa,), kx), ("M", (qb,), kz)]
            fb += [("FBX", (d[sz],), kz), ("FBZ", (d[sx],), kx)]
            part.meas[kx] = ("chk", kx)
            part.meas[kz] = ("chk", kz)
    part.add(resets)
    for l in cxl:
        part.add(l)
    part.add(rel)
    part.add(meas)
    part.add(fb)
    return part


def weight6_supports(carbon: CSSCode):
    """Return dict name -> (kind, local support tuple) for the four weight-6 checks."""
    out = {}
    for kind, H in (("X", carbon.hx), ("Z", carbon.hz)):
        j = 0
        for row in H:
            s = tuple(np.flatnonzero(row))
            if len(s) == 6:
                out[(kind, j)] = s
                j += 1
    return out


def cat3_layers(kind, q, s_, c1, c2, key, fk):
    """3-qubit cat measurement of a weight-6 check; q = 6 data names in the order
    (layer A: s,c1,c2) + (layer B: s,c1,c2)."""
    if kind == "X":
        return [[("RX", (s_,)), ("R", (c1,)), ("R", (c2,))],
                [("CX", (s_, c1))],
                [("CX", (s_, c2))],
                [("CX", (s_, q[0])), ("CX", (c1, q[1])), ("CX", (c2, q[2]))],
                [("CX", (s_, q[3])), ("CX", (c1, q[4])), ("CX", (c2, q[5]))],
                [("CX", (s_, c2))],
                [("CX", (s_, c1))],
                [("MX", (s_,), key), ("M", (c1,), fk[0]), ("M", (c2,), fk[1])]]
    return [[("R", (s_,)), ("RX", (c1,)), ("RX", (c2,))],
            [("CX", (c1, s_))],
            [("CX", (c2, s_))],
            [("CX", (q[0], s_)), ("CX", (q[1], c1)), ("CX", (q[2], c2))],
            [("CX", (q[3], s_)), ("CX", (q[4], c1)), ("CX", (q[5], c2))],
            [("CX", (c2, s_))],
            [("CX", (c1, s_))],
            [("M", (s_,), key), ("MX", (c1,), fk[0]), ("MX", (c2,), fk[1])]]


def order6(supp):
    """Order a weight-6 Carbon check (two qubits in each C4 sub-block) so that layer A
    takes one qubit of S1,S2,S3 for (s,c1,c2) and layer B takes the other qubit of
    S2,S3,S1: each cat qubit touches two different sub-blocks."""
    per = []
    for sub in SUB:
        x = [q for q in supp if q in sub]
        assert len(x) == 2, (supp, sub)
        per.append(x)
    A = [per[0][0], per[1][0], per[2][0]]
    B = [per[1][1], per[2][1], per[0][1]]
    return A + B


def level2_apart(blocks, carbon: CSSCode, name=None, stagger=True) -> APart:
    """Weight-6 checks of the given Carbon blocks; X checks and Z checks of a block are
    staggered by two layers (as the C4-Helix outer checks)."""
    w6 = weight6_supports(carbon)
    lists, offsets, meas = [], [], {}
    a = 0
    for slots in blocks:
        blk = slots[0] // 12
        for (kind, j), supp in sorted(w6.items()):
            q = [("d", slots[x]) for x in order6(supp)]
            s_, c1, c2 = ("a", a), ("a", a + 1), ("a", a + 2)
            a += 3
            key = ("c6", blk, kind, j)
            fk = [("f6", blk, kind, j, t) for t in range(2)]
            meas[key] = ("chk", key)
            for k in fk:
                meas[k] = ("flag",)
            lists.append(cat3_layers(kind, q, s_, c1, c2, key, fk))
            # X checks of a block start together; Z checks two layers later; data CNOT
            # layers of the two X checks (and of the two Z checks) must not collide
            offsets.append((0 if kind == "X" else 4) + (2 * j if not stagger else 2 * j))
    part = APart(name or "L2", n_anc=a)
    for lay in merge_layer_lists(lists, offsets):
        part.add(lay)
    part.meas = meas
    return part


LATIN = [[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]]   # layer -> cat role -> block


def outer16_order(code: CSSCode, row):
    """Data order for a weight-16 check: 4 layers x (s,f1,f2,f3); cat role r in layer l
    touches block LATIN[l][r] (its l-th qubit in support order)."""
    supp = set(np.flatnonzero(row))
    blocks = [[q for q in blk if q in supp] for blk in code.blocks]
    blocks = [b for b in blocks if b]
    assert len(blocks) == 4 and all(len(b) == 4 for b in blocks)
    order = []
    for l in range(4):
        for r in range(4):
            order.append(blocks[LATIN[l][r]][l])
    return order


def cat4_16_layers(kind, q, s_, f1, f2, f3, key, fk):
    cx = (lambda a, b: ("CX", (a, b))) if kind == "X" else (lambda a, b: ("CX", (b, a)))
    if kind == "X":
        L = [[("RX", (s_,)), ("R", (f1,)), ("R", (f2,)), ("R", (f3,))]]
    else:
        L = [[("R", (s_,)), ("RX", (f1,)), ("RX", (f2,)), ("RX", (f3,))]]
    L += [[cx(s_, f1)], [cx(s_, f2), cx(f1, f3)]]
    for l in range(4):
        L.append([cx(s_, q[4 * l]), cx(f1, q[4 * l + 1]), cx(f2, q[4 * l + 2]), cx(f3, q[4 * l + 3])])
    L += [[cx(s_, f3), cx(f1, f2)], [cx(s_, f1)]]
    if kind == "X":
        L.append([("MX", (s_,), key), ("M", (f1,), fk[0]), ("M", (f2,), fk[1]), ("M", (f3,), fk[2])])
    else:
        L.append([("M", (s_,), key), ("MX", (f1,), fk[0]), ("MX", (f2,), fk[1]), ("MX", (f3,), fk[2])])
    return L


PAIRS_K6 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def _parity_schedule(pairs):
    """CNOT layers reading cat-qubit parities: list of layers of (role, parity index), each
    role and each parity ancilla at most once per layer (bipartite edge colouring)."""
    if not pairs:
        return []
    edges = [(r, j) for j, pr in enumerate(pairs) for r in pr]
    deg = max(max(sum(1 for e in edges if e[0] == r) for r in range(4)), 2)
    cols = bipartite_edge_coloring(edges, deg)
    layers = [[] for _ in range(deg)]
    for e, c in zip(edges, cols):
        layers[c].append(e)
    return [lay for lay in layers if lay]


def cat4_16_parity_layers(kind, q, cat, par, key, fk, design):
    """4-qubit cat measurement of a weight-16 check with cat-parity readouts.

    q: the 16 data names, layer l and cat role r at q[4l + r]; cat = (s, f1, f2, f3);
    par: parity ancillas (reused across phases). design: dict with role-pair lists
    'pre' (read after the cat preparation), 'mid' (between data layers 1 and 2) and 'end'
    (after data layer 3); the cat is then unencoded and its three flags measured.
    Measurement keys: key (the check), then fk in order: pre, mid, end parities, flags.
    A cat qubit's error that persists over the data layers spreads to every later data
    qubit it touches; flags read at the end only cannot tell an error that came and went
    (two faults, up to four data errors) from none, which is what the mid readout
    detects (docs/methods.md, outer-check gadget)."""
    s_, f1, f2, f3 = cat
    if kind == "X":
        cx = lambda a, b: ("CX", (a, b))
        rc0, rc, ra, ma, ms, mf = "RX", "R", "R", "M", "MX", "M"
    else:
        cx = lambda a, b: ("CX", (b, a))
        rc0, rc, ra, ma, ms, mf = "R", "RX", "RX", "MX", "M", "MX"
    fk = list(fk)
    L = [[(rc0, (s_,)), (rc, (f1,)), (rc, (f2,)), (rc, (f3,))]]
    L += [[cx(s_, f1)], [cx(s_, f2), cx(f1, f3)]]
    pending_meas = []      # parity ancillas to measure in the next layer
    pending_reset = set()  # ancillas measured, to reset before reuse

    def phase(pairs):
        nonlocal pending_meas
        if not pairs:
            return
        sched = _parity_schedule(pairs)
        # resets of this phase's ancillas: in the layer before their first CNOT if that
        # layer does not touch them, else in an extra layer
        need = [par[j] for j in range(len(pairs))]
        prev = L[-1]
        used = {x for op in prev for x in op[1]}
        res = [(ra, (a,)) for a in need]
        if not (set(need) & used):
            prev.extend(res)
        else:
            L.append(res)
        for lay in sched:
            ops = [cx(cat[r], par[j]) for r, j in lay]
            L.append(ops)
        pending_meas = [(ma, (par[j],), fk.pop(0)) for j in range(len(pairs))]

    def data_layer(l):
        nonlocal pending_meas
        ops = [cx(cat[r], q[4 * l + r]) for r in range(4)]
        L.append(ops + pending_meas)
        pending_meas = []

    phase(design.get("pre", []))
    data_layer(0)
    data_layer(1)
    phase(design.get("mid", []))
    data_layer(2)
    data_layer(3)
    phase(design.get("end", []))
    L.append([cx(s_, f3), cx(f1, f2)] + pending_meas)
    pending_meas = []
    L.append([cx(s_, f1)])
    L.append([(ms, (s_,), key), (mf, (f1,), fk[0]), (mf, (f2,), fk[1]), (mf, (f3,), fk[2])])
    assert len(fk) == 3
    return L


OUTER_DESIGNS = {
    "latin4": None,
    "k4cyc": {"mid": [(0, 1), (1, 2), (2, 3), (0, 3)]},
    "k6": {"mid": PAIRS_K6},
    "k5pre": {"pre": [(0, 3)], "mid": [p for p in PAIRS_K6 if p != (0, 3)]},
    "k6pre": {"pre": [(0, 3)], "mid": PAIRS_K6},
}
# the same with the data order of the first gadget (edge colouring without the half
# constraint of scheduled_layers_halves)
for _k in ("k4cyc", "k6", "k5pre"):
    OUTER_DESIGNS[_k + "_e"] = dict(OUTER_DESIGNS[_k], order="edge")
DEFAULT_OUTER = "k4cyc"


def outer_design_size(design):
    """(number of parity ancillas, number of parity flags) of an outer-check design."""
    if design is None:
        return 0, 0
    ph = [design.get(k, []) for k in ("pre", "mid", "end")]
    return max(len(p) for p in ph), sum(len(p) for p in ph)


def outer_rows(code: CSSCode, n_rows=4):
    xs = [i for i, l in enumerate(code.hx_level) if l == "outer"][:n_rows]
    zs = [i for i, l in enumerate(code.hz_level) if l == "outer"][:n_rows]
    return xs, zs


def level3_apart(code: CSSCode, n_rows=4, which=None, name="L3", z_offset=4) -> APart:
    """Outer weight-16 checks. X checks in parallel, Z checks after them (their data CNOT
    layers must not overlap with the X checks' on shared data qubits)."""
    xs, zs = outer_rows(code, n_rows)
    lists, offsets, meas = [], [], {}
    a = 0
    for kind, rows in (("X", xs), ("Z", zs)):
        H = code.hx if kind == "X" else code.hz
        for i, ri in enumerate(rows):
            if which is not None and (kind, i) not in which:
                continue
            q = [("d", s) for s in outer16_order(code, H[ri])]
            s_, f1, f2, f3 = ("a", a), ("a", a + 1), ("a", a + 2), ("a", a + 3)
            a += 4
            key = ("out", kind, i)
            fk = [("f16", kind, i, t) for t in range(3)]
            meas[key] = ("chk", key)
            for k in fk:
                meas[k] = ("flag",)
            lists.append(cat4_16_layers(kind, q, s_, f1, f2, f3, key, fk))
            offsets.append(0 if kind == "X" else z_offset)
    part = APart(name, n_anc=a)
    merged = merge_layer_lists(lists, offsets)
    # resolve data-qubit collisions by splitting layers greedily
    for lay in merged:
        pending = list(lay)
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
    part.meas = meas
    return part


class CarbonHelixProtocol(Protocol):
    """Carbon-Helix memory protocol with hierarchical adaptive extraction.

    l2: 'always' | 'on_event' (block's C4 checks had an event this round) | 'never'
    l3: 'always' | 'any' (any C4 event anywhere this round) | 'ambiguous' (some block's
        Carbon syndrome change since its last weight-6 measurement is not that of a
        single-qubit error, or a flag fired) | 'never'
    T2, T3: forced intervals (all blocks' level 2 / level 3 every T2 / T3 rounds;
            None = never forced). Round 0 (projective preparation) is always full.
    Per-block level-2 parts run in parallel under depth-based (untimed) noise: a block that
    is not measured idles for the level-2 duration (part W2_b) if any block is measured.
    Under timed (Helios-like) noise, gate batches serialise and per-block parts are
    sequential, each with memory noise on every data qubit.
    """

    def __init__(self, n_rounds, basis, noise, l2="always", l3="always", T2=None, T3=None,
                 n_outer_rows=4, code=None, schedule="edge", outer=None, outer_order="XZ"):
        self.outer = outer or DEFAULT_OUTER
        self.outer_order = outer_order
        self.outer_design = OUTER_DESIGNS[self.outer]
        self.code = code or carbon_helix()
        self.carbon = carbon_code()
        c = self.code
        self.n_data = c.n
        self.basis, self.noise, self.n_rounds = basis, noise, n_rounds
        self.l2, self.l3, self.T2, self.T3 = l2, l3, T2, T3
        self.timed = getattr(noise, "timed", False)
        self.checks = {}
        w6 = weight6_supports(self.carbon)
        for b, blk in enumerate(c.blocks):
            for si, sub in enumerate(SUB):
                supp = np.array([blk[j] for j in sub])
                self.checks[("c4", b, si, "X")] = CheckDef("X", supp)
                self.checks[("c4", b, si, "Z")] = CheckDef("Z", supp)
            for (kind, j), supp in sorted(w6.items()):
                self.checks[("c6", b, kind, j)] = CheckDef(kind, np.array([blk[x] for x in supp]))
        xs, zs = outer_rows(c, n_outer_rows)
        for i, ri in enumerate(xs):
            self.checks[("out", "X", i)] = CheckDef("X", np.flatnonzero(c.hx[ri]))
        for i, ri in enumerate(zs):
            self.checks[("out", "Z", i)] = CheckDef("Z", np.flatnonzero(c.hz[ri]))
        L = c.lx if basis == "X" else c.lz
        self.logical_supports = [np.flatnonzero(r) for r in L]
        blocks = c.blocks
        if schedule == "edge":
            L2all = level2_sched_apart(blocks, self.carbon, name="L2_all")
            if self.outer_design is None:
                L3 = level3_sched_apart(c, n_outer_rows, name="L3")
            else:
                xs_, zs_ = outer_rows(c, n_outer_rows)
                L3 = outer_apart(c, [("X", i) for i in range(len(xs_))] +
                                 [("Z", i) for i in range(len(zs_))], "L3", self.outer_design,
                                 order=self.outer_order)
            L2b = {b: level2_sched_apart([blocks[b]], self.carbon, name=f"L2_{b}") for b in range(5)}
        else:
            L2all = level2_apart(blocks, self.carbon, name="L2_all")
            L3 = level3_apart(c, n_outer_rows)
            L2b = {b: level2_apart([blocks[b]], self.carbon, name=f"L2_{b}") for b in range(5)}
        self.aparts = {"L1_a": level1_apart(blocks, "a"), "L1_b": level1_apart(blocks, "b"),
                       "L2_all": L2all, "L3": L3}
        self.parts = {k: compile_part(v, noise, self.n_data) for k, v in self.aparts.items()}
        for b in range(5):
            self.aparts[f"L2_{b}"] = L2b[b]
            self.parts[f"L2_{b}"] = compile_part(L2b[b], noise, self.n_data, idle_slots=blocks[b])
            wait = APart(f"W2_{b}", n_anc=0)
            for _ in range(len(L2b[b].layers)):
                wait.add([])
            if self.timed:
                wait.layer_dts = [noise.layer_duration(
                    sum(1 for op in lay if op[0] in ("CX", "CZ")), 0,
                    sum(1 for op in lay if op[0] in ("M", "MX")),
                    sum(1 for op in lay if op[0] in ("R", "RX"))) if lay else 0.0
                    for lay in L2b[b].layers]
            self.aparts[f"W2_{b}"] = wait
            self.parts[f"W2_{b}"] = compile_part(wait, noise, self.n_data, idle_slots=blocks[b])
        self.setup()
        self._prep_rules()

    # ------------------------------------------------------------------ rule helpers
    def _prep_rules(self):
        ci = self.check_index
        self.c4_idx = {b: [ci[("c4", b, si, t)] for si in range(3) for t in "XZ"] for b in range(5)}
        self.c6_idx = {b: [ci[k] for k in self.checks if k[0] == "c6" and k[1] == b] for b in range(5)}
        # Carbon syndrome (c4 X,Z of 3 subs + 4 c6) of every single-qubit Pauli on a block
        carb = [("c4", 0, si, t) for si in range(3) for t in "XZ"] + \
               [k for k in self.checks if k[0] == "c6" and k[1] == 0]
        self.carb_names = carb
        S = np.array([self.S[ci[k]] for k in carb])
        n = self.n_data
        sigs = {tuple(np.zeros(len(carb), dtype=np.uint8))}
        for q in range(12):
            for P in ("X", "Z", "Y"):
                e = np.zeros(2 * n, dtype=np.uint8)
                if P in "XY":
                    e[q] = 1
                if P in "ZY":
                    e[n + q] = 1
                sigs.add(tuple((S @ e) % 2))
        self.weight1_sigs = sigs

    def _carbon_syndrome(self, st, idx, b):
        ci = self.check_index
        names = [("c4", b, si, t) for si in range(3) for t in "XZ"] + \
                [k for k in self.checks if k[0] == "c6" and k[1] == b]
        return st.last[np.ix_(idx, [ci[k] for k in names])]

    def _ambiguous(self, st, idx):
        """True where some block's Carbon syndrome change since its reference is not that
        of a single-qubit error, or a flag fired; updates the references."""
        trig = st.flag_event[idx].astype(bool).copy()
        ref = st.custom.setdefault("carb_ref", {})
        for b in range(5):
            cur = self._carbon_syndrome(st, idx, b)
            prev = ref.get(b)
            if prev is None:
                prev = np.zeros((st.frame.shape[0], cur.shape[1]), dtype=np.uint8)
                ref[b] = prev
            d = cur ^ prev[idx]
            ok = np.array([tuple(row) in self.weight1_sigs for row in d])
            trig |= ~ok
            prev[idx] = cur
        return trig

    def plan(self, r):
        L1 = "L1_a" if r % 2 == 0 else "L1_b"
        full = r == 0
        steps = [(L1, None)]
        force2 = full or (self.T2 is not None and r % self.T2 == 0) or self.l2 == "always"
        force3 = full or (self.T3 is not None and r % self.T3 == 0) or self.l3 == "always"
        if force2 or force3:
            steps.append(("L2_all", None))
        elif self.l2 == "on_event":
            def need(st, idx):
                return np.stack([st.round_events[np.ix_(idx, self.c4_idx[b])].any(axis=1)
                                 for b in range(5)], axis=1)
            for b in range(5):
                def cond(st, idx, b=b):
                    return need(st, idx)[:, b]
                steps.append((f"L2_{b}", cond))
            if True:
                for b in range(5):
                    def condw(st, idx, b=b):
                        nd = need(st, idx)
                        return nd.any(axis=1) & ~nd[:, b]
                    steps.append((f"W2_{b}", condw))
        if force3:
            if r == 0 or self.l3 == "always":
                steps.append(("L3", None))
            else:
                # forced full round: refresh the online references (the full measurement
                # resolves every block) and clear pending deferred escalations
                def cond_forced(st, idx):
                    if self.l3 in ("ambiguous",) or self.l3.startswith("deferred"):
                        self._ambiguous(st, idx)
                    pend = st.custom.get("pending_since")
                    if pend is not None:
                        pend[idx] = -1
                    return np.ones(len(idx), dtype=bool)
                steps.append(("L3", cond_forced))
        elif self.l3 == "any":
            allc4 = sum(self.c4_idx.values(), [])

            def cond3(st, idx):
                return st.round_events[np.ix_(idx, allc4)].any(axis=1)
            steps.append(("L3", cond3))
        elif self.l3.startswith("deferred"):
            # deferred<D>: an inner event schedules the outer checks within D rounds (one
            # outer measurement serves every event since the last one); an ambiguous change
            # escalates immediately
            D = int(self.l3[len("deferred"):] or 2)
            allc4 = sum(self.c4_idx.values(), [])

            def cond3(st, idx):
                trig = self._ambiguous(st, idx)
                pend = st.custom.setdefault("pending_since", np.full(st.frame.shape[0], -1))
                ev = st.round_events[np.ix_(idx, allc4)].any(axis=1)
                p = pend[idx]
                p = np.where((p < 0) & ev, r, p)
                due = (p >= 0) & (r - p >= D - 1)
                trig = trig | due
                p = np.where(trig, -1, p)
                pend[idx] = p
                return trig
            steps.append(("L3", cond3))
        elif self.l3 == "ambiguous":
            def cond3(st, idx):
                trig = st.flag_event[idx].astype(bool).copy()
                ref = st.custom.setdefault("carb_ref", {})
                for b in range(5):
                    cur = self._carbon_syndrome(st, idx, b)
                    prev = ref.get(b)
                    if prev is None:
                        prev = np.zeros((st.frame.shape[0], cur.shape[1]), dtype=np.uint8)
                        ref[b] = prev
                    d = cur ^ prev[idx]
                    ok = np.array([tuple(row) in self.weight1_sigs for row in d])
                    trig |= ~ok
                    # the reference becomes the current syndrome: a confident block has
                    # been corrected in the Pauli frame; an ambiguous one is resolved by
                    # the full measurement that this escalation triggers
                    prev[idx] = cur
                return trig
            steps.append(("L3", cond3))
        return steps


class CarbonBlockProtocol(Protocol):
    """A single [[12,2,4]] Carbon block with level-1 and level-2 extraction every round
    (used to check the inner circuits' fault distance)."""

    def __init__(self, n_rounds, basis, noise):
        self.carbon = carbon_code()
        c = self.carbon
        self.n_data, self.basis, self.noise, self.n_rounds = 12, basis, noise, n_rounds
        self.checks = {}
        blk = list(range(12))
        for si, sub in enumerate(SUB):
            self.checks[("c4", 0, si, "X")] = CheckDef("X", np.array(sub))
            self.checks[("c4", 0, si, "Z")] = CheckDef("Z", np.array(sub))
        for (kind, j), supp in sorted(weight6_supports(c).items()):
            self.checks[("c6", 0, kind, j)] = CheckDef(kind, np.array(supp))
        L = c.lx if basis == "X" else c.lz
        self.logical_supports = [np.flatnonzero(r) for r in L]
        self.aparts = {"L1_a": level1_apart([blk], "a"), "L1_b": level1_apart([blk], "b"),
                       "L2": level2_sched_apart([blk], c, name="L2")}
        self.parts = {k: compile_part(v, noise, 12) for k, v in self.aparts.items()}
        self.setup()

    def plan(self, r):
        return [("L1_a" if r % 2 == 0 else "L1_b", None), ("L2", None)]


# ----------------------------------------------------------------------------- schedules
def bipartite_edge_coloring(edges, ncolors):
    """Proper edge colouring of a bipartite multigraph given as list of (u, v) with
    ncolors >= max degree (Konig). Alternating-path algorithm. Returns list of colours."""
    from collections import defaultdict
    col = [-1] * len(edges)
    at = defaultdict(dict)          # vertex -> colour -> edge index
    for e, (u, v) in enumerate(edges):
        u, v = ("L", u), ("R", v)
        fu = [c for c in range(ncolors) if c not in at[u]]
        fv = [c for c in range(ncolors) if c not in at[v]]
        common = [c for c in fu if c in fv]
        if common:
            c = common[0]
        else:
            a, b = fu[0], fv[0]
            # flip the a/b alternating path starting at v
            path = []
            x, cur = v, a
            while cur in at[x]:
                ei = at[x][cur]
                path.append(ei)
                uu, vv = edges[ei]
                y = ("L", uu) if x == ("R", vv) else ("R", vv)
                x = y
                cur = b if cur == a else a
            for ei in path:
                uu, vv = ("L", edges[ei][0]), ("R", edges[ei][1])
                old = col[ei]
                del at[uu][old]
                del at[vv][old]
            for ei in path:
                uu, vv = ("L", edges[ei][0]), ("R", edges[ei][1])
                new = b if col[ei] == a else a
                col[ei] = new
            for ei in path:
                uu, vv = ("L", edges[ei][0]), ("R", edges[ei][1])
                at[uu][col[ei]] = ei
                at[vv][col[ei]] = ei
            c = a
        col[e] = c
        at[u][c] = e
        at[v][c] = e
    return col


def scheduled_layers(groups, nlayers):
    """groups: list over checks of list over 'parts' (blocks or sub-blocks) of qubit
    lists, each part having exactly nlayers qubits. Returns per check a list over layers of
    the qubit used by each part in that layer, conflict-free across checks."""
    edges, keys = [], []
    for ci, parts in enumerate(groups):
        for pi, qs in enumerate(parts):
            assert len(qs) == nlayers
            for q in qs:
                edges.append(((ci, pi), q))
                keys.append((ci, pi, q))
    cols = bipartite_edge_coloring(edges, nlayers)
    out = [[[None] * len(parts) for _ in range(nlayers)] for parts in groups]
    for (ci, pi, q), c in zip(keys, cols):
        assert out[ci][c][pi] is None
        out[ci][c][pi] = q
    return out


def scheduled_layers_halves(groups):
    """As scheduled_layers with 4 layers, for parts of 4 qubits that lie two by two in two
    C4 sub-blocks (the Carbon logicals of the outer checks): data layers 0-1 and 2-3 each
    take one qubit of each sub-block, so that the data error a cat leaves from a time cut
    in the middle of the gadget is a single-qubit error in each sub-block (seen by the
    level-1 checks) and never a C4 logical (silent at level 1). Integer program (HiGHS)."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    import scipy.sparse as sp
    ents = [(ci, pi, q) for ci, parts in enumerate(groups) for pi, qs in enumerate(parts) for q in qs]
    E = len(ents)
    nv = 4 * E
    rows, cols, lb, ub = [], [], [], []
    r = 0

    def add(idx, lo, hi):
        nonlocal r
        for i in idx:
            rows.append(r)
            cols.append(i)
        lb.append(lo)
        ub.append(hi)
        r += 1
    for e in range(E):
        add([4 * e + l for l in range(4)], 1, 1)
    by_part, by_q = {}, {}
    for e, (ci, pi, q) in enumerate(ents):
        by_part.setdefault((ci, pi), []).append(e)
        by_q.setdefault(q, []).append(e)
    for key, es in by_part.items():
        assert len(es) == 4
        for l in range(4):
            add([4 * e + l for e in es], 1, 1)
        subs = sorted({(ents[e][2] % 12) // 4 for e in es})
        assert len(subs) == 2, subs
        A = [e for e in es if (ents[e][2] % 12) // 4 == subs[0]]
        add([4 * e + l for e in A for l in (0, 1)], 1, 1)
    for q, es in by_q.items():
        for l in range(4):
            add([4 * e + l for e in es], 0, 1)
    M = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(r, nv))
    res = milp(np.zeros(nv), constraints=LinearConstraint(M, lb, ub), integrality=np.ones(nv),
               bounds=Bounds(0, 1), options={"time_limit": 120})
    assert res.x is not None, res.message
    x = np.round(res.x).astype(int).reshape(E, 4)
    out = [[[None] * len(parts) for _ in range(4)] for parts in groups]
    for e, (ci, pi, q) in enumerate(ents):
        out[ci][int(np.argmax(x[e]))][pi] = q
    return out


def level3_sched_apart(code: CSSCode, n_rows=4, name="L3s") -> APart:
    """Outer weight-16 checks with conflict-free 4-layer data schedules (X checks in
    parallel, then Z checks), 4-qubit cats whose roles visit every block once."""
    xs, zs = outer_rows(code, n_rows)
    meas = {}
    a = 0
    per_kind = {}
    for kind, rows in (("X", xs), ("Z", zs)):
        H = code.hx if kind == "X" else code.hz
        groups = []
        for ri in rows:
            supp = set(np.flatnonzero(H[ri]))
            parts = [[q for q in blk if q in supp] for blk in code.blocks]
            groups.append([p for p in parts if p])
        per_kind[kind] = (rows, scheduled_layers(groups, 4))
    lists, offsets = [], []
    for kind, off in (("X", 0), ("Z", 4)):
        rows, sched = per_kind[kind]
        for i, s in enumerate(sched):
            # layer l: cat role r -> part (block) (r + l) % 4 (Latin square)
            q = []
            for l in range(4):
                for r in range(4):
                    q.append(("d", s[l][(r + l) % 4]))
            s_, f1, f2, f3 = ("a", a), ("a", a + 1), ("a", a + 2), ("a", a + 3)
            a += 4
            key = ("out", kind, i)
            fk = [("f16", kind, i, t) for t in range(3)]
            meas[key] = ("chk", key)
            for k in fk:
                meas[k] = ("flag",)
            lists.append(cat4_16_layers(kind, q, s_, f1, f2, f3, key, fk))
            offsets.append(off)
    part = APart(name, n_anc=a)
    for lay in merge_layer_lists(lists, offsets):
        part.add(lay)
    part.meas = meas
    return part


def outer_apart(code: CSSCode, rows, name, design, order="XZ") -> APart:
    """Outer weight-16 checks rows = [(kind, i), ...] measured with the cat-parity gadget
    `design` (cat4_16_parity_layers): conflict-free 4-layer data schedules per kind (cat role
    r in data layer l touches block (r + l) % 4), all checks of the first kind in `order` in
    parallel, then all of the other kind on the same ancillas."""
    xs, zs = outer_rows(code, 4)
    npar, nflag = outer_design_size(design)
    meas = {}
    lists, offsets = [], []
    n_anc = 0
    off = 0
    for kind in order:
        idx = [i for k, i in rows if k == kind]
        if not idx:
            continue
        H = code.hx if kind == "X" else code.hz
        allrows = xs if kind == "X" else zs
        groups = []
        for i in idx:
            supp = set(np.flatnonzero(H[allrows[i]]))
            parts = [[q for q in blk if q in supp] for blk in code.blocks]
            groups.append([p for p in parts if p])
        sched = (scheduled_layers(groups, 4) if design.get("order") == "edge"
                 else scheduled_layers_halves(groups))
        a = 0
        length = 0
        for i, s in zip(idx, sched):
            q = [("d", s[l][(r + l) % 4]) for l in range(4) for r in range(4)]
            cat = tuple(("a", a + j) for j in range(4))
            par = tuple(("a", a + 4 + j) for j in range(npar))
            a += 4 + npar
            key = ("out", kind, i)
            fk = [("f16", kind, i, t) for t in range(nflag + 3)]
            meas[key] = ("chk", key)
            for k in fk:
                meas[k] = ("flag",)
            lay = cat4_16_parity_layers(kind, q, cat, par, key, fk, design)
            lists.append(lay)
            offsets.append(off)
            length = max(length, len(lay))
        n_anc = max(n_anc, a)
        off += length
    part = APart(name, n_anc=n_anc)
    for lay in merge_layer_lists(lists, offsets):
        part.add(lay)
    part.meas = meas
    return part


def level2_sched_apart(blocks, carbon: CSSCode, name=None, reuse=True, kinds=("X", "Z")) -> APart:
    """Weight-6 checks of the given blocks with conflict-free 2-layer data schedules:
    X checks of a block in parallel, then Z checks; 3-qubit cats whose roles visit two
    different C4 sub-blocks. reuse=True: the Z checks reuse the X checks' six ancillas
    after they are measured (6 ancillas per block, so all five blocks need 30 ancillas
    and fit next to the 60 data qubits on 98); reuse=False: 12 ancillas per block.
    kinds: the check types measured (('X',) or ('Z',): one type only)."""
    w6 = weight6_supports(carbon)
    lists, offsets, meas = [], [], {}
    a = 0
    for slots in blocks:
        blk = slots[0] // 12
        a_block = a
        for kind, off in (("X", 0), ("Z", 8 if reuse else 2)):
            if kind not in kinds:
                continue
            if len(kinds) == 1:
                off = 0
            if reuse:
                a = a_block
            items = sorted((j, supp) for (k, j), supp in w6.items() if k == kind)
            groups = [[[x for x in supp if x in sub] for sub in SUB] for j, supp in items]
            sched = scheduled_layers(groups, 2)
            for (j, supp), s in zip(items, sched):
                # layer l: cat role r -> sub-block (r + l) % 3
                q = []
                for l in range(2):
                    for r in range(3):
                        q.append(("d", slots[s[l][(r + l) % 3]]))
                s_, c1, c2 = ("a", a), ("a", a + 1), ("a", a + 2)
                a += 3
                key = ("c6", blk, kind, j)
                fk = [("f6", blk, kind, j, t) for t in range(2)]
                meas[key] = ("chk", key)
                for k in fk:
                    meas[k] = ("flag",)
                lists.append(cat3_layers(kind, q, s_, c1, c2, key, fk))
                offsets.append(off)
        if reuse:
            a = a_block + 6
    part = APart(name or "L2s", n_anc=a)
    for lay in merge_layer_lists(lists, offsets):
        part.add(lay)
    part.meas = meas
    return part
