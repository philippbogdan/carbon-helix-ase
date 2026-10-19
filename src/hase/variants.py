"""Carbon-Helix extraction variants beyond the main campaign's rules (frontier pushes).

CarbonHelixRotating: no periodic full measurement. Every round after the preparation
measures the inner checks (L1), the weight-6 checks of blocks with an L1 event (L2), and a
rotating share of the outer checks: in round r the group r mod `period` of the eight
weight-16 checks (period 4: one X and one Z check per round; period 8: one check per round;
each outer check every `period` rounds). A full outer measurement (L3) happens only on escalation (ambiguous-event rule),
replacing that round's rotating share. Rationale (docs/interval_theory.md): a silent inner
logical error anticommutes with several outer checks, so the rotation detects it within a
round or two - what a forced full measurement every T rounds achieves only after up to T
rounds, at a quarter of the outer checks' cost per round.
"""
from __future__ import annotations

import numpy as np

from .carbonhelix import (CarbonHelixProtocol, cat4_16_layers, level2_sched_apart, merge_layer_lists,
                          outer_apart, outer_rows, scheduled_layers)
from .engine import APart, compile_part


def level3_rows_apart(code, rows, name):
    """Weight-16 outer checks for the given (kind, i) pairs (i indexes the outer rows of
    that kind), with the same 4-qubit cats and conflict-free 4-layer data schedules as the
    full outer measurement; X checks first, Z checks after."""
    xs, zs = outer_rows(code, 4)
    meas = {}
    a = 0
    lists, offsets = [], []
    for kind, off in (("X", 0), ("Z", 4)):
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
        sched = scheduled_layers(groups, 4)
        for i, s in zip(idx, sched):
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


class CarbonHelixRotating(CarbonHelixProtocol):
    """fast_l2: the weight-6 checks of a block measured with X and Z checks in parallel
    (12 ancillas per block, 10 layers instead of 16, 46 ms less on Helios) when at most three
    blocks need them in a round (60 + 36 = 96 qubits); with four or five, the ancilla-reusing
    schedule (the base protocol's parts) keeps the round within the 98 qubits."""

    def __init__(self, n_rounds, basis, noise, period=4, escalate="ambiguous", l2="on_event",
                 schedule="edge", fast_l2=False, outer=None, l2_type=False, T2=None, T3=None, rotate=True):
        # rotate=False: no rotating outer checks (the periodic rule with parallel weight-6
        # checks: outer checks only every T3 rounds, weight-6 checks every T2 and on events)
        self.period = period
        self.fast_l2 = fast_l2
        # l2_type: a block whose inner events are all of one type (X-type C4 checks fired: a Z
        # error; Z-type: an X error) measures only its weight-6 checks of that type (fast
        # regime; 73 instead of 100 ms on Helios, 20 instead of 40 two-qubit gates)
        self.l2_type = l2_type and fast_l2
        super().__init__(n_rounds, basis, noise, l2=l2, l3=escalate, T2=T2, T3=T3,
                         schedule=schedule, outer=outer)
        # the 8 outer checks in the order X0 Z0 X1 Z1 X2 Z2 X3 Z3, split into `period`
        # consecutive groups (period 4: one X and one Z check per round; period 8: one check)
        order = [(k, i) for i in range(4) for k in ("X", "Z")]
        if period <= len(order):
            size = len(order) // period
            groups = [order[j * size:(j + 1) * size] for j in range(period)]
        else:
            # longer periods: one outer check every period/8 rounds, none in between
            step = period // len(order)
            groups = [[order[j // step]] if j % step == 0 else [] for j in range(period)]
        if not rotate:
            groups = [[] for _ in groups]
        self.rot_groups = groups
        for j, grp in enumerate(groups):
            if not grp:
                continue
            if self.outer_design is None:
                ap = level3_rows_apart(self.code, grp, f"L3r_{j}")
            else:
                ap = outer_apart(self.code, grp, f"L3r_{j}", self.outer_design)
            self.aparts[ap.name] = ap
            self.parts[ap.name] = compile_part(ap, noise, self.n_data)
        if fast_l2:
            blocks = self.code.blocks
            for b in range(5):
                ap = level2_sched_apart([blocks[b]], self.carbon, name=f"L2f_{b}", reuse=False)
                self.aparts[ap.name] = ap
                self.parts[ap.name] = compile_part(ap, noise, self.n_data, idle_slots=blocks[b])
                wait = APart(f"W2f_{b}", n_anc=0)
                for _ in range(len(ap.layers)):
                    wait.add([])
                if self.timed:
                    wait.layer_dts = [noise.layer_duration(
                        sum(1 for op in lay if op[0] in ("CX", "CZ")), 0,
                        sum(1 for op in lay if op[0] in ("M", "MX")),
                        sum(1 for op in lay if op[0] in ("R", "RX"))) if lay else 0.0
                        for lay in ap.layers]
                self.aparts[wait.name] = wait
                self.parts[wait.name] = compile_part(wait, noise, self.n_data, idle_slots=blocks[b])
            if self.l2_type:
                for b in range(5):
                    for kind in "XZ":
                        ap = level2_sched_apart([blocks[b]], self.carbon, name=f"L2f{kind}_{b}", reuse=False,
                                                kinds=(kind,))
                        self.aparts[ap.name] = ap
                        self.parts[ap.name] = compile_part(ap, noise, self.n_data, idle_slots=blocks[b])
                    # a waiting block idles for one single-type part (as the base rule's waits
                    # idle for one part: the modelling simplification of results/timing_consistency.json)
                    wait = APart(f"W2f1_{b}", n_anc=0)
                    for _ in range(len(self.aparts[f"L2fX_{b}"].layers)):
                        wait.add([])
                    if self.timed:
                        wait.layer_dts = [noise.layer_duration(
                            sum(1 for op in lay if op[0] in ("CX", "CZ")), 0,
                            sum(1 for op in lay if op[0] in ("M", "MX")),
                            sum(1 for op in lay if op[0] in ("R", "RX"))) if lay else 0.0
                            for lay in self.aparts[f"L2fX_{b}"].layers]
                    self.aparts[wait.name] = wait
                    self.parts[wait.name] = compile_part(wait, noise, self.n_data, idle_slots=blocks[b])
        self.setup()
        self._prep_rules()

    def _l2_steps(self):
        """Per-block weight-6 parts: fast (parallel X/Z) when at most three blocks need them,
        the ancilla-reusing ones otherwise; waits for the blocks that do not."""
        def need(st, idx):
            return np.stack([st.round_events[np.ix_(idx, self.c4_idx[b])].any(axis=1)
                             for b in range(5)], axis=1)
        def need_t(st, idx, t):
            # t = 0: X-type C4 checks (they see Z errors), t = 1: Z-type
            return np.stack([st.round_events[np.ix_(idx, self.c4_idx[b][t::2])].any(axis=1)
                             for b in range(5)], axis=1)

        def single(st, idx):
            """per block: needs weight-6 checks of one type only"""
            nx, nz = need_t(st, idx, 0), need_t(st, idx, 1)
            return nx ^ nz
        steps = []
        for fast in (True, False):
            for b in range(5):
                if fast and self.l2_type:
                    for name, sel in ((f"L2f_{b}", (True, True)), (f"L2fX_{b}", (True, False)),
                                      (f"L2fZ_{b}", (False, True))):
                        def cond(st, idx, b=b, sel=sel):
                            k = need(st, idx).sum(axis=1)
                            nx, nz = need_t(st, idx, 0)[:, b], need_t(st, idx, 1)[:, b]
                            return (k <= 3) & (nx == sel[0]) & (nz == sel[1])
                        steps.append((name, cond))
                    continue

                def cond(st, idx, b=b, fast=fast):
                    nd = need(st, idx)
                    k = nd.sum(axis=1)
                    return nd[:, b] & ((k <= 3) if fast else (k >= 4))
                steps.append((f"L2f_{b}" if fast else f"L2_{b}", cond))
            for b in range(5):
                if fast and self.l2_type:
                    # the wait of a block without events: one single-type part when every
                    # measured block needs one type only, else one two-type part
                    for name, one in ((f"W2f1_{b}", True), (f"W2f_{b}", False)):
                        def condw(st, idx, b=b, one=one):
                            nd = need(st, idx)
                            k = nd.sum(axis=1)
                            all_single = ~(nd & ~single(st, idx)).any(axis=1)
                            return (k >= 1) & (k <= 3) & ~nd[:, b] & (all_single == one)
                        steps.append((name, condw))
                    continue

                def condw(st, idx, b=b, fast=fast):
                    nd = need(st, idx)
                    k = nd.sum(axis=1)
                    sel = ((k >= 1) & (k <= 3)) if fast else (k >= 4)
                    return sel & ~nd[:, b]
                steps.append((f"W2f_{b}" if fast else f"W2_{b}", condw))
        return steps

    def plan(self, r):
        steps = super().plan(r)
        if r == 0:
            return steps
        out = []
        l2_done = False
        for pt, cond in steps:
            if self.fast_l2 and (pt.startswith("L2_") or pt.startswith("W2_")) and pt != "L2_all":
                if not l2_done:
                    out.extend(self._l2_steps())
                    l2_done = True
                continue
            if pt == "L3":
                def cond_rec(st, idx, cond=cond):
                    trig = cond(st, idx)
                    now = st.custom.setdefault("l3_now", np.zeros(st.frame.shape[0], dtype=bool))
                    now[:] = False
                    now[idx[np.asarray(trig, dtype=bool)]] = True
                    return trig
                out.append(("L3", cond_rec))
            else:
                out.append((pt, cond))
        j = r % self.period
        if self.rot_groups[j]:
            def cond_rot(st, idx):
                now = st.custom.get("l3_now")
                if now is None:
                    return np.ones(len(idx), dtype=bool)
                return ~now[idx]
            out.append((f"L3r_{j}", cond_rot))
        return out
