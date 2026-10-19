"""Run an (adaptive) protocol: simulate shots, decode every shot on one superset detector
error model whose priors are set by the shot's executed path (SupersetDEM +
PriorSwapFrontier), and collect logical failures and resource counts."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .decoding import PriorSwapFrontier
from .engine import SupersetDEM, simulate


class SupersetDecoder:
    def __init__(self, proto, K=512, Delta=12.0, committee=True, K2=4096, Delta2=20.0,
                 recheck_gap=8.0, stages=None, gap_stages=None):
        self.proto = proto
        plan_all = [[pt for pt, _ in proto.plan(r)] for r in range(proto.n_rounds)]
        self.plan_all = plan_all
        self.sup = SupersetDEM(proto, plan_all)
        full = [(r, pt) for r, pts in enumerate(plan_all) for pt in pts]
        base = self.sup.priors_for(self.sup.executed_mask(full))
        self.dec = PriorSwapFrontier(self.sup.H, self.sup.L, base, K=K, Delta=Delta,
                                     committee=committee)
        self.K2, self.Delta2, self.recheck_gap = K2, Delta2, recheck_gap
        # re-decoding stages (beam K, Delta), applied in turn to the shots that still meet
        # the observable-independent criteria; the last stage is (K2, Delta2)
        if stages is None:
            stages = [(1024, 16.0)] if K2 and K2 > 1024 else []
        self.stages = [tuple(x) for x in stages] + ([(K2, Delta2)] if K2 else [])
        # the low-gap criterion applies to the first `gap_stages` re-decodings only (None:
        # to all); disagreement and no-path always escalate
        self.gap_stages = gap_stages
        self.base = base
        self.dec2 = None

    def _needs_more(self, st, gap, stage=0):
        bad = np.array([s != "ok" for s in st], dtype=bool)
        disagree = getattr(self.dec, "last_disagree", np.zeros(len(st), dtype=bool))
        if self.gap_stages is not None and stage >= self.gap_stages:
            return bad | disagree
        return bad | disagree | ~(gap >= self.recheck_gap)

    def decode(self, path, D):
        pri = self.sup.priors_for(self.sup.executed_mask(path))
        pred, gap, st = self.dec.decode_batch(pri, D, cache_key=tuple(path))
        idx = np.flatnonzero(self._needs_more(st, gap))
        n_re, changed = len(idx), 0
        # deepest decoding stage used per shot (0 = primary only)
        self.last_stage = np.zeros(len(D), dtype=np.int64)
        for s_i, (K, Dl) in enumerate(self.stages):
            if not len(idx):
                break
            self.last_stage[idx] = s_i + 1
            # same native models (committee: forward + backward), larger beam
            p2, g2, s2 = self.dec.decode_batch(pri, D[idx], cache_key=tuple(path), K=K, Delta=Dl)
            changed += int(np.any(p2 != pred[idx], axis=1).sum())
            pred[idx] = p2
            gap[idx] = g2
            idx = idx[self._needs_more(s2, g2, stage=s_i + 1)]
        # shots still meeting the criteria after the last stage
        self.last_unresolved = np.zeros(len(D), dtype=bool)
        self.last_unresolved[idx] = True
        return pred, gap, n_re, changed


def run_protocol(proto, shots, rng, decoder: SupersetDecoder, inject_fn=None):
    """Simulate and decode. Returns dict with failures, per-shot two-qubit gate counts,
    executed part counts and the number of distinct paths."""
    st = simulate(proto, shots, rng, inject_fn=inject_fn)
    groups = defaultdict(list)
    for i, path in enumerate(st.paths):
        groups[tuple(path)].append(i)
    fails = np.zeros(shots, dtype=bool)
    gaps = np.zeros(shots)
    rechecked = changed = 0
    sup = decoder.sup
    for path, idx in groups.items():
        D = np.zeros((len(idx), sup.n_det), dtype=bool)
        O = np.zeros((len(idx), len(proto.logical_supports)), dtype=np.uint8)
        for a, i in enumerate(idx):
            v, o = sup.syndrome(path, st.records[i], st.final[i])
            D[a] = v
            O[a] = o
        pred, gap, nre, nch = decoder.decode(path, D)
        fails[idx] = np.any(pred != O, axis=1)
        gaps[idx] = gap
        rechecked += nre
        changed += nch
    twoq = np.array([sum(proto.parts[pt].two_q for _, pt in p) for p in st.paths])
    counts = defaultdict(int)
    for p in st.paths:
        for _, pt in p:
            counts[pt] += 1
    return dict(fails=fails, gaps=gaps, twoq=twoq, part_counts=dict(counts),
                n_paths=len(groups), rechecked=rechecked, changed=changed, state=st)


def run_stratified(proto, shots, rng, decoder: SupersetDecoder, odds_scale: float):
    """Odds-scaled superset fault sampling: returns per-shot W (number of fired sites over
    the superset) and failure indicator, plus resources, for stratified estimation."""
    from .engine import sample_superset_faults, simulate_with_faults
    faults, W = sample_superset_faults(proto, decoder.plan_all, shots, rng, odds_scale)
    st = simulate_with_faults(proto, faults, shots)
    groups = defaultdict(list)
    for i, path in enumerate(st.paths):
        groups[tuple(path)].append(i)
    fails = np.zeros(shots, dtype=bool)
    stage = np.full(shots, -1, dtype=np.int64)        # -1: no detection event, not decoded
    unresolved = np.zeros(shots, dtype=bool)
    sup = decoder.sup
    for path, idx in groups.items():
        idx = np.asarray(idx)
        D = np.zeros((len(idx), sup.n_det), dtype=bool)
        O = np.zeros((len(idx), len(proto.logical_supports)), dtype=np.uint8)
        for a, i in enumerate(idx):
            v, o = sup.syndrome(path, st.records[i], st.final[i])
            D[a] = v
            O[a] = o
        # shots without any detection event and no observable flip need no decoding
        trivial = ~D.any(axis=1)
        pred = np.zeros_like(O)
        nt = np.flatnonzero(~trivial)
        if len(nt):
            pr, gap, nre, nch = decoder.decode(path, D[nt])
            pred[nt] = pr
            stage[idx[nt]] = decoder.last_stage
            unresolved[idx[nt]] = decoder.last_unresolved
        fails[idx] = np.any(pred != O, axis=1)
    twoq = np.array([sum(proto.parts[pt].two_q for _, pt in p) for p in st.paths])
    return dict(W=W, fails=fails, twoq=twoq, faults=faults, state=st, stage=stage,
                unresolved=unresolved)


def fails_with(proto, decoder: SupersetDecoder, fault_list):
    """Simulate one shot with an explicit superset fault list and decode it.
    fault_list: dict (r, pt) -> list of term indices, plus optional 'init'/'final' lists."""
    from .engine import simulate_with_faults
    n = proto.n_data
    faults = {}
    for key, terms in fault_list.items():
        if key in ("init", "final"):
            continue
        faults[key] = (np.zeros(len(terms), dtype=np.int64), np.array(terms, dtype=np.int64))
    for nm in ("init", "final"):
        arr = np.zeros((1, n), dtype=bool)
        for s in fault_list.get(nm, []):
            arr[0, s] = True
        faults[nm] = arr
    st = simulate_with_faults(proto, faults, 1)
    path = tuple(st.paths[0])
    v, o = decoder.sup.syndrome(path, st.records[0], st.final[0])
    if not v.any():
        return bool(o.any()), path
    pred, gap, nre, nch = decoder.decode(path, v[None, :])
    return bool(np.any(pred[0] != o)), path


def minimize_failure(proto, decoder, fault_list, rng=None):
    """Greedy reduction of a failing fault configuration to a locally minimal failing
    subset. Returns (minimal fault list, its size)."""
    items = [(k, t) for k, ts in fault_list.items() for t in ts]
    if rng is not None:
        rng.shuffle(items)
    keep = list(items)
    for it in list(items):
        trial = [x for x in keep if x != it]
        fl = {}
        for k, t in trial:
            fl.setdefault(k, []).append(t)
        f, _ = fails_with(proto, decoder, fl)
        if f:
            keep = trial
    fl = {}
    for k, t in keep:
        fl.setdefault(k, []).append(t)
    return fl, len(keep)
