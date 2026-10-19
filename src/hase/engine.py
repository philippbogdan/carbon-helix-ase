"""Adaptive-circuit engine: parts over virtual qubits, compiled fault tables, Pauli-frame
simulation with per-shot branching, and per-path detector error models.

Virtual qubits: ('d', s) for data slot s and ('a', i) for the i-th local ancilla of a part.
Ops (name, qubits, key):
  R, RX            reset to |0>, |+>
  M, MX            measure Z, X (key = measurement label)
  H, S, SDG        one-qubit Cliffords
  CX, CZ           two-qubit gates
  FBX, FBZ         Pauli X/Z on a qubit conditioned on measurement `key` (noiseless)
  RELABEL          swap the names of two virtual qubits (virtual SWAP, noiseless, no time)
A part is a list of layers (lists of ops acting on disjoint qubits); every layer is one time
step for idle noise. Measurement keys are (check_name, ...) tuples; each part declares,
for each of its measurements, which check (or flag) it is.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .ir import NoiseModel

PAULI_1Q = ("X", "Y", "Z")
# single-qubit Clifford classes (Sp(2,2)) as maps (x, z) -> (x', z')
C1_MAPS = {0: ((1, 0), (0, 1)), 1: ((0, 1), (1, 0)), 2: ((1, 0), (1, 1)),
           3: ((1, 1), (0, 1)), 4: ((0, 1), (1, 1)), 5: ((1, 1), (1, 0))}
PAULI_2Q = tuple(a + b for a in "IXYZ" for b in "IXYZ")[1:]


@dataclass
class APart:
    """Abstract part. `meas` maps measurement key -> ('chk', check_name) or ('flag',)."""
    name: str
    n_anc: int
    layers: list = field(default_factory=list)
    meas: dict = field(default_factory=dict)
    layer_dts: list = None        # optional fixed per-layer durations (timed noise)

    def add(self, ops):
        qs = [q for op in ops for q in op[1]]
        assert len(qs) == len(set(qs)), f"layer acts twice on a qubit: {ops}"
        self.layers.append(list(ops))

    def two_qubit_gates(self):
        return sum(1 for lay in self.layers for op in lay if op[0] in ("CX", "CZ"))

    def n_layers(self):
        return sum(1 for lay in self.layers
                   if not lay or any(op[0] not in ("FBX", "FBZ", "RELABEL") for op in lay))


# ----------------------------------------------------------------------------- noise
def channel_terms(kind, noise: NoiseModel):
    """Return list of (pauli_string, prob) for an elementary noise location."""
    if kind == "dep1":
        p = noise.p1
        return [(P, p / 3) for P in PAULI_1Q] if p else []
    if kind == "dep2":
        if getattr(noise, "pauli2", None) is not None:
            return [(P, q) for P, q in noise.pauli2.items() if q > 0]
        p = noise.p2
        return [(P, p / 15) for P in PAULI_2Q] if p else []
    if kind == "idle":
        p = noise.pidle
        if not p:
            return []
        if noise.idle_kind == "Z":
            return [("Z", p)]
        return [(P, p / 3) for P in PAULI_1Q]
    raise ValueError(kind)


# ----------------------------------------------------------------------------- compile
@dataclass
class CompiledPart:
    name: str
    n_data: int
    meas_keys: list          # measurement keys in order
    meas_kind: list          # 'chk' or 'flag'
    meas_check: list         # check name or None
    # incoming data frame (2*n_data: x bits then z bits) -> meas flips and outgoing frame
    in_meas: np.ndarray      # [2n, m] uint8
    in_out: np.ndarray       # [2n, 2n] uint8 (permutation/identity)
    # fault terms
    site_of_term: np.ndarray     # [T] site index
    term_prob: np.ndarray        # [T]
    term_meas: sp.csr_matrix     # [T, m]
    term_out: sp.csr_matrix      # [T, 2n]
    site_prob: np.ndarray        # [S] total firing probability of each site
    site_terms: list             # list of arrays of term indices per site
    site_desc: list              # description per site
    two_q: int
    n_layers: int
    duration: float = 0.0


def compile_part(part: APart, noise: NoiseModel, n_data: int, idle_data=True,
                 idle_slots=None) -> CompiledPart:
    """Compile an abstract part into fault tables by Pauli-frame propagation.

    idle_data: apply idle noise to every data slot (of the whole code) not acted on in a
    layer, in addition to live local ancillas.
    """
    # virtual qubit indexing: data slots 0..n_data-1, ancillas n_data..n_data+n_anc-1
    nq = n_data + part.n_anc

    def vq(q):
        return q[1] if q[0] == "d" else n_data + q[1]

    # name map for RELABEL: current physical-index for each virtual name
    # we track frames on 'holders'; RELABEL swaps which holder a name refers to.
    holder = list(range(nq))

    # --- enumerate ops in order with insertion points for noise
    events = []   # list of (kind, payload)
    if idle_slots is not None:
        live = set(idle_slots)
    else:
        live = set(range(n_data)) if idle_data else set()
    timed = getattr(noise, "timed", False)
    duration = 0.0
    for li, layer in enumerate(part.layers):
        touched = set()
        is_fb_only = bool(layer) and all(op[0] in ("FBX", "FBZ", "RELABEL") for op in layer)
        n2q = sum(1 for op in layer if op[0] in ("CX", "CZ"))
        n1q = sum(1 for op in layer if op[0] in ("H", "S", "SDG") or op[0].startswith("C1_"))
        nmeas = sum(1 for op in layer if op[0] in ("M", "MX"))
        nres = sum(1 for op in layer if op[0] in ("R", "RX"))
        live_before = set(live)
        for op in layer:
            name, qs = op[0], op[1]
            key = op[2] if len(op) > 2 else None
            hs = [holder[vq(q)] for q in qs]
            if name == "RELABEL":
                a, b = vq(qs[0]), vq(qs[1])
                holder[a], holder[b] = holder[b], holder[a]
                continue
            touched.update(hs)
            events.append(("op", name, tuple(hs), key))
            if name in ("R", "RX"):
                live.add(hs[0])
                if noise.pr:
                    events.append(("noise", "reset", tuple(hs), [("X" if name == "R" else "Z", noise.pr)]))
            elif name in ("H", "S", "SDG") or name.startswith("C1_"):
                t = channel_terms("dep1", noise)
                if t:
                    events.append(("noise", "dep1", tuple(hs), t))
            elif name in ("CX", "CZ"):
                t = channel_terms("dep2", noise)
                if t:
                    events.append(("noise", "dep2", tuple(hs), t))
                lk = getattr(noise, "leak2", 0.0)
                if lk:
                    for h in hs:
                        events.append(("noise", "leak", (h,), [(P, lk / 8) for P in PAULI_1Q]))
            elif name in ("M", "MX"):
                live.discard(hs[0])
        if not is_fb_only:
            t = channel_terms("idle", noise)
            if t:
                for h in sorted(live - touched):
                    events.append(("noise", "idle", (h,), t))
            if timed:
                if part.layer_dts is not None:
                    dt = part.layer_dts[li]
                else:
                    dt = noise.layer_duration(n2q, n1q, nmeas, nres)
                duration += dt
                md, mz = noise.mem_dep_per_ms * dt, noise.mem_z_per_ms * dt
                xt = noise.xtalk_dep * nmeas
                measured = {h for e in events[-len(layer) * 4:] if e[0] == "op" and e[1] in ("M", "MX") for h in e[2]}
                for h in sorted(live | live_before):
                    terms = [("X", md / 3), ("Y", md / 3), ("Z", md / 3 + mz)]
                    if xt and h not in measured:
                        terms = [(P, q + xt / 3) for P, q in terms]
                    if md or mz or xt:
                        events.append(("noise", "mem", (h,), terms))
            else:
                duration += 1.0
    # measurement noise must act *before* the measurement: rewrite as pre-measure flips
    ev2 = []
    for e in events:
        if e[0] == "op" and e[1] in ("M", "MX") and noise.pm:
            ev2.append(("noise", "meas", e[2], [("X" if e[1] == "M" else "Z", noise.pm)]))
        ev2.append(e)
    events = ev2

    # final holder of each data slot
    final_holder = [holder[s] for s in range(n_data)]

    # --- list fault terms
    sites, terms = [], []   # sites: (event_index, desc, prob); terms: (site, paulistr, prob)
    for ei, e in enumerate(events):
        if e[0] != "noise":
            continue
        hs = e[2]
        tot = 0.0
        si = len(sites)
        for P, q in e[3]:
            terms.append((si, ei, hs, P, q))
            tot += q
        sites.append((ei, (e[1],) + tuple(hs), tot))
    T = len(terms)
    meas_keys = [e[3] for e in events if e[0] == "op" and e[1] in ("M", "MX")]
    m = len(meas_keys)
    meas_index = {k: i for i, k in enumerate(meas_keys)}

    # --- propagate: rows = T fault terms + 2*n_data incoming basis Paulis
    R = T + 2 * n_data
    X = np.zeros((R, nq), dtype=bool)
    Z = np.zeros((R, nq), dtype=bool)
    F = np.zeros((R, m), dtype=bool)
    # incoming: data qubits are holders 0..n_data-1 at the start (holder[s]=s initially)
    for s in range(n_data):
        X[T + s, s] = True
        Z[T + n_data + s, s] = True
    term_rows_at_event = {}
    for ti, (si, ei, hs, P, q) in enumerate(terms):
        term_rows_at_event.setdefault(ei, []).append((ti, hs, P))
    mi = 0
    for ei, e in enumerate(events):
        if e[0] == "noise":
            for ti, hs, P in term_rows_at_event.get(ei, []):
                for h, c in zip(hs, P):
                    if c in "XY":
                        X[ti, h] ^= True
                    if c in "ZY":
                        Z[ti, h] ^= True
            continue
        name, hs, key = e[1], e[2], e[3]
        if name == "CX":
            c, t = hs
            X[:, t] ^= X[:, c]
            Z[:, c] ^= Z[:, t]
        elif name == "CZ":
            a, b = hs
            Z[:, a] ^= X[:, b]
            Z[:, b] ^= X[:, a]
        elif name == "H":
            h = hs[0]
            X[:, h], Z[:, h] = Z[:, h].copy(), X[:, h].copy()
        elif name in ("S", "SDG"):
            h = hs[0]
            Z[:, h] ^= X[:, h]
        elif name.startswith("C1_"):
            h = hs[0]
            (a, b), (c, d) = C1_MAPS[int(name[3:])]
            x0, z0 = X[:, h].copy(), Z[:, h].copy()
            X[:, h] = (x0 & bool(a)) ^ (z0 & bool(b))
            Z[:, h] = (x0 & bool(c)) ^ (z0 & bool(d))
        elif name == "R":
            X[:, hs[0]] = False
            Z[:, hs[0]] = False
        elif name == "RX":
            X[:, hs[0]] = False
            Z[:, hs[0]] = False
        elif name == "M":
            F[:, mi] = X[:, hs[0]]
            mi += 1
        elif name == "MX":
            F[:, mi] = Z[:, hs[0]]
            mi += 1
        elif name == "FBX":
            j = meas_index[key]
            X[:, hs[0]] ^= F[:, j]
        elif name == "FBZ":
            j = meas_index[key]
            Z[:, hs[0]] ^= F[:, j]
        else:
            raise ValueError(name)
    # outgoing data frame in slot order
    fh = np.array(final_holder)
    out = np.concatenate([X[:, fh], Z[:, fh]], axis=1)
    term_meas = sp.csr_matrix(F[:T].astype(np.uint8))
    term_out = sp.csr_matrix(out[:T].astype(np.uint8))
    in_meas = F[T:].astype(np.uint8)
    in_out = out[T:].astype(np.uint8)
    site_terms = [[] for _ in sites]
    for ti, t in enumerate(terms):
        site_terms[t[0]].append(ti)
    meas_kind, meas_check = [], []
    for k in meas_keys:
        info = part.meas[k]
        meas_kind.append(info[0])
        meas_check.append(info[1] if info[0] == "chk" else None)
    return CompiledPart(
        name=part.name, n_data=n_data, meas_keys=meas_keys, meas_kind=meas_kind,
        meas_check=meas_check, in_meas=in_meas, in_out=in_out,
        site_of_term=np.array([t[0] for t in terms], dtype=np.int64),
        term_prob=np.array([t[4] for t in terms]),
        term_meas=term_meas, term_out=term_out,
        site_prob=np.array([s[2] for s in sites]),
        site_terms=[np.array(x, dtype=np.int64) for x in site_terms],
        site_desc=[s[1] for s in sites],
        two_q=part.two_qubit_gates(), n_layers=part.n_layers(), duration=duration)


# ----------------------------------------------------------------------------- protocol
@dataclass
class CheckDef:
    kind: str                 # 'X' or 'Z'
    support: np.ndarray       # data slots


class Protocol:
    """A memory protocol: init, rounds of parts (some conditional), final measurement.

    Subclasses / instances provide:
      n_data, basis, checks: dict name -> CheckDef, logical_supports: list of slot arrays,
      parts: dict type -> CompiledPart, n_rounds,
      plan(r): list of (part_type, cond) where cond is None or a function
               (sim_state, idx) -> bool mask over shots idx deciding execution.
    """

    def setup(self):
        names = list(self.checks)
        self.check_names = names
        self.check_index = {c: i for i, c in enumerate(names)}
        n = self.n_data
        S = np.zeros((len(names), 2 * n), dtype=np.uint8)
        for i, c in enumerate(names):
            d = self.checks[c]
            if d.kind == "X":       # X-type check detects Z components
                S[i, n + d.support] = 1
            else:
                S[i, d.support] = 1
        self.S = S
        Lm = np.zeros((len(self.logical_supports), 2 * n), dtype=np.uint8)
        for i, supp in enumerate(self.logical_supports):
            # memory in basis B measures B-type logicals, flipped by the other Pauli type
            if self.basis == "X":
                Lm[i, n + np.asarray(supp)] = 1
            else:
                Lm[i, np.asarray(supp)] = 1
        self.Lm = Lm
        self.final_checks = [c for c in names if self.checks[c].kind == self.basis]
        self._syn_cache = {}

    def term_syndromes(self, ptype):
        if ptype not in self._syn_cache:
            cp = self.parts[ptype]
            to = cp.term_out
            syn = (to @ sp.csr_matrix(self.S.T)).toarray() % 2
            obs = (to @ sp.csr_matrix(self.Lm.T)).toarray() % 2
            self._syn_cache[ptype] = (sp.csr_matrix(syn.astype(np.uint8)),
                                      sp.csr_matrix(obs.astype(np.uint8)))
        return self._syn_cache[ptype]


# ----------------------------------------------------------------------------- simulation
class SimState:
    """Per-shot state during simulation: data frame, last measured flip of every check,
    whether it was measured, and detection events of the most recent measurement."""

    def __init__(self, proto: Protocol, shots: int):
        self.proto = proto
        n = proto.n_data
        self.frame = np.zeros((shots, 2 * n), dtype=np.uint8)
        nc = len(proto.check_names)
        self.last = np.zeros((shots, nc), dtype=np.uint8)
        self.measured = np.zeros((shots, nc), dtype=bool)
        self.det_event = np.zeros((shots, nc), dtype=np.uint8)    # latest round's events
        self.round_events = np.zeros((shots, nc), dtype=np.uint8)
        self.flag_event = np.zeros(shots, dtype=np.uint8)
        self.paths = [[] for _ in range(shots)]
        self.records = [[] for _ in range(shots)]   # list of meas-flip arrays per instance
        self.r = 0
        self.custom = {}


def _sample_faults(cp: CompiledPart, k: int, rng):
    """Sample independent site firings for k shots; return (shot_idx, term_idx) arrays."""
    if k == 0 or len(cp.site_prob) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    counts = rng.binomial(k, cp.site_prob)
    nz = np.flatnonzero(counts)
    shots_l, terms_l = [], []
    for s in nz:
        c = counts[s]
        sh = rng.choice(k, size=c, replace=False)
        tl = cp.site_terms[s]
        pr = cp.term_prob[tl]
        tt = rng.choice(tl, size=c, p=pr / pr.sum())
        shots_l.append(sh)
        terms_l.append(tt)
    if not shots_l:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(shots_l), np.concatenate(terms_l)


def check_map_for_gate(proto, ptype):
    """For a logical-gate part: matrix T (n_checks x n_checks, GF(2)) with
    value_after(check j) = sum_i T[j, i] value_before(check i)."""
    cache = proto.__dict__.setdefault("_check_maps", {})
    if ptype in cache:
        return cache[ptype]
    from .logical import conj_pauli
    from .general_dem import invert_ops
    from .gf2 import solve
    ops = proto.gate_ops(ptype)
    n = proto.n_data
    vecs = []
    for c in proto.check_names:
        d = proto.checks[c]
        v = np.zeros(2 * n, dtype=np.uint8)
        if d.kind == "X":
            v[d.support] = 1
        else:
            v[n + d.support] = 1
        vecs.append(v)
    V = np.array(vecs)
    inv = invert_ops(ops)
    T = np.zeros((len(vecs), len(vecs)), dtype=np.uint8)
    for j, v in enumerate(vecs):
        img = conj_pauli(v, inv, n)
        x = solve(V, img)
        if x is None:
            raise ValueError("gate maps a check outside the span of measured checks")
        T[j] = x
    cache[ptype] = T
    return T


def apply_part(state: SimState, ptype: str, idx: np.ndarray, rng=None, inject=None):
    """Execute part `ptype` on shots `idx` (array). Faults are sampled with rng, or
    taken from inject = (local_shot_idx, term_idx) arrays."""
    proto = state.proto
    cp = proto.parts[ptype]
    k = len(idx)
    if k == 0:
        return
    gate_ops = proto.gate_ops(ptype) if hasattr(proto, "gate_ops") else None
    if gate_ops is not None:
        T = check_map_for_gate(proto, ptype).astype(np.int64)
        old = state.last[idx].astype(np.int64)
        state.last[idx] = ((old @ T.T) % 2).astype(np.uint8)
        was = state.measured[idx].astype(np.int64)
        state.measured[idx] = ((was @ T.T) == T.sum(axis=1)[None, :])
    fr = state.frame[idx]
    meas = (fr.astype(np.int64) @ cp.in_meas.astype(np.int64)) % 2
    out = (fr.astype(np.int64) @ cp.in_out.astype(np.int64)) % 2
    if inject is None:
        sh, tt = _sample_faults(cp, k, rng)
    else:
        sh, tt = inject
    if len(sh):
        tm = cp.term_meas[tt].toarray().astype(np.int64)
        to = cp.term_out[tt].toarray().astype(np.int64)
        np.add.at(meas, sh, tm)
        np.add.at(out, sh, to)
        meas %= 2
        out %= 2
    meas = meas.astype(np.uint8)
    state.frame[idx] = out.astype(np.uint8)
    # detection events for checks measured here
    ci = np.array([proto.check_index[c] if kd == "chk" else -1
                   for c, kd in zip(cp.meas_check, cp.meas_kind)])
    fl = np.zeros(k, dtype=np.uint8)
    for j, c in enumerate(ci):
        if c < 0:
            fl |= meas[:, j]
            continue
        prev = state.last[idx, c]
        was = state.measured[idx, c]
        ev = meas[:, j] ^ prev
        # first measurement of a random-type check carries no event
        if proto.checks[proto.check_names[c]].kind != proto.basis:
            ev = np.where(was, ev, 0)
        state.det_event[idx, c] = ev
        state.round_events[idx, c] |= ev
        state.last[idx, c] = meas[:, j]
        state.measured[idx, c] = True
    state.flag_event[idx] |= fl
    for a, i in enumerate(idx):
        state.paths[i].append((state.r, ptype))
        state.records[i].append(meas[a])


def simulate(proto: Protocol, shots: int, rng, inject_fn=None):
    """Run the protocol for `shots` shots. inject_fn(r, ptype, idx) -> (sh, tt) or None
    overrides sampling for fixed-fault studies. Returns SimState with final measurement
    flips stored in state.final (shots x n_data) for the basis component."""
    st = SimState(proto, shots)
    allidx = np.arange(shots)
    # init noise: reset flips on data (X for |0>, Z for |+>), prob pr each
    n = proto.n_data
    pr = proto.noise.pr
    init_inj = getattr(inject_fn, "init_flips", None)
    if init_inj is not None:
        if proto.basis == "X":
            st.frame[:, n:] ^= init_inj.astype(np.uint8)
        else:
            st.frame[:, :n] ^= init_inj.astype(np.uint8)
    elif pr and inject_fn is None:
        flips = (rng.random((shots, n)) < pr).astype(np.uint8)
        if proto.basis == "X":
            st.frame[:, n:] ^= flips
        else:
            st.frame[:, :n] ^= flips
    for r in range(proto.n_rounds):
        st.r = r
        st.round_events[:] = 0
        st.flag_event[:] = 0
        for ptype, cond in proto.plan(r):
            if cond is None:
                idx = allidx
            else:
                idx = allidx[cond(st, allidx)]
            inj = inject_fn(r, ptype, idx) if inject_fn else None
            apply_part(st, ptype, idx, rng=rng, inject=inj)
    # final measurement in basis
    fr = st.frame
    comp = fr[:, n:] if proto.basis == "X" else fr[:, :n]
    final = comp.copy()
    pm = proto.noise.pm
    final_inj = getattr(inject_fn, "final_flips", None)
    if final_inj is not None:
        final ^= final_inj.astype(np.uint8)
    elif pm and inject_fn is None:
        final ^= (rng.random(final.shape) < pm).astype(np.uint8)
    st.final = final
    return st


# ----------------------------------------------------------------------------- path DEM
class PathDEM:
    """Detector error model of one executed path, assembled from compiled parts.

    Detectors: for each check, consecutive-measurement comparisons (and a first-
    measurement detector for checks of the memory basis), flags (one detector each),
    and final-data detectors for checks of the memory basis.
    """

    def __init__(self, proto: Protocol, path, merge=True):
        self.proto = proto
        self.path = tuple(path)
        names = proto.check_names
        n = proto.n_data
        # occurrences per check: list of (instance k, meas index j)
        occ = {c: [] for c in names}
        flags = []
        for k, (r, ptype) in enumerate(self.path):
            cp = proto.parts[ptype]
            for j, (kd, c) in enumerate(zip(cp.meas_kind, cp.meas_check)):
                if kd == "chk":
                    occ[c].append((k, j))
                else:
                    flags.append((k, j))
        det_id = {}
        dets = []
        for c in names:
            first = 0 if proto.checks[c].kind == proto.basis else 1
            for o in range(first, len(occ[c])):
                det_id[(c, o)] = len(dets)
                dets.append(("chk", c, o))
            if proto.checks[c].kind == proto.basis:
                det_id[(c, len(occ[c]))] = len(dets)     # final detector
                dets.append(("final", c, len(occ[c])))
        for (k, j) in flags:
            det_id[("flag", k, j)] = len(dets)
            dets.append(("flag", k, j))
        self.dets = dets
        self.det_id = det_id
        self.occ = occ
        self.n_det = len(dets)
        # occurrence index of each (k, j) measurement and next occurrence after instance k
        occ_of = {}
        for c in names:
            for o, (k, j) in enumerate(occ[c]):
                occ_of[(k, j)] = (c, o)
        self.occ_of = occ_of
        cols_H, cols_L, probs = [], [], []
        owners = []
        ci = proto.check_index
        nchk = len(names)
        for k, (r, ptype) in enumerate(self.path):
            cp = proto.parts[ptype]
            m = len(cp.meas_keys)
            # M_k: meas -> detectors
            rows, colsm = [], []
            for j in range(m):
                if cp.meas_kind[j] == "chk":
                    c, o = occ_of[(k, j)]
                    for oo in (o, o + 1):
                        d = det_id.get((c, oo))
                        if d is not None:
                            rows.append(j); colsm.append(d)
                else:
                    rows.append(j); colsm.append(det_id[("flag", k, j)])
            Mk = sp.csr_matrix((np.ones(len(rows), dtype=np.int64), (rows, colsm)), shape=(m, self.n_det))
            # N_k: check residual -> detector of next occurrence after instance k
            rows, colsn = [], []
            for c in names:
                nxt = None
                for o, (kk, j) in enumerate(occ[c]):
                    if kk > k:
                        nxt = o
                        break
                if nxt is None:
                    nxt = len(occ[c])     # final detector (exists only for basis checks)
                d = det_id.get((c, nxt))
                if d is not None:
                    rows.append(ci[c]); colsn.append(d)
            Nk = sp.csr_matrix((np.ones(len(rows), dtype=np.int64), (rows, colsn)), shape=(nchk, self.n_det))
            syn, obs = proto.term_syndromes(ptype)
            Hk = (cp.term_meas.astype(np.int64) @ Mk + syn.astype(np.int64) @ Nk)
            Hk.data %= 2
            Hk.eliminate_zeros()
            cols_H.append(Hk.tocsr())
            cols_L.append(obs.astype(np.int64))
            probs.append(cp.term_prob)
            owners.extend([(k, float(q)) for q in cp.term_prob])
        # init reset flips on data and final measurement flips
        pr, pm = proto.noise.pr, proto.noise.pm
        extra_p = []
        H_extra_rows, H_extra_cols, L_extra = [], [], []
        ncol = 0
        for kind, q in (("init", pr), ("final", pm)):
            if not q:
                continue
            for s in range(n):
                e = np.zeros(2 * n, dtype=np.uint8)
                if proto.basis == "X":
                    e[n + s] = 1
                else:
                    e[s] = 1
                syn = (self.proto.S @ e) % 2
                lo = (self.proto.Lm @ e) % 2
                dl = []
                for c in np.flatnonzero(syn):
                    cname = names[c]
                    o = 0 if kind == "init" else len(occ[cname])
                    d = det_id.get((cname, o))
                    if d is not None:
                        dl.append(d)
                for d in dl:
                    H_extra_rows.append(d); H_extra_cols.append(ncol)
                L_extra.append(lo)
                extra_p.append(q)
                ncol += 1
        if ncol:
            cols_H.append(sp.csr_matrix((np.ones(len(H_extra_rows), dtype=np.int64),
                                         (H_extra_cols, H_extra_rows)), shape=(ncol, self.n_det)))
            cols_L.append(sp.csr_matrix(np.array(L_extra, dtype=np.int64)))
            probs.append(np.array(extra_p))
            owners.extend([(-1, float(q)) for q in extra_p])
        Hall = sp.vstack(cols_H).tocsr()       # [cols, dets]
        Lall = sp.vstack(cols_L).tocsr()
        pall = np.concatenate(probs)
        Hall.data %= 2
        Hall.eliminate_zeros()
        Lall.data %= 2
        Lall.eliminate_zeros()
        self.Hraw, self.Lraw, self.owner = Hall, Lall, owners
        if merge:
            self.H, self.L, self.priors = _merge_columns(Hall, Lall, pall)

    def detectors_from_records(self, records, final):
        """records: list (per instance) of meas-flip arrays for one shot; final: data
        flips in the basis component. Returns detector bool vector and observable bits."""
        proto = self.proto
        n = proto.n_data
        v = np.zeros(self.n_det, dtype=np.uint8)
        vals = {}
        for c, lst in self.occ.items():
            vals[c] = [records[k][j] for (k, j) in lst]
        for (key, d) in self.det_id.items():
            if key[0] == "flag":
                _, k, j = key
                v[d] = records[k][j]
                continue
            c, o = key
            lst = vals[c]
            if o < len(lst):
                cur = lst[o]
            else:   # final detector: parity of final data on support
                cur = int(final[proto.checks[c].support].sum() % 2)
            prev = lst[o - 1] if o >= 1 else 0
            v[d] = cur ^ prev
        obs = np.array([int(final[np.asarray(s)].sum() % 2) for s in proto.logical_supports],
                       dtype=np.uint8)
        return v, obs


def _merge_columns(H, L, p):
    """Merge identical (detector set, observable set) columns; drop empty ones."""
    H = H.tocsr()
    L = L.tocsr()
    keys = {}
    rowsH, colsH, rowsL, colsL, probs = [], [], [], [], []
    for i in range(H.shape[0]):
        d = tuple(H.indices[H.indptr[i]:H.indptr[i + 1]].tolist())
        l = tuple(L.indices[L.indptr[i]:L.indptr[i + 1]].tolist())
        if not d and not l:
            continue
        key = (tuple(sorted(d)), tuple(sorted(l)))
        q = float(p[i])
        if key in keys:
            j = keys[key]
            probs[j] = probs[j] * (1 - q) + q * (1 - probs[j])
        else:
            j = len(probs)
            keys[key] = j
            probs.append(q)
            for x in key[0]:
                rowsH.append(x); colsH.append(j)
            for x in key[1]:
                rowsL.append(x); colsL.append(j)
    ncol = len(probs)
    Hm = sp.csr_matrix((np.ones(len(rowsH), dtype=np.uint8), (rowsH, colsH)), shape=(H.shape[1], ncol))
    Lm = sp.csr_matrix((np.ones(len(rowsL), dtype=np.uint8), (rowsL, colsL)), shape=(L.shape[1], ncol))
    return Hm, Lm, np.array(probs)


# ----------------------------------------------------------------------------- superset DEM
class SupersetDEM:
    """One detector error model covering every possible path of a protocol.

    All parts of every round are present. For a given path:
      * mechanisms of non-executed part instances get prior 0,
      * every measurement that was not executed is replaced by an 'erasure' mechanism of
        prior 1/2 flipping the two detectors that compare it with its neighbours (so the
        decoder marginalises over the missing value), and its value is set to 0.
    Columns with identical signatures are merged; their prior is recombined per path from
    the members that were executed.
    """

    def __init__(self, proto: Protocol, plan_all):
        """plan_all: list over rounds of the part types that *may* execute in that round,
        in execution order."""
        self.proto = proto
        path = [(r, pt) for r, pts in enumerate(plan_all) for pt in pts]
        self.instances = path
        self.inst_index = {x: k for k, x in enumerate(path)}
        base = PathDEM(proto, path, merge=False)
        self.base = base
        n_det = base.n_det
        # erasure columns: one per (check measurement occurrence) that is optional
        rows, cols, pr_owner = [], [], []
        Hcols = [base.Hraw]
        Lcols = [base.Lraw]
        owner = list(base.owner)          # (instance k or -1, prob)
        er_rows, er_cols = [], []
        self.erasure_of = {}              # (k, j) -> erasure column index (relative)
        ne = 0
        for c, lst in base.occ.items():
            for o, (k, j) in enumerate(lst):
                ds = [base.det_id.get((c, o)), base.det_id.get((c, o + 1))]
                ds = [d for d in ds if d is not None]
                if not ds:
                    continue
                for d in ds:
                    er_rows.append(ne); er_cols.append(d)
                self.erasure_of[(k, j)] = ne
                ne += 1
        Her = sp.csr_matrix((np.ones(len(er_rows), dtype=np.int64), (er_rows, er_cols)),
                            shape=(ne, n_det))
        Ler = sp.csr_matrix((ne, base.Lraw.shape[1]), dtype=np.int64)
        Hall = sp.vstack([base.Hraw, Her]).tocsr()
        Lall = sp.vstack([base.Lraw, Ler]).tocsr()
        n_mech = base.Hraw.shape[0]
        # group identical signatures
        keys = {}
        group = np.zeros(Hall.shape[0], dtype=np.int64)
        gH_rows, gH_cols, gL_rows, gL_cols = [], [], [], []
        for i in range(Hall.shape[0]):
            d = tuple(sorted(Hall.indices[Hall.indptr[i]:Hall.indptr[i + 1]].tolist()))
            l = tuple(sorted(Lall.indices[Lall.indptr[i]:Lall.indptr[i + 1]].tolist()))
            key = (d, l)
            if key not in keys:
                g = len(keys)
                keys[key] = g
                for x in d:
                    gH_rows.append(x); gH_cols.append(g)
                for x in l:
                    gL_rows.append(x); gL_cols.append(g)
            group[i] = keys[key]
        ng = len(keys)
        self.n_groups = ng
        self.group = group
        self.H = sp.csr_matrix((np.ones(len(gH_rows), dtype=np.uint8), (gH_rows, gH_cols)), shape=(n_det, ng))
        self.L = sp.csr_matrix((np.ones(len(gL_rows), dtype=np.uint8), (gL_rows, gL_cols)),
                               shape=(base.Lraw.shape[1], ng))
        self.mech_inst = np.array([o[0] for o in base.owner] + [-2] * ne, dtype=np.int64)
        self.mech_prob = np.concatenate([np.array([o[1] for o in base.owner]), np.full(ne, 0.5)])
        self.n_mech = n_mech
        self.erasure_meas = {v + n_mech: kj for kj, v in self.erasure_of.items()}
        self.er_index = np.full(Hall.shape[0], -1, dtype=np.int64)
        for col, (k, j) in self.erasure_meas.items():
            self.er_index[col] = k
        self.n_det = n_det
        self.null_prior = 1e-30

    def priors_for(self, executed: np.ndarray):
        """executed: bool array over instances. Returns group priors."""
        ex_inst = np.concatenate([executed, [True, False]])   # -1 -> init/final (always)
        m = self.mech_inst
        on = np.where(m >= 0, ex_inst[np.clip(m, 0, None)], m == -1)
        # erasure mechanisms: active when their measurement's instance did NOT execute
        er = m == -2
        on = np.where(er, ~ex_inst[np.clip(self.er_index, 0, None)], on)
        p = np.where(on, self.mech_prob, 0.0)
        # combine within groups: p_g = (1 - prod(1-2p))/2
        f = np.log(np.clip(1 - 2 * p, 1e-300, None))
        s = np.zeros(self.n_groups)
        np.add.at(s, self.group, f)
        pg = (1 - np.exp(s)) / 2
        return np.maximum(pg, self.null_prior)

    def syndrome(self, path, records, final):
        """Superset detector vector of one shot given its executed path and records."""
        b = self.base
        executed = {x: i for i, x in enumerate(path)}
        n = self.proto.n_data
        # measurement values for every superset (k, j): 0 if not executed
        vals = {}
        for c, lst in b.occ.items():
            vals[c] = []
            for (k, j) in lst:
                inst = self.instances[k]
                if inst in executed:
                    # map superset instance k -> position in the shot's path
                    vals[c].append(records[executed[inst]][j])
                else:
                    vals[c].append(0)
        v = np.zeros(self.n_det, dtype=np.uint8)
        for key, d in b.det_id.items():
            if key[0] == "flag":
                _, k, j = key
                inst = self.instances[k]
                v[d] = records[executed[inst]][j] if inst in executed else 0
                continue
            c, o = key
            lst = vals[c]
            cur = lst[o] if o < len(lst) else int(final[self.proto.checks[c].support].sum() % 2)
            prev = lst[o - 1] if o >= 1 else 0
            v[d] = cur ^ prev
        obs = np.array([int(final[np.asarray(s)].sum() % 2) for s in self.proto.logical_supports],
                       dtype=np.uint8)
        return v, obs

    def executed_mask(self, path):
        ex = np.zeros(len(self.instances), dtype=bool)
        for x in path:
            ex[self.inst_index[x]] = True
        return ex


# ----------------------------------------------------------------------------- stim emission
def emit_stim(proto: Protocol, path, noiseless=False):
    """Emit the executed path as a stim circuit with the same noise model and detectors as
    PathDEM (used to cross-check the engine and to run stim's distance search)."""
    import stim
    nz = proto.noise
    n = proto.n_data
    c = stim.Circuit()
    phys_of_slot = list(range(n))
    free = []
    nxt = [n]

    def get():
        if free:
            return free.pop(0)
        q = nxt[0]
        nxt[0] += 1
        return q

    meas_rec = []           # list of (k, j) in order
    # init
    c.append("RX" if proto.basis == "X" else "R", list(range(n)))
    if nz.pr and not noiseless:
        c.append("Z_ERROR" if proto.basis == "X" else "X_ERROR", list(range(n)), nz.pr)
    c.append("TICK")
    rec_index = {}
    for k, (r, ptype) in enumerate(path):
        ap = proto.aparts[ptype]
        anc = {}
        name_map = {}

        def phys(q):
            if q in name_map:
                return name_map[q]
            if q[0] == "d":
                return phys_of_slot[q[1]]
            if q not in anc:
                anc[q] = get()
            return anc[q]
        live = set(phys_of_slot)
        part_meas_j = 0
        meas_keys_order = proto.parts[ptype].meas_keys
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
                    if nz.pm and not noiseless:
                        c.append(name, ps, nz.pm)
                    else:
                        c.append(name, ps)
                    j = meas_keys_order.index(key)
                    rec_index[(k, j)] = len(meas_rec)
                    meas_rec.append((k, j))
                    live.discard(ps[0])
                elif name in ("H", "S", "SDG"):
                    c.append({"SDG": "S_DAG"}.get(name, name), ps)
                    if nz.p1 and not noiseless:
                        c.append("DEPOLARIZE1", ps, nz.p1)
                elif name in ("CX", "CZ"):
                    c.append(name, ps)
                    if not noiseless:
                        if getattr(nz, "pauli2", None) is not None:
                            probs = [nz.pauli2.get(P, 0.0) for P in PAULI_2Q]
                            c.append("PAULI_CHANNEL_2", ps, probs)
                        elif nz.p2:
                            c.append("DEPOLARIZE2", ps, nz.p2)
                elif name in ("FBX", "FBZ"):
                    j = meas_keys_order.index(key)
                    back = rec_index[(k, j)] - len(meas_rec)
                    c.append("CX" if name == "FBX" else "CZ", [stim.target_rec(back), ps[0]])
            if only_virtual:
                continue
            if nz.pidle and not noiseless:
                idle = sorted(q for q in live if q not in touched)
                if idle:
                    if nz.idle_kind == "Z":
                        c.append("Z_ERROR", idle, nz.pidle)
                    else:
                        c.append("DEPOLARIZE1", idle, nz.pidle)
            c.append("TICK")
        # return this part's ancilla qubits to the pool (those not now holding data)
        used = set(anc.values()) | {v for kk, v in name_map.items() if kk[0] == "a"}
        for q in sorted(used):
            if q not in phys_of_slot and q not in free:
                free.append(q)
    # final measurement
    fin0 = len(meas_rec)
    if nz.pm and not noiseless:
        c.append("MX" if proto.basis == "X" else "M", phys_of_slot, nz.pm)
    else:
        c.append("MX" if proto.basis == "X" else "M", phys_of_slot)
    total = fin0 + n
    pd = PathDEM(proto, path)

    def rec(i):
        return stim.target_rec(i - total)
    for d in pd.dets:
        if d[0] == "flag":
            _, k, j = d
            c.append("DETECTOR", [rec(rec_index[(k, j)])])
            continue
        kind, cname, o = d
        lst = pd.occ[cname]
        targets = []
        if o < len(lst):
            targets.append(rec(rec_index[lst[o]]))
        else:
            targets += [rec(fin0 + s) for s in proto.checks[cname].support]
        if o >= 1:
            targets.append(rec(rec_index[lst[o - 1]]))
        c.append("DETECTOR", targets)
    for i, supp in enumerate(proto.logical_supports):
        c.append("OBSERVABLE_INCLUDE", [rec(fin0 + s) for s in supp], i)
    return c, pd


# ----------------------------------------------------------------------------- superset faults
def superset_sites(proto: Protocol, plan_all):
    """List of (instance_key, compiled part) for every possible part instance."""
    return [((r, pt), proto.parts[pt]) for r, pts in enumerate(plan_all) for pt in pts]


def sample_superset_faults(proto: Protocol, plan_all, shots, rng, odds_scale=1.0):
    """Sample independent faults on every site of every possible part instance (plus data
    init and final-measurement flips) with all site odds multiplied by odds_scale.

    Returns (faults, W): faults[(r, pt)] = (shot_idx, term_idx); faults['init'] and
    faults['final'] = bool arrays [shots, n_data]; W[shot] = total number of fired sites.
    Conditioned on W, the fault configuration is distributed exactly as for the unscaled
    model (to O(p)), so P(fail | W = w) estimated here is reusable at every p."""
    W = np.zeros(shots, dtype=np.int64)
    faults = {}
    for key, cp in superset_sites(proto, plan_all):
        q = cp.site_prob
        if len(q) == 0:
            continue
        odds = odds_scale * q / (1 - q)
        qs = odds / (1 + odds)
        counts = rng.binomial(shots, qs)
        nz = np.flatnonzero(counts)
        sh_l, tt_l = [], []
        for s in nz:
            c = counts[s]
            sh = rng.choice(shots, size=c, replace=False)
            tl = cp.site_terms[s]
            pr = cp.term_prob[tl]
            sh_l.append(sh)
            tt_l.append(rng.choice(tl, size=c, p=pr / pr.sum()))
        if sh_l:
            sh = np.concatenate(sh_l)
            tt = np.concatenate(tt_l)
        else:
            sh = np.zeros(0, dtype=np.int64)
            tt = np.zeros(0, dtype=np.int64)
        faults[key] = (sh, tt)
        np.add.at(W, sh, 1)
    n = proto.n_data
    for name, q in (("init", proto.noise.pr), ("final", proto.noise.pm)):
        if q:
            odds = odds_scale * q / (1 - q)
            f = rng.random((shots, n)) < odds / (1 + odds)
        else:
            f = np.zeros((shots, n), dtype=bool)
        faults[name] = f
        W += f.sum(axis=1)
    return faults, W


def simulate_with_faults(proto: Protocol, faults, shots):
    """Deterministic simulation with a pre-sampled superset fault configuration."""
    def inject(r, pt, idx):
        sh, tt = faults.get((r, pt), (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)))
        if len(sh) == 0 or len(idx) == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        pos = np.full(shots, -1, dtype=np.int64)
        pos[idx] = np.arange(len(idx))
        keep = pos[sh] >= 0
        return pos[sh[keep]], tt[keep]
    inject.init_flips = faults["init"]
    inject.final_flips = faults["final"]
    return simulate(proto, shots, None, inject_fn=inject)


def site_count_distribution(proto: Protocol, plan_all, p_scale=1.0, w_max=200):
    """Exact distribution of W (number of fired sites over the superset) as a
    Poisson-binomial over all sites at probabilities scaled by p_scale (relative to the
    protocol's noise). Returns array P[w] for w = 0..w_max."""
    probs = []
    for key, cp in superset_sites(proto, plan_all):
        probs.append(np.clip(cp.site_prob * p_scale, 0, 1))
    n = proto.n_data
    for q in (proto.noise.pr, proto.noise.pm):
        if q:
            probs.append(np.full(n, min(q * p_scale, 1.0)))
    probs = np.concatenate(probs)
    # DP in log-safe form: generating function prod (1 - q + q z), truncated at w_max
    P = np.zeros(w_max + 1)
    P[0] = 1.0
    for q in probs:
        P[1:] = P[1:] * (1 - q) + P[:-1] * q
        P[0] *= (1 - q)
    return P
