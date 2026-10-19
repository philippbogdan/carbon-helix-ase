"""A small layered circuit IR, noise models, and emission to stim.

A *part* is a list of layers; a layer is a list of ``Op`` acting on disjoint qubits.
Measurements carry a ``key`` (any hashable) so detectors can be built by name.
Feedback ops ``FBX``/``FBZ`` apply a Pauli conditioned on the result of an earlier
measurement (by key); they are noiseless Pauli-frame updates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable

import stim

ONE_Q = {"H", "S", "SDG", "X", "Y", "Z", "SQRT_X", "SQRT_X_DAG", "I"}
TWO_Q = {"CX", "CZ", "SWAPG"}            # SWAPG = physical swap gate (noisy)
RESETS = {"R", "RX"}
MEASURES = {"M", "MX"}
FEEDBACK = {"FBX", "FBZ"}


@dataclass(frozen=True)
class Op:
    name: str
    qubits: tuple
    key: Hashable = None


@dataclass
class NoiseModel:
    """Circuit-level Pauli noise.

    p2: two-qubit depolarizing after 2q gates; p1: one-qubit depolarizing after 1q gates;
    pm: measurement flip; pr: reset flip; pidle: idle error per idle layer for live qubits,
    of type idle_kind ('Z' = dephasing, 'DEP' = depolarizing).
    """
    p2: float = 0.0
    p1: float = 0.0
    pm: float = 0.0
    pr: float = 0.0
    pidle: float = 0.0
    idle_kind: str = "Z"
    name: str = "custom"

    @staticmethod
    def uniform(p: float, idle_kind: str = "Z") -> "NoiseModel":
        """C4-Helix paper (arXiv:2609.03194, App. C) model: reset p, measure p,
        1q depolarizing p/10, 2q depolarizing p, idle dephasing p/10."""
        return NoiseModel(p2=p, p1=p / 10, pm=p, pr=p, pidle=p / 10, idle_kind=idle_kind,
                          name=f"uniform(p={p:g})")

    def scaled(self, s: float) -> "NoiseModel":
        return NoiseModel(self.p2 * s, self.p1 * s, self.pm * s, self.pr * s, self.pidle * s,
                          self.idle_kind, self.name + f"*{s:g}")


@dataclass
class Part:
    """A list of layers plus bookkeeping of which qubits are live (hold state)."""
    layers: list = field(default_factory=list)
    name: str = ""

    def add_layer(self, ops):
        qs = [q for op in ops for q in op.qubits]
        assert len(qs) == len(set(qs)), f"layer acts twice on a qubit: {ops}"
        self.layers.append(list(ops))

    def ops(self):
        for layer in self.layers:
            yield from layer

    def count(self, names):
        return sum(1 for op in self.ops() if op.name in names)


class StimEmitter:
    """Emits parts into a stim.Circuit with noise, tracking measurement indices by key."""

    def __init__(self, noise: NoiseModel, live: set | None = None, tick=True):
        self.c = stim.Circuit()
        self.noise = noise
        self.meas_index = {}      # key -> absolute measurement index
        self.nmeas = 0
        self.live = set(live or ())
        self.tick = tick

    def _rec(self, key):
        return stim.target_rec(self.meas_index[key] - self.nmeas)

    def emit_layer(self, layer, noiseless=False):
        nz = self.noise
        touched = set()
        only_feedback = all(op.name in FEEDBACK for op in layer)
        for op in layer:
            n, qs = op.name, op.qubits
            touched.update(qs)
            if n in RESETS:
                self.c.append(n, qs)
                if nz.pr and not noiseless:
                    self.c.append("X_ERROR" if n == "R" else "Z_ERROR", qs, nz.pr)
                self.live.update(qs)
            elif n in MEASURES:
                if nz.pm and not noiseless:
                    self.c.append(n, qs, nz.pm)
                else:
                    self.c.append(n, qs)
                assert len(qs) == 1
                self.meas_index[op.key] = self.nmeas
                self.nmeas += 1
                self.live.discard(qs[0])
            elif n in ONE_Q:
                self.c.append(n, qs)
                if nz.p1 and not noiseless:
                    self.c.append("DEPOLARIZE1", qs, nz.p1)
            elif n in TWO_Q:
                self.c.append("SWAP" if n == "SWAPG" else n, qs)
                if nz.p2 and not noiseless:
                    self.c.append("DEPOLARIZE2", qs, nz.p2)
            elif n in FEEDBACK:
                # classically controlled Pauli: key -> qubit (noiseless frame update)
                self.c.append("CX" if n == "FBX" else "CZ", [self._rec(op.key), qs[0]])
            elif n == "MPP":     # noiseless Pauli product measurement, qs = stim targets list
                self.c.append("MPP", list(qs))
                self.meas_index[op.key] = self.nmeas
                self.nmeas += 1
            else:
                raise ValueError(n)
        if only_feedback:
            return   # classical frame update: takes no time, no idle noise, no TICK
        if nz.pidle and not noiseless:
            idle = sorted(q for q in self.live if q not in touched)
            if idle:
                if nz.idle_kind == "Z":
                    self.c.append("Z_ERROR", idle, nz.pidle)
                else:
                    self.c.append("DEPOLARIZE1", idle, nz.pidle)
        if self.tick:
            self.c.append("TICK")

    def emit_part(self, part: Part, noiseless=False):
        for layer in part.layers:
            self.emit_layer(layer, noiseless=noiseless)

    def detector(self, keys, coords=()):
        self.c.append("DETECTOR", [self._rec(k) for k in keys], list(coords))

    def observable(self, keys, index):
        self.c.append("OBSERVABLE_INCLUDE", [self._rec(k) for k in keys], index)
