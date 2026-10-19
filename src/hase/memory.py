"""Generic memory-experiment assembly on top of the layered IR.

A *scheme* provides, for each round, a list of 'blocks of work' (inner part, outer part),
each a list of layers built for the current data-slot -> physical-qubit mapping, plus the
names of the checks it measures. This module turns a sequence of executed parts into a
stim circuit with detectors and observables, keeping track of the last measurement of
every check so that detectors compare consecutive measurements of the same check.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import stim

from .ir import NoiseModel, Op, Part, StimEmitter


@dataclass
class CheckInfo:
    name: tuple            # e.g. ('in', b, 'X') or ('out', 'X', i)
    kind: str              # 'X' or 'Z'
    support: tuple         # data slots


@dataclass
class RoundRecord:
    """What a part measured: list of (check_name, meas_key) and flag keys."""
    checks: list = field(default_factory=list)
    flags: list = field(default_factory=list)


class DetectorBook:
    """Turns check measurements into detectors (consecutive comparisons)."""

    def __init__(self, checks: dict, basis: str):
        self.checks = checks           # name -> CheckInfo
        self.basis = basis             # memory basis 'X' or 'Z' (data init/final basis)
        self.last = {}                 # name -> meas key of last measurement
        self.dets = []                 # list of (keys, coords, info)

    def on_check(self, name, key, round_index, part_label):
        info = self.checks[name]
        prev = self.last.get(name)
        if prev is not None:
            self.dets.append(([key, prev], (round_index,), ("chk", name, round_index, part_label)))
        elif info.kind == self.basis:
            # deterministic first measurement (data initialised in the basis eigenstate)
            self.dets.append(([key], (round_index,), ("chk", name, round_index, part_label)))
        self.last[name] = key

    def on_flag(self, key, round_index, label):
        self.dets.append(([key], (round_index,), ("flag", key, round_index, label)))


def build_memory_circuit(scheme, rounds: int, basis: str, noise: NoiseModel,
                         full_rounds=None, noiseless_rounds=()):
    """Assemble a (possibly adaptive-path) memory experiment as a stim circuit.

    scheme: object with .n_data, .checks (dict), .logicals(basis) -> list of supports,
            .init_mapping() -> (phys list, pool), .round_parts(r, full) -> list of
            (Part, RoundRecord, label) built on the scheme's current mapping.
    full_rounds: iterable of booleans (outer part executed) per round; default all True.
    Returns (circuit, detector_infos, meta).
    """
    if full_rounds is None:
        full_rounds = [True] * rounds
    full_rounds = list(full_rounds)
    scheme.reset()
    em = StimEmitter(noise)
    book = DetectorBook(scheme.checks, basis)
    # initialise data
    init = Part(name="init")
    init.add_layer([Op("RX" if basis == "X" else "R", (q,)) for q in scheme.data_phys()])
    em.emit_part(init, noiseless=False)
    two_q = 0
    for r in range(rounds):
        for part, rec, label in scheme.round_parts(r, full_rounds[r]):
            em.emit_part(part, noiseless=(r in noiseless_rounds))
            two_q += part.count({"CX", "CZ", "SWAPG"})
            for name, key in rec.checks:
                book.on_check(name, key, r, label)
            for key in rec.flags:
                book.on_flag(key, r, label)
    # final destructive measurement
    fin = Part(name="final")
    phys = scheme.data_phys()
    fin.add_layer([Op("MX" if basis == "X" else "M", (phys[s],), ("final", s))
                   for s in range(scheme.n_data)])
    em.emit_part(fin)
    for name, info in scheme.checks.items():
        if info.kind != basis:
            continue
        keys = [("final", s) for s in info.support]
        prev = book.last.get(name)
        if prev is not None:
            keys = keys + [prev]
        book.dets.append((keys, (rounds,), ("final", name, rounds, "final")))
    for keys, coords, info in book.dets:
        em.detector(keys, coords)
    for i, supp in enumerate(scheme.logicals(basis)):
        em.observable([("final", s) for s in supp], i)
    meta = {"two_qubit_gates": two_q, "n_meas": em.nmeas}
    return em.c, [d[2] for d in book.dets], meta
