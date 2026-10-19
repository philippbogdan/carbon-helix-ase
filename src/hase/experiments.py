"""Named experiment configurations (one place that fixes every modelling choice).

`rounds` counts syndrome-extraction rounds after the projective preparation round: the
data are initialised in a product state and one full extraction round projects them into
the code (its outcomes for the random-sign checks are the reference), followed by
`rounds` rounds and a destructive measurement. Per-round error rates are normalised by
`rounds`, as in arXiv:2609.03194 (eps = 1 - (1 - p_L)^(1/r)).
"""
from __future__ import annotations

from .c4helix import C4HelixScheme
from .ir import NoiseModel
from .memory import build_memory_circuit


def scheme_for(exp: str):
    if exp.startswith("c4helix"):
        return C4HelixScheme(outer_rows=(0, 1, 2, 3), ordering="xpair")
    raise ValueError(exp)


def noise_for(exp: str, p: float) -> NoiseModel:
    if exp.endswith("_uniform"):
        return NoiseModel.uniform(p)
    raise ValueError(exp)


def build_experiment(exp: str, p: float, basis: str, rounds: int):
    sch = scheme_for(exp)
    nz = noise_for(exp, p)
    c, dets, meta = build_memory_circuit(sch, rounds + 1, basis, nz)
    meta = dict(meta, scheme=type(sch).__name__, noise=nz.name, total_se_rounds=rounds + 1)
    return c, meta
