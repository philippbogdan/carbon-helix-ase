"""Elapsed time of an executed path using its protocol's timing parameters.

Models without durations, including uniform noise, use default Helios timing.
Parts run in sequence except per-block weight-6 checks, which count as one combined
parallel schedule sharing gate batches. Idle placeholders add no separate duration.
This accounts for elapsed time, including preparation; it does not add the idle noise
missing from the individually compiled block parts.
"""
from __future__ import annotations

from functools import lru_cache

from .helios import helios_noise

_HN = helios_noise()


def _model(noise=None):
    return noise if hasattr(noise, "layer_duration") else _HN


def apart_duration(apart, noise=None) -> float:
    model = _model(noise)
    t = 0.0
    for lay in apart.layers:
        if lay and all(op[0] in ("FBX", "FBZ", "RELABEL") for op in lay):
            continue
        n2 = sum(1 for op in lay if op[0] in ("CX", "CZ"))
        n1 = sum(1 for op in lay if op[0] in ("H", "S", "SDG") or op[0].startswith("C1_"))
        nm = sum(1 for op in lay if op[0] in ("M", "MX"))
        nr = sum(1 for op in lay if op[0] in ("R", "RX"))
        t += model.layer_duration(n2, n1, nm, nr)
    return t


def _timing_key(noise=None):
    """Timing parameters distinguishing cached parallel schedules across models."""
    m = _model(noise)
    return (m.t_layer, m.t_batch, m.t_mcmr)


@lru_cache(maxsize=None)
def _l2_parallel(k: int, reuse: bool, key: tuple) -> float:
    from .carbonhelix import level2_sched_apart
    from .codes import carbon_helix, carbon_code
    from dataclasses import replace
    c = carbon_helix()
    model = replace(_HN, t_layer=key[0], t_batch=key[1], t_mcmr=key[2])
    return apart_duration(level2_sched_apart(c.blocks[:k], carbon_code(), reuse=reuse), model)


def l2_parallel_duration(k: int, reuse: bool = True, noise=None) -> float:
    return _l2_parallel(k, reuse, _timing_key(noise))


def path_time_ms(proto, path) -> float:
    """Time in ms for all supplied rounds, including preparation when present."""
    noise = getattr(proto, "noise", None)
    cache = proto.__dict__.setdefault("_part_ms", {})
    total = 0.0
    by_round = {}
    for r, pt in path:
        by_round.setdefault(r, []).append(pt)
    for r, pts in by_round.items():
        l2 = [pt for pt in pts if pt.startswith("L2_") and pt != "L2_all"]
        l2f = [pt for pt in pts if pt.startswith("L2f_")]
        for pt in pts:
            if (pt.startswith("L2_") and pt != "L2_all") or pt.startswith(("W2_", "L2f_", "W2f_")):
                continue
            if pt not in cache:
                cache[pt] = apart_duration(proto.aparts[pt], noise)
            total += cache[pt]
        if l2:
            total += l2_parallel_duration(len(l2), noise=noise)
        if l2f:
            total += l2_parallel_duration(len(l2f), reuse=False, noise=noise)
    return total


def peak_qubits(proto, path) -> int:
    """Peak simultaneously live qubits along the path (data + ancillas of the part, with
    parallel per-block L2 parts counted together)."""
    n = proto.n_data
    peak = n
    by_round = {}
    for r, pt in path:
        by_round.setdefault(r, []).append(pt)
    for r, pts in by_round.items():
        l2 = [pt for pt in pts if (pt.startswith("L2_") and pt != "L2_all") or pt.startswith("L2f_")]
        for pt in pts:
            if (pt.startswith("L2_") and pt != "L2_all") or pt.startswith(("W2_", "L2f_", "W2f_")):
                continue
            peak = max(peak, n + proto.aparts[pt].n_anc)
        if l2:
            peak = max(peak, n + sum(proto.aparts[pt].n_anc for pt in l2))
    return peak
