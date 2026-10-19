"""Wall-clock accounting of an executed path under the Helios timing model.

Parts of one round run in sequence, except that per-block weight-6 parts (L2_b) and their
waits (W2_b) of one round run in parallel: they count once, with the duration of the
parallel schedule of the k blocks that executed (gates of parallel blocks share the
batches of the gate zones)."""
from __future__ import annotations

from functools import lru_cache

from .helios import helios_noise

_HN = helios_noise()


def apart_duration(apart) -> float:
    t = 0.0
    for lay in apart.layers:
        if lay and all(op[0] in ("FBX", "FBZ", "RELABEL") for op in lay):
            continue
        n2 = sum(1 for op in lay if op[0] in ("CX", "CZ"))
        n1 = sum(1 for op in lay if op[0] in ("H", "S", "SDG") or op[0].startswith("C1_"))
        nm = sum(1 for op in lay if op[0] in ("M", "MX"))
        nr = sum(1 for op in lay if op[0] in ("R", "RX"))
        t += _HN.layer_duration(n2, n1, nm, nr)
    return t


@lru_cache(maxsize=None)
def l2_parallel_duration(k: int, reuse: bool = True) -> float:
    from .carbonhelix import level2_sched_apart
    from .codes import carbon_helix, carbon_code
    c = carbon_helix()
    return apart_duration(level2_sched_apart(c.blocks[:k], carbon_code(), reuse=reuse))


def path_time_ms(proto, path) -> float:
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
                cache[pt] = apart_duration(proto.aparts[pt])
            total += cache[pt]
        if l2:
            total += l2_parallel_duration(len(l2))
        if l2f:
            total += l2_parallel_duration(len(l2f), reuse=False)
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
