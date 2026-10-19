"""Protocol registry: build a protocol from a short name and a rule string.

Rule strings are comma-separated key=value pairs, or the shortcut 'full'.
  c4helix:      mode=full|ase|never, T=<forced interval>
  carbonhelix:  l2=always|on_event|never, l3=always|any|ambiguous|never, T2=<int>, T3=<int>,
                g3=<outer-check gadget, carbonhelix.OUTER_DESIGNS>
`rounds` counts rounds after the projective-preparation round (round 0, always full).
"""
from __future__ import annotations

from .ir import NoiseModel


def parse_rule(rule: str) -> dict:
    if rule in ("", "full"):
        return {}
    out = {}
    for item in rule.split(","):
        k, v = item.split("=")
        out[k.strip()] = v.strip()
    return out


def make_noise(noise: str, p: float) -> NoiseModel:
    if noise == "uniform":
        return NoiseModel.uniform(p)
    if noise.startswith("helios"):
        from .helios import helios_noise
        return helios_noise(noise, p)
    raise ValueError(noise)


def make_protocol(name: str, p: float, basis: str, rounds: int, rule: str = "full",
                  noise: str = "uniform"):
    nz = make_noise(noise, p)
    r = parse_rule(rule)
    if name == "c4helix":
        from .c4helix import C4HelixProtocol
        mode = r.get("mode", "full")
        T = int(r["T"]) if "T" in r else None
        return C4HelixProtocol(rounds + 1, basis, nz, mode=mode, forced=T)
    if name == "carbonhelix" and "rot" in r:
        from .variants import CarbonHelixRotating
        return CarbonHelixRotating(rounds + 1, basis, nz, period=int(r["rot"]),
                                   escalate=r.get("l3", "ambiguous"), l2=r.get("l2", "on_event"),
                                   fast_l2=r.get("l2fast", "0") == "1", outer=r.get("g3"),
                                   l2_type=r.get("l2type", "0") == "1")
    if name == "carbonhelix" and r.get("l2fast", "0") == "1":
        # parallel weight-6 checks without rotating outer checks (e.g. the periodic rule)
        from .variants import CarbonHelixRotating
        return CarbonHelixRotating(rounds + 1, basis, nz, period=8, escalate=r.get("l3", "always"),
                                   l2=r.get("l2", "on_event"), fast_l2=True, outer=r.get("g3"),
                                   l2_type=r.get("l2type", "0") == "1",
                                   T2=int(r["T2"]) if "T2" in r else None,
                                   T3=int(r["T3"]) if "T3" in r else None, rotate=False)
    if name == "carbonhelix":
        from .carbonhelix import CarbonHelixProtocol
        return CarbonHelixProtocol(rounds + 1, basis, nz, l2=r.get("l2", "always"),
                                   l3=r.get("l3", "always"),
                                   T2=int(r["T2"]) if "T2" in r else None,
                                   T3=int(r["T3"]) if "T3" in r else None, outer=r.get("g3"),
                                   outer_order=r.get("g3order", "XZ"))
    if name == "toric10":
        from .toric10 import ToricProtocol
        return ToricProtocol(rounds + 1, basis, nz)
    raise ValueError(name)
