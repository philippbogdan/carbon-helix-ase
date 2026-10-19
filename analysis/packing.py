"""Time left on the table by the round schedule (frontier: time per round is the criterion
furthest from its limit). Under the Helios timing model a layer costs 3.9 ms plus 7.6 ms per
batch of up to 8 two-qubit gates, so the sparse layers of one outer check (1-4 gates) cost
almost as much as full ones. Two ways of filling them, evaluated on the circuits used:
  * order-preserving packing: the operations of the inner checks (L1) and of the round's
    outer check (L3r_j) layered as early as each qubit allows, every qubit seeing its
    operations (including L1's relabelling of data and ancilla roles and its feedback) in the
    sequential order (the circuit is unchanged, only its timing);
  * full merge: every data CNOT of the outer check inside the four CNOT layers of L1, where
    each data qubit is busy in two of the four; possible only if each cat qubit's four data
    qubits can be given four distinct layers in which they are free.
Writes results/packing.json."""
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hase.protocols import make_protocol  # noqa: E402
from hase.timing import apart_duration  # noqa: E402
from hase.helios import helios_noise  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HN = helios_noise()


def asap(parts):
    """Layers of the parts' operations, each as early as its qubits allow, every qubit seeing
    its operations in the sequential order of the parts (relabelling and feedback operations
    included: they take no time but fix which physical qubit holds a data qubit)."""
    last, layers = {}, []
    for tag, ap in parts:
        for lay in ap.layers:
            for op in lay:
                qs = [(tag, q) if q[0] == "a" else q for q in op[1]]
                t = max(last.get(q, -1) for q in qs) + 1
                while len(layers) <= t:
                    layers.append([])
                layers[t].append(op)
                for q in qs:
                    last[q] = t
    return layers


def duration(layers):
    t = 0.0
    for lay in layers:
        if lay and all(op[0] in ("FBX", "FBZ", "RELABEL") for op in lay):
            continue
        n2 = sum(1 for op in lay if op[0] in ("CX", "CZ"))
        n1 = sum(1 for op in lay if op[0] in ("H", "S", "SDG") or op[0].startswith("C1_"))
        nm = sum(1 for op in lay if op[0] in ("M", "MX"))
        nr = sum(1 for op in lay if op[0] in ("R", "RX"))
        t += HN.layer_duration(n2, n1, nm, nr)
    return t


def main():
    pr = make_protocol("carbonhelix", 7.9e-4, "X", 2, "l2=on_event,l3=ambiguous,rot=8,l2fast=1", "helios")
    L1 = pr.aparts["L1_a"]
    cx = [i for i, l in enumerate(L1.layers) if any(op[0] in ("CX", "CZ") for op in l)]
    busy = {i: {q for op in L1.layers[i] for q in op[1] if q[0] == "d"} for i in cx}
    out = {"L1_ms": apart_duration(L1), "checks": {}}
    for j in range(8):
        ap = pr.aparts[f"L3r_{j}"]
        by_cat = {}
        for lay in ap.layers:
            for op in lay:
                if op[0] in ("CX", "CZ") and any(q[0] == "d" for q in op[1]):
                    a = [q for q in op[1] if q[0] == "a"][0]
                    by_cat.setdefault(a, []).append([q for q in op[1] if q[0] == "d"][0])
        mergeable = all(any(all(d not in busy[L] for d, L in zip(ds, perm))
                            for perm in itertools.permutations(cx, len(ds))) for ds in by_cat.values())
        out["checks"][f"L3r_{j}"] = dict(separate_ms=apart_duration(L1) + apart_duration(ap),
                                         packed_ms=duration(asap([("L1", L1), ("R", ap)])),
                                         full_merge_possible=bool(mergeable))
    v = list(out["checks"].values())
    out["saving_ms_max"] = max(x["separate_ms"] - x["packed_ms"] for x in v)
    out["saving_ms_min"] = min(x["separate_ms"] - x["packed_ms"] for x in v)
    out["full_merge_possible_any"] = any(x["full_merge_possible"] for x in v)
    json.dump(out, open(os.path.join(ROOT, "results", "packing.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("L1_ms", "saving_ms_min", "saving_ms_max", "full_merge_possible_any")}))


if __name__ == "__main__":
    main()
