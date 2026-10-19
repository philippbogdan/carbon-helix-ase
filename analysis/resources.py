"""Resource accounting per part and per round: two-qubit gates, layers, modelled Helios
wall-clock time, and peak simultaneously live qubits (data + ancillas)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hase.protocols import make_protocol  # noqa: E402


def peak_live(apart, n_data):
    """Peak number of simultaneously live qubits in a part (data slots always live;
    an ancilla is live from its reset to its measurement; relabels swap names)."""
    live_anc = set()
    peak = 0
    for lay in apart.layers:
        for op in lay:
            if op[0] in ("R", "RX") and op[1][0][0] == "a":
                live_anc.add(op[1][0])
        peak = max(peak, n_data + len(live_anc))
        for op in lay:
            if op[0] in ("M", "MX") and op[1][0][0] == "a":
                live_anc.discard(op[1][0])
    return peak


def part_table(proto_name, noise, p, rule="full", basis="X"):
    proto = make_protocol(proto_name, p, basis, 2, rule, noise)
    rows = {}
    for k, ap in proto.aparts.items():
        cp = proto.parts[k]
        rows[k] = dict(two_q=cp.two_q, layers=cp.n_layers, duration=cp.duration,
                       peak_qubits=peak_live(ap, proto.n_data), faults=float(cp.site_prob.sum()))
    return rows


if __name__ == "__main__":
    for name, rule in (("c4helix", "mode=ase"), ("carbonhelix", "l2=on_event,l3=ambiguous,T3=4"),
                       ("toric10", "full")):
        for noise, p in (("uniform", 1e-3), ("helios", 7.9e-4)):
            t = part_table(name, noise, p, rule)
            print(name, noise)
            for k, v in t.items():
                if k.startswith(("L2_", "W2_")) and k not in ("L2_all", "L2_0", "W2_0"):
                    continue
                print(f"   {k:10s} 2q={v['two_q']:4d} layers={v['layers']:3d} t={v['duration']:7.1f} "
                      f"peak_qubits={v['peak_qubits']:3d} faults/part={v['faults']:.3f}")
