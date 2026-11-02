"""Sample the checks that adaptive rule R8 executes, shot by shot, and the gates of every part."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from hase.engine import simulate  # noqa: E402
from hase.protocols import make_protocol  # noqa: E402

R8 = 'l2=on_event,l3=ambiguous,rot=8'


def label(q):
    return f'd{q[1]}' if q[0] == 'd' else f'a{q[1]}'


def part_table(proto):
    out = {}
    for name, ap in proto.aparts.items():
        layers = [[[label(q) for q in op[1]] for op in layer if op[0] in ('CX', 'CZ')] for layer in ap.layers]
        out[name] = dict(two_qubit_gates=proto.parts[name].two_q, ancillas=ap.n_anc,
                         duration_ms=float(proto.parts[name].duration),
                         gate_layers=[layer for layer in layers if layer])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--p', type=float, default=0.00079)
    ap.add_argument('--rounds', type=int, default=4)
    ap.add_argument('--shots', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--out', type=Path, default=ROOT / 'figures/data/r8_paths.json')
    a = ap.parse_args()
    proto = make_protocol('carbonhelix', a.p, 'X', a.rounds, R8, 'helios')
    full = make_protocol('carbonhelix', a.p, 'X', a.rounds, 'full', 'helios')
    state = simulate(proto, a.shots, np.random.default_rng(a.seed))
    shots = []
    for path in state.paths:
        rounds = [[] for _ in range(proto.n_rounds)]
        for r, part in path:
            rounds[r].append(part)
        shots.append(rounds)
    full_round = [part for part, _ in full.plan(1)]
    twoq = [[sum(proto.parts[p].two_q for p in rnd) for rnd in shot] for shot in shots]
    escalated = [[('L3' in rnd) for rnd in shot[1:]] for shot in shots]
    out = dict(p=a.p, noise='helios', basis='X', rule=R8, rounds_after_preparation=a.rounds, seed=a.seed,
               shots=a.shots, data_qubits=proto.n_data, blocks=[list(range(12 * b, 12 * b + 12)) for b in range(5)],
               ancilla_pool=32, qubit_capacity=98,
               full_round_parts=full_round,
               full_round_two_qubit_gates=int(sum(full.parts[p].two_q for p in full_round)),
               mean_two_qubit_gates_per_round=float(np.mean([t[1:] for t in twoq])),
               escalated_round_fraction=float(np.mean(escalated)),
               parts=part_table(proto), paths=shots)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, separators=(',', ':')) + '\n')
    print(f"{a.shots} shots: {out['mean_two_qubit_gates_per_round']:.1f} two-qubit gates per round "
          f"(full: {out['full_round_two_qubit_gates']}), {100 * out['escalated_round_fraction']:.1f}% of rounds "
          f"escalate to the full outer measurement; written to {a.out}")


if __name__ == '__main__':
    main()
