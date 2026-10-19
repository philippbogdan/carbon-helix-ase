"""Write complete memory circuits and reusable extraction parts."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from hase.carbonhelix import CarbonHelixProtocol
from hase.engine import emit_stim
from hase.ir import NoiseModel


def build(output, rounds=2, noise=None):
    output.mkdir(parents=True, exist_ok=True)
    noise = noise or NoiseModel.uniform(1e-3)
    metadata = {'noise': vars(noise), 'memory_rounds': rounds, 'preparation_rounds': 1,
                'parts': {}, 'memory': {}}
    for basis in 'XZ':
        protocol = CarbonHelixProtocol(rounds + 1, basis, noise)
        for noiseless in (False, True):
            circuit, _ = emit_stim(protocol, [(r, pt) for r in range(protocol.n_rounds) for pt, _ in protocol.plan(r)], noiseless=noiseless)
            name = f'memory_{basis}' + ('_noiseless' if noiseless else '') + '.stim'
            (output / name).write_text(str(circuit) + '\n')
            metadata['memory'][name] = {'qubits': circuit.num_qubits,
                                       'detectors': circuit.num_detectors,
                                       'observables': circuit.num_observables}
        if basis == 'X':
            for name in ('L1_a', 'L1_b', 'L2_all', 'L3'):
                part = protocol.aparts[name]
                circuit, _ = emit_stim(protocol, [(0, name)], noiseless=True)
                instructions = list(circuit)
                end = next(i for i, op in enumerate(instructions) if op.name == 'DETECTOR')
                final_readout = instructions[end - 1]
                assert final_readout.name == 'MX'
                fragment = circuit[2:end - 1]
                (output / f'{name}.stim').write_text(str(fragment) + '\n')
                compiled = protocol.parts[name]
                metadata['parts'][name] = {
                    'ancillas': part.n_anc,
                    'two_qubit_gates': compiled.two_q,
                    'layers': compiled.n_layers,
                    'data_input': list(range(protocol.n_data)),
                    'data_output': [t.value for t in final_readout.targets_copy()],
                    'measurements': [list(k) for k in compiled.meas_keys],
                    'measurement_roles': [list(part.meas[k]) for k in compiled.meas_keys],
                }
            (output / 'code.json').write_text(json.dumps({
                k: getattr(protocol.code, k).tolist() for k in ('hx', 'hz', 'lx', 'lz')
            }, indent=2) + '\n')
    (output / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'build/circuits')
    parser.add_argument('--rounds', type=int, default=2, help='Memory rounds after preparation')
    parser.add_argument('--p', type=float, default=1e-3, help='Uniform model error probability')
    parser.add_argument('--p2', type=float)
    parser.add_argument('--p1', type=float)
    parser.add_argument('--pm', type=float)
    parser.add_argument('--pr', type=float)
    parser.add_argument('--pidle', type=float)
    args = parser.parse_args()
    noise = NoiseModel.uniform(args.p)
    for key in ('p2', 'p1', 'pm', 'pr', 'pidle'):
        value = getattr(args, key)
        if value is not None:
            setattr(noise, key, value)
            noise.name = 'custom'
    if not all(0 <= getattr(noise, key) <= 1 for key in ('p2','p1','pm','pr','pidle')):
        parser.error('Noise probabilities must lie between 0 and 1.')
    result = build(args.out, args.rounds, noise)
    print(f"Built X and Z memory circuits in {args.out}")
