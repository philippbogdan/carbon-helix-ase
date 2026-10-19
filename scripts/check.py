"""Rebuild the circuits, check them, then sample and decode a short memory run."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'third_party/frontier'), str(ROOT / 'build/native')]
import numpy as np
import stim
from build import build
from check_parities import check_parities
from hase.decoding import FrontierDecoder


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shots', type=int, default=128, help='Noisy shots per basis')
    parser.add_argument('--seed', type=int, default=19019)
    args = parser.parse_args()
    if args.shots < 1:
        parser.error('--shots must be positive')
    out = ROOT / 'build/circuits'
    manifest = build(out)
    hashes = {}
    for file in sorted((ROOT / 'circuits').iterdir()):
        if file.suffix not in ('.stim', '.json'):
            continue
        require((out / file.name).read_bytes() == file.read_bytes(),
                f'Rebuilt circuit differs from circuits/{file.name}')
        hashes[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
    report = {'python': platform.python_version(), 'platform': platform.platform(),
              'versions': {k: importlib.metadata.version(k) for k in ('stim','numpy','scipy')},
              'seed': args.seed, 'circuit_sha256': hashes, 'checks': {}, 'decoded': {},
              'halfway_parities': check_parities()}
    for basis in 'XZ':
        circuit = stim.Circuit.from_file(out / f'memory_{basis}_noiseless.stim')
        circuit.detector_error_model()
        dets, obs = circuit.compile_detector_sampler(seed=args.seed).sample(1000, separate_observables=True)
        require(circuit.num_qubits == 92 and circuit.num_observables == 2, 'Unexpected circuit size')
        require(not dets.any() and not obs.any(), f'Noiseless {basis} check failed')
        report['checks'][basis] = {'shots': 1000, 'detection_events': int(dets.sum()),
                                   'logical_flips': int(obs.sum()), 'qubits': circuit.num_qubits}
        circuit = stim.Circuit.from_file(out / f'memory_{basis}.stim')
        dets, obs = circuit.compile_detector_sampler(seed=args.seed).sample(args.shots, separate_observables=True)
        print(f'{basis}: noiseless check passed; decoding {args.shots} noisy shots', flush=True)
        decoder = FrontierDecoder(circuit.detector_error_model(), K=512, Delta=12.0)
        pred, gaps, statuses = decoder.decode_batch(dets)
        require(all(status in ('ok', 'no_path') for status in statuses), f'{basis}: unexpected Frontier status')
        decoded = np.array([status == 'ok' for status in statuses])
        require(np.any(decoded & dets.any(axis=1)), f'{basis}: no nonzero syndrome was decoded')
        failures = np.any(pred != obs, axis=1) | ~decoded
        print(f'{basis}: {int(failures.sum())} failed shots, including {int((~decoded).sum())} with no decoding path', flush=True)
        np.savez_compressed(ROOT / f'build/samples_{basis}.npz', detectors=dets, observables=obs,
                            predictions=pred, gaps=gaps, statuses=np.array(statuses))
        report['decoded'][basis] = {'shots': args.shots, 'block_failures': int(failures.sum()),
                                    'decoder_no_path': int((~decoded).sum()),
                                    'shots_with_detection_events': int(dets.any(axis=1).sum()),
                                    'status_counts': dict(Counter(statuses)),
                                    'K': 512, 'Delta': 12.0, 'score_alpha': 0.8, 'committee': False}
    report['parts'] = {k: {f: v[f] for f in ('ancillas','two_qubit_gates','layers')}
                       for k,v in manifest['parts'].items()}
    (ROOT / 'build/check.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Passed. Raw samples and check.json are in build/. This is a run check, not a gain estimate.')


if __name__ == '__main__':
    main()
