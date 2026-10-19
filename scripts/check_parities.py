"""Show a cat fault that the halfway read catches and the final flags miss."""
import numpy as np
import stim
from hase.carbonhelix import cat4_16_parity_layers, OUTER_DESIGNS, DEFAULT_OUTER


def check_parities():
    results = {}
    for kind in 'XZ':
        measurements = []
        data = [('d', q) for q in range(16)]
        cat = tuple(('a', q) for q in range(4))
        parity = tuple(('a', q) for q in range(4, 8))
        flags = [('flag', q) for q in range(7)]
        layers = cat4_16_parity_layers(kind, data, cat, parity, ('check',), flags, OUTER_DESIGNS[DEFAULT_OUTER])
        circuit = stim.Circuit()
        circuit.append('RX' if kind == 'X' else 'R', range(16))
        data_layer = 0
        for layer in layers:
            touches_data = False
            for op in layer:
                name, qubits = op[:2]
                physical = [q if group == 'd' else 16 + q for group, q in qubits]
                circuit.append(name, physical)
                if name in ('M', 'MX'):
                    measurements.append(op[2])
                if name == 'CX' and any(group == 'd' for group, _ in qubits):
                    touches_data = True
            if touches_data:
                data_layer += 1
                if data_layer in (1, 3):
                    circuit.append('X' if kind == 'X' else 'Z', [16])
        samples = circuit.compile_sampler(seed=19019).sample(16)
        mid = samples[:, [measurements.index(flag) for flag in flags[:4]]]
        final = samples[:, [measurements.index(flag) for flag in flags[4:]]]
        expected = np.array([1, 0, 0, 1], dtype=bool)
        if not np.all(mid == expected) or final.any():
            raise RuntimeError(f'{kind} halfway parity check failed')
        results[kind] = {'shots': 16, 'midpoint_flags': expected.astype(int).tolist(),
                         'final_flags': [0, 0, 0],
                         'fault': kind + ' on cat role 0 after data layers 1 and 3'}
    return results
