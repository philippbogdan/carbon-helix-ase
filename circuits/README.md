# Extraction circuits

Full Carbon-Helix extraction needs 92 physical qubits and 560 two-qubit gates per round. Ancilla reuse lets the [[60,2,12]] code fit within 98 qubits. This is a simulation result; hardware compilation and execution remain to be tested.

`memory_X.stim` and `memory_Z.stim` contain preparation, two memory rounds, readout, noise, detectors and two logical observables. The `_noiseless` files remove noise.

| Reusable part | Two-qubit gates |
| --- | ---: |
| `L1_a`, `L1_b` | 120 each |
| `L2_all` | 200 |
| `L3` | 240 |

`manifest.json` records measurement order, roles and data output maps. L1 moves data, so use its output map when connecting parts. `code.json` gives stabilisers and paired logical operators.

Run `./run.sh` from the root to require exact agreement with rebuilt files. [Porting instructions](../USE.md) explain feedback and adaptive paths. Use these clean parts for integration; historical files in `results/circuits/` can include preparation and readout despite names that mention only one part.
