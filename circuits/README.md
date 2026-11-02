# Reusable extraction circuits

Full Carbon extraction uses 560 two-qubit gates and at most 92 qubits. `L1_a.stim` and `L1_b.stim` are alternating inner checks; `L2_all.stim` measures all block checks; `L3.stim` measures all outer checks with halfway cat-parity readout.

`memory_X.stim` and `memory_Z.stim` contain one preparation round, two memory rounds, final readout, detectors and two logical observables, under uniform noise at $`p=0.001`$. The `_noiseless` files omit noise. These examples differ from the headline's four-round Helios-like memories.

`manifest.json` gives noise, measurement roles and input/output qubit maps. L1 moves data into refreshed qubits: use its output map when joining parts. `code.json` contains stabilisers and logical operators. `./run.sh check` requires exact agreement with rebuilt files. [USE.md](../USE.md) explains porting and live adaptive paths.
