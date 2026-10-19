"""Physical 3-qubit GHZ under the Helios-like model (src/hase/helios.py), 32 copies in
parallel as on the hardware (arXiv:2609.03194 Sec. 4, physical baseline: Z mismatch
0.209 %, X mismatch 0.253 %): reset (preparation error), H on qubit 0, CNOT 0-1, CNOT 1-2,
measurement in Z, or H on all three first (X). Memory error per ms on every qubit during
each layer (the model's layer durations); leakage after a two-qubit gate as depolarizing on
one of its qubits. A calibration check of the noise model at the physical level (stim sampling).
Writes results/physical_ghz.json."""
import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import stim, numpy as np
from hase.helios import helios_noise
from hase.engine import PAULI_2Q
nz = helios_noise("helios", 7.9e-4)
copies = 32
def layer_mem(c, qs, dt):
    pdep = nz.mem_dep_per_ms * dt
    pz = nz.mem_z_per_ms * dt
    if pdep:
        c.append("DEPOLARIZE1", qs, pdep)
    if pz:
        c.append("Z_ERROR", qs, pz)
def build(basis):
    n = 3 * copies
    allq = list(range(n))
    c = stim.Circuit()
    c.append("R", allq)
    c.append("X_ERROR", allq, nz.pr)
    # H on the first qubit of each copy (1q layer: 0.5 ms in the model)
    h = [3 * k for k in range(copies)]
    c.append("H", h); c.append("DEPOLARIZE1", h, nz.p1); layer_mem(c, allq, 0.5)
    for a, b in ((0, 1), (1, 2)):
        pairs = [x for k in range(copies) for x in (3 * k + a, 3 * k + b)]
        c.append("CX", pairs)
        c.append("PAULI_CHANNEL_2", pairs, [nz.pauli2.get(P, 0.0) for P in PAULI_2Q])
        # leakage -> depolarizing (I, X, Y, Z equally) on one qubit of the gate, as in the engine
        c.append("DEPOLARIZE1", pairs[1::2], nz.leak2 * 0.75)
        dt = nz.layer_duration(copies, 0, 0, 0)
        layer_mem(c, allq, dt)
    if basis == "X":
        c.append("H", allq); c.append("DEPOLARIZE1", allq, nz.p1); layer_mem(c, allq, 0.5)
    c.append("M", allq, nz.pm)
    return c
out = dict(paper_hardware={"Z": 0.209e-2, "X": 0.253e-2}, copies=copies)
for basis in "ZX":
    c = build(basis)
    s = c.compile_sampler(seed=1).sample(200000).astype(np.int64)
    m = s.reshape(-1, copies, 3)
    if basis == "Z":
        bad = (m.min(axis=2) != m.max(axis=2))
    else:
        bad = (m.sum(axis=2) % 2 == 1)
    out[basis] = dict(mismatch=float(bad.mean()), states=int(bad.size))
    print(basis, "mismatch per GHZ state", bad.mean(), flush=True)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", "physical_ghz.json"), "w"), indent=1)
