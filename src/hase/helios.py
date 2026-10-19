"""A Helios-like circuit noise model assembled from published component benchmarks.

Sources (Ransford et al., arXiv:2511.05465 v1 and Nature 655, 81 (2026) + SI; Quantinuum
Helios data sheet v1.2):
  * 2Q gate (native RZZ): cycle-benchmarking Pauli probabilities per element (x1e-5,
    arXiv Table A4): {IX,IY,ZX,ZY} 4.5; {XI,YI,XZ,YZ} 5.8; {XX,XY,YX,YY} 0.06; IZ 19;
    ZI 19; ZZ 5.9; leakage 1.14e-4 per gate (CB). We convert the channel to the CNOT frame
    (CNOT = (I x H) . CZ . (I x H), CZ = RZZ(pi/2) up to S^dag x S^dag).
  * 1Q gate: average infidelity 2.5e-5 (incl. leakage) -> depolarizing 3.75e-5.
  * SPAM: average 3.3e-4 (Nature); attributed to preparation as in Quantinuum's Helios
    emulator defaults (p_init 5e-4, p_meas 1e-6 on an older SPAM figure).
  * MCMR crosstalk: 1.3e-5 infidelity per MCMR per spectator (Nature SI Eq. 10, Table
    VI, mostly leakage) -> depolarizing 2e-5 per spectator per MCMR.
  * Memory: 6e-4 per qubit per depth-1 layer of a 98-qubit dense circuit (data sheet,
    typical), which takes 55 ms (paper Fig. 4); of the linear part, 4e-4 per layer is
    transport-induced leakage. Memory per ms: 1.1e-5, split into leakage-like
    (depolarizing after repump, 7.3e-6/ms) and dephasing (3.6e-6/ms).
  * Timing (paper Fig. 4b fit, labelled an inference): a layer of 2Q gates costs
    3.9 ms + 7.6 ms per batch of 8 gates (16 qubits). Measurement/reset batch durations
    are not published; we assume `t_mcmr` ms per batch of 16 (default 2.0 ms) and report
    sensitivity.
Unpublished: per-round duration on Helios for these circuits, measurement/reset
durations, the leakage-repump efficiency, and the split between preparation and
measurement error. These are model parameters, scanned in the analysis.
"""
from __future__ import annotations

from dataclasses import dataclass

from .ir import NoiseModel

# RZZ-frame cycle-benchmarking Pauli probabilities (arXiv:2511.05465 Table A4), x1e-5
_RZZ = {}
for P in ("IX", "IY", "ZX", "ZY"):
    _RZZ[P] = 4.5e-5
for P in ("XI", "YI", "XZ", "YZ"):
    _RZZ[P] = 5.8e-5
for P in ("XX", "XY", "YX", "YY"):
    _RZZ[P] = 0.06e-5
_RZZ.update({"IZ": 19e-5, "ZI": 19e-5, "ZZ": 5.9e-5})


def _conj(P: str) -> str:
    """Conjugate a 2-qubit Pauli (control, target) error after RZZ through
    S^dag x S^dag (X<->Y on both) and H on the target (X<->Z on target)."""
    a, b = P
    sdg = {"I": "I", "X": "Y", "Y": "X", "Z": "Z"}
    a, b = sdg[a], sdg[b]
    h = {"I": "I", "X": "Z", "Z": "X", "Y": "Y"}
    return a + h[b]


def cnot_pauli_channel(scale=1.0):
    ch = {}
    for P, q in _RZZ.items():
        ch[_conj(P)] = ch.get(_conj(P), 0.0) + q * scale
    return ch


@dataclass
class HeliosNoise(NoiseModel):
    pauli2: dict = None
    leak2: float = 0.0          # per 2Q gate -> depolarizing (1/4 each of I,X,Y,Z) on one qubit
    mem_dep_per_ms: float = 0.0
    mem_z_per_ms: float = 0.0
    xtalk_dep: float = 0.0      # depolarizing on each spectator per MCMR in the layer
    t_layer: float = 3.9        # ms overhead per layer with 2Q gates
    t_batch: float = 7.6        # ms per batch of up to 8 2Q gates (16 qubits)
    t_mcmr: float = 2.0         # ms per batch of up to 16 measurements/resets (assumed)
    timed: bool = True

    def layer_duration(self, n2q: int, n1q: int, nmeas: int, nreset: int) -> float:
        t = 0.0
        if n2q:
            t += self.t_layer + self.t_batch * -(-n2q // 8)
        if nmeas or nreset:
            t += self.t_mcmr * -(-max(nmeas, nreset) // 16)
        if n1q and not n2q and not (nmeas or nreset):
            t += 0.5
        return t


def helios_noise(name: str = "helios", p: float = 7.9e-4, **overrides) -> HeliosNoise:
    """Helios-like model scaled so that its 2Q average infidelity equals `p`
    (p = 7.9e-4 reproduces the published Helios component values). Name suffixes:
    'helios'           all components scaled together
    'helios_mem<x>'    memory rates multiplied by x (sensitivity scans)
    """
    s = p / 7.9e-4
    mem = 1.0
    if "_mem" in name:
        mem = float(name.split("_mem")[1].split("_")[0])
    t_mcmr = 2.0
    if "_tm" in name:
        t_mcmr = float(name.split("_tm")[1].split("_")[0])
    ch = cnot_pauli_channel(scale=s)
    nz = HeliosNoise(p2=0.0, p1=3.75e-5 * s, pm=1e-6 * s, pr=3.3e-4 * s, pidle=0.0,
                     idle_kind="DEP", name=f"{name}(p2q={p:g})", pauli2=ch,
                     leak2=1.14e-4 * s, mem_dep_per_ms=7.3e-6 * s * mem,
                     mem_z_per_ms=3.6e-6 * s * mem, xtalk_dep=2.0e-5 * s, t_mcmr=t_mcmr)
    for k, v in overrides.items():
        setattr(nz, k, v)
    return nz
