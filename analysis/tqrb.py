"""Aggregate TQRB results and fit the logical error per Clifford as in arXiv:2609.03194:
p_L(m) = (eps0 - 1/2)(1 - 2 eps_C)^m + 1/2, with eps0 shared between the adaptive and
non-adaptive datasets (maximum likelihood on the binomial counts)."""
import glob
import json
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize

# Digitized paper values (Fig. 2f, Helios hardware TQRB, mean two-qubit gates per shot)
PAPER_GATES = {5: (1531, 1021), 7: (2140, 1430), 13: (3965, 2653)}
# Paper simulation values (App. Fig. 13 legend, uniform noise): eps_C (adaptive, non-adaptive)
PAPER_EPS = {1e-3: (2.8e-4, 4.2e-4), 5e-4: (2.6e-5, 4.7e-5)}


def load(pattern):
    agg = defaultdict(lambda: defaultdict(float))
    hist = defaultdict(lambda: defaultdict(int))
    for f in glob.glob(pattern):
        d = json.load(open(f))
        key = (d["noise"], d["p"], d["adaptive"], d["m"])
        for k in ("shots", "fails", "twoq", "dur", "s0s1", "seqs"):
            agg[key][k] += d[k]
        for g, c in d.get("hist2q", {}).items():
            hist[key][int(g)] += c
    return agg, hist


def fit_eps(counts):
    """counts: dict (adaptive, m) -> (fails, shots). Returns eps0, eps_C(adaptive), eps_C(non)."""
    ad_keys = sorted(set(a for a, m in counts))

    def nll(x):
        eps0 = x[0]
        tot = 0.0
        for i, a in enumerate(ad_keys):
            ec = x[1 + i]
            for (aa, m), (k, n) in counts.items():
                if aa != a:
                    continue
                pl = (eps0 - 0.5) * (1 - 2 * ec) ** m + 0.5
                pl = min(max(pl, 1e-15), 1 - 1e-15)
                tot -= k * np.log(pl) + (n - k) * np.log(1 - pl)
        return tot
    x0 = [1e-4] + [1e-4] * len(ad_keys)
    res = minimize(nll, x0, method="L-BFGS-B", bounds=[(0, 0.1)] * len(x0))
    return dict(eps0=res.x[0], **{f"epsC_a{a}": res.x[1 + i] for i, a in enumerate(ad_keys)}), res


def bootstrap(counts, nboot=300, rng=None):
    rng = rng or np.random.default_rng(0)
    out = []
    for _ in range(nboot):
        c2 = {k: (rng.binomial(n, k_ / n), n) for k, (k_, n) in counts.items()}
        out.append(fit_eps(c2)[0])
    return out


if __name__ == "__main__":
    agg, hist = load("results/final/tqrb_counts2/*/*/*/*.json")
    print("Gate counts and shot durations (no decoding):")
    for key in sorted(agg):
        a = agg[key]
        n = a["shots"]
        print(f"  {key}: shots {n:.0f} 2q/shot {a['twoq']/n:.0f}  S0S1/shot {a['s0s1']/n:.2f} "
              f"dur/shot {a['dur']/n:.0f}")
    agg2, _ = load("results/final/tqrb_eps2/*/*/*/*.json")
    by_np = defaultdict(dict)
    for (noise, p, ad, m), a in agg2.items():
        by_np[(noise, p)][(ad, m)] = (int(a["fails"]), int(a["shots"]))
    for (noise, p), counts in sorted(by_np.items()):
        fit, res = fit_eps(counts)
        print(f"{noise} p={p}: {counts} -> {fit}")
