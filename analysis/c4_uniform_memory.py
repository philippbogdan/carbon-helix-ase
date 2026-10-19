"""Aggregate C4-Helix uniform-noise memory runs and compare with arXiv:2609.03194 Fig. 11a."""
import glob, json, sys
from collections import defaultdict
import numpy as np
from scipy.stats import beta

# Paper values (block LER per round, gamma=0, average of X and Z bases, r=12), digitized
# from figures/frontier_extrapolation_memory.png marker centres (see analysis/digitize_fig11.py)
PAPER = {7e-4: 7.53e-6, 9e-4: 1.70e-5, 1.1e-3: 3.46e-5}

def cp(k, n, a=0.05):
    lo = beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi

def load(pattern):
    agg = defaultdict(lambda: dict(shots=0, fails=0, lo=0, re=0, ch=0, n=0))
    for f in glob.glob(pattern):
        d = json.load(open(f))
        a = agg[(d['p'], d['basis'], d['rounds'])]
        a['shots'] += d['shots']; a['fails'] += d['fails']; a['lo'] += d['fails_lowprec']
        a['re'] += d['rechecked']; a['ch'] += d['changed_by_recheck']; a['n'] += 1
    return agg

def eps(k, n, r):
    return 1 - (1 - k / n) ** (1 / r)

def main(pattern='results/raw/c4u_mem*/*/*.json'):
    agg = load(pattern)
    rows = []
    for (p, b, r), a in sorted(agg.items()):
        lo, hi = cp(a['fails'], a['shots'])
        e = eps(a['fails'], a['shots'], r)
        rows.append((p, b, r, a['shots'], a['fails'], e, eps(lo * a['shots'], a['shots'], r), eps(hi * a['shots'], a['shots'], r), a['lo']))
        print(f"p={p:.1e} basis={b} r={r}: shots={a['shots']} fails={a['fails']} (K512 {a['lo']}) eps={e:.3e} [{eps(lo*a['shots'],a['shots'],r):.2e},{eps(hi*a['shots'],a['shots'],r):.2e}]")
    print()
    for r in sorted(set(x[2] for x in rows)):
        for p in sorted(PAPER):
            ex = [x for x in rows if x[0] == p and x[2] == r]
            if len(ex) == 2:
                avg = 0.5 * (ex[0][5] + ex[1][5])
                # stat error by propagating CP half-widths (approx)
                err = 0.5 * np.hypot((ex[0][7] - ex[0][6]) / 2, (ex[1][7] - ex[1][6]) / 2) / 1.96
                print(f"r={r} p={p:.1e}: ours avg(X,Z)={avg:.3e} +- {err:.1e} (1 sigma)  paper {PAPER[p]:.3e}  ratio {avg/PAPER[p]:.3f} +- {err/PAPER[p]:.3f}")

if __name__ == '__main__':
    main(*sys.argv[1:])
