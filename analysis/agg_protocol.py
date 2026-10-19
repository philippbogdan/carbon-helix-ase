"""Aggregate job_protocol.py outputs: per (proto, rule, noise, p, basis, rounds)."""
import glob, json, sys
from collections import defaultdict
import numpy as np
from scipy.stats import beta

def cp(k, n, a=0.05):
    lo = beta.ppf(a/2, k, n-k+1) if k > 0 else 0.0
    hi = beta.ppf(1-a/2, k+1, n-k) if k < n else 1.0
    return lo, hi

def load(pattern):
    agg = defaultdict(lambda: defaultdict(float))
    for f in glob.glob(pattern):
        d = json.load(open(f))
        key = (d['proto'], d['rule'], d['noise'], d['p'], d['basis'], d['rounds'])
        a = agg[key]
        for k in ('shots', 'fails', 'twoq_sum', 'seconds', 'rechecked', 'changed'):
            a[k] += d[k]
        for k, v in d['part_counts'].items():
            a['pc_' + k] += v
        for k in ('gap_hist', 'fail_gap_hist'):
            if k in d:
                h = a.setdefault(k, np.zeros(len(d[k])))
                h += np.asarray(d[k], dtype=float)
        a['tasks'] += 1
        a['part_twoq'] = d['part_twoq']
    return agg

def eps(k, n, r):
    """Per-round rate from k failures in n shots of r rounds (without cancellation)."""
    import numpy as np
    return float(-np.expm1(np.log1p(-min(k / n, 1 - 1e-16)) / r))

if __name__ == '__main__':
    agg = load(sys.argv[1] if len(sys.argv) > 1 else 'results/final/c4_helios_mem/*/*/*.json')
    for key in sorted(agg):
        a = agg[key]
        proto, rule, noise, p, b, r = key
        n, k = int(a['shots']), int(a['fails'])
        lo, hi = cp(k, n)
        g = a['twoq_sum'] / n / (r + 1)
        print(f"{proto:12s} {rule:35s} {noise:8s} p={p:.1e} {b} r={r}: {k}/{n} eps={eps(k,n,r):.2e} [{eps(lo*n,n,r):.1e},{eps(hi*n,n,r):.1e}] 2q/round={g:.0f}  s/shot={a['seconds']/n:.2f} tasks={int(a['tasks'])}")
