"""Importance sampling of exactly-k-fault configurations: the asymptotic coefficients
b_k = sum over fault sets T with |T| = k that cause a logical failure of prod_{s in T} c_s,
where site s fires with probability q_s = c_s p. For small p the failure probability of the
experiment is sum_k b_k p^k (to leading order in each k), so b_k at the smallest failing k
is the asymptotic coefficient and b_k = 0 below the minimal failing weight.

Proposal Q: choose a window of consecutive rounds (length L, uniformly placed) and a set B
of blocks (size 2 or 3, uniformly), and draw k distinct sites with probability proportional
to prod c_s among the sites in the window whose fault footprint lies inside B (sites with
an empty footprint, e.g. outer-check measurement faults, are always eligible). Q(T) > 0 for
every configuration whose footprint spans <= 3 blocks within L rounds; the estimator
  b_k = E_Q[ 1_fail(T) prod c_s / Q(T) ]
is unbiased for that class (configurations spanning more blocks or longer windows are
counted separately by running with larger B or L)."""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hase.protocols import make_protocol  # noqa: E402
from hase.runner import SupersetDecoder  # noqa: E402
from hase.engine import simulate_with_faults  # noqa: E402


def elem_sym(c, k):
    e = np.zeros(k + 1)
    e[0] = 1.0
    for x in c:
        e[1:] = e[1:] + x * e[:-1]
    return e[k]


class KFaultIS:
    def __init__(self, proto_name, rule, basis, rounds, p_dec=1e-4, noise="uniform",
                 block_sizes=(2, 3), window=2, K=256, Delta=12, nblocks=5):
        self.proto = make_protocol(proto_name, p_dec, basis, rounds, rule, noise)
        self.dec = SupersetDecoder(self.proto, K=K, Delta=Delta, committee=True, K2=4096, Delta2=24)
        self.p_dec = p_dec
        proto = self.proto
        n = proto.n_data
        code = proto.code
        block_of_slot = np.zeros(n, dtype=np.int64)
        for b, blk in enumerate(code.blocks):
            block_of_slot[blk] = b
        self.nblocks = nblocks
        # check -> block (inner checks), outer checks -> -1
        def check_block(c):
            if c is None:
                return -1
            if c[0] in ("c4", "c6"):
                return c[1]
            if c[0] == "in":
                return c[1]
            return -1
        sites = []          # (instance key, site index, c_s, footprint frozenset, round)
        for r, pts in enumerate(self.dec.plan_all):
            for pt in pts:
                cp = proto.parts[pt]
                tm = cp.term_meas.tocsr()
                to = cp.term_out.tocsr()
                mblk = np.array([check_block(c) for c in cp.meas_check])
                for s, terms in enumerate(cp.site_terms):
                    fp = set()
                    for t in terms:
                        cols = to.indices[to.indptr[t]:to.indptr[t + 1]]
                        fp |= set(block_of_slot[cols % n].tolist())
                        mc = tm.indices[tm.indptr[t]:tm.indptr[t + 1]]
                        fp |= {int(x) for x in mblk[mc] if x >= 0}
                    sites.append(((r, pt), s, cp.site_prob[s] / p_dec, frozenset(fp), r))
        for nm, q in (("init", proto.noise.pr), ("final", proto.noise.pm)):
            if q:
                for slot in range(n):
                    rr = 0 if nm == "init" else len(self.dec.plan_all) - 1
                    sites.append(((nm,), slot, q / p_dec, frozenset({int(block_of_slot[slot])}), rr))
        self.sites = sites
        self.c = np.array([s[2] for s in sites])
        self.rounds_of = np.array([s[4] for s in sites])
        self.R = len(self.dec.plan_all)
        self.window = window
        self.block_sizes = block_sizes
        # enumerate proposal components (window start, block set)
        comps = []
        for L0 in range(0, self.R - window + 1):
            for bs in block_sizes:
                for B in itertools.combinations(range(nblocks), bs):
                    comps.append((L0, frozenset(B)))
        self.comps = comps
        self.comp_sites = []
        for L0, B in comps:
            idx = np.array([i for i, s in enumerate(sites)
                            if L0 <= s[4] < L0 + window and s[3] <= B], dtype=np.int64)
            self.comp_sites.append(idx)
        self.pi = np.full(len(comps), 1.0 / len(comps))

    def _e_k(self, k):
        cache = self.__dict__.setdefault("_ek", {})
        if k not in cache:
            cache[k] = np.array([elem_sym(self.c[idx], k) for idx in self.comp_sites])
        return cache[k]

    def sample(self, k, nsamp, rng):
        """Returns list of (site index array, weight prod c / Q)."""
        ek = self._e_k(k)
        out = []
        comp_ids = rng.choice(len(self.comps), size=nsamp, p=self.pi)
        for ci in comp_ids:
            idx = self.comp_sites[ci]
            c = self.c[idx]
            # sample k distinct sites with prob prop to prod c: sequential conditional
            # sampling via rejection from independent Bernoulli with mean ~k
            lam = k / c.sum()
            while True:
                pr = c * lam / (1 + c * lam)
                pick = np.flatnonzero(rng.random(len(c)) < pr)
                if len(pick) == k:
                    break
            T = idx[pick]
            Tset = set(T.tolist())
            # Q(T) = sum over components containing T of pi * prod c / e_k(comp)
            denom = 0.0
            for cj, (L0, B) in enumerate(self.comps):
                rr = self.rounds_of[T]
                if rr.min() < L0 or rr.max() >= L0 + self.window:
                    continue
                if all(self.sites[t][3] <= B for t in T):
                    denom += self.pi[cj] / ek[cj]
            out.append((T, 1.0 / denom))
        return out

    def faults_of(self, T):
        fl = {}
        for t in T:
            key, s, c, fp, r = self.sites[t]
            if key[0] in ("init", "final"):
                fl.setdefault(key[0], []).append(s)
                continue
            cp = self.proto.parts[key[1]]
            terms = cp.site_terms[s]
            pr = cp.term_prob[terms]
            fl.setdefault(key, []).append(int(np.random.choice(terms, p=pr / pr.sum())))
        return fl

    def run(self, k, nsamp, rng, batch=200):
        """Estimate b_k. Returns (estimate, standard error, n_fail, samples)."""
        from collections import defaultdict
        n = self.proto.n_data
        vals = []
        nfail = 0
        for start in range(0, nsamp, batch):
            draws = self.sample(k, min(batch, nsamp - start), rng)
            # build a batched superset fault configuration
            B = len(draws)
            faults = defaultdict(lambda: ([], []))
            init = np.zeros((B, n), dtype=bool)
            final = np.zeros((B, n), dtype=bool)
            for bi, (T, w) in enumerate(draws):
                fl = self.faults_of(T)
                for key, terms in fl.items():
                    if key == "init":
                        init[bi, terms] = True
                    elif key == "final":
                        final[bi, terms] = True
                    else:
                        faults[key][0].extend([bi] * len(terms))
                        faults[key][1].extend(terms)
            fz = {key: (np.array(v[0], dtype=np.int64), np.array(v[1], dtype=np.int64)) for key, v in faults.items()}
            fz["init"], fz["final"] = init, final
            st = simulate_with_faults(self.proto, fz, B)
            groups = defaultdict(list)
            for i, path in enumerate(st.paths):
                groups[tuple(path)].append(i)
            fails = np.zeros(B, dtype=bool)
            sup = self.dec.sup
            for path, idx in groups.items():
                D = np.zeros((len(idx), sup.n_det), dtype=bool)
                O = np.zeros((len(idx), len(self.proto.logical_supports)), dtype=np.uint8)
                for a, i in enumerate(idx):
                    D[a], O[a] = sup.syndrome(path, st.records[i], st.final[i])
                nt = np.flatnonzero(D.any(axis=1))
                pred = np.zeros_like(O)
                if len(nt):
                    pred[nt] = self.dec.decode(path, D[nt])[0]
                fails[idx] = np.any(pred != O, axis=1)
            for bi, (T, w) in enumerate(draws):
                vals.append(w if fails[bi] else 0.0)
                nfail += int(fails[bi])
        vals = np.array(vals)
        return vals.mean(), vals.std(ddof=1) / np.sqrt(len(vals)), nfail, len(vals)


class PatternIS(KFaultIS):
    """Mixture proposal: the generic windowed component of KFaultIS plus 'pattern'
    components (window, distinct blocks b_1..b_m, counts n_1..n_m summing to k): n_i sites
    are drawn (prob. prop. to prod c) among the sites in the window whose footprint is
    exactly {b_i}. Q(T) sums every component that can produce T, so the estimator stays
    unbiased; pattern components concentrate samples on block-structured failures."""

    PATTERNS = {5: [(2, 2, 1), (3, 2), (2, 1, 1, 1), (3, 1, 1)],
                6: [(2, 2, 2), (3, 3), (4, 2), (3, 2, 1), (2, 2, 1, 1)],
                7: [(3, 2, 2), (3, 3, 1), (4, 3), (4, 2, 1), (2, 2, 2, 1)],
                3: [(1, 1, 1), (2, 1)], 4: [(2, 2), (2, 1, 1), (3, 1)]}

    def __init__(self, *args, generic_weight=0.2, **kw):
        super().__init__(*args, **kw)
        self.generic_weight = generic_weight
        # single-block site lists per (window start, block)
        self.single = {}
        fps = [s[3] for s in self.sites]
        for L0 in range(0, self.R - self.window + 1):
            for b in range(self.nblocks):
                idx = np.array([i for i, s in enumerate(self.sites)
                                if L0 <= s[4] < L0 + self.window and fps[i] == frozenset({b})],
                               dtype=np.int64)
                self.single[(L0, b)] = idx
        self.fp_single = np.array([next(iter(f)) if len(f) == 1 else -1 for f in fps])

    def _pattern_comps(self, k):
        comps = []
        for L0 in range(0, self.R - self.window + 1):
            for pat in self.PATTERNS[k]:
                for blocks in itertools.permutations(range(self.nblocks), len(pat)):
                    # avoid double counting equal counts: require blocks increasing among equal counts
                    ok = all(not (pat[i] == pat[j] and blocks[i] > blocks[j])
                             for i in range(len(pat)) for j in range(i + 1, len(pat)))
                    if ok:
                        comps.append((L0, pat, blocks))
        return comps

    def _ek_single(self, key, n):
        cache = self.__dict__.setdefault("_eks", {})
        if (key, n) not in cache:
            cache[(key, n)] = elem_sym(self.c[self.single[key]], n)
        return cache[(key, n)]

    def q_of(self, T, k, pcomps):
        """Proposal probability density ratio: returns Q(T) / prod c (so weight = 1/that)."""
        # generic part
        ek = self._e_k(k)
        g = 0.0
        rr = self.rounds_of[T]
        for cj, (L0, B) in enumerate(self.comps):
            if rr.min() < L0 or rr.max() >= L0 + self.window:
                continue
            if all(self.sites[t][3] <= B for t in T):
                g += self.pi[cj] / ek[cj]
        tot = self.generic_weight * g
        # pattern part: T must consist of single-footprint sites
        fb = self.fp_single[T]
        if (fb < 0).any():
            return tot
        from collections import Counter
        cnt = Counter(fb.tolist())
        pw = (1 - self.generic_weight) / len(pcomps)
        for (L0, pat, blocks) in pcomps:
            if rr.min() < L0 or rr.max() >= L0 + self.window:
                continue
            if Counter(dict(zip(blocks, pat))) != cnt:
                continue
            val = 1.0
            for b, n in zip(blocks, pat):
                val /= self._ek_single((L0, b), n)
            tot += pw * val
        return tot

    def sample(self, k, nsamp, rng):
        pcomps = self.__dict__.setdefault("_pc", {}).setdefault(k, self._pattern_comps(k))
        out = []
        for _ in range(nsamp):
            if rng.random() < self.generic_weight:
                T, _w = super().sample(k, 1, rng)[0]
            else:
                L0, pat, blocks = pcomps[rng.integers(len(pcomps))]
                parts = []
                for b, n in zip(blocks, pat):
                    idx = self.single[(L0, b)]
                    c = self.c[idx]
                    lam = n / c.sum()
                    while True:
                        pr = c * lam / (1 + c * lam)
                        pick = np.flatnonzero(rng.random(len(c)) < pr)
                        if len(pick) == n:
                            break
                    parts.append(idx[pick])
                T = np.concatenate(parts)
            out.append((T, 1.0 / self.q_of(T, k, pcomps)))
        return out


if __name__ == "__main__":
    import json
    import time
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--proto", default="carbonhelix")
    ap.add_argument("--rule", default="full")
    ap.add_argument("--basis", default="X")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--blocks", default="2,3")
    ap.add_argument("--p-dec", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--patterns", type=int, default=1)
    a = ap.parse_args()
    t0 = time.time()
    cls = PatternIS if a.patterns else KFaultIS
    est = cls(a.proto, a.rule, a.basis, a.rounds, p_dec=a.p_dec, window=a.window,
              block_sizes=tuple(int(x) for x in a.blocks.split(",")))
    rng = np.random.default_rng(a.seed)
    b, se, nf, ns = est.run(a.k, a.samples, rng)
    res = dict(vars(a), b_k=b, se=se, n_fail=nf, n_samples=ns, seconds=time.time() - t0,
               n_sites=len(est.sites), n_components=len(est.comps))
    print(json.dumps(res))
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(res, open(a.out, "w"))


