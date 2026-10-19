"""Stratified (fault-count) analysis of job_strat.py outputs.

For one protocol configuration, the failure probability conditioned on the number W of
fired fault sites in the protocol's superset, f_w = P(fail | W = w), does not depend on p
(to O(p)). Then
    p_L(p) = sum_w P_p(W = w) f_w,
with P_p the exact Poisson-binomial distribution of W over the superset sites at noise p.

Estimators (docs/methods.md):
  * hybrid (reported): measured f_w where a stratum has >= 3 failures, the fitted
    polynomial model (PolyModel) elsewhere; statistical uncertainty by a smoothed
    parametric bootstrap (every stratum resampled from its hybrid f_w, so strata observed
    without failures vary too; model refitted each time);
  * data only (no model): Jeffreys posterior Beta(f + 1/2, n - f + 1/2) for every sampled
    stratum, unsampled strata at f_w = 1: a one-sided 95 % upper bound that contains the
    zero-count strata's uncertainty the plug-in estimate f_w = f/n omits.
Per-round rates: 1 - (1 - p_L)^(1/r), evaluated without cancellation (per_round).
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hase.protocols import make_protocol  # noqa: E402
from hase.engine import site_count_distribution  # noqa: E402


def per_round(pl, r):
    """Per-round rate 1 - (1 - pl)^(1/r) of a per-shot probability pl over r rounds, without
    the cancellation of the direct formula at rare-event probabilities."""
    x = np.clip(np.asarray(pl, dtype=float), 0.0, 1.0 - 1e-16)
    out = -np.expm1(np.log1p(-x) / r)
    return float(out) if np.ndim(out) == 0 else out


def load(pattern):
    """Return dict (proto, rule, noise, basis, rounds, p_dec) -> dict(n, f, g, l3, shots)."""
    out = {}
    for fpath in glob.glob(pattern):
        d = json.load(open(fpath))
        key = (d["proto"], d["rule"], d["noise"], d["basis"], d["rounds"], d["p_dec"])
        a = out.setdefault(key, dict(n=np.zeros(400), f=np.zeros(400), g=np.zeros(400),
                                     l3=np.zeros(400), t=np.zeros(400), tn=np.zeros(400),
                                     q=np.zeros(400), shots=0, seconds=0.0, tasks=0))
        m = len(d["n"])
        a["n"][:m] += d["n"]
        a["f"][:m] += d["f"]
        a["g"][:m] += d["g"]
        a["l3"][:m] += d["l3"]
        if "t_ms" in d:
            a["t"][:m] += d["t_ms"]
            a["tn"][:m] += d["n"]
            a["q"][:m] = np.maximum(a["q"][:m], d["peak_qubits"])
        a["shots"] += d["shots"]
        a["seconds"] += d["seconds"]
        a["tasks"] += 1
    return out


class StratModel:
    def __init__(self, key, data):
        proto_name, rule, noise, basis, rounds, p_dec = key
        self.key = key
        self.rounds = rounds
        self.p_dec = p_dec
        self.proto = make_protocol(proto_name, p_dec, basis, rounds, rule, noise)
        self.plan_all = [[pt for pt, _ in self.proto.plan(r)] for r in range(self.proto.n_rounds)]
        self.n, self.f, self.g, self.l3 = data["n"], data["f"], data["g"], data["l3"]
        self.t, self.tn, self.q = data.get("t"), data.get("tn"), data.get("q")
        self.shots = data["shots"]
        # two-qubit gates of the always-full preparation round
        self.g_prep = sum(self.proto.parts[pt].two_q for pt, c in self.proto.plan(0))

    def pw(self, p, wmax=399):
        """Exact distribution of W at noise p (cached; shared by shallow copies)."""
        cache = self.__dict__.setdefault("_pw_cache", {})
        key = (float(p), int(wmax))
        if key not in cache:
            cache[key] = site_count_distribution(self.proto, self.plan_all, p_scale=p / self.p_dec,
                                                 w_max=wmax)
        return cache[key]

    def pL(self, p):
        """Plug-in estimate (f_w = f/n on sampled strata, 0 on unsampled ones) of the per-shot
        failure probability at noise p and its binomial standard error. Strata observed
        without failures contribute nothing to either: use direct_posterior or the hybrid
        estimate for intervals."""
        P = self.pw(p)
        n, f = self.n, self.f
        m = min(len(P), len(n))
        ok = n[:m] > 0
        fw = np.where(ok, f[:m] / np.maximum(n[:m], 1), 0.0)
        est = float((P[:m] * fw).sum())
        var = float((P[:m] ** 2 * np.where(ok, fw * (1 - fw) / np.maximum(n[:m], 1), 0.0)).sum())
        return est, np.sqrt(var)

    def eps(self, p):
        e, s = self.pL(p)
        r = self.rounds
        return per_round(e, r), (per_round(e + s, r) - per_round(max(e - s, 0.0), r)) / 2

    def twoq_per_round(self, p):
        """Expected two-qubit gates per round after the preparation round, conditional on the
        sampled strata; returns (value, coverage = P_p(sampled strata))."""
        P = self.pw(p)
        n, g = self.n, self.g
        m = (n > 0)
        gw = np.where(m, g / np.maximum(n, 1), 0.0)
        cov = P[m].sum()
        tot = (P[m] * gw[m]).sum() / max(cov, 1e-300)
        return (tot - self.g_prep) / self.rounds, cov

    def time_per_round(self, p):
        """Expected Helios-timed duration (ms) per round after the preparation round,
        conditional on the sampled strata (coverage: twoq_per_round); the reported resources
        come from direct path sampling (scripts/job_resources.py)."""
        if self.t is None or not np.any(self.tn):
            return None
        from hase.timing import path_time_ms
        P = self.pw(p)
        m = self.tn > 0
        tw = np.where(m, self.t / np.maximum(self.tn, 1), 0.0)
        cov = P[m].sum()
        tot = (P[m] * tw[m]).sum() / max(cov, 1e-300)
        prep = path_time_ms(self.proto, [(0, pt) for pt, c in self.proto.plan(0)])
        return (tot - prep) / self.rounds

    def peak_qubits(self):
        return int(self.q.max()) if self.q is not None and np.any(self.q) else None

    def outer_fraction(self, p):
        P = self.pw(p)
        n, l3 = self.n, self.l3
        m = (n > 0)
        lw = np.where(m, l3 / np.maximum(n, 1), 0.0)
        cov = P[m].sum()
        tot = (P[m] * lw[m]).sum() / max(cov, 1e-300)
        return (tot - 1) / self.rounds        # minus the preparation round


def local_exponent(model, p, h=0.05):
    e1 = model.eps(p * np.exp(-h))[0]
    e2 = model.eps(p * np.exp(h))[0]
    return (np.log(e2) - np.log(e1)) / (2 * h)


if __name__ == "__main__":
    pat = sys.argv[1] if len(sys.argv) > 1 else "results/final/ch_main/*/*/*/*.json"
    data = load(pat)
    for key in sorted(data):
        m = StratModel(key, data[key])
        print(key, "shots", int(m.shots), "fails", int(m.f.sum()))
        for p in [3e-3, 2e-3, 1.5e-3, 1e-3, 7e-4, 5e-4, 3e-4]:
            e, s = m.eps(p)
            g, cov = m.twoq_per_round(p)
            print(f"   p={p:.1e}: plug-in eps={e:.3e} +- {s:.1e}  2q/round={g:.0f}"
                  f" (sampled-strata coverage {cov:.3f})  outer frac={m.outer_fraction(p):.3f}")


# --------------------------------------------------------------------------- extrapolation
def falling(w, k):
    out = np.ones_like(np.asarray(w, dtype=float))
    for i in range(k):
        out = out * (np.asarray(w, dtype=float) - i)
    return np.where(np.asarray(w) >= k, out, 0.0)


class PolyModel:
    """f_w = sum_{k=k0}^{k0+J} a_k [w]_k / [N]_k fitted by binomial maximum likelihood on
    the strata w in `use` (a_k >= 0). The negative log-likelihood is convex in a (f is
    linear in a), so the bounded quasi-Newton fit below finds the global optimum. k0 is the
    smallest number of faults that can cause a failure."""

    def __init__(self, sm: "StratModel", k0=6, J=3, use=None, min_n=1, x0=None):
        from scipy.optimize import minimize
        self.sm, self.k0, self.J = sm, k0, J
        n, f = sm.n, sm.f
        self.N = int(sum(len(sm.proto.parts[pt].site_prob) for pts in sm.plan_all for pt in pts)
                     + (2 * sm.proto.n_data if sm.proto.noise.pr else 0))
        ws = np.array([w for w in range(len(n)) if n[w] >= min_n and (use is None or w in use)])
        self.ws = ws
        ks = list(range(k0, k0 + J + 1))
        self.ks = ks
        G = np.array([[falling(w, k) / falling(self.N, k) for k in ks] for w in ws])
        scale = np.maximum(G.max(axis=0), 1e-300)        # columns of order one
        Gs = G / scale
        nn, ff = n[ws], f[ws]

        def fun(b):
            fw = np.clip(Gs @ b, 1e-300, 1 - 1e-12)
            nll = -(ff * np.log(fw) + (nn - ff) * np.log(1 - fw)).sum()
            grad = -Gs.T @ (ff / fw - (nn - ff) / (1 - fw))
            return nll, grad
        b0 = np.full(len(ks), max(ff.sum() / max(nn.sum(), 1), 1e-9) / len(ks))
        if x0 is not None and len(x0) == len(ks):
            b0 = np.maximum(np.exp(np.asarray(x0, float)) * scale, 1e-300)
        r = minimize(fun, b0, jac=True, method="L-BFGS-B", bounds=[(0, None)] * len(ks),
                     options=dict(maxiter=5000, ftol=1e-13, gtol=1e-10))
        self.a = r.x / scale
        self.x = np.log(np.maximum(self.a, 1e-300))
        self.nll = float(r.fun)
        self.G = G

    def fw_all(self, L=400):
        """f_w for w = 0..L-1 (vectorized; cached)."""
        c = self.__dict__.get("_fw_all")
        if c is None or len(c) < L:
            ws = np.arange(L)
            phi = np.stack([falling(ws, k) / falling(self.N, k) for k in self.ks], axis=1)
            c = np.clip(phi @ np.asarray(self.a, float), 0.0, 1.0)
            self._fw_all = c
        return c[:L]

    def fw(self, w):
        return float(self.fw_all()[w]) if 0 <= w < 400 else \
            float(np.clip(sum(a * falling(w, k) / falling(self.N, k) for a, k in zip(self.a, self.ks)), 0, 1))

    def pL(self, p, wmax=399):
        P = self.sm.pw(p, wmax)
        return float(P @ self.fw_all(len(P)))

    def eps(self, p):
        return per_round(self.pL(p), self.sm.rounds)


def sample_quantile_cut(sm, q=0.75):
    """Pre-declared held-out cut: the strata below it hold the fraction q of the sampled shots
    (by W). It depends on the sample sizes only, not on where failures occurred."""
    c = np.cumsum(sm.n)
    return int(np.searchsorted(c, c[-1] * q)) + 1


def heldout(sm, make, q=0.75, nboot=200, seed=0, min_power=3.0):
    """Held-out check of the extrapolation model: fit on the strata at or above the cut (the
    upper quarter of the sampled shots by W), predict the failures of the strata below it
    (three quarters of the shots, at the fault counts closest to the extrapolation),
    compare with the observed count. The cut is fixed by the sample sizes before outcomes
    are used (sample_quantile_cut). Predictive distribution: for every parametric-bootstrap
    refit of the model, each held-out stratum's failures drawn from Binomial(n_w, f_w).
    Reported: the two-sided predictive tail probability of the observed count ('not
    rejected' when >= 0.05; a non-rejection, not a proof), and the 95 % interval of the
    calibration factor c (observed = c x predicted) - the size of a model error the
    held-out data could have hidden. 'uninformative' when predicted and observed counts are
    both below min_power."""
    import copy
    cut = sample_quantile_cut(sm, q)
    ws = [w for w in range(len(sm.n)) if sm.n[w] > 0]
    sb = copy.copy(sm)
    sb.n = sm.n.copy()
    sb.f = sm.f.copy()
    sb.n[:cut] = 0
    sb.f[:cut] = 0
    try:
        m = make(sb)
    except Exception as ex:
        return {"error": str(ex)}
    held = np.array([w for w in ws if w < cut], dtype=int)
    nh = sm.n[held].astype(np.int64)
    obs = int(round(float(sm.f[held].sum())))
    rng = np.random.default_rng(seed)
    fh = np.clip(hybrid_fhat(sb, m), 0.0, 1.0)
    fits = [m]
    for _ in range(nboot):
        s2 = copy.copy(sb)
        s2.f = rng.binomial(sb.n.astype(np.int64), fh).astype(float)
        try:
            fits.append(make(s2, x0=getattr(m, "x", None)))
        except Exception:
            continue
    F = np.array([np.clip(np.asarray(f_.fw_all(len(sm.n)))[held], 0.0, 1.0) for f_ in fits])   # fits x strata
    pred = float((nh * F[0]).sum())
    lam = (F * nh).sum(axis=1)

    def tail(c, ndraw=20):
        # predictive draws: each refit, each stratum Binomial(n_w, min(1, c f_w))
        Fc = np.clip(c * F, 0.0, 1.0)
        draws = rng.binomial(np.broadcast_to(nh, (ndraw,) + Fc.shape), np.broadcast_to(Fc, (ndraw,) + Fc.shape)).sum(axis=2).ravel()
        lo = float(np.mean(draws <= obs))
        hi = float(np.mean(draws >= obs))
        return float(min(1.0, 2 * min(lo, hi)))
    pv = tail(1.0)
    cs = np.exp(np.linspace(np.log(0.05), np.log(20.0), 61))
    ok = [c for c in cs if tail(c, 5) >= 0.05]
    informative = max(pred, obs) >= min_power
    return dict(cut=int(cut), cut_rule=f"strata below the {q:.0%} quantile of sampled W held out",
                predicted=pred, predicted_95=[float(np.percentile(lam, 2.5)), float(np.percentile(lam, 97.5))],
                observed=obs, p_value=pv, informative=bool(informative),
                # 95 % interval of the calibration factor; 0 / None where the data do not bound it
                calibration_95=([0.0 if min(ok) <= cs[0] else float(min(ok)),
                                 None if max(ok) >= cs[-1] else float(max(ok))] if ok else None),
                direction=("over-predicts" if obs < pred else "under-predicts" if obs > pred else "exact"),
                status=("not rejected" if pv >= 0.05 else "rejected") if informative else "uninformative",
                heldout_shots=int(nh.sum()), fit_failures=float(sb.f.sum()),
                fit_strata=[int(w) for w in ws if w >= cut], heldout_strata=[int(w) for w in held])


# --------------------------------------------------------------------------- hybrid estimate
def hybrid_eps(sm, model, p, min_fail=3):
    """eps(p) from the measured f_w where a stratum has at least `min_fail` failures and
    from the model elsewhere. Returns (eps, share of the per-shot failure probability that
    comes from measured strata)."""
    P = sm.pw(p)
    L = len(P)
    meas = np.zeros(L, dtype=bool)
    m = min(L, len(sm.n))
    meas[:m] = (sm.f[:m] >= min_fail) & (sm.n[:m] > 0)
    fm = np.zeros(L)
    fm[:m] = np.where(meas[:m], sm.f[:m] / np.maximum(sm.n[:m], 1), 0.0)
    fx = model.fw_all(L) if hasattr(model, "fw_all") else np.array([model.fw(w) for w in range(L)])
    pl_m = float(P[meas] @ fm[meas])
    pl_x = float(P[~meas] @ fx[~meas])
    pl = pl_m + pl_x
    return per_round(pl, sm.rounds), (pl_m / pl if pl > 0 else 0.0)


def hybrid_fhat(sm, model, min_fail=3):
    """Per-stratum failure probability of the hybrid estimate (measured or model)."""
    fh = np.array(model.fw_all(len(sm.n)) if hasattr(model, "fw_all") else [model.fw(w) for w in range(len(sm.n))],
                  dtype=float)
    meas = (sm.f >= min_fail) & (sm.n > 0)
    fh[meas] = sm.f[meas] / sm.n[meas]
    return fh


def hybrid_boot(sm, make_model, ps, nboot=100, seed=0, min_fail=3, model=None, perturb=None):
    """Hybrid estimate at each p with its statistical uncertainty: smoothed parametric
    bootstrap - every stratum's failures redrawn from Binomial(n_w, fhat_w) with fhat the
    hybrid per-stratum probability (so strata observed with no or few failures vary
    according to the fitted model), the model refitted (warm start), the hybrid estimate
    recomputed. make_model(sm, x0=None) -> fitted model. perturb(rng) -> per-stratum factors
    multiplying fhat in each replica (an earlier estimated stage, e.g. the decoder audit's
    classification of the failures, resampled jointly).
    Returns (estimates [(eps, share)], replicas array [nboot, len(ps)], base model)."""
    import copy
    rng = np.random.default_rng(seed)
    base = model if model is not None else make_model(sm)
    est = [hybrid_eps(sm, base, p, min_fail) for p in ps]
    fh = np.clip(hybrid_fhat(sm, base, min_fail), 0.0, 1.0)
    nn = sm.n.astype(np.int64)
    reps = []
    for _ in range(nboot):
        sb = copy.copy(sm)
        fr = fh if perturb is None else np.clip(fh * perturb(rng), 0.0, 1.0)
        sb.f = rng.binomial(nn, fr).astype(float)
        try:
            m = make_model(sb, x0=getattr(base, "x", None))
        except Exception:
            continue
        reps.append([hybrid_eps(sb, m, p, min_fail)[0] for p in ps])
    return est, np.array(reps).reshape(-1, len(ps)), base


def boot_summary(reps, i):
    col = reps[:, i] if len(reps) else np.array([np.nan])
    return float(np.nanpercentile(col, 2.5)), float(np.nanpercentile(col, 97.5))


def slope_summary(reps, est, ps, i, j):
    """Local exponent d ln eps / d ln p between ps[i] and ps[j]: point value and bootstrap
    95 % interval. Replicas with a zero rate count as the bound they imply (+inf/-inf), so
    the interval is not conditional on non-zero replicas."""
    a, b = ps[i], ps[j]
    with np.errstate(divide="ignore", invalid="ignore"):
        s0 = (np.log(est[i]) - np.log(est[j])) / (np.log(a) - np.log(b)) if est[i] > 0 and est[j] > 0 else np.nan
        if len(reps):
            sl = (np.log(reps[:, i]) - np.log(reps[:, j])) / (np.log(a) - np.log(b))
            sl = sl[~np.isnan(sl)]
        else:
            sl = np.array([])
    if not len(sl):
        return float(s0), None, None
    return float(s0), float(np.percentile(sl, 2.5)), float(np.percentile(sl, 97.5))


# --------------------------------------------------------------------------- data only
def direct_posterior(sm, ps, k0=None, ndraw=4000, seed=0):
    """Model-free estimate of eps(p): Jeffreys posterior Beta(f + 1/2, n - f + 1/2) of f_w on
    every sampled stratum, unsampled strata at f_w = 1. No smallest failing
    fault count is assumed unless k0 is given (strata below k0 without failures then count
    as 0; a lower bound on the failing fault count is not proved for the Carbon-Helix
    circuits and decoder, so the analyses pass None). Returns {p: dict(median, upper95,
    unsampled_mass)} of the per-round rate; upper95 is the one-sided 95 % upper bound with
    unsampled strata at f_w = 1 (the lower side would be set by the prior)."""
    rng = np.random.default_rng(seed)
    n, f = sm.n, sm.f
    L = len(n)
    ws = np.arange(L)
    floor = 0 if k0 is None else k0
    samp = (n > 0) & ((ws >= floor) | (f > 0))
    draws = np.zeros((ndraw, L))
    draws[:, samp] = rng.beta(f[samp] + 0.5, n[samp] - f[samp] + 0.5, size=(ndraw, int(samp.sum())))
    out = {}
    for p in ps:
        P = sm.pw(p, wmax=L - 1)
        un = float(P[(n == 0) & (ws >= floor)].sum())
        pl = draws @ P
        # only the upper side is a statement of the data: the lower side of the posterior of
        # strata observed without failures is set by the prior, not by the data
        out[p] = dict(median=per_round(float(np.median(pl)), sm.rounds),
                      upper95=per_round(float(np.percentile(pl, 95.0)) + un, sm.rounds),
                      unsampled_mass=un, k0_assumed=k0)
    return out


# --------------------------------------------------------------------------- resources
def _sample_linear(proto, plan_all, shots, rng, scale):
    """Faults on every superset site with probability scale x q (the law P_p(W) of pw uses),
    in the format of hase.engine.sample_superset_faults."""
    from hase.engine import superset_sites
    W = np.zeros(shots, dtype=np.int64)
    faults = {}
    for key, cp in superset_sites(proto, plan_all):
        q = np.clip(cp.site_prob * scale, 0.0, 1.0)
        if len(q) == 0:
            continue
        counts = rng.binomial(shots, q)
        sh_l, tt_l = [], []
        for s_ in np.flatnonzero(counts):
            c = counts[s_]
            sh_l.append(rng.choice(shots, size=c, replace=False))
            tl = cp.site_terms[s_]
            pr = cp.term_prob[tl]
            tt_l.append(rng.choice(tl, size=c, p=pr / pr.sum()))
        sh = np.concatenate(sh_l) if sh_l else np.zeros(0, dtype=np.int64)
        tt = np.concatenate(tt_l) if tt_l else np.zeros(0, dtype=np.int64)
        faults[key] = (sh, tt)
        np.add.at(W, sh, 1)
    for name, q in (("init", proto.noise.pr), ("final", proto.noise.pm)):
        f = rng.random((shots, proto.n_data)) < min((q or 0.0) * scale, 1.0)
        faults[name] = f
        W += f.sum(axis=1)
    return faults, W


def path_resources(sm, ps, shots=4000, seed=0):
    """Expected two-qubit gates, Helios-model time (ms), outer checks measured and full outer
    measurements (L3) per round after the preparation round at each p, by direct sampling
    of the executed paths at noise p (every site at q p/p_dec, the law of pw): the path
    depends on the syndromes only, so no decoding is needed and the expectation covers the
    whole fault distribution (unlike averages over the sampled strata). Returns {p: dict(
    twoq, twoq_se, time_ms, time_ms_se, outer_checks, full_outer_fraction, shots)}."""
    from hase.engine import simulate_with_faults
    from hase.timing import path_time_ms
    proto = sm.proto
    prep_g = sm.g_prep
    prep_t = path_time_ms(proto, [(0, pt) for pt, c in proto.plan(0)])
    nchk = {}
    for name, ap in proto.aparts.items():
        if name in ("L3", "outer") or name.startswith("L3r_"):
            nchk[name] = sum(1 for v in ap.meas.values() if v[0] == "chk")
    prep_chk = sum(nchk.get(pt, 0) for pt, c in proto.plan(0))
    out = {}
    for i, p in enumerate(ps):
        rng = np.random.default_rng([seed, i, 9177])
        g, t, l3, oc = [], [], [], []
        done = 0
        while done < shots:
            b = min(1000, shots - done)
            faults, W = _sample_linear(proto, sm.plan_all, b, rng, p / sm.p_dec)
            st = simulate_with_faults(proto, faults, b)
            for path in st.paths:
                g.append(sum(proto.parts[pt].two_q for _, pt in path))
                t.append(path_time_ms(proto, path))
                l3.append(sum(1 for _, pt in path if pt in ("L3", "outer")))
                oc.append(sum(nchk.get(pt, 0) for _, pt in path))
            done += b
        g, t, l3, oc = (np.array(x, dtype=float) for x in (g, t, l3, oc))
        r = sm.rounds
        out[p] = dict(twoq=float((g.mean() - prep_g) / r), twoq_se=float(g.std(ddof=1) / np.sqrt(len(g)) / r),
                      time_ms=float((t.mean() - prep_t) / r), time_ms_se=float(t.std(ddof=1) / np.sqrt(len(t)) / r),
                      outer_checks=float((oc.mean() - prep_chk) / r),
                      full_outer_fraction=float((l3.mean() - 1) / r), shots=int(len(g)))
    return out


# --------------------------------------------------------------------------- fault order
def local_order(sm, q=0.75, use_all=False):
    """Order of the failure probability in the number of faults, straight from the strata:
    f_w = A [w]_k / [N]_k with k free (real; [x]_k = Gamma(x+1)/Gamma(x-k+1)) fitted by
    binomial likelihood on the strata below the pre-declared cut (the lower fraction q of
    the sampled shots, as for the held-out check) or on all strata; profile-likelihood 95 %
    interval. k is a shape parameter of f_w over the observed strata (for an integer k and
    one order alone it is exactly the exponent in p, since E[[W]_k] = [N]_k p^k). The
    exponent in p it implies is computed from the integrated rate, eps(p) = sum_w P_p(w) f_w,
    as d ln eps / d ln p at p_star, the p whose mean fault count equals the failures' mean
    count (mean-count mapping), for the fit and, as its interval, the range over k's interval."""
    from scipy.optimize import minimize
    from scipy.special import gammaln
    cut = len(sm.n) if use_all else sample_quantile_cut(sm, q)
    ws = np.array([w for w in range(min(cut, len(sm.n))) if sm.n[w] > 0])
    nn, ff = sm.n[ws], sm.f[ws]
    if ff.sum() < 3:
        return dict(k=None, note=f"{ff.sum():.0f} failures below the cut", cut=int(cut))
    w0 = float(np.average(ws, weights=np.maximum(ff, 1e-9)))

    def logphi(k, w):
        # log([w]_k / [w0]_k): the level A absorbs the normalisation; -inf where w < k - 1
        w = np.asarray(w, float)
        val = (gammaln(w + 1) - gammaln(np.maximum(w - k + 1, 1e-9))) - \
              (gammaln(w0 + 1) - gammaln(max(w0 - k + 1, 1e-9)))
        return np.where(w - k + 1 > 0, val, -np.inf)

    def nll(th, k=None):
        la, kk = (th[0], th[1]) if k is None else (th[0], k)
        lf = np.minimum(la + logphi(kk, ws), -1e-12)
        f = np.exp(lf)
        ll = np.where(np.isfinite(lf), ff * np.where(np.isfinite(lf), lf, 0.0) + (nn - ff) * np.log1p(-f),
                      np.where(ff > 0, -1e6, 0.0))
        return -ll.sum()
    la0 = np.log(max(ff.sum() / nn.sum(), 1e-12))
    best = None
    for k_init in (3.0, 6.0, 9.0):
        r = minimize(nll, [la0, k_init], method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-9, maxiter=4000))
        if best is None or r.fun < best.fun:
            best = r
    la, k = best.x
    fit_la = lambda kk: minimize(lambda t: nll(t, kk), [la], method="Nelder-Mead").x[0]
    prof = lambda kk: nll([fit_la(kk)], kk) - best.fun
    grid = np.linspace(max(k - 8, 0.0), k + 8, 801)
    ok = [g for g in grid if prof(g) <= 1.92]
    bounded = bool(ok) and min(ok) > grid[0] and max(ok) < grid[-1]
    # implied exponent in p at p_star (mean-count mapping), from the integrated rate
    P0 = sm.pw(sm.p_dec)
    ew = float((np.arange(len(P0)) * P0).sum())
    p_star = w0 / ew * sm.p_dec
    wall = np.arange(len(sm.n))

    def slope(lak, kk, h=0.05):
        fw = np.exp(np.minimum(lak + logphi(kk, wall), 0.0))
        e = [float(sm.pw(p_star * np.exp(s_)) @ fw) for s_ in (-h, h)]
        return float((np.log(e[1]) - np.log(e[0])) / (2 * h)) if min(e) > 0 else None
    s0 = slope(la, k)
    s_ci = None
    if bounded:
        # the implied exponent is not monotonic in k: its range over k's whole interval (which
        # contains the fit) rather than its values at the interval's ends
        ks = np.unique(np.concatenate([np.linspace(min(ok), max(ok), 17), [k]]))
        sl = [slope(fit_la(kk), kk) for kk in ks]
        sl = [x for x in sl if x is not None]
        if sl:
            s_ci = [float(min(sl)), float(max(sl))]
    wf = ws[ff > 0]
    return dict(k=float(k), ci95=[float(min(ok)), float(max(ok))] if bounded else None, cut=int(cut),
                strata=[int(ws.min()), int(ws.max())], failures=float(ff.sum()), shots=int(nn.sum()),
                w_mean_failures=w0, p_star=float(p_star), slope_at_p_star=s0, slope_ci95=s_ci,
                p_equiv=[float(wf.min() / ew * sm.p_dec), float(wf.max() / ew * sm.p_dec)] if len(wf) else None,
                p_equiv_note="mean-count mapping: the p whose expected number of faults equals the stratum's")


def order_bound(sm, k=5, J=4, ps=(), level=0.95):
    """What the strata allow for a failure mode with k faults (below the order the analyses
    assume): profile likelihood of a_k in f_w = sum_{j=k}^{k+J} a_j [w]_j/[N]_j (a_j >= 0, the
    higher orders free), one-sided upper limit at `level`. Returns the limit, the share of
    the probability of failing it allows at each p in ps (its contribution sum_w P_p(w) a_k
    [w]_k/[N]_k over the k..k+J fit's total), and the largest p at which that bounded mode
    could equal the rest of the fit (below it, such a mode, if present, could dominate)."""
    from scipy.optimize import minimize
    from scipy.stats import chi2
    n, f = sm.n, sm.f
    N = int(sum(len(sm.proto.parts[pt].site_prob) for pts in sm.plan_all for pt in pts)
            + (2 * sm.proto.n_data if sm.proto.noise.pr else 0))
    ws = np.array([w for w in range(len(n)) if n[w] > 0])
    ks = list(range(k, k + J + 1))
    G = np.array([[falling(w, j) / falling(N, j) for j in ks] for w in ws])
    scale = np.maximum(G.max(axis=0), 1e-300)
    Gs = G / scale
    nn, ff = n[ws], f[ws]

    def nll_fixed(b0):
        def fun(b):
            bb = np.concatenate([[b0], b])
            fw = np.clip(Gs @ bb, 1e-300, 1 - 1e-12)
            val = -(ff * np.log(fw) + (nn - ff) * np.log(1 - fw)).sum()
            grad = -(Gs[:, 1:].T @ (ff / fw - (nn - ff) / (1 - fw)))
            return val, grad
        start = np.full(J, max(ff.sum() / max(nn.sum(), 1), 1e-9) / J)
        r = minimize(fun, start, jac=True, method="L-BFGS-B", bounds=[(0, None)] * J,
                     options=dict(maxiter=5000, ftol=1e-13, gtol=1e-10))
        return float(r.fun), r.x
    base, rest0 = nll_fixed(0.0)
    full = PolyModel(sm, k0=k, J=J)
    best = min(base, full.nll)
    thr = chi2.ppf(2 * level - 1, 1) / 2          # one-sided
    hi = 1e-6
    while nll_fixed(hi)[0] - best < thr and hi < 1e3:
        hi *= 4
    lo = 0.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if nll_fixed(mid)[0] - best < thr:
            lo = mid
        else:
            hi = mid
    b_up = lo
    a_up = b_up / scale[0]
    _, rest = nll_fixed(b_up)
    out = dict(k=k, a_k_upper=float(a_up), level=level, shares={})
    wall = np.arange(len(n))
    phi_k = falling(wall, k) / falling(N, k)
    phi_rest = np.stack([falling(wall, j) / falling(N, j) for j in ks[1:]], axis=1) @ (rest / scale[1:])
    for p in ps:
        P = sm.pw(p)
        mode = float(P @ (a_up * phi_k))
        others = float(P @ phi_rest)
        out["shares"][f"{p:g}"] = mode / (mode + others) if mode + others > 0 else None
    # largest p (scanning down) where the bounded k-fault mode equals the rest of the fit
    grid = np.exp(np.linspace(np.log(5e-3), np.log(1e-6), 200))
    pc = None
    for p in grid:
        P = sm.pw(p)
        if float(P @ (a_up * phi_k)) >= float(P @ phi_rest):
            pc = float(p)
            break
    out["p_cross"] = pc
    return out
