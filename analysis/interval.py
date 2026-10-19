"""Interval study: logical error per round versus the forced full-measurement interval T.

Data: stratified results of the `interval` campaign (results/final/ch_interval/<tag>/X/...),
8 rounds after the preparation round, tags full (T = 1), B_T<T> / B_Tinf (escalation on
ambiguous events), C_T<T> / C_Tinf (escalation on any event), R4 / R8 (one outer check pair
or one outer check per round: every outer check once per 4 or 8 rounds).

Estimator: the hybrid estimate of FINDINGS section 3 with its smoothed parametric bootstrap
(strat.hybrid_boot). A point p counts as data-driven for a comparison when at least half of
every compared configuration's value comes from measured strata (>= 3 failures); where the
order-6 model dominates, ratios between configurations are p-independent by construction,
so no p-dependence is read from such points.

Quantities (docs/interval_theory.md):
  * R(p, T) = eps(p, T) / eps(p, full) for every configuration;
  * the elasticity E(p) = d ln eps / d ln T of the ambiguous-event family between T = 2 and
    T = 8 (and between 4 and 8): under an interval growing as 1/p (the unmasking rule of
    arXiv:2502.14835) the exponent of eps in p drops by E, under a constant interval it
    does not change;
  * gamma = d E / d ln p over the data-driven points: 0 when window failures need as many
    faults as in-round ones (the window penalty does not fade as p falls, T* ~ p^0); a
    window penalty that needed one more fault would fade in proportion to p (E(p) ~ p,
    gamma ~ E);
  * the fault-count test (fault_order_test): in bins of the fault density x = w/N, the
    failure probability of the longer interval relative to the shorter one, f_B/f_A ~
    x^gamma_w; window failures needing one more fault than the shorter interval's failures
    would give gamma_w ~ 1, the same number gamma_w ~ 0. The strata with failures span a
    wider range of fault densities than the p at which the hybrid estimates are data-driven;
  * T*(delta, p): the largest T in the observed range 2..8 (log-linear interpolation in T)
    with eps(p, T) <= (1 + delta) eps(p, full), reported as '>= 8' when T = 8 qualifies;
    no extrapolation beyond the observed T;
  * the 1/p rule evaluated inside the observed T range: T(p) = 10 (1e-3 / p) lies in 2..8
    for p in [1.25e-3, 5e-3], where eps(p, T(p)) is interpolated in T.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from strat import load, StratModel, PolyModel, hybrid_boot, local_order, path_resources  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# the high end is where the sampled failures lie (their strata stand for p ~ 2e-3 to 6e-3)
PS = (5e-3, 4e-3, 3e-3, 2.5e-3, 2e-3, 1.5e-3, 1.2e-3, 1e-3, 7e-4, 5e-4)


def tag_T(rule):
    if rule == "full":
        return "full", 1
    if "rot=" in rule:
        return "R", int(rule.split("rot=")[1].split(",")[0])
    fam = "B" if "ambiguous" in rule else ("C" if "any" in rule else ("P" if "l3=never" in rule else "?"))
    T = int(rule.split("T3=")[1].split(",")[0]) if "T3=" in rule else np.inf
    return fam, T


def models(pattern=None):
    pattern = pattern or os.path.join(ROOT, "results/final/ch_interval/*/*/*/*.json")
    out = {}
    for key, d in load(pattern).items():
        fam, T = tag_T(key[1])
        out[(fam, T, key[3])] = StratModel(key, d)
    return out


def make_model(sm, x0=None):
    return PolyModel(sm, k0=6, J=3, x0=x0)


def boot_all(M, ps, nboot, seed):
    """Hybrid estimates, shares and bootstrap replicas of every configuration."""
    out = {}
    for i, (k, sm) in enumerate(sorted(M.items(), key=lambda kv: (kv[0][0], float(kv[0][1])))):
        est, reps, _ = hybrid_boot(sm, make_model, ps, nboot=nboot, seed=seed + 101 * i)
        n = min(len(reps), nboot)
        out[k] = dict(eps=np.array([e for e, s in est]), share=np.array([s for e, s in est]),
                      reps=reps[:n], shots=int(sm.shots), fails=int(sm.f.sum()))
    m = min(len(v["reps"]) for v in out.values())
    for v in out.values():
        v["reps"] = v["reps"][:m]
    return out


def ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if len(x) else None


def pct(a):
    """Column-wise 95 % interval of bootstrap replicas, ignoring undefined (0/0) replicas."""
    a = np.where(np.isnan(a), np.nan, a)
    with np.errstate(invalid="ignore"):
        return [np.nanpercentile(a, 2.5, axis=0).tolist(), np.nanpercentile(a, 97.5, axis=0).tolist()]


def interp_logT(Ts, eps, T):
    """eps at interval T by linear interpolation of ln eps in ln T inside the observed Ts."""
    lt = np.log(np.asarray(Ts, float))
    return float(np.exp(np.interp(np.log(T), lt, np.log(eps))))


def fault_order_test(smA, smB, nbins=6, k_ref=6):
    """Descriptive: how B's failure probability grows with the fault density relative to
    A's. With x = w/N (fired fraction of each configuration's superset; P_p(W) peaks at
    w ~ N p), the model f_B(x) / f_A(x) = A (x / x0)^gamma is fitted by the binomial
    likelihood of how each x-bin's failures split between A and B given their total (the
    bin's common level cancels): within a bin A's exposure is sum n x^k_ref and B's
    sum n x^k_ref (x/x0)^gamma, each over its own strata, so different spreads of A's and
    B's shots inside a bin do not bias gamma. gamma describes the relative growth only: a
    longer interval whose extra failures need one more fault (f_B = f_A + C x^(k+1)) still
    gives gamma -> 0 where f_A dominates, so gamma cannot identify the order of the extra
    failures (the minimal failing sets do). Returns gamma, its profile 95 % interval, bins."""
    from scipy.optimize import minimize
    NA, NB = PolyModel(smA, k0=6, J=0).N, PolyModel(smB, k0=6, J=0).N
    wa, wb = np.arange(len(smA.n)), np.arange(len(smB.n))
    xa, xb = wa / NA, wb / NB
    pool_x = np.concatenate([xa[smA.n > 0], xb[smB.n > 0]])
    pool_n = np.concatenate([smA.n[smA.n > 0], smB.n[smB.n > 0]])
    o = np.argsort(pool_x)
    c = np.cumsum(pool_n[o]) / pool_n.sum()
    lo_x, hi_x = pool_x[o][np.searchsorted(c, 0.05)], pool_x[o][min(np.searchsorted(c, 0.995), len(c) - 1)]
    edges = np.exp(np.linspace(np.log(lo_x), np.log(hi_x), nbins + 1))
    edges[0], edges[-1] = 0.0, np.inf
    rows, bins = [], []
    for j in range(nbins):
        ma = (xa >= edges[j]) & (xa < edges[j + 1])
        mb = (xb >= edges[j]) & (xb < edges[j + 1])
        fa, fb = smA.f[ma].sum(), smB.f[mb].sum()
        na, nb = smA.n[ma].sum(), smB.n[mb].sum()
        xm = ((smA.n[ma] * xa[ma]).sum() + (smB.n[mb] * xb[mb]).sum()) / max(na + nb, 1)
        rows.append(dict(x=float(xm), fails_A=float(fa), fails_B=float(fb), shots_A=float(na), shots_B=float(nb)))
        bins.append((smA.n[ma] * xa[ma] ** k_ref, smB.n[mb], xb[mb]))
    use = [j for j, r in enumerate(rows) if r["fails_A"] + r["fails_B"] > 0 and bins[j][0].sum() > 0 and bins[j][1].sum() > 0]
    if len(use) < 2:
        return dict(gamma=None, bins=rows, note="fewer than two bins with failures")
    x0 = float(np.exp(np.average(np.log([rows[j]["x"] for j in use]),
                                 weights=[rows[j]["fails_A"] + rows[j]["fails_B"] for j in use])))
    fa = np.array([rows[j]["fails_A"] for j in use])
    fb = np.array([rows[j]["fails_B"] for j in use])
    lEA = np.log(np.array([bins[j][0].sum() for j in use]))

    def lEB(gm):
        return np.log(np.array([(bins[j][1] * bins[j][2] ** k_ref * (bins[j][2] / x0) ** gm).sum() for j in use]))

    def nll(th, g=None):
        la, gm = (th[0], th[1]) if g is None else (th[0], g)
        eta = la + lEB(gm) - lEA                # log odds that a failure of the bin is B's
        return -(fb * eta - (fa + fb) * np.logaddexp(0, eta)).sum()
    r0 = minimize(nll, [0.0, 0.0], method="BFGS")
    la, g = r0.x
    best = r0.fun
    prof = lambda gg: minimize(lambda t: nll(t, gg), [la], method="BFGS").fun - best
    grid = np.linspace(g - 12, g + 12, 2401)
    ok = [gg for gg in grid if prof(gg) <= 1.92]
    if ok and (min(ok) <= grid[0] or max(ok) >= grid[-1]):
        ok = []                        # the likelihood does not bound gamma
    PA = smA.pw(smA.p_dec)
    ew = float((np.arange(len(PA)) * PA).sum())
    p_of_x = lambda x: float(x * NA / ew * smA.p_dec)
    for r in rows:
        r["p_equiv"] = p_of_x(r["x"])
    xs_f = [rows[j]["x"] for j in use]
    return dict(gamma=float(g), A=float(np.exp(la)), x0=x0, gamma_ci95=[float(min(ok)), float(max(ok))] if ok else None,
                bins=rows, N_A=int(NA), N_B=int(NB), k_ref=k_ref, failures=[float(fa.sum()), float(fb.sum())],
                p_range=[p_of_x(min(xs_f)), p_of_x(max(xs_f))], note="descriptive; does not identify the order "
                "of the longer interval's extra failures")


def study(fam="B", basis="X", ps=PS, nboot=100, seed=0, deltas=(0.1, 0.5, 1.0), gate=0.5, B=None):
    M = models()
    if ("full", 1, basis) not in M:
        return {"missing": "full baseline"}
    B = B if B is not None else boot_all({k: v for k, v in M.items() if k[2] == basis}, list(ps), nboot, seed)
    full = B[("full", 1, basis)]
    Ts = sorted(T for (f, T, b) in B if f == fam and b == basis and np.isfinite(T))
    keys = [(fam, T, basis) for T in Ts] + ([(fam, np.inf, basis)] if (fam, np.inf, basis) in B else [])
    out = {"ps": list(ps), "T": Ts, "eps_full": full["eps"].tolist(), "eps": {}, "ratio": {}, "ratio_ci95": {},
           "measured_share": {"full": full["share"].tolist()}, "data": {}}
    out["data"]["full"] = dict(shots=full["shots"], fails=full["fails"])
    out["fault_order_k"] = {"full": local_order(M[("full", 1, basis)], use_all=True)}
    for k in keys:
        b = B[k]
        tk = "inf" if not np.isfinite(k[1]) else str(k[1])
        out["eps"][tk] = b["eps"].tolist()
        out["ratio"][tk] = (b["eps"] / full["eps"]).tolist()
        with np.errstate(divide="ignore", invalid="ignore"):
            rr = b["reps"] / full["reps"]
        out["ratio_ci95"][tk] = pct(rr)
        out["measured_share"][tk] = b["share"].tolist()
        out["data"][tk] = dict(shots=b["shots"], fails=b["fails"])
        try:
            out.setdefault("fault_order_k", {})[tk] = local_order(M[k], use_all=True)
        except Exception as ex:
            out.setdefault("fault_order_k", {})[tk] = {"error": repr(ex)}
    # data-driven points for each comparison
    def driven(cfgs):
        return [i for i in range(len(ps)) if min(B[c]["share"][i] for c in cfgs) >= gate]
    out["gate"] = f"every compared configuration >= {gate:.0%} measured"
    out["ratio_driven"] = {tk: [ps[i] for i in driven([("full", 1, basis), (fam, (np.inf if tk == 'inf' else int(tk)), basis)])]
                           for tk in out["ratio"]}
    if len(Ts) >= 2:
        lo_T, hi_T = Ts[0], Ts[-1]
        fin = [(fam, T, basis) for T in Ts]
        idx = driven(fin)
        out["elasticity_points"] = [ps[i] for i in idx]
        E = {}
        for a, b in ((lo_T, hi_T),) + (((Ts[-2], hi_T),) if len(Ts) >= 3 else ()):
            ka, kb = (fam, a, basis), (fam, b, basis)
            e0 = np.log(B[kb]["eps"] / B[ka]["eps"]) / np.log(b / a)
            with np.errstate(divide="ignore", invalid="ignore"):
                er = np.log(B[kb]["reps"] / B[ka]["reps"]) / np.log(b / a)
            E[f"{a}-{b}"] = dict(value=e0.tolist(), ci95=pct(er), driven=[ps[i] for i in driven([ka, kb])])
        out["elasticity"] = E
        # p-dependence of the elasticity over the data-driven points (2..8)
        key = f"{lo_T}-{hi_T}"
        di = [i for i in range(len(ps)) if ps[i] in E[key]["driven"]]
        if len(di) >= 2:
            x = np.log(np.array([ps[i] for i in di]))
            e0 = np.log(B[(fam, hi_T, basis)]["eps"] / B[(fam, lo_T, basis)]["eps"]) / np.log(hi_T / lo_T)
            with np.errstate(divide="ignore", invalid="ignore"):
                er = np.log(B[(fam, hi_T, basis)]["reps"] / B[(fam, lo_T, basis)]["reps"]) / np.log(hi_T / lo_T)
            g0 = float(np.polyfit(x, e0[di], 1)[0])
            gr = np.array([np.polyfit(x, row[di], 1)[0] for row in er if np.all(np.isfinite(row[di]))])
            out["gamma"] = dict(value=g0, ci95=ci(gr), points=[ps[i] for i in di],
                                note="d E / d ln p, E the elasticity of eps in T between "
                                     f"T = {lo_T} and {hi_T}")
        else:
            out["gamma"] = dict(value=None, ci95=None, points=[ps[i] for i in di],
                                note="fewer than two data-driven points")
        # T*(delta): largest T in the observed range with eps(T) <= (1 + delta) eps(full)
        tstar = {}
        for d in deltas:
            row = {}
            for i in range(len(ps)):
                if ps[i] not in out["elasticity_points"] or min(full["share"][i], 1.0) < gate:
                    continue
                eT = [B[(fam, T, basis)]["eps"][i] for T in Ts]
                lim = (1 + d) * full["eps"][i]
                if eT[-1] <= lim:
                    row[f"{ps[i]:g}"] = f">= {hi_T}"
                    continue
                grid = np.exp(np.linspace(np.log(lo_T), np.log(hi_T), 400))
                ok = [T for T in grid if interp_logT(Ts, eT, T) <= lim]
                row[f"{ps[i]:g}"] = f"{max(ok):.1f}" if ok else f"< {lo_T}"
            tstar[str(d)] = row
        out["T_star"] = tstar
        # the 1/p unmasking rule inside the observed T range
        rule = {}
        for i, p in enumerate(ps):
            Tp = 10 * 1e-3 / p
            if not (lo_T <= Tp <= hi_T):
                continue
            eT = [B[(fam, T, basis)]["eps"][i] for T in Ts]
            rule[f"{p:g}"] = dict(T=Tp, eps=interp_logT(Ts, eT, Tp), ratio_full=interp_logT(Ts, eT, Tp) / full["eps"][i],
                                  driven=p in out["elasticity_points"])
        out["inverse_p_rule"] = dict(points=rule, note="T(p) = 10 (1e-3/p), arXiv:2502.14835; only p with T(p) "
                                                       f"inside the observed {lo_T}..{hi_T}")
    # windows that always reach T (periodic family): the penalty per round of window length,
    # R(p, T) = a + c (T - 1), fitted at each data-driven p over the observed T (replicas give
    # the intervals); with equal orders c is the p-independent cost of one more round of window
    if fam == "P" and len(Ts) >= 2:
        lin = {}
        X = np.array([T - 1.0 for T in Ts])
        for i, p in enumerate(ps):
            cfgs = [("full", 1, basis)] + [(fam, T, basis) for T in Ts]
            if min(B[c_]["share"][i] for c_ in cfgs) < gate:
                continue
            Rv = np.array([B[(fam, T, basis)]["eps"][i] / full["eps"][i] for T in Ts])
            a0, c0 = np.polyfit(X, Rv, 1)[::-1]
            with np.errstate(divide="ignore", invalid="ignore"):
                Rr = np.stack([B[(fam, T, basis)]["reps"][:, i] / full["reps"][:, i] for T in Ts], axis=1)
            ok = np.all(np.isfinite(Rr), axis=1)
            fits = np.array([np.polyfit(X, row, 1)[::-1] for row in Rr[ok]]) if ok.any() else np.zeros((0, 2))
            lin[f"{p:g}"] = dict(a=float(a0), c=float(c0), a_ci95=ci(fits[:, 0]) if len(fits) else None,
                                 c_ci95=ci(fits[:, 1]) if len(fits) else None, R=Rv.tolist())
        out["linear_penalty"] = dict(points=lin, T=Ts, note="R(p, T) = a + c (T - 1) over the observed T at "
                                     "each data-driven p; c = extra error per round of window, relative to full "
                                     "extraction")
    # fault-count dependence of the interval penalty, straight from the strata
    if len(Ts) >= 2:
        pairs = [(Ts[0], Ts[-1])] + ([(Ts[-2], Ts[-1])] if len(Ts) >= 3 else [])
        out["fault_order"] = {}
        for a, b in pairs:
            try:
                out["fault_order"][f"{a}-{b}"] = fault_order_test(M[(fam, a, basis)], M[(fam, b, basis)])
            except Exception as ex:
                out["fault_order"][f"{a}-{b}"] = {"error": repr(ex)}
    # how often an escalation measures the full syndrome without forcing (rule without forced
    # measurements): windows end at an escalation after about 1/rate rounds, so a forced
    # interval T matters only where 1/rate exceeds T
    if (fam, np.inf, basis) in M:
        res = path_resources(M[(fam, np.inf, basis)], list(ps), shots=2000)
        out["escalation_per_round"] = [res[p]["full_outer_fraction"] for p in ps]
        # the same down to p = 1e-4 (direct path sampling: a measurement at every p): without
        # forcing, a window lasts ~1 / rate rounds; the rate's exponent in p tells how windows
        # lengthen as p falls
        pe = [5e-3, 3e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4]
        rr = path_resources(M[(fam, np.inf, basis)], pe, shots=8000)
        rate = np.array([rr[p]["full_outer_fraction"] for p in pe])
        lo = [k for k, p in enumerate(pe) if p <= 1e-3 and rate[k] > 0]
        slope = float(np.polyfit(np.log([pe[k] for k in lo]), np.log(rate[lo]), 1)[0]) if len(lo) >= 2 else None
        out["escalation_scan"] = dict(p=pe, per_round=rate.tolist(), shots=8000, exponent_low_p=slope,
                                      note="fraction of rounds with a full outer measurement without forcing "
                                           "(direct path sampling, 8 rounds); exponent fitted for p <= 1e-3")
    if (fam, np.inf, basis) in B and Ts:
        ka, kb = (fam, Ts[-1], basis), (fam, np.inf, basis)
        r0 = B[kb]["eps"] / B[ka]["eps"]
        with np.errstate(divide="ignore", invalid="ignore"):
            rr = B[kb]["reps"] / B[ka]["reps"]
        out["inf_over_maxT"] = dict(value=r0.tolist(), ci95=pct(rr), driven=[ps[i] for i in driven([ka, kb])])
    return out


def study_all(basis="X", ps=PS, nboot=100, seed=0):
    M = models()
    if ("full", 1, basis) not in M:
        return {"missing": "full baseline"}
    B = boot_all({k: v for k, v in M.items() if k[2] == basis}, list(ps), nboot, seed)
    res = {fam: study(fam, basis, ps, nboot, seed, B=B) for fam in ("B", "C", "R", "P")
           if any(k[0] == fam for k in B)}
    return res


if __name__ == "__main__":
    import json
    r = study_all(nboot=int(sys.argv[1]) if len(sys.argv) > 1 else 50)
    for fam, v in r.items():
        print(fam, json.dumps({k: v[k] for k in v if k not in ("eps", "ratio_ci95")}, default=float)[:3000])
