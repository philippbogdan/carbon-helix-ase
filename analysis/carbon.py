"""Carbon-Helix logical error per round across p, its exponent, and the extrapolation
model's validation, for every stratified configuration of a campaign.

For each configuration (rule, noise, basis, rounds):
  * eps(p) on a grid: hybrid estimate (measured f_w where a stratum has >= 3 failures, the
    order-6 polynomial model elsewhere), the share of it that is measured, statistical 95 %
    interval from the smoothed parametric bootstrap (strata redrawn from the hybrid f_w,
    model refitted); 'measured' when >= 90 % of it comes from measured strata, otherwise
    'extrapolated'; the model-free (Jeffreys) one-sided 95 % upper bound; the model range (fits with
    only the sixth- and only the seventh-order term);
  * local exponent d ln eps / d ln p between neighbouring grid points (bootstrap);
  * held-out validation: the model fitted on the strata above the median sampled W
    predicts the failures observed below it (cut fixed by the sample sizes);
  * resources per round from direct path sampling at each p (strat.path_resources);
  * asymptotic exponent: the smallest minimal failing set (core) found, with its genuineness
    (exact most-likely-error decoding also fails).
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from strat import (load, StratModel, PolyModel, hybrid_eps, hybrid_boot, heldout, boot_summary,  # noqa: E402
                   slope_summary, direct_posterior, path_resources, local_order)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MIN_FAILS = 5        # fewer failures in a configuration: no estimate (the fit is unconstrained)
# the high end is where the sampled failures lie (their strata stand for p ~ 2e-3 to 6e-3)
GRID = {"uniform": [5e-3, 4e-3, 3e-3, 2e-3, 1.5e-3, 1e-3, 7e-4, 5e-4, 3e-4, 2e-4, 1e-4],
        "helios": [3.2e-3, 2.4e-3, 1.6e-3, 1.2e-3, 7.9e-4, 5e-4, 3e-4, 2e-4, 1e-4]}


def minimal_records(campaign, key):
    rows = []
    for f in glob.glob(os.path.join(ROOT, "results", "minimal", "*.json")):
        d = json.load(open(f))
        if (d.get("campaign"), d.get("rule"), d.get("noise"), d.get("basis"), d.get("rounds")) == \
                (campaign, key[1], key[2], key[3], key[4]):
            rows += d.get("rows", [])
    return rows


def model_factory(cores=None, k0_default=6):
    """Extrapolation model f_w = sum_{k=k0}^{k0+3} a_k [w]_k/[N]_k fitted to the stratified
    failure counts alone (binomial likelihood). Lowest order k0 = 6 for Carbon-Helix:
    six-fault failures exist (constructed, and a genuine six-fault minimal failing set was
    found) and no smaller one was found; 3 for C4-Helix (distance 6). The sizes of minimal
    failing sets are not used in the fit: greedy reduction stops at locally minimal sets,
    often larger than the smallest failing subset, so their sizes are biased upward (a fit
    using them failed its held-out check). make(sm, x0=None)."""
    def make(sm, x0=None):
        return PolyModel(sm, k0=k0_default, J=3, x0=x0)
    return make


def audit_for(campaign, key):
    """Decoder audit of a configuration (analysis/redecode.py), if it exists."""
    f = os.path.join(ROOT, "results", "redecode", f"{campaign}__{TAG_OF.get(key[1], key[1])}__{key[3]}.json")
    return json.load(open(f)) if os.path.exists(f) else None


TAG_OF = {"full": "full", "l2=on_event,l3=any": "C", "l2=on_event,l3=ambiguous,T3=4": "B4",
          "l2=on_event,l3=ambiguous,rot=4": "R4", "l2=on_event,l3=ambiguous,rot=8": "R8",
          "l2=on_event,l3=ambiguous,rot=8,l2fast=1": "R8F", "l2=on_event,l3=ambiguous,rot=16,l2fast=1": "R16F"}


def strong_corrected(sm, audit):
    """Failure counts confirmed by the strong decoder, per stratum. Every stored failure
    (W <= the saving cut) is re-decoded, so the confirmed ('genuine') count of an audited
    stratum is known exactly; failures not covered by the audit (strata above the cut, or
    stored after the audit ran) are counted with the pooled confirmed fraction, whose
    uncertainty (Beta posterior of the audited counts) is resampled with the strata by
    `perturb` (for hybrid_boot). Successes that the strong decoder would fail are measured
    on the stored random sample of successes; their rate and its 95 % upper limit are
    reported for the audit's own sample (its fault counts), not added:
        f_w(strong) = genuine_w + g (f_w - audited_w)."""
    import copy
    tot = audit["totals"]
    g = tot["genuine"] / tot["fail"] if tot["fail"] else 1.0
    sc = copy.copy(sm)
    f = sm.f.copy().astype(float)
    conf = np.zeros(len(f))
    unaud = np.zeros(len(f))
    cover = {}
    for w in range(len(f)):
        v = audit["per_w"].get(str(w))
        if sm.f[w] == 0:
            continue
        a = min(v["fail"], sm.f[w]) if v else 0
        gen = v["fail_strong"] * (a / v["fail"]) if v and v["fail"] else 0.0
        conf[w], unaud[w] = gen, sm.f[w] - a
        f[w] = gen + g * unaud[w]
        cover[str(w)] = dict(fails=int(sm.f[w]), audited=int(a), genuine=float(gen))
    sc.f = f
    m, k = tot["ok_sampled"], tot["ok_strong_fail"]
    from scipy.stats import beta

    def perturb(rng):
        gs = rng.beta(tot["genuine"] + 0.5, tot["fail"] - tot["genuine"] + 0.5)
        drawn = conf + rng.binomial(unaud.astype(np.int64), gs)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(f > 0, drawn / np.maximum(f, 1e-300), 1.0)
    info = dict(genuine_fraction=g, audited_fails=int(tot["fail"]), genuine=int(tot["genuine"]),
                campaign_fails=int(sm.f.sum()), unaudited_fails=float(unaud.sum()),
                coverage=float(1 - unaud.sum() / max(sm.f.sum(), 1)),
                ok_sampled=int(m), ok_strong_fail=int(k),
                ok_to_fail_rate_hi95=float(beta.ppf(0.95, k + 1, m - k)) if m else None,
                ok_to_fail_note="per stored success of the audit sample (its fault counts), not at a target p",
                per_w=cover)
    return sc, info, perturb


def analyse(campaign, key, d, nboot=100, seed=0, decoder="campaign", resources=True):
    sm = StratModel(key, d)
    corr = perturb = None
    if decoder == "strong":
        audit = audit_for(campaign, key)
        if audit is None:
            return None
        sm, corr, perturb = strong_corrected(sm, audit)
    if sm.f.sum() < MIN_FAILS:
        return None                   # too few failures to fit anything (reported as missing)
    rows = minimal_records(campaign, key)
    cores = [(r["W"], r["size"]) for r in rows]
    make = model_factory(cores)
    grid = GRID["helios" if key[2].startswith("helios") else "uniform"]
    est, reps, base = hybrid_boot(sm, make, grid, nboot=nboot, seed=seed, perturb=perturb)
    # model-free interval: no smallest failing fault count assumed; not defined for the
    # confirmed-failure counts (they are estimates, not binomial observations)
    dp = direct_posterior(sm, grid, k0=None, seed=seed) if decoder == "campaign" else None
    res = path_resources(sm, grid) if resources else {}
    rec = dict(shots=int(sm.shots), fails=float(sm.f.sum()), cores=len(cores),
               core_sizes={str(k): int(v) for k, v in zip(*np.unique([c[1] for c in cores], return_counts=True))}
               if cores else {}, model=type(base).__name__, model_orders=list(getattr(base, "ks", [])),
               model_coefficients=[float(x) for x in base.a], points={}, slopes={},
               bootstrap="smoothed parametric (hybrid f_w), %d replicas" % len(reps))
    for i, p in enumerate(grid):
        e, share = est[i]
        lo, hi = boot_summary(reps, i)
        rp = res.get(p, {})
        rec["points"][f"{p:g}"] = dict(
            eps=e, eps_LQ=e / 2, lo95=lo, hi95=hi, measured_share=share,
            status="measured" if share >= 0.9 else "extrapolated",
            data_only_upper_LQ=dp[p]["upper95"] / 2 if dp else None,
            twoq_per_round=rp.get("twoq"), time_ms_per_round=rp.get("time_ms"),
            outer_checks=rp.get("outer_checks"), full_outer_fraction=rp.get("full_outer_fraction"),
            twoq_se=rp.get("twoq_se"), time_ms_se=rp.get("time_ms_se"), peak_qubits=sm.peak_qubits())
    for i in range(len(grid) - 1):
        a, b = grid[i], grid[i + 1]
        s0, lo, hi = slope_summary(reps, [x[0] for x in est], grid, i, i + 1)
        rec["slopes"][f"{a:g}-{b:g}"] = dict(
            slope=s0, lo95=lo, hi95=hi,
            status="measured" if min(est[i][1], est[i + 1][1]) >= 0.9 else "extrapolated")
    # model systematics: the strata do not yet separate the leading order when few failures
    # have few faults; bracket the extrapolation by fits with only the sixth-order term and
    # with only the seventh-order term (six-fault failures exist, so the physical answer
    # lies at or above the 'only 6' side asymptotically)
    alt = {}
    for k0, J in ((6, 0), (7, 0)):
        try:
            mJ = PolyModel(sm, k0=k0, J=J)
            alt[(k0, J)] = [hybrid_eps(sm, mJ, p)[0] for p in grid]
            rec.setdefault("model_alternatives", {})[f"only_order_{k0}"] = dict(nll=mJ.nll)
        except Exception:
            pass
    rec.setdefault("model_alternatives", {})["orders_6_to_9"] = dict(nll=base.nll)
    for i, p in enumerate(grid):
        vals = [est[i][0]] + [v[i] for v in alt.values()]
        rec["points"][f"{p:g}"]["model_range_LQ"] = [min(vals) / 2, max(vals) / 2]
    rec["heldout"] = heldout(sm, make)
    # the exponent straight from the strata (no model order imposed)
    rec["fault_order_all"] = local_order(sm, use_all=True)
    rec["fault_order_low"] = local_order(sm)
    rec["decoder"] = decoder
    rec["audit"] = corr
    genuine = [r["size"] for r in rows if r.get("mle_fails") or r.get("strong_fails")]
    rec["min_core"] = int(min(c[1] for c in cores)) if cores else None
    rec["min_core_genuine"] = int(min(genuine)) if genuine else None
    return rec


def campaign_summary(campaign, nboot=100, decoder="campaign"):
    out = {}
    for key, d in load(os.path.join(ROOT, f"results/final/{campaign}/*/*/*/*.json")).items():
        r = analyse(campaign, key, d, nboot=nboot, decoder=decoder, resources=(decoder == "campaign"))
        if r is not None:
            out["|".join(map(str, key))] = r
    return out


if __name__ == "__main__":
    camp = sys.argv[1] if len(sys.argv) > 1 else "ch_main"
    res = campaign_summary(camp, nboot=int(sys.argv[2]) if len(sys.argv) > 2 else 50)
    for k, r in res.items():
        h = r["heldout"] or {}
        print(k, "shots", r["shots"], "fails", r["fails"], "cores", r["core_sizes"],
              "heldout", {x: h.get(x) for x in ("cut", "predicted", "observed", "p_value", "status")})
        for p, x in r["points"].items():
            print(f"   p={p}: eps/LQ={x['eps_LQ']:.2e} [{x['lo95']/2:.1e},{x['hi95']/2:.1e}] {x['status']} "
                  f"({100*x['measured_share']:.0f}% measured) data-only < {x['data_only_upper_LQ']:.1e}"
                  f" model [{x['model_range_LQ'][0]:.1e},{x['model_range_LQ'][1]:.1e}]"
                  f" 2q={x['twoq_per_round']} t={x['time_ms_per_round']}")
        for ab, x in r["slopes"].items():
            print(f"   slope {ab}: {x['slope']:.2f} [{x['lo95']}, {x['hi95']}] {x['status']}")
