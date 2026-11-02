"""Rebuild results/summary.json from samples, saved audits and reference readings.

A full rebuild needs no previous summary. --check compares against the existing
summary without writing; --only updates named sections in an existing summary,
or compares just those sections when combined with --check.
The c4_uniform section also reads results/reference/c4helix_fig11a.csv.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def per_round(pl, r):
    from strat import per_round as pr
    return pr(pl, r)




def sec_c4_uniform(nboot=200):
    """Compare full-extraction C4 with readings of the authors' Fig. 11a.

    Use c4_repro: uniform noise, 12 memory rounds, no postselection (gamma = 0).
    Convert each basis's block failure probability to 1 - (1 - p_L)^(1/12),
    then average X and Z. Decoder priors remain at the samples' p_dec.
    Observed fractions supply strata with at least three failures; a polynomial
    starting at order three supplies the rest. Average paired bootstrap replicas
    to obtain the conditional 95% interval (200 refits per basis by default).

    The CSV supplies evaluation points and reference rates. Its estimated
    reading uncertainty is not propagated into intervals or containment checks.
    Also return alternative-fit ranges (single terms of order three or four,
    enclosing the central estimate), the mean of the two basiswise Jeffreys
    upper limits, held-out checks and free-order fits. The averaged upper limits
    are not a joint interval. This does not validate adaptive C4 or Helios noise.
    """
    from strat import load, StratModel, PolyModel, hybrid_boot, hybrid_eps, direct_posterior, heldout, local_order
    import csv
    with open(os.path.join(ROOT, "results", "reference", "c4helix_fig11a.csv")) as fh:
        PAPER = {float(r["p"]): float(r["error_per_round"]) for r in csv.DictReader(fh)}
    data = load(os.path.join(ROOT, "results/final/c4_repro/*/*/*/*.json"))
    models = {key[3]: StratModel(key, d) for key, d in data.items()}
    if set(models) != {"X", "Z"}:
        return {"missing": "results/final/c4_repro (X and Z)"}
    ps = sorted(PAPER)
    make = lambda s, x0=None: PolyModel(s, k0=3, J=3, x0=x0)
    est, reps, share, alt, dp, ho, lo_ = {}, {}, {}, {}, {}, {}, {}
    for i, (b, sm) in enumerate(sorted(models.items())):
        e, r, base = hybrid_boot(sm, make, ps, nboot=nboot, seed=11 + i)
        est[b] = np.array([x[0] for x in e])
        share[b] = [x[1] for x in e]
        reps[b] = r
        alt[b] = [np.array([hybrid_eps(sm, PolyModel(sm, k0=k, J=0), p)[0] for p in ps]) for k in (3, 4)]
        dp[b] = direct_posterior(sm, ps, k0=None, seed=11 + i)
        ho[b] = {k: v for k, v in heldout(sm, make).items() if k not in ("fit_strata", "heldout_strata")}
        lo_[b] = dict(all=local_order(sm, use_all=True), low=local_order(sm))
    m = min(len(reps["X"]), len(reps["Z"]))
    avg = (est["X"] + est["Z"]) / 2
    ravg = (reps["X"][:m] + reps["Z"][:m]) / 2
    out = {"reference_values": {str(k): v for k, v in PAPER.items()}, "rounds": 12, "points": {},
           "estimator": "hybrid (order-3 model where a stratum has < 3 failures), smoothed parametric bootstrap",
           "shots": {b: int(sm.shots) for b, sm in models.items()},
           "fails": {b: int(sm.f.sum()) for b, sm in models.items()}, "heldout": ho, "fault_order": lo_}
    for i, p in enumerate(ps):
        lo, hi = np.percentile(ravg[:, i], [2.5, 97.5])
        mlo = min((alt["X"][j][i] + alt["Z"][j][i]) / 2 for j in range(2))
        mhi = max((alt["X"][j][i] + alt["Z"][j][i]) / 2 for j in range(2))
        dhi = (dp["X"][p]["upper95"] + dp["Z"][p]["upper95"]) / 2
        out["points"][str(p)] = dict(eps_X=float(est["X"][i]), eps_Z=float(est["Z"][i]), eps_avg=float(avg[i]),
                                     measured_share_X=share["X"][i], measured_share_Z=share["Z"][i],
                                     ci95=[float(lo), float(hi)], model_range=[float(min(mlo, avg[i])), float(max(mhi, avg[i]))],
                                     data_only_upper95=float(dhi),
                                     paper=PAPER[p], ratio=float(avg[i] / PAPER[p]),
                                     paper_inside_ci=bool(lo <= PAPER[p] <= hi),
                                     paper_below_data_only_upper=bool(PAPER[p] <= dhi))
    out["slope_ours"] = float(np.polyfit(np.log(ps), np.log(avg), 1)[0])
    sl = [np.polyfit(np.log(ps), np.log(r), 1)[0] for r in ravg if np.all(r > 0)]
    out["slope_ours_ci95"] = [float(np.percentile(sl, 2.5)), float(np.percentile(sl, 97.5))] if sl else None
    out["slope_replicas_used"] = [len(sl), int(m)]
    out["slope_paper"] = float(np.polyfit(np.log(ps), np.log([PAPER[p] for p in ps]), 1)[0])
    return out


RULE_SHORT = {"full": "non-adaptive", "C": "any-event C", "B4": "ambiguous-event B4", "R4": "rotation R4",
              "R8": "rotation R8", "R8F": "rotation R8F (parallel weight-6)"}

TAGS = {"full": "full", "l2=on_event,l3=any": "C", "l2=on_event,l3=ambiguous,T3=4": "B4",
        "l2=on_event,l3=ambiguous,rot=4": "R4", "l2=on_event,l3=ambiguous,rot=8": "R8",
        "l2=on_event,l3=ambiguous,rot=8,l2fast=1": "R8F"}


def sec_carbon():
    """Carbon-Helix per configuration, flattened for the documents:
    carbon/<noise>/<basis>/<tag>/{eps_LQ,lo95_LQ,hi95_LQ,status,measured_share,twoq,time_ms,
    outer_fraction}/<p> plus slopes, held-out validation, cores, shots, failures."""
    from carbon import campaign_summary
    out = {}
    for camp, noise, dec in (("ch_main", "uniform", "campaign"), ("ch_helios", "helios", "campaign"),
                             ("ch_main", "uniform", "strong"), ("ch_helios", "helios", "strong")):
        r = campaign_summary(camp, nboot=100, decoder=dec)
        for key, rec in r.items():
            proto, rule, nz, basis, rounds, pdec = key.split("|")
            tag = TAGS.get(rule, rule)
            flat = dict(rule=rule, rounds=int(rounds), shots=rec["shots"], fails=rec["fails"],
                        cores=rec["cores"], core_sizes=rec["core_sizes"], min_core=rec["min_core"],
                        min_core_genuine=rec["min_core_genuine"], model=rec["model"],
                        model_orders=rec["model_orders"], heldout=rec["heldout"],
                        fault_order_all=rec.get("fault_order_all"), fault_order_low=rec.get("fault_order_low"))
            for fld, src in (("eps_LQ", "eps_LQ"), ("status", "status"), ("measured_share", "measured_share"),
                             ("twoq", "twoq_per_round"), ("time_ms", "time_ms_per_round"),
                             ("outer_checks", "outer_checks"), ("full_outer_fraction", "full_outer_fraction"),
                             ("peak_qubits", "peak_qubits")):
                flat[fld] = {p: x[src] for p, x in rec["points"].items()}
            flat["model_range_LQ"] = {p: x.get("model_range_LQ") for p, x in rec["points"].items()}
            flat["data_only_upper_LQ"] = {p: x.get("data_only_upper_LQ") for p, x in rec["points"].items()}
            flat["lo95_LQ"] = {p: x["lo95"] / 2 for p, x in rec["points"].items()}
            flat["hi95_LQ"] = {p: x["hi95"] / 2 for p, x in rec["points"].items()}
            flat["slope"] = {ab: x["slope"] for ab, x in rec["slopes"].items()}
            flat["slope_lo95"] = {ab: x["lo95"] for ab, x in rec["slopes"].items()}
            flat["slope_hi95"] = {ab: x["hi95"] for ab, x in rec["slopes"].items()}
            flat["slope_status"] = {ab: x["status"] for ab, x in rec["slopes"].items()}
            flat["audit"] = rec.get("audit")
            if dec == "campaign":
                out.setdefault(noise, {}).setdefault(basis, {})[tag] = flat
            else:
                out.setdefault(noise + "_strong", {}).setdefault(basis, {})[tag] = flat
    return out or {"missing": "Carbon-Helix stratified results"}


def sec_interval():
    from interval import study_all
    return study_all()


def sec_value():
    from value import table
    return table()


def sec_circuits():
    """Per part: two-qubit gates, layers, ancillas, peak live qubits, Helios-model duration and
    expected faults per execution (uniform p = 1e-3 and Helios model)."""
    from hase.protocols import make_protocol
    from resources import peak_live
    out = {}
    for name, rule in (("carbonhelix", "l2=on_event,l3=ambiguous,rot=4"), ("c4helix", "mode=ase")):
        pu = make_protocol(name, 1e-3, "X", 2, rule, "uniform")
        ph = make_protocol(name, 7.9e-4, "X", 2, rule, "helios")
        rec = {}
        for k, ap in pu.aparts.items():
            if k.startswith(("L2_", "W2_")) and k not in ("L2_all", "L2_0", "W2_0"):
                continue
            if k.startswith("L3r_") and k != "L3r_0":
                continue
            rec[k] = dict(two_q=pu.parts[k].two_q, layers=pu.parts[k].n_layers, ancillas=ap.n_anc,
                          peak_qubits=peak_live(ap, pu.n_data), helios_ms=ph.parts[k].duration,
                          faults_uniform_1e3=float(pu.parts[k].site_prob.sum()),
                          faults_helios=float(ph.parts[k].site_prob.sum()))
        out[name] = rec
    return out


def sec_round_totals(S):
    c = S.get("circuits", {})
    if "carbonhelix" not in c:
        return {}, {}
    ch, c4 = c["carbonhelix"], c["c4helix"]
    full = dict(two_q=ch["L1_a"]["two_q"] + ch["L2_all"]["two_q"] + ch["L3"]["two_q"],
                helios_ms=ch["L1_a"]["helios_ms"] + ch["L2_all"]["helios_ms"] + ch["L3"]["helios_ms"])
    c4r = dict(two_q=c4["inner_a"]["two_q"] + c4["outer"]["two_q"],
               helios_ms=c4["inner_a"]["helios_ms"] + c4["outer"]["helios_ms"])
    return full, c4r


def sec_sensitivity():
    """Unpublished Helios quantities: error per logical qubit per round (X basis, 4 rounds)
    at the model's own p for memory rate x0.5 / x2 and measurement-reset 4 ms, for
    Carbon-Helix (non-adaptive, R4) and C4-Helix (non-adaptive, ASE); hybrid estimates with
    their statistical interval, and time per round from direct path sampling using
    each variant's timing parameters. Round means exclude full preparation, which
    scripts/headline.py times with the same variant for its per-second ratios."""
    from strat import load, StratModel, hybrid_boot, boot_summary, path_resources
    from carbon import model_factory
    out = {}
    for camp, code in (("ch_helios_sens", "Carbon-Helix"), ("c4_helios_sens", "C4-Helix")):
        for key, d in load(os.path.join(ROOT, f"results/final/{camp}/*/*/*/*/*.json")).items():
            sm = StratModel(key, d)
            make = model_factory(k0_default=6 if code == "Carbon-Helix" else 3)
            est, reps, base = hybrid_boot(sm, make, [key[5]], nboot=100, seed=5)
            lo, hi = boot_summary(reps, 0)
            res = path_resources(sm, [key[5]])[key[5]]
            out[f"{code}|{key[1]}|{key[2]}"] = dict(eps_LQ=est[0][0] / 2, ci95_LQ=[lo / 2, hi / 2],
                                                   measured_share=est[0][1], time_ms_per_round=res["time_ms"],
                                                   twoq_per_round=res["twoq"],
                                                   shots=int(sm.shots), fails=int(sm.f.sum()))
    return out or {"missing": "Helios sensitivity campaign"}


def sec_direct(S):
    """Direct (unstratified) sampling of the non-adaptive Carbon-Helix memory at p = 2e-3 and
    3e-3 (scripts/campaigns.py ch_direct) against the stratified hybrid estimate at the same
    p: a check of the whole rare-event method where both are feasible."""
    from agg_protocol import load
    agg = load(os.path.join(ROOT, "results/final/ch_direct/*/*/*.json"))
    if not agg:
        return {"missing": "results/final/ch_direct"}
    car = (S.get("carbon", {}) or {}).get("uniform", {}).get("X", {}).get("full", {})
    out = {}
    for key, a in agg.items():
        proto, rule, noise, p, b, r = key
        n, k = int(a["shots"]), int(a["fails"])
        lo, hi = wilson(k, n)
        rec = dict(shots=n, fails=k, eps_LQ=per_round(k / n, r) / 2,
                   ci95_LQ=[per_round(lo, r) / 2, per_round(hi, r) / 2])
        st = (car.get("eps_LQ") or {}).get(f"{p:g}")
        if st is not None:
            slo, shi = car["lo95_LQ"][f"{p:g}"], car["hi95_LQ"][f"{p:g}"]
            rec.update(stratified_LQ=st, stratified_ci95_LQ=[slo, shi], ratio=rec["eps_LQ"] / st if st else None,
                       intervals_overlap=bool(rec["ci95_LQ"][0] <= shi and slo <= rec["ci95_LQ"][1]),
                       stratified_status=car["status"][f"{p:g}"])
        out[f"{p:g}"] = rec
    return out


def sec_basis():
    """Which basis is worse: the X-basis failure probability per stratum (hybrid: measured
    where a stratum has >= 3 failures, the order-6 model elsewhere) applied to the strata the
    Z-basis run sampled gives the failures X would have had there; compared with Z's observed
    count (ratio with an exact Poisson 95 % interval on the observed count)."""
    from strat import load, StratModel, hybrid_fhat
    from carbon import model_factory
    from scipy.stats import chi2
    out = {}
    for camp, noise in (("ch_main", "uniform"), ("ch_helios", "helios")):
        X = load(os.path.join(ROOT, f"results/final/{camp}/full/X/*/*.json"))
        Z = load(os.path.join(ROOT, f"results/final/{camp}/full/Z/*/*.json"))
        if not X or not Z:
            continue
        smx = StratModel(*next(iter(X.items())))
        smz = StratModel(*next(iter(Z.items())))
        fx = hybrid_fhat(smx, model_factory()(smx))
        L = min(len(fx), len(smz.n))
        exp = float((smz.n[:L] * fx[:L]).sum())
        obs = int(smz.f.sum())
        lo = chi2.ppf(0.025, 2 * obs) / 2 if obs > 0 else 0.0
        hi = chi2.ppf(0.975, 2 * obs + 2) / 2
        out[noise] = dict(z_shots=int(smz.shots), z_fails=obs, x_expected_on_z_strata=exp,
                          z_over_x=obs / exp if exp > 0 else None, z_over_x_ci95=[lo / exp, hi / exp] if exp > 0 else None)
    return out or {"missing": "Z-basis sets"}


def sec_distance():
    """Circuit fault distance at the end of the circuit (scripts/path_distance.py --method
    restricted; scripts/distance_suite.py): for every place an outer-check part can end a
    circuit, the minimum number of fault mechanisms of an undetectable logical error among
    the mechanisms seen only by the final measurement and the flags (exact integer
    programming: optimal = proved), minimised over the logical classes; and the same for the
    outer-check gadget designs compared (results/distance/design__*)."""
    files = sorted(glob.glob(os.path.join(ROOT, "results", "distance", "*.json")))
    if not files:
        return {"missing": "results/distance (scripts/path_distance.py)"}
    ends, designs, whole = {}, {}, {}
    for f in files:
        nm = os.path.basename(f)[:-5]
        d = json.load(open(f))
        parts = nm.split("__")
        if parts[0] == "whole" and d.get("method") == "ilp":
            whole[f"{parts[2]}|{parts[3]}"] = dict(best=d.get("ilp_best"), lower_bound=d.get("ilp_lower_bound"),
                                                  optimal=d.get("ilp_status") == 0, seconds=d.get("seconds"),
                                                  mechanisms=d.get("mechanisms"))
            continue
        if d.get("method") not in ("restricted", "restricted_l1"):
            continue
        rec = dict(weight=d.get("weight"), optimal=d.get("ilp_status") == 0,
                   lower_bound=d.get("ilp_lower_bound"), combo="".join(map(str, d["combo"])),
                   mechanisms=d.get("restricted_mechanisms"), seconds=d.get("seconds"))
        if parts[0] == "end":
            ends.setdefault(f"{parts[1]}|{parts[2]}|{parts[3]}", []).append(rec)
        elif parts[0] == "design":
            designs.setdefault(parts[1], []).append(rec)

    def summ(recs):
        ws = [r["weight"] for r in recs if r["weight"] is not None]
        return dict(min_weight=min(ws) if ws else None, all_optimal=all(r["optimal"] for r in recs),
                    combos=sorted(r["combo"] for r in recs),
                    min_lower_bound=min((r["lower_bound"] for r in recs if r["lower_bound"] is not None), default=None),
                    restricted_mechanisms=max((r["mechanisms"] or 0) for r in recs))
    out = {"ends": {k: summ(v) for k, v in sorted(ends.items())},
           "designs": {k: summ(v) for k, v in sorted(designs.items())}, "whole": whole}
    complete = [v for v in out["ends"].values() if len(v["combos"]) == 3 and v["all_optimal"]]
    if complete:
        out["end_distance"] = min(v["min_weight"] for v in complete)
        out["end_failure_faults"] = (out["end_distance"] + 1) // 2
        out["end_cases_complete"] = len(complete)
    out["end_cases"] = len(out["ends"])
    gg = os.path.join(ROOT, "results", "gadget_gain.json")
    if os.path.exists(gg):
        out["gadget_gain"] = json.load(open(gg))
    return out


def sec_inner_window():
    """One Carbon block inside a window between full measurements
    (scripts/inner_window_distance.py): exact circuit distance of its inner logicals over
    every placement of its weight-6 checks, and the structural checks it relies on."""
    files = sorted(glob.glob(os.path.join(ROOT, "results", "inner_window", "T*_*.json")))
    out = {"windows": {}}
    for f in files:
        d = json.load(open(f))
        out["windows"][f"T{d['T']}_{d['basis']}"] = dict(T=d["T"], basis=d["basis"], min_distance=d["min_distance"],
                                                         all_optimal=d["all_optimal"], paths=len(d["paths"]))
    st = os.path.join(ROOT, "results", "inner_window", "structure.json")
    if os.path.exists(st):
        d = json.load(open(st))
        out["structure"] = {k: v for k, v in d.items() if k not in ("provenance", "versions", "path")}
    if not files:
        out["missing"] = "results/inner_window (scripts/inner_window_distance.py)"
    return out


def sec_relaxed_distance():
    """Circuit fault distance by decomposition (scripts/relaxed_distance.py, docs/methods.md):
    for each path kind, basis and outer-check order, whether every subset of active outer-check
    instances (at most four; more cost at least 11 by the counting bound) has been searched,
    and whether any undetected logical error of at most W faults exists in the relaxed problem.
    None in every logical class proves a fault distance of at least W + 1 on every path the
    kind stands for."""
    out = {}
    for d_ in sorted(glob.glob(os.path.join(ROOT, "results", "relaxed_distance", "*"))):
        files = sorted(glob.glob(os.path.join(d_, "task*.json")))
        if not files:
            continue
        recs = [json.load(open(f)) for f in files]
        r0 = recs[0]
        covered, found = set(), []
        for r in recs:
            covered |= set(r["subsets"])
            for sub, v in r["results"].items():
                for combo, m in v["min"].items():
                    if isinstance(m, int):
                        found.append(dict(subset=sub, combo=combo, weight=m))
        ng = next((r["no_gadget"] for r in recs if "no_gadget" in r), None)
        if ng:
            for combo, m in ng.items():
                if isinstance(m, int):
                    found.append(dict(subset="none", combo=combo, weight=m))
        complete = len(covered) == r0["n_subsets"] and ng is not None
        out[os.path.basename(d_)] = dict(
            basis=r0["basis"], order=r0["order"], kind=r0["kind"], instances=r0["instances"], W=r0["W"],
            n_subsets=r0["n_subsets"], covered=len(covered), complete=complete, found=found,
            no_gadget=ng, lower_bound=(r0["W"] + 1) if complete and not found else None,
            cheapest_pure_class=r0.get("cheapest_pure_class"), tasks=len(recs),
            cpu_hours=round(sum(r.get("seconds", 0) for r in recs) / 3600, 2))
    return out or {"missing": "results/relaxed_distance (scripts/relaxed_distance.py)"}


def sec_files(sub):
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "results", sub, "*.json"))):
        out[os.path.basename(f)[:-5]] = json.load(open(f))
    return out or {"missing": f"results/{sub}"}


def _load(name):
    return json.load(open(os.path.join(ROOT, "results", name)))


SECTIONS = {
    "checks": lambda S: _load("checks.json"),
    "c4_uniform": lambda S: sec_c4_uniform(),
    "circuits": lambda S: sec_circuits(),
    "carbon": lambda S: sec_carbon(),
    "value": lambda S: sec_value(),
    "interval": lambda S: sec_interval(),
    "sensitivity": lambda S: sec_sensitivity(),
    "basis": lambda S: sec_basis(),
    "decoder_benchmark": lambda S: _load("decoder_benchmark.json"),
    "decoder_audit": lambda S: sec_files("redecode"),
    "timing_consistency": lambda S: _load("timing_consistency.json"),
    "minimal": lambda S: sec_files("minimal"),
    "core_census": lambda S: __import__("core_census").census(),
    "distance": lambda S: sec_distance(),
    "inner_window": lambda S: sec_inner_window(),
    "relaxed_distance": lambda S: sec_relaxed_distance(),
    "direct": lambda S: sec_direct(S),
}


def same(a, b, rtol=1e-9):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k], rtol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y, rtol) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            return a is b
        if np.isnan(a) and np.isnan(b):
            return True
        return bool(np.isclose(a, b, rtol=rtol, atol=1e-15))
    return a == b


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    path = os.path.join(ROOT, "results", "summary.json")
    saved = json.load(open(path)) if (a.check or a.only) else {}
    names = a.only.split(",") if a.only else list(SECTIONS)
    S = dict(saved) if (a.check or a.only) else {}
    bad = []
    for name in names:
        value = json.loads(json.dumps(SECTIONS[name](S), default=float))
        if a.check:
            ok = same(value, saved[name])
            print(f"{name}: {'matches' if ok else 'DIFFERS'}", flush=True)
            if not ok:
                bad.append(name)
        else:
            S[name] = value
            print(f"{name}: rebuilt", flush=True)
    if not a.check:
        S["circuits_full_round"], S["circuits_c4_round"] = sec_round_totals(S)
        with open(path, "w") as f:
            json.dump(S, f, indent=1, default=float)
    elif bad:
        sys.exit("differs from results/summary.json: " + ", ".join(bad))


if __name__ == "__main__":
    main()
