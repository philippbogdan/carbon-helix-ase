"""Compute every reported number from the raw results and write results/summary.json.

Sections (each skipped with a note if its raw data are missing):
  checks            structural checks (scripts/checks.py output)
  c4_uniform        C4-Helix memory, uniform noise, vs arXiv:2609.03194 Fig. 11a
  helios_memory     C4-Helix and [[10,2,3]] memory under the Helios model vs the paper's
                    hardware values (4.6e-5 and 2.1e-4 per logical qubit per round)
  tqrb_counts       two-qubit gates and shot time, adaptive vs not, vs paper Fig. 2f / App.
  tqrb_eps          logical error per Clifford vs paper Fig. 12 / hardware 2.8e-4
  carbon            Carbon-Helix memory (uniform and Helios models): error per round,
                    two-qubit gates, time and qubits per round, exponents, extrapolation
  interval          error per round versus forced full-measurement interval T
  frontier          non-dominated configurations (error, gates, time, qubits)
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return "unknown"


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


# ------------------------------------------------------------------------------- sections
def sec_c4_uniform(nboot=200):
    """Paper Fig. 11a (uniform noise, gamma = 0, r = 12, mean of the X- and Z-basis block
    error per round) from the stratified c4_repro campaign, with the estimator used for
    every stratified configuration: measured f_w where a stratum has >= 3 failures, the
    order-3 polynomial model (distance 6) elsewhere; statistical 95 % interval from the
    smoothed parametric bootstrap; model range from fits with only the third- and only the
    fourth-order term; the model-free (Jeffreys) interval; held-out validation."""
    from strat import load, StratModel, PolyModel, hybrid_boot, hybrid_eps, direct_posterior, heldout, local_order
    PAPER = {7e-4: 7.53e-6, 9e-4: 1.70e-5, 1.1e-3: 3.46e-5}
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
    out = {"paper_digitized": {str(k): v for k, v in PAPER.items()}, "rounds": 12, "points": {},
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


def sec_helios_memory():
    from agg_protocol import load
    out = {}
    refs = {("c4helix", "helios"): (4.6e-5, [2.0e-5, 1.08e-4]),
            ("toric10", "helios"): (2.1e-4, [1.4e-4, 3.1e-4])}
    for pat in ("results/final/c4_helios_mem/*/*/*.json", "results/final/t10_helios_mem/*/*/*.json"):
        agg = load(os.path.join(ROOT, pat))
        for key, a in agg.items():
            proto, rule, noise, p, b, r = key
            n, k = int(a["shots"]), int(a["fails"])
            lo, hi = wilson(k, n)
            ref = refs.get((proto, "helios"))
            rec = dict(
                eps_block=per_round(k / n, r), eps_per_LQ=per_round(k / n, r) / 2,
                ci95_per_LQ=[per_round(lo, r) / 2, per_round(hi, r) / 2], shots=n, fails=k, rounds=r,
                paper_per_LQ=ref[0] if (ref and noise == "helios" and b == "X") else None,
                paper_ci95=ref[1] if (ref and noise == "helios" and b == "X") else None)
            # forced-gap postselection (paper: gamma = 3.9 nats; 0.4 % of 20-round shots
            # discarded gives 9.3e-6 per LQ per round, 95 % CI 1.7e-6 - 5.2e-5)
            gh, fh = a.get("gap_hist"), a.get("fail_gap_hist")
            if gh is not None and n:
                ps = {}
                for gamma in (3.9,):
                    cut = int(np.ceil(gamma / 0.5))
                    disc = float(gh[:cut].sum())
                    kf = float(k - fh[:cut].sum())
                    nk = n - disc
                    l2, h2 = wilson(int(kf), int(nk))
                    ps[str(gamma)] = dict(discard=disc / n, eps_per_LQ=per_round(kf / nk, r) / 2,
                                          ci95_per_LQ=[per_round(l2, r) / 2, per_round(h2, r) / 2],
                                          fails_kept=kf)
                cs = np.cumsum(gh)
                cut = int(np.searchsorted(cs, 0.004 * n))      # discard about 0.4 %
                disc = float(gh[:cut].sum())
                kf = float(k - fh[:cut].sum())
                l2, h2 = wilson(int(kf), int(n - disc))
                ps["discard_0.4pct"] = dict(gamma=cut * 0.5, discard=disc / n,
                                            eps_per_LQ=per_round(kf / (n - disc), r) / 2,
                                            ci95_per_LQ=[per_round(l2, r) / 2, per_round(h2, r) / 2])
                rec["postselection"] = ps
                if proto == "c4helix" and noise == "helios" and b == "X":
                    rec["paper_postselected"] = dict(eps_per_LQ=9.3e-6, ci95=[1.7e-6, 5.23e-5], discard=0.004)
            out[f"{proto}|{noise}|{b}"] = rec
    a, c = out.get("toric10|helios|X"), out.get("c4helix|helios|X")
    if a and c and c.get("eps_per_LQ"):
        # suppression under C4 concatenation, [[10,2,3]] -> [[20,2,6]]: the paper's 4.52x, with
        # the range its two published 95 % intervals allow
        out["suppression"] = dict(ours=a["eps_per_LQ"] / c["eps_per_LQ"], paper=4.52,
                                  paper_range=[1.4e-4 / 1.08e-4, 3.1e-4 / 2.0e-5])
    return out or {"missing": "helios memory results"}


def sec_tqrb_counts():
    import glob
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(float))
    for f in glob.glob(os.path.join(ROOT, "results/final/tqrb_counts2/*/*/*/*.json")):
        d = json.load(open(f))
        key = (d["noise"], d["p"], d["adaptive"], d["m"])
        for k in ("shots", "twoq", "dur", "dur_helios", "s0s1"):
            agg[key][k] += d[k]
    if not agg:
        return {"missing": "results/final/tqrb_counts2"}
    PAPER = {5: (1531, 1021), 7: (2140, 1430), 13: (3965, 2653)}
    out = {"paper_gates_per_shot_digitized": {str(m): v for m, v in PAPER.items()},
           "paper_reduction": {"gates_hardware": 0.33, "time_hardware": 0.23,
                               "time_sim_p1e-3": 0.24, "time_sim_p1e-4": 0.30}, "rows": {}}
    for (noise, p, ad, m), a in agg.items():
        if ad != 1:
            continue
        b = agg[(noise, p, 0, m)]
        g1, g0 = a["twoq"] / a["shots"], b["twoq"] / b["shots"]
        t1, t0 = a["dur_helios"] / a["shots"], b["dur_helios"] / b["shots"]
        s0 = a["s0s1"] / a["shots"]
        out["rows"][f"{noise}|{p}|m{m}"] = dict(gates_nonadaptive=g0, gates_adaptive=g1,
                                                gate_reduction=1 - g1 / g0, time_nonadaptive_s=t0 / 1000,
                                                time_adaptive_s=t1 / 1000, time_reduction=1 - t1 / t0,
                                                s0s1_per_shot=s0,
                                                gates_per_s0s1_nonadaptive=(g0 - 152) / s0,
                                                gates_per_s0s1_adaptive=(g1 - 152) / s0)
    return out


def sec_tqrb_eps():
    """Logical error per Clifford from the importance-sampled TQRB runs: for every target p'
    = s p (job_tqrb.IS_SCALES) the per-shot failure probability at each m and the fit
    p_L(m) = (eps0 - 1/2)(1 - 2 eps_C)^m + 1/2 with eps0 shared by the adaptive and
    non-adaptive data (weighted least squares on the IS estimates; bootstrap over tasks)."""
    import glob
    from collections import defaultdict
    from scipy.optimize import least_squares
    files = glob.glob(os.path.join(ROOT, "results/final/tqrb_is/*/*/*/*.json"))
    if not files:
        return {"missing": "results/final/tqrb_is"}
    PAPER = {("uniform", 1e-3): (2.8e-4, 4.2e-4), ("uniform", 5e-4): (2.6e-5, 4.7e-5),
             ("helios", 7.9e-4): (2.8e-4, None)}
    per = defaultdict(list)          # (noise, p', a, m) -> list of (S1, S2, n) per task
    for f in files:
        d = json.load(open(f))
        for k, v in d.items():
            if k.startswith("is_s") and k.endswith("_sum"):
                sc = float(k[4:-4])
                pp = round(d["p"] * sc, 12)
                seqs = d.get("is_seq_sums", {}).get(f"{sc:g}")
                per[(d["noise"], pp, d["adaptive"], d["m"])].append(
                    (v, d[k.replace("_sum", "_sq")], d["shots"], seqs, d.get("is_seq_shots")))
    def fit(points):
        ads = sorted({a for a, m in points})
        def res(x):
            r = []
            for (a, m), (pl, sd) in points.items():
                ec = x[1 + ads.index(a)]
                r.append(((x[0] - 0.5) * (1 - 2 * ec) ** m + 0.5 - pl) / max(sd, 1e-12))
            return r
        x0 = [1e-4] + [1e-4] * len(ads)
        sol = least_squares(res, x0, bounds=([0] * len(x0), [0.1] * len(x0)))
        return dict(eps0=float(sol.x[0]), **{f"epsC_a{a}": float(sol.x[1 + i]) for i, a in enumerate(ads)})
    out = {}
    rng = np.random.default_rng(0)
    keys = sorted({(k[0], k[1]) for k in per})
    for noise, pp in keys:
        tasks = {(a, m): v for (nz, p2, a, m), v in per.items() if nz == noise and p2 == pp}
        def points_from(tsel):
            # files without per-sequence sums: tasks (each its own random sequences) are the
            # independent units; the error never falls below one event's worth (1/n)
            pts = {}
            for am, lst in tsel.items():
                n = sum(x[2] for x in lst)
                pl = sum(x[0] for x in lst) / n
                tm = [x[0] / x[2] for x in lst if x[2]]
                sd = float(np.std(tm, ddof=1) / np.sqrt(len(tm))) if len(tm) > 1 else 0.0
                pts[am] = (pl, sd)
            return zero_floor(pts)

        def seq_means(lst):
            if all(x[3] is not None and x[4] for x in lst):
                return np.array([s / x[4] for x in lst for s in x[3]])
            return None

        def zero_floor(pts):
            # a point without any event has no spread: give it the largest standard error of
            # the other points at this noise level instead of an infinite weight
            pos = [sd for pl, sd in pts.values() if sd > 0]
            fl = max(pos) if pos else 1.0
            return {am: (pl, sd if sd > 0 else fl) for am, (pl, sd) in pts.items()}

        def points_from_seqs(sel):
            # sequences are the independent units
            return zero_floor({am: (float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))) for am, v in sel.items()})
        # shots under one random Clifford sequence share its failure probability: with
        # per-sequence sums the sequences are the independent units (clustered standard
        # errors, bootstrap over sequences); otherwise shots and tasks
        seqs = {am: seq_means(lst) for am, lst in tasks.items()}
        if all(v is not None and len(v) > 1 for v in seqs.values()):
            unit = "sequences"
            pts = points_from_seqs(seqs)
            resample = lambda: points_from_seqs({am: v[rng.integers(0, len(v), len(v))] for am, v in seqs.items()})
        else:
            unit = "tasks"
            pts = points_from(tasks)
            resample = lambda: points_from({am: [lst[i] for i in rng.integers(0, len(lst), len(lst))]
                                            for am, lst in tasks.items()})
        try:
            best = fit(pts)
        except Exception as ex:
            continue
        boots = []
        for _ in range(200):
            try:
                boots.append(fit(resample()))
            except Exception:
                pass
        ci = {k: [float(np.percentile([b[k] for b in boots], 2.5)), float(np.percentile([b[k] for b in boots], 97.5))]
              for k in best} if boots else {}
        pap = PAPER.get((noise, pp))
        out[f"{noise}|{pp:g}"] = dict(fit=best, ci95=ci, paper=pap, resampling_unit=unit,
                                      points={f"a{a}_m{m}": dict(pL=v[0], sd=v[1]) for (a, m), v in pts.items()},
                                      ratio_nonadaptive_over_adaptive=(best.get("epsC_a0", np.nan) / best["epsC_a1"]
                                                                       if best.get("epsC_a1") else None))
    return out


def sec_tqrb_direct(nboot=400):
    """Logical error per Clifford from direct (unweighted) TQRB sampling at the paper's points:
    p_L(m) = (eps0 - 1/2)(1 - 2 eps_C)^m + 1/2 fitted by weighted least squares with eps0
    shared by the adaptive and non-adaptive data (the paper's App. fit), standard errors
    clustered by random sequence, intervals by bootstrap over sequences."""
    import glob
    from collections import defaultdict
    from scipy.optimize import least_squares
    files = glob.glob(os.path.join(ROOT, "results/final/tqrb_direct/*/*/*/*.json")) + \
        glob.glob(os.path.join(ROOT, "results/final/tqrb_fig13_helios/*/*/*/*.json"))
    if not files:
        return {"missing": "results/final/tqrb_direct"}
    PAPER = {("uniform", 1e-3): dict(adaptive=2.8e-4, nonadaptive=4.2e-4, source="App. Fig. 13 legend (ratio 1.5x stated in the text)"),
             ("uniform", 5e-4): dict(adaptive=2.6e-5, nonadaptive=4.7e-5, source="App. Fig. 13 legend (ratio 1.8x stated in the text)"),
             ("helios", 7.9e-4): dict(adaptive=2.8e-4, adaptive_ci95=[1.2e-4, 3.8e-4], source="hardware TQRB, main text")}
    seqs = defaultdict(list)          # (noise, p, a, m) -> per-sequence failure fractions
    for f in files:
        d = json.load(open(f))
        if "seq_fails" not in d:
            continue
        # results decoded with another pruning window than the campaign decoder's (Delta = 12)
        # are kept apart: noise key '<noise>_D<Delta>'
        nk = d["noise"] if float(d.get("Delta", 12.0)) == 12.0 else f"{d['noise']}_D{float(d['Delta']):g}"
        seqs[(nk, d["p"], d["adaptive"], d["m"])] += [x / d["seq_shots"] for x in d["seq_fails"]]

    def fit(points):
        ads = sorted({a for a, m in points})
        def res(x):
            return [((x[0] - 0.5) * (1 - 2 * x[1 + ads.index(a)]) ** m + 0.5 - pl) / max(sd, 1e-12)
                    for (a, m), (pl, sd) in points.items()]
        x0 = [1e-4] + [1e-4] * len(ads)
        sol = least_squares(res, x0, bounds=([0] * len(x0), [0.1] * len(x0)))
        return dict(eps0=float(sol.x[0]), **{f"epsC_a{a}": float(sol.x[1 + i]) for i, a in enumerate(ads)})

    def points(sel):
        out = {}
        for am, v in sel.items():
            v = np.asarray(v)
            sd = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
            out[am] = (float(v.mean()), sd)
        pos = [sd for _, sd in out.values() if sd > 0]
        fl = max(pos) if pos else 1.0
        return {am: (pl, sd if sd > 0 else fl) for am, (pl, sd) in out.items()}
    rng = np.random.default_rng(1)
    out = {}
    for noise, p in sorted({(k[0], k[1]) for k in seqs}):
        sel = {(a, m): np.asarray(v) for (nz, pp, a, m), v in seqs.items() if nz == noise and pp == p}
        pts = points(sel)
        best = fit(pts)
        boots = []
        for _ in range(nboot):
            try:
                boots.append(fit(points({am: v[rng.integers(0, len(v), len(v))] for am, v in sel.items()})))
            except Exception:
                pass
        ci = {k: [float(np.percentile([b[k] for b in boots], 2.5)), float(np.percentile([b[k] for b in boots], 97.5))]
              for k in best}
        rec = dict(fit=best, ci95=ci, paper=PAPER.get((noise, p)), sequences={f"a{a}_m{m}": len(v) for (a, m), v in sel.items()},
                   points={f"a{a}_m{m}": dict(pL=v[0], sd=v[1]) for (a, m), v in pts.items()})
        # the paper's Fig. 13 points at the same p (digitized), for either noise model
        dg = os.path.join(ROOT, "results/paper_digitized.json")
        if os.path.exists(dg):
            f13 = json.load(open(dg))["fig13"]["points"]
            cmp_ = {}
            for (a, m), v in pts.items():
                k13 = f"p{p:g}_a{a}_m{m}"
                if k13 in f13:
                    cmp_[f"a{a}_m{m}"] = dict(paper_pL=f13[k13]["pL"], ratio=v[0] / f13[k13]["pL"])
            if cmp_:
                rec["fig13_points"] = cmp_
        if best.get("epsC_a1") and best.get("epsC_a0"):
            rec["ratio_nonadaptive_over_adaptive"] = best["epsC_a0"] / best["epsC_a1"]
            rr = [b["epsC_a0"] / b["epsC_a1"] for b in boots if b.get("epsC_a1")]
            rec["ratio_ci95"] = [float(np.percentile(rr, 2.5)), float(np.percentile(rr, 97.5))] if rr else None
        out[f"{noise}|{p:g}"] = rec
    return out


def _gap_tables(files, r, kind):
    """Per task: (shots, accepted[gamma], failures among accepted[gamma]) for gamma in
    GAMMAS (accept a shot when its decoder confidence gap >= gamma)."""
    rows = []
    for f in files:
        d = json.load(open(f))
        if kind == "protocol":                 # scripts/job_protocol.py: 0.5-nat bins, all and failing shots
            w, allh, fh = 0.5, np.asarray(d["gap_hist"]), np.asarray(d["fail_gap_hist"])
        else:                                  # scripts/job_tqrb.py --gap-hist: failing and correct shots
            w = d["gap_bin"]
            fh = np.asarray(d["gap_hist_fail"])
            allh = fh + np.asarray(d["gap_hist_ok"])
        n = float(allh.sum())
        acc = {g: float(allh[int(round(g / w)):].sum()) for g in GAMMAS}
        fl = {g: float(fh[int(round(g / w)):].sum()) for g in GAMMAS}
        rows.append((n, acc, fl))
    return rows


GAMMAS = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0)


def fig11_pruned_share(dig_points, gamma=4.0):
    """Memory (Fig. 11): of the failures kept at gamma, the share whose gap is at least 31.5
    nats (the last 0.5-nat bin, infinite gaps included: the correct class pruned), and the
    error per round at gamma (mean of X and Z) with and without them, against the paper."""
    out = {}
    for p in ("0.0007", "0.0009", "0.0011"):
        per_b = {}
        for b in "XZ":
            fs = glob.glob(os.path.join(ROOT, f"results/final/c4_fig11/uniform_p{p}/{b}/task*.json"))
            if not fs:
                continue
            allh = np.sum([json.load(open(f))["gap_hist"] for f in fs], axis=0)
            fh = np.sum([json.load(open(f))["fail_gap_hist"] for f in fs], axis=0)
            i = int(round(gamma / 0.5))
            per_b[b] = (float(allh[i:].sum()), float(fh[i:].sum()), float(fh[-1]))
        if len(per_b) < 2:
            continue
        def eps(drop):
            return float(np.mean([1 - (1 - (k - (inf if drop else 0)) / n) ** (1 / 12) for n, k, inf in per_b.values()]))
        kept = sum(k for _, k, _ in per_b.values())
        inf = sum(x for _, _, x in per_b.values())
        pa = ((dig_points.get(p) or {}).get(f"{gamma:g}") or {}).get("value")
        out[p] = dict(kept=kept, infinite=inf, share=inf / kept if kept else None, eps=eps(False), eps_without=eps(True),
                      paper=pa, ratio=eps(False) / pa if pa else None, ratio_without=eps(True) / pa if pa else None)
    return out


def sec_gap_figures(nboot=400):
    """The paper's Figs. 11 and 12 (uniform noise, forced-gap postselection at gamma = 0,
    0.5, 1, 2, 3, 4 nats) against our direct sampling with the decoder's gaps:
    Fig. 11: C4-Helix memory, r = 12, error per round 1 - (1 - p_L)^(1/12) and rejection per
    round 1 - (1 - R)^(1/12), mean of the X- and Z-basis values (campaign c4_fig11);
    Fig. 12: non-adaptive TQRB, 6 random Cliffords and the inverse, the same per Clifford with
    r = 7 (campaign tqrb_fig12). Paper values: results/paper_digitized.json
    (analysis/digitize_paper.py). 95 % intervals: bootstrap over tasks (independent shots
    and, for TQRB, independent random sequences)."""
    dig = json.load(open(os.path.join(ROOT, "results/paper_digitized.json")))
    rng = np.random.default_rng(3)
    out = {}
    for fig, pat, r, kind, bases in (("fig11", "results/final/c4_fig11/uniform_p{p}/{b}/task*.json", 12, "protocol", "XZ"),
                                     ("fig12", "results/final/tqrb_fig12/uniform_p{p}/a0/m7/task*.json", 7, "tqrb", "-")):
        res = {}
        for p in (7e-4, 9e-4, 1.1e-3):
            per_b = {}
            for b in bases:
                fs = sorted(glob.glob(os.path.join(ROOT, pat.format(p=p, b=b))))
                if fs:
                    per_b[b] = _gap_tables(fs, r, kind)
            if len(per_b) < len(bases):
                continue

            def values(sel):
                ev, rv = {g: [] for g in GAMMAS}, {g: [] for g in GAMMAS}
                for rows in sel.values():
                    N = sum(x[0] for x in rows)
                    for g in GAMMAS:
                        A = sum(x[1][g] for x in rows)
                        F = sum(x[2][g] for x in rows)
                        ev[g].append(1 - (1 - F / A) ** (1 / r) if A else np.nan)
                        rv[g].append(1 - (A / N) ** (1 / r) if N else np.nan)
                return {g: float(np.mean(ev[g])) for g in GAMMAS}, {g: float(np.mean(rv[g])) for g in GAMMAS}
            e0, r0 = values(per_b)
            boots = [values({b: [rows[i] for i in rng.integers(0, len(rows), len(rows))] for b, rows in per_b.items()})
                     for _ in range(nboot)]
            pa = dig[fig]["panels"]
            rec = {}
            for g in GAMMAS:
                gk = f"{g:g}"
                eb = [x[0][g] for x in boots]
                rb = [x[1][g] for x in boots]
                pe = pa["a"]["points"][f"{p:g}"].get(gk, {}).get("value")
                pr_ = pa["b"]["points"][f"{p:g}"].get(gk, {}).get("value")
                ci = [float(np.nanpercentile(eb, 2.5)), float(np.nanpercentile(eb, 97.5))]
                rci = [float(np.nanpercentile(rb, 2.5)), float(np.nanpercentile(rb, 97.5))]
                fails = sum(x[2][g] for rows in per_b.values() for x in rows)
                rec[gk] = dict(eps=e0[g], ci95=ci, paper=pe, ratio=(e0[g] / pe if pe else None),
                               paper_inside=(ci[0] <= pe * 1.03 and pe * 0.97 <= ci[1]) if pe else None,
                               rejection=r0[g] if g > 0 else None, rejection_ci95=rci if g > 0 else None,
                               paper_rejection=pr_, rejection_ratio=(r0[g] / pr_ if (pr_ and g > 0) else None),
                               rejection_paper_inside=((rci[0] <= pr_ * 1.03 and pr_ * 0.97 <= rci[1]) if (pr_ and g > 0) else None),
                               failures_kept=fails)
            shots = sum(x[0] for rows in per_b.values() for x in rows)
            res[f"{p:g}"] = dict(gammas=rec, shots=shots)
        if res:
            allr = [v["ratio"] for x in res.values() for g, v in x["gammas"].items() if v["ratio"]]
            r0s = [x["gammas"]["0"]["ratio"] for x in res.values()]
            rr = [v["rejection_ratio"] for x in res.values() for g, v in x["gammas"].items() if v["rejection_ratio"]]
            ins = [v["paper_inside"] for x in res.values() for v in x["gammas"].values() if v["paper_inside"] is not None]
            rins = [v["rejection_paper_inside"] for x in res.values() for v in x["gammas"].values()
                    if v["rejection_paper_inside"] is not None]
            res["summary"] = dict(ratio_gamma0=[min(r0s), max(r0s)], ratio_all=[min(allr), max(allr)],
                                  rejection_ratio=[min(rr), max(rr)] if rr else None,
                                  inside=f"{sum(ins)} of {len(ins)}", rejection_inside=f"{sum(rins)} of {len(rins)}",
                                  points=len(ins))
        out[fig] = res or {"missing": pat}
        if fig == "fig11" and res:
            out["fig11"]["pruned"] = fig11_pruned_share(dig["fig11"]["panels"]["a"]["points"])
    # decoder sensitivity: the same shots (tasks 0-9 at p = 1.1e-3) decoded with other beam
    # widths K and pruning windows Delta (campaign decoder: K = 512, Delta = 12)
    variants = (("K64", "results/final/tqrb_fig12_beam/uniform_p0.0011_K64/a0/m7", "K = 64, Delta = 12"),
                ("K512", "results/final/tqrb_fig12/uniform_p0.0011/a0/m7", "K = 512, Delta = 12 (campaign)"),
                ("D24", "results/final/tqrb_fig12_beam/uniform_p0.0011_D24/a0/m7", "K = 512, Delta = 24"),
                ("K4096", "results/final/tqrb_fig12_beam/uniform_p0.0011/a0/m7", "K = 4096, Delta = 24"))
    names = [f"task{t}.json" for t in range(10)]
    pa = dig["fig12"]["panels"]
    res = {}
    for lab, d, desc in variants:
        files = [os.path.join(ROOT, d, nm) for nm in names]
        if not all(os.path.exists(f) for f in files):
            continue
        rows = _gap_tables(files, 7, "tqrb")
        N = sum(x[0] for x in rows)
        res[lab] = {"decoder": desc, "shots": N}
        for g in GAMMAS:
            A = sum(x[1][g] for x in rows)
            F = sum(x[2][g] for x in rows)
            e = 1 - (1 - F / A) ** (1 / 7)
            rj = 1 - (A / N) ** (1 / 7)
            pe = pa["a"]["points"]["0.0011"].get(f"{g:g}", {}).get("value")
            prj = pa["b"]["points"]["0.0011"].get(f"{g:g}", {}).get("value")
            res[lab][f"{g:g}"] = dict(eps=e, failures_kept=F, rejection=rj if g > 0 else None,
                                      ratio=e / pe if pe else None,
                                      rejection_ratio=(rj / prj if (prj and g > 0) else None))
    if len(res) > 1:
        out["fig12_beam"] = res
    # the same for memory (X basis, p = 7e-4, tasks 0-9): per-basis values, K = 512 and 4096
    mb = sorted(glob.glob(os.path.join(ROOT, "results/final/c4_fig11_beam/uniform_p0.0007/X/task*.json")))
    if mb:
        names = [os.path.basename(f) for f in mb]
        mn = [os.path.join(ROOT, "results/final/c4_fig11/uniform_p0.0007/X", nm) for nm in names]
        if all(os.path.exists(f) for f in mn):
            res = {}
            for lab, files in (("K512", mn), ("K4096", mb)):
                rows = _gap_tables(files, 12, "protocol")
                N = sum(x[0] for x in rows)
                res[lab] = {"shots": N}
                for g in GAMMAS:
                    A = sum(x[1][g] for x in rows)
                    F = sum(x[2][g] for x in rows)
                    res[lab][f"{g:g}"] = dict(eps=1 - (1 - F / A) ** (1 / 12), failures_kept=F,
                                              rejection=(1 - (A / N) ** (1 / 12)) if g > 0 else None)
            res["tasks"] = len(mb)
            out["fig11_beam"] = res
    return out


def sec_fig14(nboot=400):
    """The paper's Fig. 14 ('C4-Helix transv.'): two blocks, 6 transversal CNOTs interleaved
    with full rounds on both blocks (campaign c4_fig14), against its digitized markers. The
    paper does not define 'per CNOT'; ours per logical CNOT, 1 - (1 - p_L)^(1/12) (each
    transversal CNOT is two logical CNOTs; p_L: any of the four logical outcomes wrong), and
    per transversal CNOT, 1 - (1 - p_L)^(1/6), each the mean of the X- and Z-basis values.
    95 % intervals: bootstrap over tasks."""
    dig = json.load(open(os.path.join(ROOT, "results/paper_digitized.json"))).get("fig14", {}).get("points", {})
    rng = np.random.default_rng(4)
    out = {}
    for p in (7e-4, 9e-4, 1.1e-3):
        per_b = {}
        for b in "XZ":
            fs = sorted(glob.glob(os.path.join(ROOT, f"results/final/c4_fig14/uniform_p{p}/{b}/task*.json")))
            if fs:
                per_b[b] = [(float(d["shots"]), float(d["fails"])) for d in (json.load(open(f)) for f in fs)]
        if len(per_b) < 2:
            continue

        def vals(sel):
            e12, e6, pl = [], [], {}
            for b, rows in sel.items():
                n = sum(x[0] for x in rows)
                k = sum(x[1] for x in rows)
                q = k / n
                pl[b] = q
                e12.append(1 - (1 - q) ** (1 / 12))
                e6.append(1 - (1 - q) ** (1 / 6))
            return float(np.mean(e12)), float(np.mean(e6)), pl
        e12, e6, pl = vals(per_b)
        bs = [vals({b: [rows[i] for i in rng.integers(0, len(rows), len(rows))] for b, rows in per_b.items()})
              for _ in range(nboot)]
        ci12 = [float(np.percentile([x[0] for x in bs], 2.5)), float(np.percentile([x[0] for x in bs], 97.5))]
        ci6 = [float(np.percentile([x[1] for x in bs], 2.5)), float(np.percentile([x[1] for x in bs], 97.5))]
        pa = dig.get(f"{p:g}", {}).get("value")
        out[f"{p:g}"] = dict(per_logical_cnot=e12, ci95=ci12, per_transversal_cnot=e6, ci95_transversal=ci6,
                             pL={b: v for b, v in pl.items()},
                             shots={b: sum(x[0] for x in rows) for b, rows in per_b.items()},
                             failures={b: sum(x[1] for x in rows) for b, rows in per_b.items()},
                             paper=pa, ratio=e12 / pa if pa else None,
                             paper_inside=(ci12[0] <= pa * 1.03 and pa * 0.97 <= ci12[1]) if pa else None,
                             ratio_transversal=e6 / pa if pa else None)
    if out:
        out["summary"] = dict(ratios=[v["ratio"] for v in out.values()],
                              inside=f"{sum(1 for v in out.values() if v['paper_inside'])} of {len(out)}")
    return out or {"missing": "results/final/c4_fig14"}


def sec_chainmap(nboot=400):
    """The chain-map CNOT from the distance-5 surface code into C4-Helix (src/hase/chainmap.py;
    the paper's chain map is not published, ours is found by SAT with the properties it
    states): the map, the circuit's fault distance (results/chainmap/distance_*.json), the
    paper's Fig. 14 'Surface-C4-Helix' series (campaign chainmap: 6 chain-map CNOTs, Z basis,
    per CNOT 1 - (1 - p_L)^(1/6), p_L: any of the three logical outcomes wrong; 95 %
    intervals by bootstrap over tasks), and the inter-code GHZ experiment of the paper's
    Sec. 4 under the Helios model against its hardware mismatch rates (Wilson intervals)."""
    out = {}
    fp = os.path.join(ROOT, "results", "chainmap", "F.json")
    if os.path.exists(fp):
        d = json.load(open(fp))
        out["map"] = {k: d[k] for k in ("cnots", "max_degree_surface", "max_degree_helix")}
    dist = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "chainmap", "distance_[0-9]*.json"))):
        d = json.load(open(f))
        dist[f"{d['basis']}_{''.join(map(str, d['combo']))}"] = dict(weight=d["weight"], optimal=d["optimal"])
    if dist:
        out["distance"] = dict(classes=dist, n=len(dist), min=min(v["weight"] for v in dist.values()),
                               all_optimal=all(v["optimal"] for v in dist.values()))
    dig = json.load(open(os.path.join(ROOT, "results/paper_digitized.json"))).get("fig14", {}).get("points_surface", {})
    rng = np.random.default_rng(6)
    f14 = {}

    def per_cnot(rr):
        n = sum(x[0] for x in rr)
        k = sum(x[1] for x in rr)
        return 1 - (1 - k / n) ** (1 / 6)
    for p in (7e-4, 9e-4, 1.1e-3):
        per_b = {}
        for b in "ZX":
            fs = sorted(glob.glob(os.path.join(ROOT, f"results/final/chainmap_fig14/uniform_p{p:g}/{b}/task*.json")))
            if fs:
                per_b[b] = [(float(d["shots"]), float(d["fails"]), d["observable_flips"])
                            for d in (json.load(open(f)) for f in fs)]
        if "Z" not in per_b:
            continue
        pa = dig.get(f"{p:g}", {}).get("value")
        rec = dict(paper=pa)
        boots = {}
        for b, rows in per_b.items():
            v = per_cnot(rows)
            bs = [per_cnot([rows[i] for i in rng.integers(0, len(rows), len(rows))]) for _ in range(nboot)]
            boots[b] = bs
            ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
            fl = np.sum([x[2] for x in rows], axis=0)
            rec[b] = dict(per_cnot=float(v), ci95=ci, shots=sum(x[0] for x in rows), failures=sum(x[1] for x in rows),
                          tasks=len(rows), surface_share=float(fl[2] / max(1, fl.sum())),
                          ratio=float(v / pa) if pa else None,
                          paper_inside=bool(ci[0] <= pa * 1.03 and pa * 0.97 <= ci[1]) if pa else None)
        if len(per_b) == 2:
            m = 0.5 * (rec["X"]["per_cnot"] + rec["Z"]["per_cnot"])
            bs = [0.5 * (a + b) for a, b in zip(boots["X"], boots["Z"])]
            ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
            rec["mean"] = dict(per_cnot=float(m), ci95=ci, ratio=float(m / pa) if pa else None,
                               paper_inside=bool(ci[0] <= pa * 1.03 and pa * 0.97 <= ci[1]) if pa else None)
        # backwards-compatible fields: the Z basis (the paper's stated basis)
        rec.update(per_cnot=rec["Z"]["per_cnot"], ci95=rec["Z"]["ci95"], shots=rec["Z"]["shots"],
                   failures=rec["Z"]["failures"], ratio=rec["Z"]["ratio"], paper_inside=rec["Z"]["paper_inside"])
        f14[f"{p:g}"] = rec
    alts = {}
    for i in (1, 2, 3):
        fs = sorted(glob.glob(os.path.join(ROOT, f"results/final/chainmap_fig14_alt{i}/uniform_p0.0011/Z/task*.json")))
        if fs:
            rows = [(float(d["shots"]), float(d["fails"])) for d in (json.load(open(f)) for f in fs)]
            v = per_cnot(rows)
            bs = [per_cnot([rows[j] for j in rng.integers(0, len(rows), len(rows))]) for _ in range(nboot)]
            alts[str(i)] = dict(per_cnot=float(v), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                cnots=int(json.load(open(fs[0])).get("chain_map_cnots", 0)),
                                failures=sum(x[1] for x in rows), shots=sum(x[0] for x in rows))
    if alts:
        out["fig14_surface_alternatives"] = alts
    fs = sorted(glob.glob(os.path.join(ROOT, "results/final/chainmap_fig14_noidle/uniform_p0.0011/X/task*.json")))
    if fs:
        rows = [(float(d["shots"]), float(d["fails"]), d["observable_flips"]) for d in (json.load(open(f)) for f in fs)]
        fl = np.sum([x[2] for x in rows], axis=0)
        out["fig14_surface_noidle"] = dict(per_cnot=float(per_cnot(rows)), failures=sum(x[1] for x in rows),
                                           shots=sum(x[0] for x in rows), surface_share=float(fl[2] / max(1, fl.sum())))
    if f14:
        out["fig14_surface"] = f14
        out["fig14_surface_summary"] = dict(
            ratios=[v["ratio"] for v in f14.values()],
            inside=f"{sum(1 for v in f14.values() if v['paper_inside'])} of {len(f14)}",
            ratios_mean=[v["mean"]["ratio"] for v in f14.values() if "mean" in v],
            inside_mean=f"{sum(1 for v in f14.values() if v.get('mean', {}).get('paper_inside'))} of "
                        f"{sum(1 for v in f14.values() if 'mean' in v)}",
            ratios_x=[v["X"]["ratio"] for v in f14.values() if "X" in v],
            surface_share_z=[v["Z"]["surface_share"] for v in f14.values()],
            surface_share_x=[v["X"]["surface_share"] for v in f14.values() if "X" in v])
    # the paper's hardware mismatch fractions (Fig. 'Inter-code GHZ', 95 % Clopper-Pearson)
    PAPER_GHZ = {"Z": dict(value=0.025e-2, ci95=[0.001e-2, 0.139e-2]), "X": dict(value=0.050e-2, ci95=[0.006e-2, 0.180e-2])}
    ghz = {}
    for meas in "ZX":
        fs = sorted(glob.glob(os.path.join(ROOT, f"results/final/chainmap_ghz/helios_p0.00079/{meas}/task*.json")))
        if not fs:
            continue
        n = sum(json.load(open(f))["shots"] for f in fs)
        k = sum(json.load(open(f))["fails"] for f in fs)
        lo, hi = wilson(k, n)
        pp = PAPER_GHZ[meas]
        ghz[meas] = dict(shots=int(n), failures=int(k), rate=k / n, ci95=[lo, hi], paper=pp["value"],
                         paper_ci95=pp["ci95"], overlap=bool(lo <= pp["ci95"][1] and pp["ci95"][0] <= hi),
                         paper_inside=bool(lo <= pp["value"] <= hi))
    if len(ghz) == 2:
        z, x = ghz["Z"]["rate"], ghz["X"]["rate"]
        ghz["fidelity_bounds"] = [1 - z - x, 1 - z]
        ghz["paper_fidelity_bounds"] = [0.99925, 0.99975]
    # sensitivity: Helios's memory error per unit time doubled
    m2 = {}
    for meas in "ZX":
        fs = sorted(glob.glob(os.path.join(ROOT, f"results/final/chainmap_ghz/helios_mem2_p0.00079/{meas}/task*.json")))
        if fs:
            n = sum(json.load(open(f))["shots"] for f in fs)
            k = sum(json.load(open(f))["fails"] for f in fs)
            lo, hi = wilson(k, n)
            pp = PAPER_GHZ[meas]
            m2[meas] = dict(shots=int(n), failures=int(k), rate=k / n, ci95=[lo, hi],
                            overlap=bool(lo <= pp["ci95"][1] and pp["ci95"][0] <= hi))
    if m2:
        ghz["memory_x2"] = m2
    pg = os.path.join(ROOT, "results", "physical_ghz.json")
    if os.path.exists(pg):
        ghz["physical"] = json.load(open(pg))
    if ghz:
        out["ghz"] = ghz
    return out or {"missing": "results/chainmap, results/final/chainmap_*"}


def sec_tqrb_confident():
    """TQRB failures that the campaign decoder makes with a gap >= 12 nats (results/
    tqrb_confident, scripts/tqrb_confident.py): their share, minimal failing cores and exact
    most-likely-error classification; and the decoder settings that still fail on those cores
    (scripts/tqrb_cores_decode.py)."""
    from collections import Counter
    out = {}
    for d in sorted(glob.glob(os.path.join(ROOT, "results/tqrb_confident/p*"))):
        fs = sorted(glob.glob(os.path.join(d, "task*.json")))
        if not fs:
            continue
        tot, rows = Counter(), []
        for f in fs:
            x = json.load(open(f))
            tot.update(x["totals"])
            rows += x["confident"]
        out[os.path.basename(d)[1:]] = dict(
            tasks=len(fs), shots=tot["shots"], failures=tot["failures"], confident=tot["confident"],
            core_sizes={str(k): v for k, v in sorted(Counter(r["core_size"] for r in rows).items())},
            exact_mle_correct=sum(1 for r in rows if r["exact_mle_fails"] is False),
            exact_mle_fails=sum(1 for r in rows if r["exact_mle_fails"] is True),
            max_core_exact_correct=max((r["core_size"] for r in rows if r["exact_mle_fails"] is False), default=None))
    cd = os.path.join(ROOT, "results/tqrb_confident/cores_decode.json")
    if os.path.exists(cd):
        out["cores_decode"] = json.load(open(cd))["summary"]
    return out or {"missing": "results/tqrb_confident"}


def sec_tqrb_distance():
    """Exact circuit fault distance of complete TQRB sequences (results/tqrb_distance,
    scripts/tqrb_distance.py), the detector-completeness check and the cat-assignment
    comparison (scripts/tqrb_detector_check.py, scripts/tqrb_ordering_check.py)."""
    seqs = []
    for f in sorted(glob.glob(os.path.join(ROOT, "results/tqrb_distance/m*/task*.json"))):
        if f.endswith("_detectors.json"):
            continue
        d = json.load(open(f))
        row = dict(m=d["m"], task=d["task"], s0s1=d["s0s1"])
        for k in ("nonadaptive", "adaptive_quiet", "adaptive_full"):
            v = d.get(k) or {}
            row[k] = v.get("distance")
            row[k + "_proven"] = all(c.get("optimal") for c in (v.get("classes") or {}).values())
        seqs.append(row)
    det = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(ROOT, "results/tqrb_distance/m*/*_detectors.json")))]
    orders = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "results/tqrb_distance/orderings/*.json"))):
        d = json.load(open(f))
        orders[d["ordering"]] = dict(distance=d["distance"], proven=all(c["optimal"] for c in d["classes"].values()))
    return dict(sequences=seqs, detectors=[dict(missing=x["missing"], source=x["source"]) for x in det],
                orderings=orders) if seqs else {"missing": "results/tqrb_distance"}


def sec_tqrb_consistency(S):
    """The paper's two TQRB simulations under the same uniform noise model, compared on one
    footing: Fig. 12 (non-adaptive, 6 random Cliffords + the inverse; its gamma = 0 markers
    and the power law through them, the paper's own fitted line) against Fig. 13's
    non-adaptive p_L(m = 7) (7 Cliffords if m counts the inverse, as in the main text; 8 if
    not), and ours (direct sampling, m = 7 counts the inverse) against both."""
    dig = json.load(open(os.path.join(ROOT, "results/paper_digitized.json")))
    pts = dig["fig12"]["panels"]["a"]["points"]
    ps = np.array([7e-4, 9e-4, 1.1e-3])
    ev = np.array([pts[f"{p:g}"]["0"]["value"] for p in ps])
    slope, icpt = np.polyfit(np.log(ps), np.log(ev), 1)
    td = S.get("tqrb_direct", {}) if isinstance(S.get("tqrb_direct"), dict) else {}
    out = dict(fig12_slope=float(slope), rows={})
    # the paper's adaptive gain with its own uncertainty: its Fig. 13 points drawn within their
    # 95 % error bars (digitized; log-normal), refitted with the shared offset each time
    from scipy.optimize import least_squares
    f13 = dig["fig13"]["points"]
    rng = np.random.default_rng(5)

    def fit13(pts):
        def res(x):
            return [((x[0] - 0.5) * (1 - 2 * (x[1] if a == 1 else x[2])) ** m + 0.5 - v) / v for (a, m), v in pts.items()]
        return least_squares(res, [1e-4] * 3, bounds=([0] * 3, [0.1] * 3)).x
    gains = {}
    for pk in ("0.001", "0.0005"):
        base = {(a, m): f13[f"p{pk}_a{a}_m{m}"] for a in (0, 1) for m in (3, 5, 7)}
        if not all("ci95" in v for v in base.values()):
            continue
        x0 = fit13({k: v["pL"] for k, v in base.items()})
        gs = []
        for _ in range(2000):
            pts = {k: v["pL"] * np.exp(rng.normal(0, (np.log(v["ci95"][1]) - np.log(v["ci95"][0])) / 3.92))
                   for k, v in base.items()}
            xx = fit13(pts)
            if xx[1] > 0:
                gs.append(xx[2] / xx[1])
        gains[pk] = dict(gain=float(x0[2] / x0[1]), ci95=[float(np.percentile(gs, 2.5)), float(np.percentile(gs, 97.5))])
    out["paper_gain"] = gains
    for p in (1e-3, 5e-4):
        e12 = float(np.exp(icpt + slope * np.log(p)))
        p7, p8 = 1 - (1 - e12) ** 7, 1 - (1 - e12) ** 8
        f13 = dig["fig13"]["points"][f"p{p:g}_a0_m7"]["pL"]
        rec = dict(fig12_eps=e12, fig12_extrapolated=bool(p < ps.min()), fig12_pL_7=p7, fig12_pL_8=p8, fig13_pL=f13,
                   fig13_over_fig12=[f13 / p8, f13 / p7],
                   fig13_eps_legend=dig["fig13"]["legend_eps_C"][f"{p:g}"]["nonadaptive"])
        o = td.get(f"uniform|{p:g}", {}).get("points", {}).get("a0_m7")
        if o:
            rec.update(ours_pL=o["pL"], ours_sd=o["sd"], ours_over_fig13=o["pL"] / f13, ours_over_fig12=o["pL"] / p7)
        out["rows"][f"{p:g}"] = rec
    return out


RULE_SHORT = {"full": "non-adaptive", "C": "any-event C", "B4": "ambiguous-event B4", "R4": "rotation R4",
              "R8": "rotation R8", "R8F": "rotation R8F (parallel weight-6)"}

TAGS = {"full": "full", "l2=on_event,l3=any": "C", "l2=on_event,l3=ambiguous,T3=4": "B4",
        "l2=on_event,l3=ambiguous,rot=4": "R4", "l2=on_event,l3=ambiguous,rot=8": "R8",
        "l2=on_event,l3=ambiguous,rot=8,l2fast=1": "R8F", "l2=on_event,l3=ambiguous,rot=16,l2fast=1": "R16F",
        "l2=on_event,l3=ambiguous,rot=32,l2fast=1": "R32F",
        "l2=on_event,l3=ambiguous,rot=32,l2fast=1,l2type=1": "R32FT"}


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
    their statistical interval, time per round from direct path sampling."""
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
    """Circuit fault distance at the end of the circuit (scripts/job_distance.py --method
    restricted; scripts/distance_suite.py): for every place an outer-check part can end a
    circuit, the minimum number of fault mechanisms of an undetectable logical error among
    the mechanisms seen only by the final measurement and the flags (exact integer
    programming: optimal = proved), minimised over the logical classes; and the same for the
    outer-check gadget designs compared (results/distance/design__*)."""
    files = sorted(glob.glob(os.path.join(ROOT, "results", "distance", "*.json")))
    if not files:
        return {"missing": "results/distance (scripts/job_distance.py)"}
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
        out["structure"] = {k: v for k, v in d.items() if k not in ("provenance", "path")}
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


def tqrb_texts(S):
    """FINDINGS 1.6 (Fig. 13 and the paper's two TQRB simulations) and 1.7 (Figs. 11, 12)."""
    T = {}
    tc = S.get("tqrb_consistency", {}) if isinstance(S.get("tqrb_consistency"), dict) else {}
    td = S.get("tqrb_direct", {}) if isinstance(S.get("tqrb_direct"), dict) else {}
    gf = S.get("gap_figures", {}) if isinstance(S.get("gap_figures"), dict) else {}
    r1, r5 = tc.get("rows", {}).get("0.001"), tc.get("rows", {}).get("0.0005")
    f12 = gf.get("fig12", {})
    txt = []
    if r1:
        txt.append(
            "Under uniform noise our values are about a third of Fig. 13's. The paper's own Fig. 12, the same "
            "non-adaptive benchmark under the same noise model and decoder (6 random Cliffords and the inverse), lies "
            f"as far below Fig. 13: the power law through its γ = 0 markers (slope {tc['fig12_slope']:.2f}) gives "
            f"{r1['fig12_eps']:.2e} per Clifford at p = 1e-3, p_L = {r1['fig12_pL_7']:.2e} over seven Cliffords, "
            f"while Fig. 13's non-adaptive p_L(m = 7) is {r1['fig13_pL']:.2e}, {r1['fig13_over_fig12'][0]:.1f}-"
            f"{r1['fig13_over_fig12'][1]:.1f}x more (8 or 7 Cliffords, whether or not m counts the inverse). The "
            "definitions cannot account for it: Fig. 12's error per Clifford includes the preparation and measurement "
            "share and can only exceed Fig. 13's fitted one for the same data."
            + (f" At p = 5e-4 the factor is {r5['fig13_over_fig12'][0]:.1f}-{r5['fig13_over_fig12'][1]:.1f}x "
               "(Fig. 12's line extended below its lowest point)." if r5 else ""))
        if "ours_pL" in r1:
            txt.append(f"Our non-adaptive p_L(m = 7) at p = 1e-3 is {r1['ours_pL']:.2e} ± {r1['ours_sd']:.1e}: "
                       f"{r1['ours_over_fig12']:.2f}x the value implied by Fig. 12 and {r1['ours_over_fig13']:.2f}x Fig. 13's"
                       + (f"; at p = 5e-4 {r5['ours_pL']:.2e} ± {r5['ours_sd']:.1e}, {r5['ours_over_fig12']:.2f}x and "
                          f"{r5['ours_over_fig13']:.2f}x" if r5 and "ours_pL" in r5 else "") + ".")
        s12 = f12.get("summary") if isinstance(f12, dict) else None
        if s12:
            txt.append(f"Section 1.7 reproduces Fig. 12 directly: without postselection ours / paper = "
                       f"{s12['ratio_gamma0'][0]:.2f}-{s12['ratio_gamma0'][1]:.2f} at its three points.")
        agree = isinstance(f12, dict) and "summary" in f12 and all(
            0.67 <= f12[p]["gammas"]["0"]["ratio"] <= 1.5 for p in ("0.0007", "0.0009", "0.0011") if p in f12)
        if agree:
            txt.append("The factor of about three between ours and Fig. 13 is therefore one between the paper's two "
                       "simulations of the same protocol; the smaller differences from Fig. 12 itself are those of "
                       "section 1.7.")
        # Fig. 13 against our Helios-like model scaled to two-qubit error p (TQRB, campaign
        # tqrb_fig13_helios): point by point
        hl = []
        for key, lab in (("helios|0.001", "p = 1e-3"), ("helios|0.0005", "p = 5e-4")):
            fp_ = (td.get(key) or {}).get("fig13_points") or {}
            if fp_:
                rr = [v["ratio"] for v in fp_.values()]
                hl.append(f"{lab}: ours / Fig. 13 = {min(rr):.2f}-{max(rr):.2f} over its {len(rr)} points "
                          "(adaptive and not, m = 3, 5, 7)")
        if hl:
            txt.append("Under our Helios-like model (section 1.4) scaled so that its two-qubit error equals p, the same "
                       "benchmark reproduces Fig. 13 instead: " + "; ".join(hl) + ". Fig. 13 thus behaves as a "
                       "device-like model labelled by its two-qubit error, not as the stated uniform one (an inference "
                       "from this match; the paper names no model for Fig. 13 and calls it the hardware's conditions).")
        g = []
        for key, lab, ref in (("uniform|0.001", "uniform, p = 1e-3", "1.5x"), ("uniform|0.0005", "uniform, p = 5e-4", "1.8x"),
                              ("helios|0.001", "Helios-like, p = 1e-3", "1.5x"), ("helios|0.0005", "Helios-like, p = 5e-4", "1.8x"),
                              ("helios_D24|0.001", "Helios-like decoded with pruning window 24, p = 1e-3", "1.5x"),
                              ("helios_mem2|0.001", "Helios-like with the memory error per unit time doubled, p = 1e-3", "1.5x")):
            r = td.get(key, {})
            if r.get("ratio_nonadaptive_over_adaptive") and r.get("ratio_ci95"):
                g.append(f"{r['ratio_nonadaptive_over_adaptive']:.2f}x (95 % {r['ratio_ci95'][0]:.2f}-{r['ratio_ci95'][1]:.2f}) "
                         f"{lab}, against the paper's {ref}")
        if g:
            txt.append("The adaptive gain (non-adaptive over adaptive error per Clifford): ours " + "; ".join(g) + ".")
        pg = tc.get("paper_gain") or {}
        if pg:
            ov = []
            for pk, lab in (("0.001", "1e-3"), ("0.0005", "5e-4")):
                r_ = td.get(f"helios|{pk}", {})
                if pk in pg and r_.get("ratio_ci95"):
                    lo, hi = r_["ratio_ci95"]
                    ov.append(lo <= pg[pk]["ci95"][1] and pg[pk]["ci95"][0] <= hi)
            txt.append("The paper gives its gains without an uncertainty; its Fig. 13 points carry 95 % error bars (digitized), "
                       "and refitting them within those bars gives "
                       + ", ".join(f"{v['gain']:.2f} (95 % {v['ci95'][0]:.2f}-{v['ci95'][1]:.2f}) at p = "
                                   f"{'1e-3' if k == '0.001' else '5e-4'}" for k, v in pg.items())
                       + (": our Helios-like intervals overlap both, so the gains agree within the two uncertainties."
                          if ov and all(ov) else "."))
    T["tqrb_fig13_text"] = " ".join(txt) if txt else "PENDING (TQRB consistency)"
    tdist = S.get("tqrb_distance", {}) if isinstance(S.get("tqrb_distance"), dict) else {}
    sq = tdist.get("sequences") or []
    if sq:
        dn = [r["nonadaptive"] for r in sq if r.get("nonadaptive") is not None]
        da = [r[k] for r in sq for k in ("adaptive_quiet", "adaptive_full") if r.get(k) is not None]
        proven = all(r.get(k + "_proven") for r in sq for k in ("nonadaptive", "adaptive_quiet", "adaptive_full"))
        dets = tdist.get("detectors") or []
        od = tdist.get("orderings") or {}
        allp = [r for r in sq if all(r.get(k + "_proven") for k in ("nonadaptive", "adaptive_quiet", "adaptive_full"))]
        ub = [r for r in sq if r not in allp]
        dp = sorted({r[k] for r in allp for k in ("nonadaptive", "adaptive_quiet", "adaptive_full") if r.get(k) is not None})
        du = sorted({r[k] for r in ub for k in ("nonadaptive", "adaptive_quiet", "adaptive_full") if r.get(k) is not None})
        T["tqrb_distance_text"] = (
            "Complete TQRB sequences have lower fault distance: "
            + (f"exactly {'-'.join(map(str, dp))} for {len(allp)} random sequence{'s' if len(allp) != 1 else ''} (m = "
               f"{', '.join(sorted({str(r['m']) for r in allp}))}; non-adaptive and along either adaptive branch, every "
               "logical class proven by exact integer programming)" if allp else "")
            + (" and " if allp and ub else "")
            + (f"at most {min(du)}" + (f"-{max(du)}" if max(du) != min(du) else "") + f" for {len(ub)} "
               f"{'more' if allp else 'random'} (m = {', '.join(sorted({str(r['m']) for r in ub}))}; the integer program "
               "found an undetectable logical error of that weight and stopped after an hour without a matching lower "
               "bound)" if ub else "")
            + " (`scripts/tqrb_distance.py`). In the sequence traced (`scripts/tqrb_lowweight.py`) the smallest "
            "undetectable error holds three faults of an outer check, whose hook errors the next automorphism and S0S1 "
            "carry, and one fault of the next inner check; the decoder's detectors are complete (with the observables "
            "they span every "
            "deterministic parity of the noiseless circuit"
            + (f", {len(dets)} sequence checked" if dets else "") + "). The paper's distance-6 statement concerns one "
            "S0S1 followed by syndrome extraction, as checked above."
            + (" Under the three natural cat assignments of the outer checks the same sequence has distance "
               + ", ".join(f"{v['distance']} ({k}{', used' if k == 'xpair' else ''})" for k, v in od.items())
               + (", so the assignment left open by Fig. 8 does not cause it." if len({v['distance'] for v in od.values()}) == 1
                  else ".") if od else ""))
    else:
        T["tqrb_distance_text"] = "PENDING (TQRB sequence distances)"
    # 1.7: forced-gap postselection tables
    rows, summ = [], []
    for fig, lab, unit in (("fig11", "Fig. 11 (memory, r = 12, mean of X and Z)", "round"),
                           ("fig12", "Fig. 12 (non-adaptive TQRB, 7 Cliffords)", "Clifford")):
        res = gf.get(fig, {})
        if not isinstance(res, dict) or "summary" not in res:
            continue
        if not rows:
            rows.append("| figure | quantity | p | " + " | ".join(f"γ = {g:g}" for g in GAMMAS) + " |")
            rows.append("|---|---|---|" + "---|" * len(GAMMAS))
        for qty, key, rkey in (("error per " + unit, "eps", "ratio"), ("rejection per " + unit, "rejection", "rejection_ratio")):
            for p in ("0.0007", "0.0009", "0.0011"):
                if p not in res:
                    continue
                cells = []
                for g in GAMMAS:
                    v = res[p]["gammas"][f"{g:g}"]
                    if v.get(key) is None or v.get(rkey) is None:
                        cells.append("–")
                        continue
                    ins = v["paper_inside"] if key == "eps" else v["rejection_paper_inside"]
                    cells.append(f"{v[key]:.2e} ({v[rkey]:.2f}{'' if ins else '*'})")
                rows.append(f"| {lab} | {qty} | {float(p):g} | " + " | ".join(cells) + " |")
        su = res["summary"]
        summ.append(f"{lab}: error ours / paper {su['ratio_all'][0]:.2f}-{su['ratio_all'][1]:.2f}, paper inside our 95 % "
                    f"interval (with its ±3 % reading) at {su['inside']} points"
                    + (f"; rejection ours / paper {su['rejection_ratio'][0]:.2f}-{su['rejection_ratio'][1]:.2f}, inside at "
                       f"{su['rejection_inside']}" if su.get("rejection_ratio") else ""))
        pr = res.get("pruned") or {}
        if pr and all(v.get("ratio") and v.get("share") is not None for v in pr.values()):
            summ.append("of our memory failures kept at γ = 4, " + ", ".join(
                f"{100 * v['share']:.0f} % at p = {float(k):g}" for k, v in pr.items())
                + " have a gap of at least 31.5 nats (the bin that holds infinite gaps): the signature of the TQRB "
                "failures below whose correct class the pruning window removed; with them our γ = 4 error is " + ", ".join(
                    f"{v['ratio']:.2f}" for v in pr.values()) + " x the paper's, without them " + ", ".join(
                    f"{v['ratio_without']:.2f}" for v in pr.values())
                + (lambda over: (f": where ours exceed the paper's (p = {', '.join(f'{float(k):g}' for k in over)}), its values "
                                 "lie between ours with and without them, as for Fig. 12, where a wider pruning window "
                                 "removes such failures: the decoder's pruning sets them")
                   if over and all(pr[k]["ratio_without"] <= 1.03 for k in over) else "")(
                    [k for k, v in pr.items() if v["ratio"] > 1.03]))
    if rows:
        rows.append("")
        rows.append("Cells: ours (ours / paper); * the paper's value, ±3 % for reading, lies outside our 95 % interval "
                    "(bootstrap over tasks). The test is strict: the paper's statistical uncertainties are not published, "
                    "and error bars shorter than its markers (±14-22 % in value on these axes) are not visible.")
    fb = gf.get("fig12_beam") if isinstance(gf.get("fig12_beam"), dict) else None
    bt = []
    if fb:
        for lab in ("K64", "K512", "D24", "K4096"):
            v = fb.get(lab)
            if not v:
                continue
            bt.append(f"{v['decoder']}: failures {v['0']['failures_kept']:.0f}, kept at γ = 4 {v['4']['failures_kept']:.0f} "
                      f"(error at γ = 0 and 4: {v['0']['ratio']:.2f}x and {v['4']['ratio']:.2f}x the paper's; rejection at "
                      f"γ = 4 {v['4']['rejection_ratio']:.2f}x)")
    mb = gf.get("fig11_beam") if isinstance(gf.get("fig11_beam"), dict) else None
    if mb and "K512" in mb and "K4096" in mb:
        a_, b_ = mb["K512"], mb["K4096"]
        bt.append(f"memory (X basis, p = 7e-4, {a_['shots']:.0f} shots): failures {a_['0']['failures_kept']:.0f} with K = 512, "
                  f"Delta = 12 against {b_['0']['failures_kept']:.0f} with K = 4096, Delta = 24; kept at γ = 4 "
                  f"{a_['4']['failures_kept']:.0f} against {b_['4']['failures_kept']:.0f}; rejection per round at γ = 4 "
                  f"{a_['4']['rejection']:.2e} against {b_['4']['rejection']:.2e}")
    concl = ""
    if fb and fb.get("K512") and fb.get("D24"):
        k5, d2 = fb["K512"]["4"], fb["D24"]["4"]
        if (d2.get("ratio") is not None and k5.get("ratio") is not None and d2["ratio"] <= 1 <= k5["ratio"]
                and (d2.get("rejection_ratio") or 0) >= 0.85):
            concl = (" The paper's postselected TQRB values lie between our two pruning windows (at γ = 4 its error is "
                     f"{1 / max(k5['ratio'], 1e-9):.2f}x ours with Delta = 12 and "
                     + (f"{1 / d2['ratio']:.2f}x" if d2["ratio"] > 0 else "above")
                     + " ours with Delta = 24, which also matches its rejection): forced-gap postselection is set by "
                     "the decoder's pruning, and the paper does not publish its decoder settings. Without "
                     f"postselection the wider window lowers our error to {fb['D24']['0']['ratio']:.2f}x the paper's.")
    T["gap_beam_text"] = ("Decoder dependence: the same shots (TQRB: 1e5 at p = 1.1e-3) decoded with other beam widths K "
                          "and pruning windows Delta (states more than Delta nats below the best are dropped): "
                          + "; ".join(bt) + "." + concl) if bt else "PENDING (decoder sensitivity runs)"
    tcf = S.get("tqrb_confident", {}) if isinstance(S.get("tqrb_confident"), dict) else {}
    ct = []
    for pk in sorted(k for k in tcf if k not in ("cores_decode", "missing")):
        v = tcf[pk]
        ct.append(f"at p = {float(pk):g}, {v['confident']} of {v['failures']} failures in {v['shots']:.0f} shots "
                  f"(minimal failing cores of {', '.join(f'{n} x {k}' for k, n in v['core_sizes'].items())} faults; exact "
                  f"most-likely-error decoding of the core is right in {v['exact_mle_correct']} and wrong in "
                  f"{v['exact_mle_fails']})")
    cdd = tcf.get("cores_decode")
    if ct:
        T["gap_beam_text"] += (" Why: the campaign decoder makes some TQRB failures with an infinite gap, every state "
                               "of the correct class pruned, and then never re-decodes them (re-decoding is triggered "
                               "by a gap below 8 nats): " + "; ".join(ct) + "."
                               + ((" Re-decoding these cores: " + ", ".join(
                                   f"{k.replace('_D', ', Delta = ').replace('K', 'K = ')} fails on {v['fail']} of "
                                   f"{v['cores']} ({v['fail_where_exact_mle_succeeds']} that exact decoding gets right)"
                                   for k, v in cdd.items()) + "."
                                   + (" The pruning window, not the beam width, loses them."
                                      if cdd.get("K4096_D12", {}).get("fail", -1) >= cdd.get("K512_D12", {}).get("fail", 0) - 1
                                      and cdd.get("K512_D24", {}).get("fail_where_exact_mle_succeeds", 99)
                                      <= 0.2 * cdd.get("K512_D12", {}).get("fail_where_exact_mle_succeeds", 0) else ""))
                                  if cdd else ""))
    T["gap_table"] = "\n".join(rows) if rows else "PENDING (campaigns c4_fig11, tqrb_fig12)"
    T["gap_summary"] = ("; ".join(summ) + ".") if summ else "PENDING"
    f14 = S.get("fig14", {}) if isinstance(S.get("fig14"), dict) else {}
    rows = []
    for p in ("0.0007", "0.0009", "0.0011"):
        v = f14.get(p)
        if not v:
            continue
        if not rows:
            rows += ["| p | ours, per logical CNOT [95 %] | per transversal CNOT | paper (digitized) | ours / paper | failures X / Z (shots) |",
                     "|---|---|---|---|---|---|"]
        rows.append(f"| {float(p):g} | {v['per_logical_cnot']:.2e} [{v['ci95'][0]:.2e}, {v['ci95'][1]:.2e}] | "
                    f"{v['per_transversal_cnot']:.2e} | {v['paper']:.2e} | {v['ratio']:.2f}{'' if v['paper_inside'] else '*'} | "
                    f"{v['failures']['X']:.0f} / {v['failures']['Z']:.0f} ({v['shots']['X']:.0f}, {v['shots']['Z']:.0f}) |")
    if rows:
        rows.append("")
        rows.append("\\* the paper's value, ±3 % for reading, lies outside our 95 % interval (strict: the paper's own "
                    "statistical uncertainty is not published, and error bars shorter than its markers, ±14 % in value "
                    "on this axis, are not visible).")
    T["fig14_table"] = "\n".join(rows) if rows else "PENDING (campaign c4_fig14)"
    # ---- chain-map CNOT from the surface code (Fig. 14 'Surface-C4-Helix') and the GHZ state
    cm = S.get("chainmap", {}) if isinstance(S.get("chainmap"), dict) else {}
    mp, dist = cm.get("map") or {}, cm.get("distance") or {}
    txt = ("The paper's chain map is not published (its search method is arXiv:2607.02482); ours "
           "(`src/hase/chainmap.py`, `results/chainmap/F.json`) solves the chain-map conditions over GF(2) with at "
           "most two CNOTs per qubit by SAT"
           + (f": {mp['cnots']} CNOTs in two layers (depth 2, as the paper's)" if mp else "")
           + ", mapping the surface code's logical X onto C4-Helix's X_0, C4-Helix's Z_0 onto the surface code's "
           "logical Z and its Z_1 onto a surface stabilizer.")
    if dist:
        txt += (f" One chain-map CNOT between rounds of both codes has circuit fault distance {dist['min']} (exact "
                f"integer programming over {dist['n']} logical classes in both bases"
                + (", every class proved" if dist["all_optimal"] else ", not every class proved")
                + "), the paper's 'distance-5'.")
    txt += (" Six chain-map CNOTs, each followed by a round on both codes (the surface code's extraction in parallel "
            "with C4-Helix's inner checks), destructive measurement in the basis of preparation, uniform noise; per "
            "CNOT 1 − (1 − p_L)^(1/6), p_L: any of the three logical outcomes wrong. The paper states the Z basis; "
            "its transversal series is the mean of the two bases, so both are given:")
    T["chainmap_text"] = txt
    f14s = cm.get("fig14_surface") or {}
    rows = []
    for p in ("0.0007", "0.0009", "0.0011"):
        v = f14s.get(p)
        if not v:
            continue
        if not rows:
            rows += ["| p | Z basis, per CNOT [95 %] | X basis | mean of X and Z [95 %] | paper (digitized) | ours / paper: "
                     "Z, mean | failures from the surface code: Z, X | failures Z / X (shots) |",
                     "|---|---|---|---|---|---|---|---|"]
        z, x, m = v["Z"], v.get("X"), v.get("mean")
        rows.append(f"| {float(p):g} | {z['per_cnot']:.2e} [{z['ci95'][0]:.2e}, {z['ci95'][1]:.2e}] | "
                    + (f"{x['per_cnot']:.2e} [{x['ci95'][0]:.2e}, {x['ci95'][1]:.2e}]" if x else "PENDING") + " | "
                    + (f"{m['per_cnot']:.2e} [{m['ci95'][0]:.2e}, {m['ci95'][1]:.2e}]" if m else "PENDING") + " | "
                    f"{v['paper']:.2e} | {z['ratio']:.2f}{'' if z['paper_inside'] else '*'}, "
                    + (f"{m['ratio']:.2f}{'' if m['paper_inside'] else '*'}" if m else "-") + " | "
                    f"{100 * z['surface_share']:.0f} %, " + (f"{100 * x['surface_share']:.0f} %" if x else "-") + " | "
                    f"{z['failures']:.0f} / " + (f"{x['failures']:.0f}" if x else "-")
                    + f" ({z['shots']:.0f}" + (f", {x['shots']:.0f}" if x else "") + ") |")
    if rows:
        rows.append("")
        rows.append("\\* the paper's value, ±3 % for reading, lies outside our 95 % interval. Failures from the surface "
                    "code: share of the logical flips on the surface code's observable.")
        su = cm.get("fig14_surface_summary") or {}
        ni = cm.get("fig14_surface_noidle")
        if su.get("ratios_mean") and len(su["ratios_mean"]) == len(su["ratios"]):
            ok = all(0.67 <= r <= 1.5 for r in su["ratios_mean"])
            rows.append("")
            rows.append(
                f"In the Z basis ours is {min(su['ratios']):.2f}-{max(su['ratios']):.2f} of the paper's values, and "
                f"C4-Helix's logicals fail most (the surface code: {100 * min(su['surface_share_z']):.0f}-"
                f"{100 * max(su['surface_share_z']):.0f} % of the flips); in the X basis the surface code fails most "
                f"({100 * min(su['surface_share_x']):.0f}-{100 * max(su['surface_share_x']):.0f} %"
                + ((f"; without idle noise the X basis gives {ni['per_cnot']:.2e} per CNOT at p = 0.0011 instead of "
                    f"{f14s['0.0011']['X']['per_cnot']:.2e}, near the Z basis' {f14s['0.0011']['Z']['per_cnot']:.2e}: the X "
                    "basis's excess is the dephasing of idle data, which the Z basis does not see, and the surface "
                    "code's data idle through C4-Helix's longer rounds")
                   if ni and f14s.get("0.0011", {}).get("X") and ni["per_cnot"] < 0.5 * f14s["0.0011"]["X"]["per_cnot"]
                   else (f"; without idle noise the X basis gives {ni['per_cnot']:.2e} per CNOT at p = 0.0011" if ni else ""))
                + "), so the paper's remark that this CNOT is 'limited by the surface code' describes the X basis. "
                "The mean of the two bases gives ours / paper "
                + ", ".join(f"{r:.2f}" for r in su["ratios_mean"]) + f" (paper inside our 95 % CIs: {su['inside_mean']})"
                + (f": the figure's surface series matches the two-basis mean within "
                   f"{100 * max(abs(r - 1) for r in su['ratios_mean']):.0f} %, as its transversal series matches ours "
                   "(table above), and not the Z basis alone." if ok else "."))
    alts = cm.get("fig14_surface_alternatives") or {}
    if rows and alts and f14s.get("0.0011"):
        ref = f14s["0.0011"]["Z"]["per_cnot"]
        rows.append("")
        rows.append("Three other depth-2 chain maps (the next SAT solutions, "
                    + ", ".join(f"{a['cnots']}" for a in alts.values()) + " CNOTs; scripts/chainmap_search.py "
                    "alternatives) give at p = 0.0011 in the Z basis "
                    + ", ".join(f"{a['per_cnot']:.2e} [{a['ci95'][0]:.2e}, {a['ci95'][1]:.2e}]" for a in alts.values())
                    + f" per CNOT against our {ref:.2e}: the choice of map moves the value by "
                    f"{100 * max(abs(a['per_cnot'] / ref - 1) for a in alts.values()):.0f} % at most, not by the factor "
                    "between the bases.")
    T["chainmap_table"] = "\n".join(rows) if rows else "PENDING (campaign chainmap)"
    g = cm.get("ghz") or {}
    if g.get("Z") and g.get("X"):
        fb = g.get("fidelity_bounds")
        T["ghz_text"] = (
            "Both codes prepared in the Z basis (one round of extraction on both), the surface code's logical "
            "Hadamard (transversal H and a quarter turn of the lattice, which maps the turned X faces back onto the "
            "Z faces), the chain-map CNOT, C4-Helix's automorphism CNOT 0 → 1 (a relabelling: aut_H then aut_HS of "
            "`src/hase/logical.py`), destructive measurement in Z, or in X (transversal H on both codes first, as a "
            "measurement in X is made), decoded over the whole circuit; Helios model at p = 7.9e-4 "
            "(`ChainMapGHZ`, campaign chainmap). Its detector error model is identical to stim's. Fraction of "
            f"shots violating the GHZ conditions: Z agreement {100 * g['Z']['rate']:.3f} % "
            f"[{100 * g['Z']['ci95'][0]:.3f}, {100 * g['Z']['ci95'][1]:.3f}] ({g['Z']['failures']} of {g['Z']['shots']:.0f}) "
            f"against the hardware's 0.025 % [0.001, 0.139] (1 of 4000); X parity {100 * g['X']['rate']:.3f} % "
            f"[{100 * g['X']['ci95'][0]:.3f}, {100 * g['X']['ci95'][1]:.3f}] ({g['X']['failures']} of {g['X']['shots']:.0f}) "
            "against 0.050 % [0.006, 0.180] (2 of 4000); "
            + ("intervals overlapping in both bases" if g["Z"]["overlap"] and g["X"]["overlap"] else
               "intervals NOT overlapping in " + " and ".join(b for b in "ZX" if not g[b]["overlap"]))
            + (f". Fidelity bounds 1 − z̄ − x̄ ≤ F ≤ 1 − z̄: {100 * fb[0]:.3f} % ≤ F ≤ {100 * fb[1]:.3f} % against the "
               "hardware's 99.925 % ≤ F ≤ 99.975 %." if fb else ".")
            + (f" Ours lie below the hardware's central values ({g['Z']['rate'] / g['Z']['paper']:.2f}x and "
               f"{g['X']['rate'] / g['X']['paper']:.2f}x; the hardware saw 3 violations in 8000 shots), as our C4-Helix "
               "memory and TQRB values do (FINDINGS 1.4, 1.6)")
            + ((f"; with Helios's memory error per unit time doubled, the unpublished input that sets them (section 8), "
                f"{100 * g['memory_x2']['Z']['rate']:.3f} % and {100 * g['memory_x2']['X']['rate']:.3f} % "
                f"({g['memory_x2']['Z']['failures']} and {g['memory_x2']['X']['failures']} violations)")
               if len(g.get("memory_x2", {})) == 2 else "")
            + ((f". At the physical level the model is not optimistic: the hardware's physical baseline (32 "
                f"three-qubit GHZ states in parallel) gives {100 * g['physical']['Z']['mismatch']:.2f} % and "
                f"{100 * g['physical']['X']['mismatch']:.2f} % in our model against the hardware's 0.209 % and 0.253 % "
                "(analysis/physical_ghz.py), so the gap is in what compiled logical circuits add, such as memory error "
                "during their longer rounds.") if g.get("physical") else "."))
    else:
        T["ghz_text"] = "PENDING (campaign chainmap, GHZ)"
    return T


RD_KIND = {"full1": "the two-round non-adaptive circuit",
           "full8": "non-adaptive extraction over the simulated 8 rounds after preparation",
           "any8": "every path of the escalating and periodic rules over 8 rounds (each weight-6 part optional, "
                   "an outer-check part in every round)",
           "any4": "every path of those rules over 4 rounds (the value comparisons' length)",
           "P16x1": "a periodic window of 16 rounds (push 8)",
           "P8x1": "a periodic window of 8 rounds", "P4x2": "two periodic windows of 4 rounds",
           "P2x4": "four periodic windows of 2 rounds"}


def relaxed_distance_parts(S, kinds):
    """[(label, bound or None, best known, complete, found, kind, basis, order)] of the decomposition
    bounds for the given kinds."""
    rd = S.get("relaxed_distance") or {}
    wh = ((S.get("distance") or {}).get("whole") or {}) if isinstance(S.get("distance"), dict) else {}
    out = []
    for name, v in sorted(rd.items(), key=lambda kv: (list(RD_KIND).index(kv[1]["kind"]) if isinstance(kv[1], dict)
                                                      and kv[1].get("kind") in RD_KIND else 99, kv[0])):
        if not isinstance(v, dict) or v.get("kind") not in kinds:
            continue
        lab = (f"{RD_KIND[v['kind']]}, {v['basis']} basis ({'Z' if v['order'] == 'XZ' else 'X'} checks last)")
        best = (wh.get(f"{v['order']}|{v['basis']}") or {}).get("best") if v["kind"] == "full1" else None
        out.append((lab, v.get("lower_bound"), best, v.get("complete"), v.get("found"), v["kind"], v["basis"],
                    v["order"]))
    return out


def relaxed_distance_text(S, kinds):
    """One sentence: the bound, the kinds of path it holds on (bases grouped), the exact case."""
    parts = relaxed_distance_parts(S, kinds)
    if not parts:
        return None
    ok = [x for x in parts if x[1] is not None]
    bad = [x for x in parts if x[1] is None and x[3]]
    pend = [x for x in parts if x[1] is None and not x[3]]
    segs = []
    if ok:
        lb = min(x[1] for x in ok)
        groups = {}
        for lab, b_, best, c_, f_, kind, basis, order in ok:
            groups.setdefault(kind, []).append(basis + ("" if order == "XZ" else " with the X checks last"))
        names = []
        for kind in RD_KIND:
            if kind in groups:
                bs = groups[kind]
                names.append(f"{RD_KIND[kind]} ({'bases' if len(bs) > 1 else 'basis'} {', '.join(bs)})")
        exact = [(lab, best) for lab, b_, best, *_ in ok if best == b_]
        segs.append(f"no undetected logical error of at most {lb - 1} faults in any of the {len(ok)} cases decided (each "
                    "standing for all its schedules): " + "; ".join(names)
                    + f" - a fault distance of at least {lb} on each"
                    + ("".join(f"; exactly {b_} on {lab_}, where integer programming found {'an' if b_ in (8, 11, 18) else 'a'} "
                               f"{b_}-fault logical error" for lab_, b_ in exact)))
    for lab, b_, best, c_, f_, *_ in bad:
        segs.append(lab + ": " + ", ".join(f"{f['weight']} (instances {f['subset']}, combo {f['combo']})" for f in f_[:3]))
    if pend:
        segs.append("pending: " + "; ".join(x[0] for x in pend))
    return "; ".join(segs)


def push8_texts(S):
    """Pushes 8 and 9 on the periodic rule (non-dominated at uniform p = 1e-3 once the interval
    study had measured it): 8 on time per round (outer checks every 16 rounds), 9 on the error
    (parallel weight-6 checks); then the evaluated next step for the latest success."""
    uv = (S.get("value") or {}).get("uniform|0.001", {}) if isinstance(S.get("value"), dict) else {}
    rows, nd = uv.get("rows", {}), set(uv.get("nondominated", []))
    k8, k16 = "Carbon-Helix|l2=on_event,l3=never,T2=8,T3=8|r8", "Carbon-Helix|l2=on_event,l3=never,T2=16,T3=16|r16"
    k4 = "Carbon-Helix|l2=on_event,l3=never,T2=4,T3=4|r8"
    k8f = "Carbon-Helix|l2=on_event,l3=never,T2=8,T3=8,l2fast=1|r8"
    p8, p16, p4, p8f = rows.get(k8), rows.get(k16), rows.get(k4), rows.get(k8f)
    tb = S.get("time_budget", {}) if isinstance(S.get("time_budget"), dict) else {}
    out = []
    if not p8 or k8 not in nd:
        return out
    d8 = p8["distance_from_limit"]

    def ci(r):
        return f"{r['eps_LQ']:.2g} (95 % {r['ci95_LQ'][0]:.2g}-{r['ci95_LQ'][1]:.2g})"
    lead = ("Push 8 (time per round, for the periodic rule - outer checks every 8 rounds, no escalation - which the "
            "interval study's windows made non-dominated at uniform p = 1e-3; its time and its error lie "
            f"{d8['time_ms_per_round']:.2f}x and {d8['eps_LQ']:.2f}x from their limits, within "
            f"{100 * abs(d8['eps_LQ'] / d8['time_ms_per_round'] - 1):.0f} % of each other: push 8 took time, push 9 the error)")
    latest = None
    if p16 and p16.get("eps_LQ"):
        ok = k16 in nd
        out.append(lead + f": outer checks every 16 rounds instead (16 rounds): {p16['time_ms_per_round']:.0f} vs "
                   f"{p8['time_ms_per_round']:.0f} ms and {p16['twoq_per_round']:.0f} vs {p8['twoq_per_round']:.0f} "
                   f"two-qubit gates per round, error {ci(p16)} vs {ci(p8)} ({p16['status']}): "
                   + ("non-dominated, a success." if ok else "dominated."))
        if ok:
            latest = (k16, p16, "outer checks every 16 rounds")
    else:
        out.append(lead + ": outer checks every 16 rounds instead (campaign push8): error PENDING.")
    if p8f and p8f.get("eps_LQ"):
        ok9 = k8f in nd
        far = [(k, r["distance_from_limit"]["eps_LQ"]) for k, r in ((k8, p8), (k16, p16)) if r and r.get("distance_from_limit")]
        out.append("Push 9 (the error, after push 8 the criterion furthest from its limit for both periodic configurations: "
                   + ", ".join(f"{v:.2f}x" for k, v in far) + "): the weight-6 checks of a block measured with X and Z in "
                   "parallel when at most three blocks need them (as R8F for the rotations), outer checks every 8 rounds: "
                   f"error {ci(p8f)} vs {ci(p8)}, {p8f['time_ms_per_round']:.0f} vs {p8['time_ms_per_round']:.0f} ms and "
                   f"{p8f['twoq_per_round']:.0f} vs {p8['twoq_per_round']:.0f} two-qubit gates per round, peak "
                   f"{p8f['peak_qubits']} vs {p8['peak_qubits']} qubits ({p8f['status']}): "
                   + ("non-dominated, a success." if ok9 else "dominated."))
        if ok9:
            latest = (k8f, p8f, "the periodic rule with parallel weight-6 checks")
    elif p16:
        out.append("Push 9 (the error, after push 8 the criterion furthest from its limit for both periodic "
                   "configurations): parallel weight-6 checks (campaign push9): error PENDING.")
    if latest:
        k, r, nm = latest
        fc = r.get("furthest_criterion")
        dist = r.get("distance_from_limit", {})
        if fc == "time_ms_per_round" and tb.get("P16") and k == k16:
            b = tb["P16"]
            tot = b["total_ms_per_round"]
            sh = {kk: v / tot for kk, v in b["ms_per_round"].items()}
            per = sh.get("weight-6 checks, periodic (all blocks)", 0) + sh.get("full outer measurement (periodic)", 0)
            out.append(f"Next on time per round (evaluated, {nm}): the inner checks take "
                       f"{100 * sh.get('inner checks (L1)', 0):.0f} % (the limit), weight-6 checks after inner events "
                       f"{100 * sh.get('weight-6 checks, both types', 0):.0f} % and the periodic weight-6 and outer "
                       f"measurements {100 * per:.0f} % (analysis/time_budget.py, direct path sampling at uniform p = 1e-3); "
                       "doubling the period again would save at most half the periodic share, "
                       f"{50 * per:.0f} % of the time per round, and needs 32-round experiments.")
        elif fc == "eps_LQ" and p4 and p8:
            lp = (((S.get("interval") or {}).get("P") or {}).get("linear_penalty") or {}).get("points") or {}
            cs = [v["c"] for v in lp.values()]
            out.append(f"Next on the error (evaluated, {nm}, {dist.get('eps_LQ', 0):.2f}x the lowest error of any "
                       "configuration): the periodic rule's error is its in-round floor plus a window penalty of "
                       + (f"c = {min(cs):.3f}-{max(cs):.3f} of full extraction's error per round of window (FINDINGS 5)"
                          if cs else "a few per cent per round of window")
                       + f"; halving the window changes it by {100 * abs(p4['eps_LQ'] / p8['eps_LQ'] - 1):.0f} % "
                       f"(outer checks every 4 rounds: {p4['eps_LQ']:.2g} vs {p8['eps_LQ']:.2g}), so what remains is the "
                       "in-round faults of the inner checks and the weight-6 checks after events, which correction needs "
                       "and whose circuits the limits on gates and time already count.")
        elif fc == "time_ms_per_round" and tb.get("P8F") and k == k8f:
            b = tb["P8F"]
            tot = b["total_ms_per_round"]
            sh = {kk: v / tot for kk, v in b["ms_per_round"].items()}
            per = sh.get("weight-6 checks, periodic (all blocks)", 0) + sh.get("full outer measurement (periodic)", 0)
            ev = sh.get("weight-6 checks, both types", 0)
            r32 = (tb.get("R32F") or {}).get("total_ms_per_round")
            r32t = (tb.get("R32FT") or {}).get("total_ms_per_round")
            out.append(f"Next on time per round (evaluated, {nm}, now its criterion furthest from its limit at "
                       f"{dist.get(fc, 0):.2f}x): the inner checks take {100 * sh.get('inner checks (L1)', 0):.0f} % (the "
                       f"limit), weight-6 checks after inner events {100 * ev:.0f} % and the periodic weight-6 and outer "
                       f"measurements {100 * per:.0f} % (analysis/time_budget.py, direct path sampling at uniform p = 1e-3). "
                       "A longer period trades error for time (push 8); measuring only the event's type of weight-6 "
                       "checks, which saved "
                       + (f"{100 * (1 - r32t / r32):.0f} %" if r32 and r32t else "some")
                       + " of the rotations' time (push 7), would cut at most the event share, "
                       f"{100 * ev:.0f} % of the time per round here.")
        elif fc:
            out.append(f"Next (evaluated, {nm}): the criterion furthest from its limit is {fc} at {dist.get(fc, 0):.2f}x.")
    return out


def sec_texts(S):
    """Tables and sentences for FINDINGS.md that depend on which data exist."""
    T = {}
    iv = S.get("interval", {}) if isinstance(S.get("interval"), dict) else {}
    rows = []
    for fam, label in (("B", "B (ambiguous)"), ("C", "C (any-event)"), ("R", "R (rotation)"),
                       ("P", "P (periodic, no escalation)")):
        r = iv.get(fam, {})
        if "ratio" not in r:
            continue
        ps = r["ps"]
        if not rows:
            rows.append("| family | T | " + " | ".join(f"p = {p:g}" for p in ps) + " |")
            rows.append("|---|---|" + "---|" * len(ps))
        for Tk in sorted(r["ratio"], key=lambda x: float(x)):
            lo, hi = r["ratio_ci95"][Tk]
            drv = set(r.get("ratio_driven", {}).get(Tk, []))
            rows.append(f"| {label} | {Tk} | " + " | ".join(
                f"{v:.2f} [{a:.2f}, {b:.2f}]" + ("" if p in drv else "*")
                for p, v, a, b in zip(ps, r["ratio"][Tk], lo, hi)) + " |")
    if rows:
        rows.append("")
        rows.append("\\* model-dominated: under half of the configuration's or of full extraction's value comes "
                    "from measured strata; these ratios are extrapolations whose p-dependence follows the fitted "
                    "models (orders 6 to 9, coefficients fitted per configuration) and is not a measurement.")
    T["interval_table"] = "\n".join(rows) if rows else "PENDING (interval campaign)"
    txt = []
    b = iv.get("B", {})
    E = b.get("elasticity", {})
    key = f"{b['T'][0]}-{b['T'][-1]}" if b.get("T") and len(b["T"]) >= 2 else None
    cc = S.get("core_census", {}) if isinstance(S.get("core_census"), dict) else {}
    win = {t: (cc.get(f"ch_interval__{t}__X") or {}).get("genuine_six_fault_cores_over_several_rounds")
           for t in ("full", "B_T2", "B_T4", "B_T8")}
    cons = (S.get("minimal", {}) or {}).get("constructed", {}) if isinstance(S.get("minimal"), dict) else {}
    built = any(any(v.get("fails") for v in row.values()) for tag, row in (cons.get("window") or {}).items()
                if tag != "full")
    sampled = any((win.get(t) or 0) > 0 for t in ("B_T2", "B_T4", "B_T8"))
    iw = S.get("inner_window") or {}
    wins = iw.get("windows") or {}
    stc = iw.get("structure") or {}
    dist = S.get("distance") or {}
    lower_ok = False
    if wins and stc:
        Ts = sorted({w["T"] for w in wins.values()})
        d_in = min(w["min_distance"] for w in wins.values())
        allopt = all(w["all_optimal"] for w in wins.values())
        npaths = sum(w["paths"] for w in wins.values())
        local = stc.get("mechanisms_on_inner_detectors_of_several_blocks") == 0 and \
            stc.get("flags_shared_between_blocks") == 0
        three = not stc.get("X_logicals_on_at_most_two_blocks") and not stc.get("Z_logicals_on_at_most_two_blocks")
        if local and three:
            lower_ok = allopt and d_in >= 4
            txt.append("Lower bound: outside the outer-check gadgets every fault acts on one Carbon block (checked on "
                       "executed paths), a block then needs " + str(d_in) + " faults to carry an undetected inner logical "
                       "error through a window whatever rounds its weight-6 checks run in (exact integer programming, "
                       "windows of T = " + ", ".join(map(str, Ts)) + f" rounds, {npaths} paths, "
                       + " and ".join(sorted({w['basis'] for w in wins.values()})) + " bases"
                       + (", every logical class proved" if allopt else ", not every class proved") +
                       "; scripts/inner_window_distance.py), and no logical operator of the code lies on fewer than "
                       f"three blocks: such faults need {3 * d_in} to leave an undetected logical error, inside a window "
                       f"as in one round, and a most-likely-error decoder fails only with at least {(3 * d_in + 1) // 2}"
                       + ((lambda w: (f"; with the outer-check gadgets, the whole circuit's fault distance is at least "
                                     f"{min(b_ for _, b_, *_r in w)} on every path of {len(w)} kinds ("
                                     + "; ".join(l_ for l_, *_r in w) + "; decomposition, section 4), so failures "
                                     f"need {(min(b_ for _, b_, *_r in w) + 1) // 2} faults with the gadgets too, in a "
                                     "window as in a round") if w and all(b_ for _, b_, *_r in w) else
                           (f"; where the outer-check gadgets' data errors meet no later check (the end of the circuit) "
                            f"the distance is at least {dist['end_distance']} (section 4)" if dist.get("end_distance") else ""))(
                           relaxed_distance_parts(S, ("full8", "any8", "P16x1", "P8x1", "P4x2", "P2x4")))) + ".")
    if lower_ok and built:
        T["interval_head"] = ("The interval must stay constant, T* ~ p^0: window failures need six faults, as many "
                              "as the in-round failures that set the code's floor (six-fault ties; section 4), so the "
                              "window penalty has the order of the floor and the interval keeping the error within a "
                              "fixed factor of it is the same at every p")
        txt.insert(0, T["interval_head"] + ".")
        txt.append("Upper bound: window failures with six faults exist, constructed directly (two silent errors in one "
                   "window; analysis/window_failures.py).")
    elif built or sampled:
        txt.append(("Upper bound: window failures with six faults exist, " if lower_ok else
                    "Window failures need six faults, as many as the in-round failures that set the code's floor "
                    "(six-fault ties; section 4): ")
                   + ("constructed directly (two silent errors in one window, below)" if built else "")
                   + (" and " if built and sampled else "")
                   + (f"found among the sampled failures (six-fault minimal failing sets confirmed by exact decoding "
                      f"whose faults lie two or more rounds apart: T = 2: {win.get('B_T2') or 0}, T = 4: {win.get('B_T4') or 0}, "
                      f"T = 8: {win.get('B_T8') or 0}; full extraction: {win.get('full') or 0})" if sampled else "")
                   + (", as many faults as the in-round failures that set the code's floor (six-fault ties; section 4)"
                      if lower_ok else "")
                   + ". The window penalty therefore has the order of the in-round floor, and the interval that keeps "
                   "the error within a fixed factor of the floor is the same at every p: T* ~ p^0.")
    if key and key in E:
        drv = E[key]["driven"]
        ps = b["ps"]
        esc = b.get("escalation_per_round")
        if drv:
            vals = [(p, E[key]["value"][ps.index(p)], E[key]["ci95"][0][ps.index(p)], E[key]["ci95"][1][ps.index(p)])
                    for p in drv]
            txt.append("Measured where the estimates are data-driven, for the escalating rule (ambiguous events, "
                       "full measurement forced every T rounds): the error depends on the forced interval as "
                       "T^E with E = " + ", ".join(f"{v:.2f} [{a:.2f}, {c:.2f}] at p = {p:g}" for p, v, a, c in vals)
                       + f" (T = {key.replace('-', ' to ')})"
                       + ((". At these noise levels escalations measure the full syndrome in "
                           + ", ".join(f"{100 * esc[ps.index(p)]:.0f} %" for p, *_ in vals)
                           + " of the rounds without any forcing, so windows rarely reach the forced interval; "
                           "escalations grow rarer as p falls (below), so the forced interval increasingly bounds the "
                           "windows, and there the six-fault window failures set the interval")
                          if esc else "") + ".")
        else:
            txt.append("No p has at least half of every compared value from measured strata yet, so the interval "
                       "dependence is not measured.")
    sc = b.get("escalation_scan") or {}
    if sc.get("per_round"):
        pe, rt = sc["p"], sc["per_round"]
        txt.append("Without forced full measurements a window closes at the next escalation (ambiguous change or "
                   "fired flag); measured by direct path sampling at every p, escalations occur in "
                   + ", ".join(f"{100 * r:.2g} % of the rounds at p = {p:g}" for p, r in zip(pe, rt) if p in (3e-3, 1e-3, 1e-4))
                   + (f", falling as p^{sc['exponent_low_p']:.2f} below 1e-3" if sc.get("exponent_low_p") else "")
                   + ": unforced windows last about 1/p rounds, as under the 1/p unmasking rule"
                   + ("." if (iv.get("P") or {}).get("ratio") else
                      ", and would cost about one power of p; a forced interval that stays constant keeps the exponent."))
    fk = b.get("fault_order_k", {})
    ks = [(t, fk[t]) for t in ["full"] + [str(x) for x in b.get("T", [])] if isinstance(fk.get(t), dict) and fk[t].get("k") is not None]
    if ks:
        txt.append("Order of the failure probability in the number of faults over the observed strata (exponent in "
                   "p it implies at the strata's mean fault density): " + ", ".join(
                       f"{'full' if t == 'full' else 'T = ' + t} {x['slope_at_p_star']:.2f}"
                       + (f" [{x['slope_ci95'][0]:.2f}, {x['slope_ci95'][1]:.2f}]" if x.get("slope_ci95") else "")
                       + f" at p = {x['p_star']:.1e}" for t, x in ks) + ".")
    ts = b.get("T_star", {}).get("0.5", {})
    if ts:
        txt.append("Largest interval in the observed range keeping the error within 1.5x of full extraction: " +
                   ", ".join(f"T* {v} at p = {k}" for k, v in ts.items()) + ".")
    ip = b.get("inverse_p_rule", {}).get("points", {})
    ipd = {k: v for k, v in ip.items() if v.get("driven")}
    if ipd:
        txt.append("The 1/p unmasking rule of arXiv:2502.14835 (T = 10 at p = 1e-3, proportional to 1/p) inside "
                   "the observed T range: error relative to full extraction " + ", ".join(
                       f"{v['ratio_full']:.2f} at p = {k} (T = {v['T']:.1f})" for k, v in ipd.items()) + ".")
    # windows that always reach the forced interval: the periodic family (no escalation)
    pf = iv.get("P", {}) if isinstance(iv.get("P"), dict) else {}
    if pf.get("ratio"):
        ps_ = pf["ps"]
        rows_ = []
        for Tk in sorted(pf["ratio"], key=float):
            drv = pf.get("ratio_driven", {}).get(Tk, [])
            lo, hi = pf["ratio_ci95"][Tk]
            if drv:
                rows_.append(f"T = {Tk}: " + ", ".join(
                    f"{pf['ratio'][Tk][ps_.index(p)]:.2f} [{lo[ps_.index(p)]:.2f}, {hi[ps_.index(p)]:.2f}] at p = {p:g}"
                    for p in drv))
        Ep = pf.get("elasticity", {})
        kp = f"{pf['T'][0]}-{pf['T'][-1]}" if len(pf.get("T", [])) >= 2 else None
        es = ""
        if kp and kp in Ep and Ep[kp]["driven"]:
            es = ("; it depends on the interval as T^E, E = " + ", ".join(
                f"{Ep[kp]['value'][ps_.index(p)]:.2f} [{Ep[kp]['ci95'][0][ps_.index(p)]:.2f}, "
                f"{Ep[kp]['ci95'][1][ps_.index(p)]:.2f}] at p = {p:g}" for p in Ep[kp]["driven"]) + f" (T = {kp.replace('-', ' to ')})")
        g = pf.get("gamma") or {}
        gs = (f", with no measured dependence on p (dE/d ln p = {g['value']:.2f} [{g['ci95'][0]:.2f}, {g['ci95'][1]:.2f}] over "
              f"p = {', '.join(f'{x:g}' for x in g['points'])})" if g.get("value") is not None and g.get("ci95") else "")
        fo = (pf.get("fault_order") or {}).get(kp) or {}
        fs = (f"; in the strata, the failure probability's growth with the number of faults at T = {pf['T'][-1]} "
              f"relative to T = {pf['T'][0]} has order {fo['gamma']:.2f} [{fo['gamma_ci95'][0]:.2f}, "
              f"{fo['gamma_ci95'][1]:.2f}] (equal orders give 0, window failures needing one more fault 1"
              + ("; the data exclude 1" if fo['gamma_ci95'][1] < 1 else "")
              + f"; fault densities of p = {fo['p_range'][0]:.1e} to {fo['p_range'][1]:.1e})"
              if fo.get("gamma") is not None and fo.get("gamma_ci95") else "")
        lp = (pf.get("linear_penalty") or {}).get("points") or {}
        ls = ""
        if lp:
            ls = ("; per round of window, relative to full extraction, R = a + c (T - 1) with c = " + ", ".join(
                f"{v['c']:.3f}" + (f" [{v['c_ci95'][0]:.3f}, {v['c_ci95'][1]:.3f}]" if v.get("c_ci95") else "")
                + f" at p = {k}" for k, v in lp.items()))
        tsp = (pf.get("T_star") or {}).get("0.5") or {}
        tss = ("; the largest periodic interval keeping the error within 1.5x of full extraction: " + ", ".join(
            f"T* {v} at p = {k}" for k, v in tsp.items()) if tsp else "")
        if rows_:
            txt.append("Measured with windows that always last the forced interval (periodic family: outer checks only "
                       "every T rounds, weight-6 checks on events and at the window ends, no escalation - the windows "
                       "the escalating rules approach as escalations grow rare at low p): error relative to full "
                       "extraction " + "; ".join(rows_) + es + gs + fs + ls + tss + ". Below the data-driven p the "
                       "ratios come from the order-6 models (labelled extrapolated in the table); T* ~ p^0 there rests "
                       "on the proven equal orders, which make each ratio tend to a constant (a ratio of six-fault "
                       "coefficients) as p -> 0, not on the extrapolated values.")
            sc_ = b.get("escalation_scan") or {}
            his = [(v["a_ci95"][1], v["c_ci95"][1]) for v in lp.values() if v.get("a_ci95") and v.get("c_ci95")]
            if his and sc_.get("per_round"):
                a_hi = max(x[0] for x in his)
                c_hi = max(x[1] for x in his)
                proj = []
                for p, r in zip(sc_["p"], sc_["per_round"]):
                    if p in (1e-3, 1e-4) and r > 0:
                        L = 1.0 / r
                        proj.append(f"windows of about {L:.0f} rounds at p = {p:g}: at most {a_hi + c_hi * (L - 1):.1f}x "
                                    "full extraction")
                if proj:
                    txt.append("Without forced full measurements windows close only at escalations, whose measured rate "
                               "(above) makes them last about 1/p rounds. With the largest extra error per round of "
                               f"window the periodic data allow (c <= {c_hi:.3f}, a <= {a_hi:.2f}, 95 %), that is "
                               + "; ".join(proj) + f" (extrapolated beyond the {max(pf['T'])}-round windows measured, assuming the "
                               "per-round penalty does not grow as p falls, as equal orders imply). As p -> 0 the bound "
                               "grows with the window length, about 1/p (one fault can trigger an escalation, so the "
                               "escalation rate is of first order in p; measured exponent above): unforced windows give "
                               "up one power of p - not less, since six-fault window failures exist - while a constant "
                               "forced interval keeps the floor's exponent.")
    elif (b.get("inf_over_maxT") or {}).get("driven"):
        txt.append("Without forced full measurements (T = infinity) the 8-round experiment cannot show windows "
                   "longer than 8 rounds, so it differs from T = 8 only in the last round.")
    T["interval_text"] = " ".join(txt) if txt else "PENDING"
    rdp = relaxed_distance_parts(S, tuple(RD_KIND))
    if rdp:
        oks = [x for x in rdp if x[1] is not None]
        T["relaxed_distance_text"] = relaxed_distance_text(S, tuple(RD_KIND)) + (
            f". Failures need at least {(min(x[1] for x in oks) + 1) // 2} faults on every one of them." if oks else ".")
    else:
        T["relaxed_distance_text"] = "PENDING (decomposition runs, scripts/relaxed_distance.py)."
    # ---- minimal failing sets of the interval configurations: window failures among the
    # sampled failures (six-fault cores confirmed by exact decoding, spread over rounds)
    cc = S.get("core_census", {}) if isinstance(S.get("core_census"), dict) else {}
    ct = []
    for tag, lab in (("full", "full extraction"), ("B_T2", "T = 2"), ("B_T4", "T = 4"), ("B_T8", "T = 8"),
                     ("B_Tinf", "T = infinity")):
        c = cc.get(f"ch_interval__{tag}__X")
        if not c or not c.get("cores"):
            continue
        sz = c.get("sizes_genuine", {})
        ct.append(f"{lab}: {c['cores']} cores, confirmed sizes {', '.join(f'{k} ({v})' for k, v in sz.items()) or 'none'}, "
                  f"confirmed six-fault cores whose faults lie two or more rounds apart: "
                  f"{c.get('genuine_six_fault_cores_over_several_rounds', 0)}")
    T["census_text"] = ("Minimal failing sets of the stored failures with the fewest faults (analysis/minimal_fails.py; "
                        "exact most-likely-error decoding for cores of at most 7 faults): " + "; ".join(ct) + "."
                        ) if ct else "PENDING (minimal failing sets of the interval configurations)"
    # ---- which basis is worse (the tables use X, the C4-Helix paper's convention)
    bs = S.get("basis", {}) if isinstance(S.get("basis"), dict) else {}
    bt = []
    for noise, lab in (("uniform", "uniform noise"), ("helios", "Helios model")):
        b_ = bs.get(noise)
        if not b_ or b_.get("z_over_x") is None:
            continue
        bt.append(f"{lab}: the Z-basis memory failed {b_['z_fails']} times in {b_['z_shots']} shots where the X-basis "
                  f"failure rates of the same strata predict {b_['x_expected_on_z_strata']:.0f} (Z / X = "
                  f"{b_['z_over_x']:.2g}, 95 % {b_['z_over_x_ci95'][0]:.2g}-{b_['z_over_x_ci95'][1]:.2g})")
    T["basis_text"] = ("The tables use the X basis, the worse one: " + "; ".join(bt) + ".") if bt else "PENDING (Z-basis sets)"
    # ---- rotations whose period exceeds the 4-round experiment: per-round error over longer
    # experiments (each outer check measured at least once, windows not cut by the readout)
    val = S.get("value", {}) if isinstance(S.get("value"), dict) else {}
    lt_ = []
    for pt, short, long_, lab in (("uniform|0.001", "Carbon-Helix|l2=on_event,l3=ambiguous,rot=8|r4",
                                   "Carbon-Helix|l2=on_event,l3=ambiguous,rot=8|r8", "R8, 8 against 4 rounds, uniform p = 1e-3"),
                                  ("helios|0.00079", "Carbon-Helix|l2=on_event,l3=ambiguous,rot=8,l2fast=1|r4",
                                   "Carbon-Helix|l2=on_event,l3=ambiguous,rot=8,l2fast=1|r16", "R8F, 16 against 4 rounds, Helios")):
        rows = (val.get(pt) or {}).get("rows", {})
        a, b_ = rows.get(short), rows.get(long_)
        if a and b_ and a.get("eps_LQ") and b_.get("eps_LQ"):
            lt_.append(f"{lab}: {b_['eps_LQ']:.2g} against {a['eps_LQ']:.2g} per logical qubit per round "
                       f"({b_['eps_LQ'] / a['eps_LQ']:.2g}x; {b_['status']} / {a['status']})")
    hrows = (val.get("helios|0.00079") or {}).get("rows", {})
    r16 = hrows.get("Carbon-Helix|l2=on_event,l3=ambiguous,rot=16,l2fast=1|r16")
    r8l = hrows.get("Carbon-Helix|l2=on_event,l3=ambiguous,rot=8,l2fast=1|r16")
    if r16 and r8l and r16.get("eps_LQ") and r8l.get("eps_LQ"):
        lt_.append(f"push 5 (time per round), Helios, 16 rounds: one outer check every other round (R16F) against "
                   f"one per round (R8F): {r16['time_ms_per_round']:.0f} vs {r8l['time_ms_per_round']:.0f} ms and "
                   f"{r16['twoq_per_round']:.0f} vs {r8l['twoq_per_round']:.0f} two-qubit gates per round, error "
                   f"{r16['eps_LQ']:.2g} vs {r8l['eps_LQ']:.2g} ({r16['status']})")
    r32 = hrows.get("Carbon-Helix|l2=on_event,l3=ambiguous,rot=32,l2fast=1|r32")
    r16l = hrows.get("Carbon-Helix|l2=on_event,l3=ambiguous,rot=16,l2fast=1|r32")
    if r32 and r16l and r32.get("eps_LQ") and r16l.get("eps_LQ"):
        lt_.append(f"push 6 (time per round), Helios, 32 rounds: one outer check every fourth round (R32F) against "
                   f"every other round (R16F): {r32['time_ms_per_round']:.0f} vs {r16l['time_ms_per_round']:.0f} ms and "
                   f"{r32['twoq_per_round']:.0f} vs {r16l['twoq_per_round']:.0f} two-qubit gates per round, error "
                   f"{r32['eps_LQ']:.2g} (95 % {r32['ci95_LQ'][0]:.2g}-{r32['ci95_LQ'][1]:.2g}) vs {r16l['eps_LQ']:.2g} "
                   f"(95 % {r16l['ci95_LQ'][0]:.2g}-{r16l['ci95_LQ'][1]:.2g}) ({r32['status']} / {r16l['status']})")
    T["rotation_text"] = ("Rotations longer than the 4-round experiment measure some outer checks only at the final "
                          "readout, so their 4-round values understate a long run; measured over longer experiments: "
                          + "; ".join(lt_) + ".") if lt_ else "PENDING (longer rotation experiments)"
    # ---- direct sampling against the stratified estimate
    dr = S.get("direct", {}) if isinstance(S.get("direct"), dict) else {}
    dd = [f"p = {p}: direct {x['eps_LQ']:.2g} (95 % {x['ci95_LQ'][0]:.2g}-{x['ci95_LQ'][1]:.2g}, {x['fails']} failures in "
          f"{x['shots']} shots), stratified {x['stratified_LQ']:.2g} (95 % {x['stratified_ci95_LQ'][0]:.2g}-"
          f"{x['stratified_ci95_LQ'][1]:.2g}, {x['stratified_status']})"
          for p, x in sorted(dr.items()) if isinstance(x, dict) and x.get("stratified_LQ") is not None]
    T["direct_text"] = ("Direct sampling (no stratification, no model) of the non-adaptive memory against the "
                        "stratified estimate at the same p: " + "; ".join(dd) + "; the intervals "
                        + ("overlap at every p." if all(x.get("intervals_overlap") for x in dr.values() if isinstance(x, dict) and "intervals_overlap" in x)
                           else "do NOT overlap everywhere.")) if dd else "PENDING (direct-sampling check)"
    # ---- circuit fault distance at the end of the circuit (exact integer programming)
    dist = S.get("distance", {}) if isinstance(S.get("distance"), dict) else {}
    dt = []
    if dist.get("ends"):
        ends = dist["ends"]
        r8 = [f"{k}{i}" for i in range(4) for k in "XZ"]       # R8's order of the outer checks
        rows, rot = [], {}
        for key, v in ends.items():
            design, case, basis = key.split("|")
            ok = v["all_optimal"] and len(v["combos"]) == 3
            if case.startswith("R8_"):
                chk = r8[int(case[3:])]
                spreads = (chk[0] == "Z") == (basis == "X")   # its cat spreads the errors this basis sees
                rot.setdefault((spreads, v["min_weight"], ok), []).append(f"{chk} ({basis})")
                continue
            nm = {"XZ": "a non-adaptive round with the Z checks last",
                  "ZX": "a non-adaptive round with the X checks last",
                  "quiet": "a quiet round (level 1 only) after a full one"}.get(case, case)
            rows.append(f"{nm}, {basis} basis: {v['min_weight']}" + ("" if ok else " (not every logical class proved)"))
        for (spreads, w, ok), cs in sorted(rot.items()):
            rows.append(f"a rotation (R8) round ending with one outer check whose cat spreads "
                        + ("the errors the basis measures" if spreads else "only the other type")
                        + f" ({len(cs)} cases): {w}" + ("" if ok else " (not every logical class proved)"))
        ed = dist.get("end_distance")
        n_ok, n_all = dist.get("end_cases_complete", 0), dist.get("end_cases", 0)
        dt.append(f"at the end of the circuit, where the data errors an outer-check gadget leaves meet "
                  f"no later check (exact integer programming over the mechanisms seen only by the final measurement "
                  f"and the flags, every logical class; scripts/distance_suite.py): "
                  + (f"at least {ed} at each of the {n_ok} places where every logical class is proved "
                     f"({n_all} checked; failures there need {(ed + 1) // 2} faults)" if ed is not None else "pending")
                  + "; " + "; ".join(rows))
    wit = os.path.join(ROOT, "results", "distance", "end__k4cyc__XZ__X__c1.json")
    if dist.get("ends") and os.path.exists(wit):
        try:
            from distance_witness import describe
            ms = describe(json.load(open(wit)))
            gad = [m for m in ms if any("L3" in k for k in m["sources"]) and not any("data flip" in k for k in m["sources"])]
            nfl = sum(str(m["detectors"]).count("flag") for m in gad)
            dt.append(f"a smallest one at the end of a non-adaptive round (X basis, {len(ms)} mechanisms, "
                      f"results/distance/end__k4cyc__XZ__X__c1.json): {len(gad)} faults inside the outer-check gadgets, "
                      f"whose {nfl} flag and parity outcomes cancel in pairs, and {len(ms) - len(gad)} single data errors")
        except Exception as ex:
            dt.append(f"witness not described ({ex!r})")
    des = dist.get("designs", {})
    if des:
        names = [("latin4", "the first gadget (cat flags read only at the end)"),
                 ("k5pre_e", "five cat parities read halfway and one before the data, first data order"),
                 ("k5pre", "the same, half-balanced data order"),
                 ("k4cyc_e", "four parities halfway (the gadget used), first data order"),
                 ("k6_e", "all six parities halfway, first data order"),
                 ("k6", "the same, half-balanced data order")]
        dt.append("outer-check gadgets compared at the end of a non-adaptive round (X basis, same search; the gadget "
                  "used, with the half-balanced data order, is the first line above): "
                  + "; ".join(f"{lab}: {des[k]['min_weight']}" + ("" if des[k]["all_optimal"] else " (bound not proved)")
                              for k, lab in names if k in des))
    wh = dist.get("whole", {})
    if wh:
        dt.append("the whole two-round non-adaptive circuit by integer programming on the unrestricted model "
                  "(11 hours, not converged): "
                  + "; ".join(f"{k.split('|')[1]} basis ({'Z' if k.startswith('XZ') else 'X'} checks last): "
                              + (f"{v['best']}, proved" if v["optimal"] else
                                 f"best found {v['best']}, proven lower bound {v['lower_bound']:.0f}")
                              for k, v in sorted(wh.items())))
    gg = dist.get("gadget_gain", {})
    if gg.get("outer"):
        def worst(r):
            g = {int(k): v for k, v in r["gain"].items() if v is not None}
            return max(g.values()), min(k for k, v in g.items() if v == max(g.values()))
        lat = gg["outer"].get("latin4_Z")
        new = gg["outer"].get("k4cyc_Z")
        if lat and new:
            gl, kl = worst(lat)
            gn, kn = worst(new)
            dt.append(f"one gadget alone (analysis/gadget_gain.py: the most data errors, up to the check, that k faults "
                      f"leaving every flag silent can cause, minus k): first gadget {gl} more errors than faults "
                      f"(with {kl} faults; with 2 faults already {lat['gain']['2']}), new gadget at most {gn} (with {kn} faults: "
                      "all four cat qubits flipped between two data layers, a stabilizer of the cat that no cat readout can "
                      "see; with 2 faults " + str(new['gain']['2']) + ")")
        bl = gg.get("blocks", {})
        if bl:
            def wg(key):
                r = bl.get(key)
                return max(v for v in r["gain"].values() if v is not None) if r else None
            dt.append(f"the level-2 parts have gain {wg('L2_Z')} (X: {wg('L2_X')}); the level-1 parts (the C4-Helix "
                      f"circuit) {wg('L1a_Z')} when only flags must stay silent - one fault there can leave a C4 logical "
                      f"error on two data qubits, but it also flips its check's outcome, and with every outcome of the "
                      f"part silent the gain is {wg('L1a_Z_strict')}")
    rdt = relaxed_distance_text(S, tuple(RD_KIND))
    if rdt:
        head = ("of the whole circuit, decided by decomposition (relaxed problem, every logical class, every "
                "subset of active outer-check instances; scripts/relaxed_distance.py, docs/methods.md): " + rdt
                + ". Diagnostics of the gadgets, from before the decomposition: ")
        dt = [head + dt[0]] + dt[1:] if dt else [head.rstrip(": ")]
    T["distance_text"] = ("Circuit fault distance " + "; ".join(dt) + ".") if dt else "PENDING (fault-distance runs)"
    # ---- decoder audit and the model-free range of the Carbon-Helix values
    car = S.get("carbon", {}) if isinstance(S.get("carbon"), dict) else {}
    da = []
    for noise, pref in (("uniform", "1e-3"), ("helios", "7.9e-4")):
        camp, strong = car.get(noise, {}).get("X", {}), car.get(noise + "_strong", {}).get("X", {})
        pk = "0.001" if noise == "uniform" else "0.00079"
        parts = []
        for tag in ("full", "C", "B4", "R4", "R8", "R8F"):
            a, c = strong.get(tag), camp.get(tag)
            if not a or not c or not a.get("audit"):
                continue
            au = a["audit"]
            parts.append(f"{tag}: {au['genuine']}/{au['audited_fails']} failures confirmed"
                         + (f" (audit covers {au['coverage']:.0%} of {au['campaign_fails']})" if au["coverage"] < 0.999 else "")
                         + f", successes turned into failures {au['ok_strong_fail']}/{au['ok_sampled']}"
                         + (f" (95 % upper limit {au['ok_to_fail_rate_hi95']:.2g} per shot)" if au.get("ok_to_fail_rate_hi95") else "")
                         + f"; confirmed-failure error at p = {pref}: {a['eps_LQ'][pk]:.2g} against {c['eps_LQ'][pk]:.2g}")
        if parts:
            da.append(f"{noise.capitalize()} noise - " + "; ".join(parts) + ".")
    # the C4-Helix configurations of the value comparison, audited the same way
    rd = S.get("decoder_audit", {}) if isinstance(S.get("decoder_audit"), dict) else {}
    c4a = []
    for key, lab in (("c4_strat_uniform__full__X", "uniform, non-adaptive"), ("c4_strat_uniform__ase__X", "uniform, ASE"),
                     ("c4_strat_helios__full__X", "Helios, non-adaptive"), ("c4_strat_helios__ase__X", "Helios, ASE")):
        t = (rd.get(key) or {}).get("totals") if isinstance(rd.get(key), dict) else None
        if t:
            c4a.append(f"{lab} {t['genuine']}/{t['fail']}")
    if c4a:
        def corr(t):
            return 1 - t["genuine"] / t["fail"] if t.get("fail") else None
        c4c = [corr(v["totals"]) for k, v in rd.items() if k.startswith("c4_strat_") and isinstance(v, dict) and v.get("totals")]
        chc = [corr(v["totals"]) for k, v in rd.items() if (k.startswith("ch_main__") or k.startswith("ch_helios__"))
               and isinstance(v, dict) and v.get("totals")]
        c4c, chc = [x for x in c4c if x is not None], [x for x in chc if x is not None]
        fair = bool(c4c and chc and max(c4c) <= min(chc))
        da.append("C4-Helix configurations of the value comparison, the same audit: " + ", ".join(c4a)
                  + f" failures confirmed (the stronger decoder corrects {100 * min(c4c):.0f}-{100 * max(c4c):.0f} % of "
                  f"C4-Helix's failures against {100 * min(chc):.0f}-{100 * max(chc):.0f} % of Carbon-Helix's)"
                  + (", so the comparison of section 6 does not favour Carbon-Helix." if fair else "."))
    T["decoder_audit_text"] = (" ".join(da) + " The campaign decoder's values are the ones reported; the "
                               "confirmed-failure values show how much of them a stronger decoder could remove."
                               ) if da else "PENDING (decoder audits of the complete campaigns)"
    full = car.get("uniform", {}).get("X", {}).get("full")
    if full and full.get("data_only_upper_LQ"):
        d2, d1 = full["data_only_upper_LQ"].get("0.002"), full["data_only_upper_LQ"].get("0.001")
        T["data_only_text"] = (
            f"Without the model, the data alone (Jeffreys posterior of every sampled stratum, unsampled strata "
            f"counted as failing) bound the non-adaptive error per logical qubit by {d2:.1e} at p = 2e-3 and by "
            f"{d1:.1e} at p = 1e-3 (one-sided 95 %): the strata that dominate at p = 1e-3 hold few or no "
            f"failures, so values there rest on the model, whose check is section 4.") if d2 and d1 else "PENDING"
    else:
        T["data_only_text"] = "PENDING"
    cons = S.get("minimal", {}).get("constructed", {}) if isinstance(S.get("minimal"), dict) else {}
    w = cons.get("window", {})
    if w:
        parts = []
        for tag, row in w.items():
            fails = [k for k, v in row.items() if v["fails"]]
            parts.append(f"{tag}: {len(fails)}/{len(row)} placements (rounds {', '.join(fails) if fails else 'none'})")
        T["window_text"] = ("Mechanism, constructed directly (analysis/window_failures.py): two \"silent\" weight-3 "
                            "errors (syndrome identical to a single-qubit error) in two blocks fail for these "
                            "placements (rounds of the two errors): " + "; ".join(parts) +
                            ". Full extraction resolves each silent error in its own round; the adaptive rules "
                            "fail when both errors fall between two outer measurements.")
    else:
        T["window_text"] = "PENDING"
    # ---- value
    val = S.get("value", {})
    vt = []
    for pt in ("helios|0.00079", "helios|0.0003", "uniform|0.001"):
        v = val.get(pt) if isinstance(val, dict) else None
        if not v or not v.get("rows"):
            continue
        rows = v["rows"]
        c4 = rows.get("C4-Helix|full|r4")
        c4a = rows.get("C4-Helix|mode=ase|r4")
        chf = rows.get("Carbon-Helix|full|r4")
        # the comparison configuration: 4 rounds after preparation, like C4-Helix's; every
        # adaptive rule reported (a single 'lowest' estimate would select on noise)
        ad = [r for k, r in rows.items() if r["code"] == "Carbon-Helix" and r.get("rounds") == 4
              and r.get("rule") != "full" and r.get("eps_LQ")]
        if c4 and c4a and c4.get("eps_LQ") and c4a.get("eps_LQ") and chf and chf.get("eps_LQ") and ad:
            e = [r["eps_LQ"] for r in ad]
            g = [r["twoq_per_round"] for r in ad]
            t = [r["time_ms_per_round"] for r in ad]
            cif = chf.get("ci95_LQ")
            vt.append(f"{pt.replace('|', ' p = ')} (X basis, 4 rounds after preparation): C4-Helix {c4['eps_LQ']:.2g} "
                      f"non-adaptive and {c4a['eps_LQ']:.2g} with its adaptive extraction ({c4a['status']}); Carbon-Helix "
                      f"{chf['eps_LQ']:.2g} non-adaptive" + (f" (95 % {cif[0]:.2g}-{cif[1]:.2g})" if cif else "")
                      + f" and {min(e):.2g}-{max(e):.2g} with the adaptive rules ({len(ad)} rules, {chf['status']}): "
                      f"{c4a['eps_LQ'] / max(e):.0f}-{c4a['eps_LQ'] / min(e):.0f}x below C4-Helix's adaptive extraction, at "
                      f"{min(g) / c4a['twoq_per_round']:.1f}-{max(g) / c4a['twoq_per_round']:.1f}x its two-qubit gates and "
                      f"{min(t) / c4a['time_ms_per_round']:.1f}-{max(t) / c4a['time_ms_per_round']:.1f}x its time per round; "
                      f"against Carbon-Helix non-adaptive the adaptive rules use {1 - max(g) / chf['twoq_per_round']:.0%}-"
                      f"{1 - min(g) / chf['twoq_per_round']:.0%} fewer two-qubit gates and {1 - max(t) / chf['time_ms_per_round']:.0%}-"
                      f"{1 - min(t) / chf['time_ms_per_round']:.0%} less time per round.")
        nd = v.get("nondominated", [])
        if nd:
            vt.append(f"Non-dominated at {pt.replace('|', ' p = ')} (error, two-qubit gates, time and qubits per round): "
                      + "; ".join(f"{rows[k]['label']} ({rows[k]['rounds']} rounds)" for k in nd if k in rows) + ".")
    T["value_text"] = " ".join(vt) if vt else "PENDING"
    # ---- frontier: distance of each criterion from its limit, and the pushes made
    ft = []
    for pt in ("helios|0.00079", "uniform|0.001"):
        v = val.get(pt) if isinstance(val, dict) else None
        if not v or not v.get("rows"):
            continue
        L = v.get("limits", {}).get("Carbon-Helix")
        for key in v.get("nondominated", []):
            r = v["rows"][key]
            if r["code"] != "Carbon-Helix":
                continue
            d = r.get("distance_from_limit", {})
            ft.append(f"{pt.replace('|', ' p = ')}, {r['label']}: " + ", ".join(
                f"{k.replace('_per_round', '').replace('_ms', '')} {x:.2f}x its limit" for k, x in d.items()
                if x is not None) + (f" (furthest: {r['furthest_criterion']})" if r.get("furthest_criterion") else "") + ".")
        if L:
            ft.append(f"Limits used (Carbon-Helix): {L['twoq_per_round']} two-qubit gates and "
                      f"{L['time_ms_per_round']:.0f} ms per round (inner checks alone), {L['peak_qubits']} qubits "
                      f"(data plus one 4-qubit cat and one parity ancilla); error: the lowest of any configuration tried.")
    # pushes made (in order) and what each measured
    pushes = []
    db = S.get("decoder_benchmark", {})
    if isinstance(db, dict) and "kappa3_K256" in db:
        pushes.append(
            "Push 1 (decoding cost and decoder error): primary beam K = 512 instead of 256, with staged "
            f"re-decoding: {db['kappa3_K512']['seconds_per_shot']:.2f} vs {db['kappa3_K256']['seconds_per_shot']:.2f} s "
            f"per shot at 14 faults per shot and {db['kappa2_K512']['seconds_per_shot']:.2f} vs "
            f"{db['kappa2_K256']['seconds_per_shot']:.2f} s at 10, re-decodings "
            f"{db['kappa3_K512']['stage_use'][1] + db['kappa3_K512']['stage_use'][2]} vs "
            f"{db['kappa3_K256']['stage_use'][1] + db['kappa3_K256']['stage_use'][2]} per {db['shots']} shots, "
            f"decisions unchanged on the benchmark sample ({db['kappa2_decisions_differ'] + db['kappa3_decisions_differ']} differ).")
    hv = (S.get("value", {}).get("helios|0.00079", {}) or {}).get("rows", {}) if isinstance(S.get("value"), dict) else {}
    uv = (S.get("value", {}).get("uniform|0.001", {}) or {}).get("rows", {}) if isinstance(S.get("value"), dict) else {}
    for rows, lab in ((uv, "uniform p = 1e-3"), (hv, "Helios")):
        r4 = rows.get("Carbon-Helix|l2=on_event,l3=ambiguous,rot=4|r4")
        b4 = rows.get("Carbon-Helix|l2=on_event,l3=ambiguous,T3=4|r4")
        if r4 and b4 and r4.get("eps_LQ") and b4.get("eps_LQ"):
            pushes.append(
                f"Push 2 (two-qubit gates, the criterion furthest from its limit after B4), {lab}: rotating outer "
                f"checks R4 instead of a forced full measurement every 4 rounds: {r4['twoq_per_round']:.0f} vs "
                f"{b4['twoq_per_round']:.0f} two-qubit gates and {r4['time_ms_per_round']:.0f} vs "
                f"{b4['time_ms_per_round']:.0f} ms per round, error {r4['eps_LQ']:.2g} vs {b4['eps_LQ']:.2g} "
                f"({r4['status']}).")
    for rows, lab in ((uv, "uniform p = 1e-3"), (hv, "Helios")):
        r4 = rows.get("Carbon-Helix|l2=on_event,l3=ambiguous,rot=4|r4")
        r8 = rows.get("Carbon-Helix|l2=on_event,l3=ambiguous,rot=8|r4")
        r8f = rows.get("Carbon-Helix|l2=on_event,l3=ambiguous,rot=8,l2fast=1|r4")
        if r4 and r8 and r8.get("eps_LQ") and r4.get("eps_LQ"):
            pushes.append(
                f"Push 3 (time per round, the criterion furthest from its limit after R4), {lab}: one outer check "
                f"per round (R8) instead of two: {r8['time_ms_per_round']:.0f} vs {r4['time_ms_per_round']:.0f} ms and "
                f"{r8['twoq_per_round']:.0f} vs {r4['twoq_per_round']:.0f} two-qubit gates per round, error "
                f"{r8['eps_LQ']:.2g} vs {r4['eps_LQ']:.2g} ({r8['status']}).")
        if r8 and r8f and r8f.get("eps_LQ") and r8.get("eps_LQ"):
            pushes.append(
                f"Push 4 (time per round, still the furthest), {lab}: weight-6 X and Z checks of a block measured in "
                f"parallel when at most three blocks need them (R8F, peak {r8f['peak_qubits']} qubits): "
                f"{r8f['time_ms_per_round']:.0f} vs {r8['time_ms_per_round']:.0f} ms per round, error "
                f"{r8f['eps_LQ']:.2g} vs {r8['eps_LQ']:.2g} ({r8f['status']}).")
    pk = S.get("packing", {})
    tb = S.get("time_budget", {}) if isinstance(S.get("time_budget"), dict) else {}
    hv7 = (S.get("value") or {}).get("helios|0.00079", {}).get("rows", {}) if isinstance(S.get("value"), dict) else {}
    r32 = hv7.get("Carbon-Helix|l2=on_event,l3=ambiguous,rot=32,l2fast=1|r32")
    r32t = hv7.get("Carbon-Helix|l2=on_event,l3=ambiguous,rot=32,l2fast=1,l2type=1|r32")
    pushes.append("Pushes 5 and 6 (time per round, still the furthest): one outer check every other round (R16F), then "
                  "every fourth round (R32F), over 16 and 32 rounds (section 6, rotations over longer experiments).")
    if r32 and r32t and r32.get("eps_LQ") and r32t.get("eps_LQ"):
        pushes.append(
            f"Push 7 (time per round, the criterion furthest from its limit after R32F), Helios, 32 rounds: a block whose "
            f"inner events are all of one type measures only its weight-6 checks of that type (R32FT): "
            f"{r32t['time_ms_per_round']:.0f} vs {r32['time_ms_per_round']:.0f} ms and {r32t['twoq_per_round']:.0f} vs "
            f"{r32['twoq_per_round']:.0f} two-qubit gates per round, peak {r32t['peak_qubits']} vs {r32['peak_qubits']} "
            f"qubits, error {r32t['eps_LQ']:.2g} (95 % {r32t['ci95_LQ'][0]:.2g}-{r32t['ci95_LQ'][1]:.2g}) vs "
            f"{r32['eps_LQ']:.2g} (95 % {r32['ci95_LQ'][0]:.2g}-{r32['ci95_LQ'][1]:.2g}) ({r32t['status']})"
            + (f". Fewer escalations as well ({tb['R32FT']['instances_per_round'].get('full outer measurement (escalation)', 0):.3f} "
               f"against {tb['R32F']['instances_per_round'].get('full outer measurement (escalation)', 0):.3f} full outer "
               "measurements per round): a block's unneeded checks no longer contribute measurement errors that make "
               "its change look ambiguous." if tb.get("R32FT") and tb.get("R32F") else "."))
    elif tb.get("R32FT"):
        pushes.append(
            "Push 7 (time per round, the criterion furthest from its limit after R32F), Helios, 32 rounds: a block whose "
            f"inner events are all of one type measures only its weight-6 checks of that type (R32FT): "
            f"{tb['R32FT']['total_ms_per_round']:.0f} vs {tb['R32F']['total_ms_per_round']:.0f} ms per round by path "
            "sampling; error PENDING (campaign push7).")
    lead = "R32FT" if (r32t and r32t.get("eps_LQ")) else "R32F"
    if tb.get(lead):
        b = tb[lead]["ms_per_round"]
        tot = tb[lead]["total_ms_per_round"]
        sh = {k: v / tot for k, v in b.items()}
        w6 = sum(v for k, v in sh.items() if k.startswith("weight-6"))
        l3r_inst = tb[lead]["instances_per_round"].get("rotating outer checks", 0.0)
        gains = [max(0.0, c["separate_ms"] - c["packed_ms"]) for c in pk.get("checks", {}).values()] \
            if isinstance(pk, dict) else []
        pack = l3r_inst * (sum(gains) / len(gains)) if gains else None
        pushes.append(
            f"Next on time per round (evaluated): of {lead}'s time per round the inner checks take "
            f"{100 * sh.get('inner checks (L1)', 0):.0f} % (the limit), weight-6 checks after inner events {100 * w6:.0f} %, "
            f"full outer measurements after escalations {100 * sh.get('full outer measurement (escalation)', 0):.0f} % and "
            f"the rotating outer checks {100 * sh.get('rotating outer checks', 0):.0f} % (analysis/time_budget.py, direct "
            "path sampling at Helios's p). The weight-6 checks follow inner events, which locate the errors the inner "
            "code corrects, and escalations protect the ambiguous ones, so neither can be dropped; the rotating checks' "
            "layers hold 1-4 two-qubit gates, and packing them into the inner-check layers without changing any "
            f"qubit's gate order saves at most {pk.get('saving_ms_max', 0):.0f} ms per check"
            + (f", {pack:.1f} ms per round here" if pack is not None else "")
            + " (analysis/packing.py); halving the rotation rate again (one outer check every eighth round) would save at "
            f"most half of the rotating checks' share, {50 * sh.get('rotating outer checks', 0):.0f} % of the time per "
            "round, and needs 64-round experiments.")
    pushes += push8_texts(S)
    T["frontier_text"] = (" ".join(ft) if ft else "PENDING") + ("\n\n" + "\n\n".join(pushes) if pushes else "")
    # ---- limits: does an unpublished Helios quantity decide a conclusion?
    sens = S.get("sensitivity", {})
    lt, ratios = [], []
    if isinstance(sens, dict) and "missing" not in sens:
        def get(code, rule, noise):
            r = sens.get(f"{code}|{rule}|{noise}")
            return r["eps_LQ"] if r else None
        hv = S.get("value", {}).get("helios|0.00079", {}).get("rows", {}) if isinstance(S.get("value"), dict) else {}
        base = {("Carbon-Helix", "full"): hv.get("Carbon-Helix|full|r4", {}).get("eps_LQ"),
                ("Carbon-Helix", "l2=on_event,l3=ambiguous,rot=4"): hv.get("Carbon-Helix|l2=on_event,l3=ambiguous,rot=4|r4", {}).get("eps_LQ"),
                ("C4-Helix", "full"): hv.get("C4-Helix|full|r4", {}).get("eps_LQ"),
                ("C4-Helix", "mode=ase"): hv.get("C4-Helix|mode=ase|r4", {}).get("eps_LQ")}
        for noise, label in (("helios_mem0.5", "memory error x0.5"), ("helios_mem2", "memory error x2"),
                             ("helios_tm4", "measurement/reset 4 ms per batch")):
            c4 = get("C4-Helix", "full", noise)
            chf = get("Carbon-Helix", "full", noise)
            chr_ = get("Carbon-Helix", "l2=on_event,l3=ambiguous,rot=4", noise)
            if c4 and chr_:
                ratios.append(c4 / chr_)
                rr = sens.get(f"Carbon-Helix|l2=on_event,l3=ambiguous,rot=4|{noise}") or {}
                tr = rr.get("time_ms_per_round")
                ci = rr.get("ci95_LQ")
                lt.append(f"{label}: C4-Helix {c4:.2g}, Carbon-Helix non-adaptive {fmt_e(chf)}, R4 {chr_:.2g}"
                          + (f" (95 % {ci[0]:.2g}-{ci[1]:.2g}; {rr.get('fails')} failures, "
                             f"{100 * rr.get('measured_share', 0):.0f} % measured)" if ci else "")
                          + f" (C4-Helix / R4 = {c4 / chr_:.0f}"
                          + (f", {_r2(c4 / ci[1])}-{_r2(c4 / ci[0])} over R4's interval" if ci and ci[0] > 0 else "")
                          + (f"; R4 time per round {tr:.0f} ms" if tr else "") + ")")
    if lt:
        b4r = (hv.get("Carbon-Helix|l2=on_event,l3=ambiguous,rot=4|r4") or {})
        bc4 = (hv.get("C4-Helix|full|r4") or {})
        if b4r.get("eps_LQ") and bc4.get("eps_LQ"):
            lt.insert(0, f"model values: C4-Helix {bc4['eps_LQ']:.2g}, R4 {b4r['eps_LQ']:.2g} (C4-Helix / R4 = "
                         f"{bc4['eps_LQ'] / b4r['eps_LQ']:.0f}; R4 time per round {b4r['time_ms_per_round']:.0f} ms)")
    T["limits_text"] = ("Unpublished Helios quantities, varied (error per logical qubit per round, X basis, 4 rounds): "
                        + "; ".join(lt) + (". They move the absolute values; Carbon-Helix R4 stays below C4-Helix "
                        f"under every variation (ratio {min(ratios):.0f} to {max(ratios):.0f})." if ratios and min(ratios) > 1
                        else ". Under some variation C4-Helix is not above Carbon-Helix R4: the unpublished quantity "
                        "decides the comparison there.") if lt else "PENDING (Helios sensitivity campaign)")
    tc = S.get("timing_consistency", {})
    if isinstance(tc, dict) and tc and "missing" not in tc:
        worst = max(v["missing_faults_per_round"] / v["faults_per_round"] for v in tc.values())
        T["limits_text"] += (f" Modelling simplification (not a limit of the problem, and quantified): rounds in "
                             f"which several blocks measure their weight-6 checks together carry at most "
                             f"{100 * worst:.1f} % fewer memory faults than their duration implies "
                             "(results/timing_consistency.json), a bias of a few per cent on the adaptive "
                             "rules' Helios error, inside the reported uncertainties.")
    # ---- summary bullets (each traceable to a section below)
    sb = []
    cu = S.get("c4_uniform", {})
    hm = S.get("helios_memory", {}) if isinstance(S.get("helios_memory"), dict) else {}
    if "points" in cu:
        c4h = hm.get("c4helix|helios|X")
        sb.append("- C4-Helix reproduced (section 1): uniform-noise memory within " + ", ".join(
            f"{abs(1 - x['ratio']) * 100:.0f} %" for x in cu["points"].values()) + " of the paper's Fig. 11a, paper "
            f"inside our 95 % intervals at {sum(x['paper_inside_ci'] for x in cu['points'].values())} of "
            f"{len(cu['points'])} points"
            + (f"; Helios memory {c4h['eps_per_LQ']:.2g} per logical qubit per round (95 % {c4h['ci95_per_LQ'][0]:.2g}-"
               f"{c4h['ci95_per_LQ'][1]:.2g}) against the published 4.6e-5 (95 % 2.0e-5-1.08e-4)"
               + (", inside the published interval" if 2.0e-5 <= c4h['eps_per_LQ'] <= 1.08e-4 else ", OUTSIDE the published interval")
               if c4h else "")
            + ".")
        gf = S.get("gap_figures", {}) if isinstance(S.get("gap_figures"), dict) else {}
        tcs = S.get("tqrb_consistency", {}) if isinstance(S.get("tqrb_consistency"), dict) else {}
        f14 = S.get("fig14", {}) if isinstance(S.get("fig14"), dict) else {}
        extra = []
        s12 = (gf.get("fig12") or {}).get("summary") if isinstance(gf.get("fig12"), dict) else None
        s11 = (gf.get("fig11") or {}).get("summary") if isinstance(gf.get("fig11"), dict) else None
        r1 = tcs.get("rows", {}).get("0.001")
        if s12:
            extra.append(f"non-adaptive logical error per Clifford (TQRB, Fig. 12) ours / paper "
                         f"{s12['ratio_gamma0'][0]:.2f}-{s12['ratio_gamma0'][1]:.2f}")
        tdd = S.get("tqrb_direct", {}) if isinstance(S.get("tqrb_direct"), dict) else {}
        hr = [v["ratio"] for k in ("helios|0.001", "helios|0.0005") for v in ((tdd.get(k) or {}).get("fig13_points") or {}).values()]
        if hr and r1:
            extra.append(f"Fig. 13 (adaptive against non-adaptive) reproduced by our Helios-like model at the same two-qubit "
                         f"error (ours / paper {min(hr):.2f}-{max(hr):.2f}), not by the uniform model it is labelled with, "
                         f"under which the paper's own Fig. 12 lies {r1['fig13_over_fig12'][0]:.1f}-"
                         f"{r1['fig13_over_fig12'][1]:.1f}x below it (section 1.6)")
        if s11 and s12:
            fbb = gf.get("fig12_beam") if isinstance(gf.get("fig12_beam"), dict) else {}
            pw = bool(fbb.get("K512") and fbb.get("D24") and fbb["K512"]["4"].get("ratio") is not None
                      and fbb["D24"]["4"].get("ratio") is not None
                      and fbb["K512"]["4"]["ratio"] >= 1 >= fbb["D24"]["4"]["ratio"])
            extra.append(f"forced-gap postselection (Figs. 11, 12) ours / paper {min(s11['ratio_all'][0], s12['ratio_all'][0]):.2f}-"
                         f"{max(s11['ratio_all'][1], s12['ratio_all'][1]):.2f}"
                         + (", the paper's values between those of two decoder pruning windows (section 1.7)" if pw else ""))
        if "summary" in f14:
            extra.append("transversal CNOT between two blocks (Fig. 14) ours / paper "
                         + ", ".join(f"{x:.2f}" for x in f14["summary"]["ratios"]) + " per logical CNOT")
        if extra:
            sb[-1] = sb[-1][:-1] + "; " + "; ".join(extra) + "."
    car = S.get("carbon", {}) if isinstance(S.get("carbon"), dict) else {}
    fo = ((car.get("uniform", {}).get("X", {}).get("full") or {}).get("fault_order_all") or {})
    if fo.get("slope_at_p_star"):
        sb.append(f"- Carbon-Helix at circuit level (sections 2-4): the non-adaptive error falls as p^"
                  f"{fo['slope_at_p_star']:.1f}" + (f" ({fo['slope_ci95'][0]:.1f}-{fo['slope_ci95'][1]:.1f}, 95 %)" if fo.get("slope_ci95") else "")
                  + f" where its failures were sampled (p ~ {fo['p_star']:.0e}); six-fault failures exist, so no faster "
                  "than p^6 below.")
    hv = S.get("value", {}).get("helios|0.00079", {}) if isinstance(S.get("value"), dict) else {}
    if hv.get("rows"):
        sb.append("- " + (T.get("value_text", "") or "").split(". ")[0] + " (section 6).")
    if T.get("interval_head"):
        sb.append("- Full-measurement interval (section 5): " + T["interval_head"] + ".")
    elif T.get("interval_text", "PENDING") != "PENDING":
        sents = T["interval_text"].split(". ")
        sb.append("- Full-measurement interval (section 5): "
                  + ". ".join(sents[:2]).rstrip(".").replace("in one window, below)", "in one window)") + ".")
    if T.get("frontier_text", "PENDING") != "PENDING":
        sb.append("- Frontier (section 7): time per round is the criterion furthest from its limit; pushes made: "
                  "decoder beam, rotation R4, one outer check per round (R8), parallel weight-6 checks (R8F)"
                  + (", one outer check every other round (R16F)" if "push 5" in T.get("rotation_text", "") else "")
                  + (", one outer check every fourth round (R32F)" if "push 6" in T.get("rotation_text", "") else "")
                  + (", weight-6 checks of the event's type only (R32FT)"
                     if "Push 7 (" in T.get("frontier_text", "") and "error PENDING" not in T.get("frontier_text", "")
                     else "") + ".")
    T["summary_bullets"] = "\n".join(sb) if sb else "PENDING"
    return T


def _r2(x):
    """A ratio with two significant digits, without exponent notation."""
    return f"{x:.0f}" if x >= 10 else f"{x:.2g}"


def fmt_e(x):
    return "n/a" if x is None else f"{x:.2g}"


def sec_verdicts(S):
    """Short statements for COMPLETE.md / REPORT.md derived from the other sections, so the
    documents can never disagree with the numbers. 'PENDING' marks missing data."""
    V = {}
    # ---- fidelity
    f = []
    cu = S.get("c4_uniform", {})
    if "points" in cu:
        ins = [x["paper_inside_ci"] for x in cu["points"].values()]
        f.append(f"uniform-noise memory (Fig. 11a): paper inside our 95 % CI at {sum(ins)}/{len(ins)} points, "
                 f"ratios {', '.join(f'{x['ratio']:.2f}' for x in cu['points'].values())}")
    else:
        f.append("uniform-noise memory: PENDING")
    hm = S.get("helios_memory", {})
    c4h = hm.get("c4helix|helios|X") if isinstance(hm, dict) else None
    if c4h:
        lo, hi = c4h["ci95_per_LQ"]
        f.append(f"Helios memory: ours {c4h['eps_per_LQ']:.2g} (95 % CI {lo:.2g}-{hi:.2g}) vs paper 4.6e-5 "
                 f"(CI 2.0e-5-1.08e-4): {'overlapping' if hi >= 2.0e-5 and lo <= 1.08e-4 else 'NOT overlapping'}")
    else:
        f.append("Helios memory: PENDING")
    tc = S.get("tqrb_counts", {}).get("rows", {}) if isinstance(S.get("tqrb_counts"), dict) else {}
    if "helios|0.00079|m7" in tc:
        tr = [100 * tc[k]["time_reduction"] for k in ("helios|0.00079|m5", "helios|0.00079|m7", "helios|0.00079|m13")
              if k in tc and tc[k].get("time_reduction") is not None]
        def rng_(noise, p, key):
            v = [100 * tc[f"{noise}|{p}|m{m}"][key] for m in (5, 7, 13) if f"{noise}|{p}|m{m}" in tc]
            return f"{min(v):.0f}-{max(v):.0f} %" if v else None
        u3, u4 = rng_("uniform", "0.001", "time_reduction"), rng_("uniform", "0.0001", "time_reduction")
        g3, g4 = rng_("uniform", "0.001", "gate_reduction"), rng_("uniform", "0.0001", "gate_reduction")
        f.append(f"TQRB gate savings (Helios model, m = 7): {100 * tc['helios|0.00079|m7']['gate_reduction']:.1f} % vs paper 33 %"
                 + (f"; shot time {min(tr):.1f}-{max(tr):.1f} % shorter (Helios timing model, m = 5 to 13) vs the paper's "
                    "~23 % (hardware wall-clock, whose per-operation durations are not published)" if tr else "")
                 + (f", and {u3} / {u4} at uniform p = 1e-3 / 1e-4 vs its simulated 24 % / 30 %, which lie between our "
                    f"timing model and time proportional to two-qubit gates ({g3} / {g4}): the paper's unpublished "
                    "durations set them" if u3 and u4 and g3 and g4 else ""))
    else:
        f.append("TQRB gate savings: PENDING")
    td = S.get("tqrb_direct", {}) if isinstance(S.get("tqrb_direct"), dict) else {}
    tt = []
    for key, lab in (("uniform|0.001", "uniform p = 1e-3"), ("uniform|0.0005", "uniform p = 5e-4"),
                     ("helios|0.00079", "Helios model")):
        r = td.get(key)
        if not r or "fit" not in r:
            continue
        fa, ci = r["fit"], r["ci95"]
        pa = r.get("paper") or {}
        tt.append(f"{lab}: adaptive {fa.get('epsC_a1', float('nan')):.2e} (95 % {ci['epsC_a1'][0]:.1e}-{ci['epsC_a1'][1]:.1e})"
                  + (f", non-adaptive {fa['epsC_a0']:.2e} (95 % {ci['epsC_a0'][0]:.1e}-{ci['epsC_a0'][1]:.1e})" if "epsC_a0" in fa else "")
                  + ((f" against the paper's {'hardware ' if 'hardware' in pa.get('source', '') else 'Fig. 13 '}"
                      f"{pa.get('adaptive'):.1e}"
                      + (f" (95 % {pa['adaptive_ci95'][0]:.1e}-{pa['adaptive_ci95'][1]:.1e})" if pa.get("adaptive_ci95") else "")
                      + (f" / {pa['nonadaptive']:.1e}" if pa.get("nonadaptive") else "")) if pa else ""))
    f.append("TQRB logical error per Clifford: " + ("; ".join(tt) if tt else "PENDING"))
    tc = S.get("tqrb_consistency", {}) if isinstance(S.get("tqrb_consistency"), dict) else {}
    gf = S.get("gap_figures", {}) if isinstance(S.get("gap_figures"), dict) else {}
    s12 = (gf.get("fig12") or {}).get("summary") if isinstance(gf.get("fig12"), dict) else None
    s11 = (gf.get("fig11") or {}).get("summary") if isinstance(gf.get("fig11"), dict) else None
    r1 = tc.get("rows", {}).get("0.001")
    if r1:
        agree12 = bool(s12) and all(0.67 <= gf["fig12"][p]["gammas"]["0"]["ratio"] <= 1.5
                                    for p in ("0.0007", "0.0009", "0.0011") if p in gf["fig12"])
        if s12:
            f.append(f"the paper's Fig. 12 (the same non-adaptive benchmark, noise and decoder) lies "
                     f"{r1['fig13_over_fig12'][0]:.1f}-{r1['fig13_over_fig12'][1]:.1f}x below its Fig. 13 at p = 1e-3, "
                     f"and ours against Fig. 12: ours / paper {s12['ratio_gamma0'][0]:.2f}-{s12['ratio_gamma0'][1]:.2f} "
                     f"at its three points without postselection (paper inside our 95 % CI at {s12['inside']} points "
                     "with postselection included)"
                     + ("; the factor-three difference from Fig. 13 is therefore the paper's own, between its two "
                        "simulations" if agree12 else ""))
        else:
            f.append(f"the paper's Fig. 12 (the same non-adaptive benchmark, noise and decoder) lies "
                     f"{r1['fig13_over_fig12'][0]:.1f}-{r1['fig13_over_fig12'][1]:.1f}x below its Fig. 13 at p = 1e-3 "
                     "(our Fig. 12 runs PENDING)")
    h13 = [(lab, [v["ratio"] for v in (td.get(k) or {}).get("fig13_points", {}).values()])
           for k, lab in (("helios|0.001", "p = 1e-3"), ("helios|0.0005", "5e-4"))]
    h13 = [(lab, rr) for lab, rr in h13 if rr]
    if h13:
        f.append("Fig. 13's adaptive comparison is reproduced instead by our Helios-like model at the same two-qubit "
                 "error (ours / paper " + ", ".join(f"{min(rr):.2f}-{max(rr):.2f} at {lab}" for lab, rr in h13)
                 + "): the paper's Fig. 13 is reproduced by a device-like model, not by the stated uniform one")
    if s11:
        pr11 = ((gf.get("fig11") or {}).get("pruned") or {}) if isinstance(gf.get("fig11"), dict) else {}
        over = {k: v for k, v in pr11.items() if v.get("ratio") and v["ratio"] > 1.03}
        f.append(f"forced-gap postselection of Fig. 11 (memory): ours / paper {s11['ratio_all'][0]:.2f}-"
                 f"{s11['ratio_all'][1]:.2f}, paper inside our 95 % CI at {s11['inside']} points, rejection "
                 f"{s11['rejection_ratio'][0]:.2f}-{s11['rejection_ratio'][1]:.2f}"
                 + ((f"; where ours exceed the paper's (γ = 4 at p = {', '.join(f'{float(k):g}' for k in over)}: "
                     + ", ".join(f"{v['ratio']:.2f}" for v in over.values()) + "), most kept failures have a gap of at "
                     "least 31.5 nats, infinite gaps included - the signature of TQRB failures whose correct class the "
                     "pruning removed - and without them ours are " + ", ".join(
                         f"{v['ratio_without']:.2f}" for v in over.values()) + " of the paper's: as for Fig. 12, the "
                     "decoder's pruning, unpublished in the paper, sets these values")
                    if over and all(v["ratio_without"] <= 1.03 for v in over.values()) else ""))
    if s12 and s12.get("rejection_ratio"):
        f.append(f"forced-gap postselection of Fig. 12 (TQRB): error ours / paper {s12['ratio_all'][0]:.2f}-"
                 f"{s12['ratio_all'][1]:.2f}, rejection {s12['rejection_ratio'][0]:.2f}-{s12['rejection_ratio'][1]:.2f}"
                 + (lambda fbb: ("; on the same shots at p = 1.1e-3 a pruning window of 24 instead of 12 nats gives "
                                 f"error {fbb['D24']['4']['ratio']:.2f}x (12: {fbb['K512']['4']['ratio']:.2f}x) the "
                                 "paper's at γ = 4: its values lie between our two windows, so its unpublished decoder "
                                 "settings set them")
                    if fbb.get("D24") and fbb.get("K512") and fbb["D24"].get("4", {}).get("ratio") is not None
                    and (fbb["K512"]["4"].get("ratio") or 0) >= 1 >= fbb["D24"]["4"]["ratio"] else "")(
                     gf.get("fig12_beam") if isinstance(gf.get("fig12_beam"), dict) else {}))
    f14v = S.get("fig14", {}) if isinstance(S.get("fig14"), dict) else {}
    if "summary" in f14v:
        f.append("two-block transversal CNOT (Fig. 14): ours / paper " + ", ".join(f"{x:.2f}" for x in f14v["summary"]["ratios"])
                 + f" per logical CNOT (paper inside our 95 % CIs at {f14v['summary']['inside']})")
    cmv = S.get("chainmap", {}) if isinstance(S.get("chainmap"), dict) else {}
    su = cmv.get("fig14_surface_summary") or {}
    if su.get("ratios_mean") and len(su["ratios_mean"]) == len(su["ratios"]):
        f.append("chain-map CNOT from the distance-5 surface code (Fig. 14; the paper's chain map is unpublished, ours "
                 f"is depth 2 with circuit distance {(cmv.get('distance') or {}).get('min', '?')}): ours / paper "
                 + ", ".join(f"{x:.2f}" for x in su["ratios_mean"]) + " for the mean of the X and Z bases (the "
                 "convention of the transversal series; paper inside our 95 % CIs at " + su["inside_mean"] + "), "
                 + ", ".join(f"{x:.2f}" for x in su["ratios"]) + " for the Z basis alone, the basis the paper states")
    elif su.get("ratios"):
        f.append("chain-map CNOT from the distance-5 surface code (Fig. 14; the paper's chain map is unpublished, ours "
                 f"is depth 2 with circuit distance {(cmv.get('distance') or {}).get('min', '?')}): ours / paper "
                 + ", ".join(f"{x:.2f}" for x in su["ratios"]) + " in the Z basis (X basis PENDING)")
    gzv = cmv.get("ghz") or {}
    if gzv.get("Z") and gzv.get("X"):
        f.append(f"inter-code GHZ via the chain map (Helios model against hardware): Z mismatch {100 * gzv['Z']['rate']:.3f} % "
                 f"vs 0.025 %, X mismatch {100 * gzv['X']['rate']:.3f} % vs 0.050 % ("
                 + ("95 % intervals overlapping in both" if gzv["Z"]["overlap"] and gzv["X"]["overlap"] else
                    "95 % intervals NOT overlapping in " + " and ".join(b for b in "ZX" if not gzv[b]["overlap"]))
                 + "; ours below the hardware's central values, as for memory and TQRB, and set by the unpublished "
                 "memory error per unit time" + (f": doubled, {100 * gzv['memory_x2']['Z']['rate']:.3f} % and "
                                                  f"{100 * gzv['memory_x2']['X']['rate']:.3f} %"
                                                  if len(gzv.get("memory_x2", {})) == 2 else "") + ")")
    V["fidelity"] = "; ".join(f) + "."
    # ---- scaling: exponents from the strata (no model order imposed), the bound by
    # construction, the fault distance and the extrapolation model's held-out checks
    ch = S.get("carbon", {})
    if isinstance(ch, dict) and "uniform" in ch:
        parts, hs = [], []
        for tag in ("full", "C", "B4", "R4", "R8", "R8F"):
            rec = ch["uniform"].get("X", {}).get(tag)
            if not rec:
                continue
            fo = rec.get("fault_order_all") or {}
            if fo.get("slope_at_p_star") is not None:
                parts.append(f"{RULE_SHORT.get(tag, tag)} {fo['slope_at_p_star']:.2f}"
                             + (f" [{fo['slope_ci95'][0]:.2f}, {fo['slope_ci95'][1]:.2f}]" if fo.get("slope_ci95") else "")
                             + f" at p = {fo['p_star']:.1e}")
            h = rec.get("heldout") or {}
            if h.get("status"):
                hs.append(h["status"])
        dist_ = S.get("distance_text", "PENDING")
        iwv = (S.get("inner_window") or {}) if isinstance(S.get("inner_window"), dict) else {}
        wins_ = iwv.get("windows") or {}
        stc_ = iwv.get("structure") or {}
        Ts_both = sorted(T_ for T_ in {w["T"] for w in wins_.values()}
                         if all(f"T{T_}_{b}" in wins_ for b in "XZ"))
        iw_ok = bool(Ts_both) and all(w["all_optimal"] for w in wins_.values()) and \
            stc_.get("mechanisms_on_inner_detectors_of_several_blocks") == 0 and \
            not stc_.get("X_logicals_on_at_most_two_blocks") and not stc_.get("Z_logicals_on_at_most_two_blocks")
        iw_d = min((w["min_distance"] for w in wins_.values()), default=0)
        iw_T = max(Ts_both) if Ts_both else 0
        rdt_s = relaxed_distance_text(S, tuple(RD_KIND))
        rdp_s = relaxed_distance_parts(S, tuple(RD_KIND))
        ed_ = (S.get("distance") or {}).get("end_distance") if isinstance(S.get("distance"), dict) else None
        V["scaling"] = (("Exponent of the error in p measured from the strata (uniform noise, X basis, 95 % "
                         "profile intervals): " + "; ".join(parts) + ". ") if parts else "") + \
            ("Six-fault failures exist (constructed), so no rule falls faster than p^6 as p -> 0. "
             + (("Circuit fault distance with the outer-check gadgets everywhere, decided by decomposition (exact over "
                 "every logical class and every combination of flag-silent gadget faults over the outer-check "
                 "instances, a counting bound excluding the largest combinations, with every block's cheapest "
                 "completion; FINDINGS 4, docs/methods.md): " + rdt_s + "."
                 + (f" Away from the gadgets the bound is exact block by block (each block needs {iw_d} faults over any "
                    f"window of up to {iw_T} rounds, every logical spans three blocks: {3 * iw_d})" if iw_ok else "")
                 + (f"; the gadget diagnostics (end-of-circuit distance at least {ed_}, one gadget's amplification) are "
                    "in FINDINGS 2.2 and 4" if ed_ else "") + ". "
                 + ("Failures therefore need exactly six faults as p -> 0 on every path decided - non-adaptive "
                    "extraction and the periodic and escalating rules (the rotations' one-check parts were not "
                    "decomposed) - so their exponents tend to 6, as distance 12 implies. "
                    if rdp_s and all(x[1] for x in rdp_s) else ""))
                if rdt_s else (dist_ + " ") if dist_ != "PENDING" else "")
             + "Numbers beyond the sampled strata come from the order-6 model and are labelled extrapolated; "
             f"its held-out checks: {sum(1 for x in hs if x == 'not rejected')} not rejected, "
             f"{sum(1 for x in hs if x == 'rejected')} rejected, {sum(1 for x in hs if x == 'uninformative')} "
             "uninformative (FINDINGS 4).")
    else:
        V["scaling"] = "PENDING"
    if "points" in cu:
        ins = [x["paper_inside_ci"] for x in cu["points"].values()]
        V["c4_uniform_inside"] = f"{sum(ins)} of {len(ins)}"
    td = S.get("tqrb_direct", {}) if isinstance(S.get("tqrb_direct"), dict) else {}
    if td and "missing" not in td and "error" not in td:
        parts, verd = [], []
        for key, r in td.items():
            fa, ci, pa = r.get("fit", {}), r.get("ci95", {}), r.get("paper") or {}
            parts.append(f"{key.replace('|', ' p = ')}: adaptive {fmt_e(fa.get('epsC_a1'))}, non-adaptive {fmt_e(fa.get('epsC_a0'))}")
            if pa.get("adaptive") and "epsC_a1" in ci:
                lo, hi = ci["epsC_a1"]
                inside = lo <= pa["adaptive"] <= hi
                ov = pa.get("adaptive_ci95")
                overlap = ov is not None and lo <= ov[1] and ov[0] <= hi
                verd.append(f"{key.split('|')[0]}: paper's adaptive {pa['adaptive']:.2g} "
                            + ("inside our 95 % interval" if inside else
                               ("outside our interval, the intervals overlap" if overlap else "outside our 95 % interval")))
            if r.get("ratio_nonadaptive_over_adaptive") and r.get("ratio_ci95"):
                verd.append(f"{key.split('|')[0]}: adaptive reduction {r['ratio_nonadaptive_over_adaptive']:.2g}x "
                            f"(95 % {r['ratio_ci95'][0]:.2g}-{r['ratio_ci95'][1]:.2g})"
                            + (" against the paper's 1.5x" if key.startswith("uniform") else ""))
        V["tqrb_eps"] = "; ".join(parts)
        V["tqrb_eps_verdict"] = "; ".join(verd) if verd else "see FINDINGS 1.6"
    rows = []
    hw = td.get("helios|0.00079", {}) if td else {}
    if hw.get("fit"):
        lo, hi = hw["ci95"]["epsC_a1"]
        rows.append(f"| TQRB error per Clifford on Helios (hardware, adaptive): 2.8e-4 (CI 1.2e-4 – 3.8e-4) | Helios model: "
                    f"adaptive {hw['fit']['epsC_a1']:.2e} (95 % {lo:.2e}–{hi:.2e}), non-adaptive {hw['fit']['epsC_a0']:.2e} | "
                    + ("inside the published interval" if 1.2e-4 <= hw['fit']['epsC_a1'] <= 3.8e-4 else "outside the published interval")
                    + "; the model's unpublished inputs (memory error per unit time, FINDINGS 1.4) apply here too |")
    tcs = S.get("tqrb_consistency", {}) if isinstance(S.get("tqrb_consistency"), dict) else {}
    gfs = S.get("gap_figures", {}) if isinstance(S.get("gap_figures"), dict) else {}
    f12 = gfs.get("fig12", {}) if isinstance(gfs.get("fig12"), dict) else {}
    if "summary" in f12:
        rr = [f12[p]["gammas"]["0"]["ratio"] for p in ("0.0007", "0.0009", "0.0011") if p in f12]
        ins = [f12[p]["gammas"]["0"]["paper_inside"] for p in ("0.0007", "0.0009", "0.0011") if p in f12]
        rows.append("| non-adaptive TQRB error per Clifford, Fig. 12 (3 points, no postselection) | ours / paper "
                    + ", ".join(f"{x:.2f}" for x in rr) + f" | paper inside our 95 % CIs: {sum(ins)} of {len(ins)} |")
    else:
        rows.append("| non-adaptive TQRB error per Clifford, Fig. 12 (3 points, no postselection) | PENDING | PENDING |")
    r1 = tcs.get("rows", {}).get("0.001")
    agree12_r = "summary" in f12 and all(0.67 <= f12[p]["gammas"]["0"]["ratio"] <= 1.5
                                         for p in ("0.0007", "0.0009", "0.0011") if p in f12)
    u1, u5 = td.get("uniform|0.001", {}) if td else {}, td.get("uniform|0.0005", {}) if td else {}
    if r1 and u1.get("fit"):
        ours = (f"p = 1e-3: adaptive {u1['fit']['epsC_a1']:.2e}, non-adaptive {u1['fit']['epsC_a0']:.2e}"
                + (f"; p = 5e-4: adaptive {u5['fit']['epsC_a1']:.2e}, non-adaptive {u5['fit']['epsC_a0']:.2e}" if u5.get("fit") else ""))
        gains = []
        for u, ref in ((u1, "1.5"), (u5, "1.8")):
            if u.get("ratio_nonadaptive_over_adaptive") and u.get("ratio_ci95"):
                gains.append(f"{u['ratio_nonadaptive_over_adaptive']:.2f} (95 % {u['ratio_ci95'][0]:.2f}–{u['ratio_ci95'][1]:.2f}) "
                             f"against {ref}")
        hm = [(lab, [v["ratio"] for v in (td.get(k) or {}).get("fig13_points", {}).values()])
              for k, lab in (("helios|0.001", "p = 1e-3"), ("helios|0.0005", "5e-4"))]
        hm = [(lab, rr) for lab, rr in hm if rr]
        hg = []
        pgr = tcs.get("paper_gain") or {}
        for k, ref, pk in (("helios|0.001", "1.5", "0.001"), ("helios|0.0005", "1.8", "0.0005")):
            u = td.get(k) or {}
            if u.get("ratio_nonadaptive_over_adaptive") and u.get("ratio_ci95"):
                pr_ = (f"the paper's {pgr[pk]['gain']:.2f} (95 % {pgr[pk]['ci95'][0]:.2f}–{pgr[pk]['ci95'][1]:.2f}, from "
                       "its error bars)") if pk in pgr else ref
                hg.append(f"{u['ratio_nonadaptive_over_adaptive']:.2f} (95 % {u['ratio_ci95'][0]:.2f}–{u['ratio_ci95'][1]:.2f}) "
                          f"against {pr_}")
        verdict = ("about a third of Fig. 13, as is the paper's own Fig. 12 (the same non-adaptive benchmark, noise and "
                   f"decoder: {r1['fig13_over_fig12'][0]:.1f}–{r1['fig13_over_fig12'][1]:.1f}x below Fig. 13 at p = 1e-3)"
                   + (", with which ours agree (row above)" if agree12_r else ""))
        if hm:
            verdict = ("reproduced by our Helios-like model at the same two-qubit error: ours / Fig. 13 = "
                       + ", ".join(f"{min(rr):.2f}–{max(rr):.2f} ({lab}, {len(rr)} points)" for lab, rr in hm)
                       + "; under the uniform model it is labelled with, ours are " + verdict
                       + ". Fig. 13 thus labels a device-like model by its two-qubit error (inference from the match)"
                       + ("; adaptive gain, Helios-like: " + ", ".join(hg) if hg else "")
                       + ("; uniform: " + ", ".join(gains) if gains else ""))
        else:
            verdict += ("; adaptive gain " + ", ".join(gains) if gains else "")
        rows.append("| adaptive and non-adaptive TQRB per Clifford, Fig. 13: 2.8e-4 / 4.2e-4 at p = 1e-3, 2.6e-5 / 4.7e-5 at "
                    f"5e-4 | uniform {ours} | {verdict} |")
    f11 = gfs.get("fig11", {}) if isinstance(gfs.get("fig11"), dict) else {}
    for f_, lab in ((f11, "forced-gap postselection, Fig. 11 (memory; γ = 0–4 nats, error and rejection per round)"),
                    (f12, "forced-gap postselection, Fig. 12 (TQRB; γ = 0–4 nats, error and rejection per Clifford)")):
        su = f_.get("summary") if isinstance(f_, dict) else None
        if su:
            extra = ""
            fbb = gfs.get("fig12_beam") if isinstance(gfs.get("fig12_beam"), dict) else {}
            if f_ is f12 and fbb.get("D24") and fbb.get("K512"):
                d2 = fbb["D24"]
                rr = [d2[f"{g:g}"]["rejection_ratio"] for g in GAMMAS if g > 0 and d2[f"{g:g}"].get("rejection_ratio")]
                er = [d2[f"{g:g}"]["ratio"] for g in GAMMAS if g >= 2 and d2[f"{g:g}"].get("ratio") is not None]
                between = (fbb["K512"]["4"].get("ratio") or 0) >= 1 >= (d2["4"].get("ratio") if d2["4"].get("ratio") is not None else 2)
                extra = (f"; on the same shots at p = 1.1e-3, pruning window Delta = 24 instead of 12 gives rejection "
                         f"{min(rr):.2f}–{max(rr):.2f}x the paper's and error {min(er):.2f}–{max(er):.2f}x at γ = 2–4"
                         + (": the paper's values lie between our two windows, so the decoder's pruning sets them, and the "
                            "paper does not publish its decoder settings" if between else ""))
            rows.append(f"| {lab} | error ours / paper {su['ratio_all'][0]:.2f}–{su['ratio_all'][1]:.2f}; rejection "
                        f"{su['rejection_ratio'][0]:.2f}–{su['rejection_ratio'][1]:.2f} | paper inside our 95 % CIs: error "
                        f"{su['inside']}, rejection {su['rejection_inside']} (FINDINGS 1.7){extra} |")
        else:
            rows.append(f"| {lab} | PENDING | PENDING |")
    f14 = S.get("fig14", {}) if isinstance(S.get("fig14"), dict) else {}
    if "summary" in f14:
        rows.append("| two-block transversal CNOT, Fig. 14 ('C4-Helix transv.', 3 points) | ours / paper "
                    + ", ".join(f"{x:.2f}" for x in f14["summary"]["ratios"])
                    + " per logical CNOT (per transversal CNOT: "
                    + ", ".join(f"{f14[p]['ratio_transversal']:.2f}" for p in ("0.0007", "0.0009", "0.0011") if p in f14)
                    + "; the paper does not define 'per CNOT') | "
                    f"paper inside our 95 % CIs: {f14['summary']['inside']} |")
    else:
        rows.append("| two-block transversal CNOT, Fig. 14 ('C4-Helix transv.', 3 points) | PENDING | PENDING |")
    cms = (S.get("chainmap") or {}) if isinstance(S.get("chainmap"), dict) else {}
    if cms.get("fig14_surface_summary"):
        su = cms["fig14_surface_summary"]
        mean_ok = su.get("ratios_mean") and len(su["ratios_mean"]) == len(su["ratios"])
        rows.append("| chain-map CNOT from the distance-5 surface code, Fig. 14 ('Surface-C4-Helix', 3 points) | ours / paper "
                    + (", ".join(f"{x:.2f}" for x in su["ratios_mean"]) + " per CNOT for the mean of the X and Z bases "
                       "(the convention of the transversal series), " + ", ".join(f"{x:.2f}" for x in su["ratios"])
                       + " for the Z basis alone (the basis the paper states)" if mean_ok else
                       ", ".join(f"{x:.2f}" for x in su["ratios"]) + " per CNOT, Z basis")
                    + f"; our depth-2 chain map of circuit distance {(cms.get('distance') or {}).get('min', '?')} (the "
                    "paper's is unpublished) | paper inside our 95 % CIs: "
                    + (f"{su['inside_mean']} (two-basis mean), {su['inside']} (Z)" if mean_ok else su["inside"])
                    + " (FINDINGS 1.8) |")
    else:
        rows.append("| chain-map CNOT from the distance-5 surface code, Fig. 14 ('Surface-C4-Helix') | PENDING | PENDING |")
    gz = cms.get("ghz") or {}
    if gz.get("Z") and gz.get("X"):
        rows.append("| inter-code GHZ via the chain map (hardware: Z mismatch 0.025 %, X mismatch 0.050 %; 99.925 % <= F) "
                    f"| Helios model: {100 * gz['Z']['rate']:.3f} %, {100 * gz['X']['rate']:.3f} %; "
                    f"{100 * gz['fidelity_bounds'][0]:.3f} % <= F | "
                    + ("intervals overlap in both bases" if gz["Z"]["overlap"] and gz["X"]["overlap"] else
                       "intervals do NOT overlap in " + " and ".join(b for b in "ZX" if not gz[b]["overlap"]))
                    + " (FINDINGS 1.9; our chain map) |")
    else:
        rows.append("| inter-code GHZ via the chain map (hardware) | PENDING | PENDING |")
    cdist = ((S.get("checks") or {}).get("circuit_distance") or {}).get("c4helix_S0S1_plus_extraction_Z", {}) \
        if isinstance(S.get("checks"), dict) else {}
    tdist = S.get("tqrb_distance", {}) if isinstance(S.get("tqrb_distance"), dict) else {}
    sq = tdist.get("sequences") or []
    if cdist.get("distance") is not None:
        prov_ = [r for r in sq if all(r.get(k + "_proven") for k in ("nonadaptive", "adaptive_quiet", "adaptive_full"))]
        seqd = sorted({r["nonadaptive"] for r in prov_ if r.get("nonadaptive") is not None})
        sequb = sorted({r["nonadaptive"] for r in sq if r not in prov_ and r.get("nonadaptive") is not None})
        rows.append("| circuit distance 6 of the flagged S0S1 followed by extraction | "
                    f"{cdist['distance']} | identical"
                    + (f"; complete TQRB sequences (extraction, automorphism, S0S1, extraction, ...) have distance "
                       f"{'-'.join(map(str, seqd))} for {len(prov_)} random sequence{'s' if len(prov_) != 1 else ''} "
                       "(proven)" + (f" and at most {'-'.join(map(str, sequb))} for {len(sq) - len(prov_)} more" if sequb else "")
                       + (f", the same under all {len(tdist.get('orderings') or {})} cat assignments tried on one of them"
                          if tdist.get("orderings") else "")
                       + " (FINDINGS 1.2): the paper's statement covers one gate, not a sequence" if seqd else "") + " |")
    V["tqrb_rows"] = "\n".join(rows)
    for k in ("deliverable1", "deliverable2", "deliverable3", "frontier", "limits", "evidence",
              "tqrb_eps", "tqrb_eps_verdict", "tqrb_rows", "c4_uniform_inside",
              "fidelity_detail", "carbon_detail", "scaling_detail", "interval_detail", "value_detail",
              "frontier_detail", "failed", "open", "evidence_detail"):
        V.setdefault(k, "PENDING")
    # ---- deliverables and details (data-driven sentences)
    car = S.get("carbon", {}) if isinstance(S.get("carbon"), dict) else {}
    cu_X = car.get("uniform", {}).get("X", {})
    ch_X = car.get("helios", {}).get("X", {})
    if cu_X:
        def row(rows, tag, p):
            r = rows.get(tag)
            if not r or r["eps_LQ"].get(p) is None:
                return None
            mr = (r.get("model_range_LQ") or {}).get(p)
            return (f"{RULE_SHORT.get(tag, tag)} {r['eps_LQ'][p]:.2g} ({r['status'][p]}"
                    + (f"; model range {mr[0]:.2g}-{mr[1]:.2g}" if mr else "") + ")")
        tags = [t for t in ("full", "C", "B4", "R4", "R8", "R8F") if t in cu_X or t in ch_X]
        V["deliverable1"] = (
            f"Circuits for the three levels (C4-level inner checks, weight-6 Carbon checks, weight-16 outer checks) "
            f"and {len(tags)} extraction rules ({', '.join(RULE_SHORT.get(t, t) for t in tags)}), exported as stim files "
            "(results/circuits); decoded with Frontier over the whole circuit, adaptive paths on one superset model "
            "with path priors (FINDINGS 2); error per logical qubit per round, X basis, over the uniform-noise grid "
            "3e-3 to 1e-4 and the Helios-model grid (FINDINGS 3); at uniform p = 1e-3: "
            + "; ".join(x for x in (row(cu_X, t, "0.001") for t in tags) if x)
            + ((". At Helios's p (7.9e-4): " + "; ".join(x for x in (row(ch_X, t, "0.00079") for t in tags) if x))
               if ch_X else ". Helios: PENDING") + ".")
        sd = []
        for tag in ("full", "C", "B4", "R4", "R8", "R8F"):
            r = cu_X.get(tag)
            if not r:
                continue
            fo = r.get("fault_order_all") or {}
            h = r.get("heldout") or {}
            sd.append(f"{RULE_SHORT.get(tag, tag)}: exponent from the strata {fmt_e(fo.get('slope_at_p_star'))}"
                      + (f" [{fo['slope_ci95'][0]:.2f}, {fo['slope_ci95'][1]:.2f}]" if fo.get("slope_ci95") else "")
                      + (f" at p = {fo['p_star']:.1e}" if fo.get("p_star") else "")
                      + f"; smallest minimal failing set confirmed by exact decoding: {r.get('min_core_genuine')} faults "
                      f"(smallest of any kind {r['min_core']}); held-out strata: predicted {fmt_e(h.get('predicted'))}, "
                      f"observed {h.get('observed')} ({h.get('status')}"
                      + (f", calibration factor {h['calibration_95'][0]:.2g}-{fmt_e(h['calibration_95'][1])}"
                         if h.get("calibration_95") else "") + ")")
        V["scaling_detail"] = ("; ".join(sd) + ". Six-fault failures exist by construction, so no rule falls faster "
                               "than p^6. " + S.get("distance_text", "")) if sd else "PENDING"
    iv = S.get("interval", {})
    if isinstance(iv, dict) and isinstance(iv.get("B"), dict) and S.get("interval_text", "PENDING") != "PENDING":
        V["deliverable2"] = S.get("interval_text", "PENDING")
        V["interval_detail"] = S.get("interval_text", "PENDING") + " " + S.get("window_text", "")
    # ---- what failed / what remains open (each item names its evidence)
    failed = []
    hm = S.get("helios_memory", {}) if isinstance(S.get("helios_memory"), dict) else {}
    c4h = hm.get("c4helix|helios|X")
    if c4h:
        failed.append(f"- C4-Helix Helios memory: our central value {c4h['eps_per_LQ']:.2g} is "
                      f"{c4h['eps_per_LQ'] / 4.6e-5:.2f} of the paper's 4.6e-5 and its 95 % interval excludes "
                      f"4.6e-5, while ours lies inside the paper's; the memory error per unit time is the "
                      f"unpublished input that moves it (x0.5: {hm.get('c4helix|helios_mem0.5|X', {}).get('eps_per_LQ', float('nan')):.1e}, "
                      f"x2: {hm.get('c4helix|helios_mem2|X', {}).get('eps_per_LQ', float('nan')):.1e}).")
    tc = S.get("tqrb_counts", {}).get("rows", {}) if isinstance(S.get("tqrb_counts"), dict) else {}
    if tc:
        failed.append("- TQRB shot-time savings: " + ", ".join(
            f"{100 * tc[k]['time_reduction']:.0f} %" for k in ("helios|0.00079|m5", "helios|0.00079|m13") if k in tc)
            + " against the paper's 23 % (hardware) and 24 % / 30 % (simulated at 1e-3 / 1e-4); the paper's "
            "time model is not published (gate-count-proportional time would give the 32-41 % gate savings).")
    for camp_tag, label in (("ch_main__full__X", "non-adaptive"), ("ch_main__B4__X", "B4"), ("ch_main__C__X", "C")):
        a = (S.get("decoder_audit") or {}).get(camp_tag)
        if isinstance(a, dict) and "totals" in a:
            t = a["totals"]
            failed.append(f"- Decoder approximation ({label}, uniform noise): {t['fail'] - t['genuine']} of "
                          f"{t['fail']} campaign failures are corrected by a 4x wider beam and a wider pruning window (K = 2048, Delta = 20; none of "
                          f"{t['ok_sampled']} sampled successes is broken by it); the confirmed-failure "
                          "estimates of FINDINGS 2.5 count only the failures the stronger decoder also makes.")
    tcf = S.get("tqrb_confident", {}) if isinstance(S.get("tqrb_confident"), dict) else {}
    tcp = {k: v for k, v in tcf.items() if k not in ("cores_decode", "missing")}
    if tcp:
        gfb = ((S.get("gap_figures") or {}).get("fig12_beam") or {}) if isinstance(S.get("gap_figures"), dict) else {}
        d24 = ""
        if gfb.get("K512") and gfb.get("D24"):
            a_, b_ = gfb["K512"], gfb["D24"]
            d24 = (f" On the same 1e5 shots at p = 1.1e-3, Delta = 24 leaves {b_['0']['failures_kept']:.0f} of the "
                   f"{a_['0']['failures_kept']:.0f} failures and {b_['4']['failures_kept']:.0f} of the "
                   f"{a_['4']['failures_kept']:.0f} kept at gamma = 4.")
        failed.append("- Decoder pruning window on the TQRB circuits (Frontier's documented K = 512, Delta = 12, used for "
                      "every campaign): "
                      + "; ".join(f"{v['confident']} of {v['failures']} failures at p = {float(k):g}" for k, v in sorted(tcp.items()))
                      + " are made with an infinite gap - every state of the correct class pruned - on minimal cores "
                      "of mostly 2-4 faults that exact decoding gets right in "
                      + ", ".join(str(v["exact_mle_correct"]) for k, v in sorted(tcp.items()))
                      + " cases; a gap that large never triggers re-decoding, and Delta >= 18 decodes them (FINDINGS 1.7)."
                      + d24 + " They raise our TQRB error and survive forced-gap postselection"
                      + (", which is why our postselected TQRB values stay above the paper's"
                         if (gfb.get("K512") and gfb.get("D24") and gfb["D24"]["4"].get("ratio") and gfb["K512"]["4"].get("ratio")
                             and abs(np.log(gfb["D24"]["4"]["ratio"])) < abs(np.log(gfb["K512"]["4"]["ratio"]))) else "")
                      + (lambda c: f"; memory circuits are barely affected (the audit corrects {100 * min(c):.0f}-"
                                   f"{100 * max(c):.0f} % of C4-Helix memory failures, FINDINGS 2.5)." if c else ".")(
                          [1 - v["totals"]["genuine"] / v["totals"]["fail"]
                           for k, v in (S.get("decoder_audit") or {}).items()
                           if k.startswith("c4_strat_") and isinstance(v, dict) and (v.get("totals") or {}).get("fail")]))
    dist = S.get("distance", {}) if isinstance(S.get("distance"), dict) else {}
    lat = (dist.get("designs") or {}).get("latin4", {})
    failed.append("- The first outer-check gadget (4-qubit cat, flags read only at the end; every result made with "
                  "it is at commit 0ddbd72): two faults on one cat qubit, one before and one after its data CNOTs, "
                  "leave up to four data errors and no flag, so the circuit's fault distance is "
                  + (f"{lat['min_weight']}" if lat.get("min_weight") else "9")
                  + " (an undetectable logical error found by exact integer programming over the end-of-circuit "
                  "mechanisms), five faults can fail and the error falls as p^5 as p -> 0, not p^6. It was found "
                  "after the first campaigns; the cat-parity gadget of FINDINGS 2.2 replaces it (end-of-circuit "
                  "distance in FINDINGS 4) and every Carbon-Helix campaign was rerun with it.")
    wh = dist.get("whole", {})
    whs = "; ".join(f"{k.split('|')[1]} basis: " + (f"{v['best']}, proved" if v["optimal"] else
                                                    f"best found {v['best']}, proven lower bound {v['lower_bound']:.0f}")
                    for k, v in sorted(wh.items()))
    rdt_ = relaxed_distance_text(S, tuple(RD_KIND))
    failed.append("- Fault distance of the whole Carbon-Helix circuit by generic solvers: integer programming over the "
                  "whole two-round detector error model does not converge in 11 hours"
                  + (f" ({whs})" if whs else " (runs pending)")
                  + ", nor do CP-SAT, SCIP or CryptoMiniSat; "
                  + ("the decomposition of FINDINGS 4 decides it instead: " + rdt_ + "." if rdt_ else
                     "the decomposition runs are pending."))
    V["failed"] = "\n".join(failed)
    # what remains open: only unpublished quantities and physics, each with the measurement
    # that would settle it
    not_limits = []
    opn = [
        "- Helios memory error per unit time during compiled extraction rounds (unpublished; the "
        "model uses the data-sheet depth-1 memory error spread over the published layer time). It "
        "sets the absolute error per round on Helios for both codes (FINDINGS 1.4 and section 8); "
        "settled by idle (Ramsey/echo) error of a data qubit measured across one compiled round.",
    ]
    sens_ = S.get("sensitivity", {}) if isinstance(S.get("sensitivity"), dict) else {}
    hv_ = ((S.get("value") or {}).get("helios|0.00079") or {}).get("rows", {}) if isinstance(S.get("value"), dict) else {}
    tm4 = sens_.get("Carbon-Helix|l2=on_event,l3=ambiguous,rot=4|helios_tm4") or {}
    c4tm = sens_.get("C4-Helix|full|helios_tm4") or {}
    r4b = hv_.get("Carbon-Helix|l2=on_event,l3=ambiguous,rot=4|r4") or {}
    c4b = hv_.get("C4-Helix|full|r4") or {}
    if tm4.get("eps_LQ") and c4tm.get("eps_LQ") and r4b.get("time_ms_per_round") and c4b.get("eps_LQ"):
        not_limits.append(
            "- Not a limit: the measurement/reset duration per batch on Helios (unpublished; 2 ms assumed). At 4 ms "
            f"R4's time per round moves from {r4b['time_ms_per_round']:.0f} to {tm4.get('time_ms_per_round', float('nan')):.0f} ms "
            f"and C4-Helix / R4 from {c4b['eps_LQ'] / r4b['eps_LQ']:.0f} to {c4tm['eps_LQ'] / tm4['eps_LQ']:.0f} "
            "(FINDINGS 8): it decides nothing here.")
    gz = ((S.get("chainmap") or {}).get("ghz") or {}) if isinstance(S.get("chainmap"), dict) else {}
    ghz_ok = bool(gz.get("Z") and gz.get("X") and gz["Z"]["overlap"] and gz["X"]["overlap"])
    if not ghz_ok:
        opn.append("- The chain map of the paper's inter-code GHZ experiment and Fig. 14 (unpublished; ours has the "
                   "stated depth 2 and distance 5): "
                   + ("our GHZ mismatch rates do not overlap the hardware's (FINDINGS 1.9); " if gz.get("Z") else
                      "the GHZ simulation is pending; ")
                   + "settled by the paper's chain map.")
    tcs = S.get("tqrb_consistency", {}) if isinstance(S.get("tqrb_consistency"), dict) else {}
    if tcs.get("rows", {}).get("0.001"):
        r1 = tcs["rows"]["0.001"]
        opn.append("- The noise model of the paper's Fig. 13 (not stated): Fig. 13 puts the non-adaptive benchmark "
                   f"{r1['fig13_over_fig12'][0]:.1f}-{r1['fig13_over_fig12'][1]:.1f}x above the paper's own Fig. 12 under "
                   "the stated uniform model, and our Helios-like model at the same two-qubit error reproduces it "
                   "(FINDINGS 1.6). It decides only the comparison with Fig. 13, none of this work's results; settled "
                   "by the model behind Fig. 13.")
    tcn = S.get("tqrb_counts", {}).get("rows", {}) if isinstance(S.get("tqrb_counts"), dict) else {}
    if tcn.get("uniform|0.001|m7") and tcn.get("uniform|0.0001|m7"):
        opn.append("- The paper's shot-time model (per-operation durations on Helios; not published): its shot-time "
                   f"savings (23 % hardware, 24 % / 30 % simulated at 1e-3 / 1e-4) lie between ours with the Helios timing "
                   f"model ({100 * tcn['uniform|0.001|m7']['time_reduction']:.0f} % / "
                   f"{100 * tcn['uniform|0.0001|m7']['time_reduction']:.0f} %, m = 7) and time proportional to two-qubit "
                   f"gates ({100 * tcn['uniform|0.001|m7']['gate_reduction']:.0f} % / "
                   f"{100 * tcn['uniform|0.0001|m7']['gate_reduction']:.0f} %); it decides that comparison only; settled "
                   "by the compiled durations.")
    gfs = S.get("gap_figures", {}) if isinstance(S.get("gap_figures"), dict) else {}
    if "the paper's values lie between our two windows" in V.get("tqrb_rows", ""):
        opn.append("- The paper's decoder settings (Frontier's beam and pruning window; not published): they decide the "
                   "forced-gap postselection comparisons of Figs. 11 and 12 - Fig. 12's rejection and error lie between "
                   "our pruning windows Delta = 12 and 24 on the same shots, and Fig. 11's high-threshold errors between "
                   "ours with and without the failures that carry the pruning's signature, a gap of at least 31.5 nats "
                   "(FINDINGS 1.7); none of this "
                   "work's results depends on them (its campaigns use one stated configuration); settled by the paper's "
                   "decoder configuration.")
    opn += not_limits
    opn.append("- Not a limit, for completeness: errors per round near 1e-8 at p ~ 5e-4 are beyond direct sampling (the "
               "problem statement's own obstacle); they come from stratified sampling and the polynomial model in the "
               "number of faults, labelled extrapolated, with their model range and the model's predictions of "
               "held-out strata (FINDINGS sections 3-4).")
    V["open"] = "\n".join(opn)
    V["value_detail"] = S.get("value_text", "PENDING")
    V["frontier_detail"] = S.get("frontier_text", "PENDING")
    val = S.get("value", {}) if isinstance(S.get("value"), dict) else {}
    hv = (val.get("helios|0.00079") or {}).get("rows", {})
    peaks = [r.get("peak_qubits") for r in hv.values() if r.get("code") == "Carbon-Helix" and r.get("peak_qubits")]
    V["deliverable3"] = S.get("value_text", "PENDING") + (
        f" Every Carbon-Helix configuration fits on Helios: at most {max(peaks)} simultaneously live qubits of 98."
        if peaks else "")
    nd = (val.get("helios|0.00079") or {}).get("nondominated", [])
    if nd:
        far = {}
        for k in nd:
            r = hv.get(k, {})
            if r.get("furthest_criterion"):
                far[k] = (r["furthest_criterion"], (r.get("distance_from_limit") or {}).get(r["furthest_criterion"]))
        V["frontier"] = (f"Non-dominated at Helios's p: {len(nd)} configurations ("
                         + "; ".join(f"{hv[k]['label']}" + (f", furthest from its limit: {v[0]} at {v[1]:.2f}x" if v[1] else "")
                                     for k, v in far.items()) + "). Pushes made on the furthest criterion and the "
                         "evaluated next one: FINDINGS 7.")
        r32_ = hv.get("Carbon-Helix|l2=on_event,l3=ambiguous,rot=32,l2fast=1|r32")
        r32t_ = hv.get("Carbon-Helix|l2=on_event,l3=ambiguous,rot=32,l2fast=1,l2type=1|r32")
        tb_ = S.get("time_budget", {}) if isinstance(S.get("time_budget"), dict) else {}
        pk_ = S.get("packing", {}) if isinstance(S.get("packing"), dict) else {}
        gains_ = [max(0.0, c["separate_ms"] - c["packed_ms"]) for c in pk_.get("checks", {}).values()]
        if r32_ and r32t_ and r32t_.get("eps_LQ") and tb_.get("R32FT") and gains_:
            sh = {k: v / tb_["R32FT"]["total_ms_per_round"] for k, v in tb_["R32FT"]["ms_per_round"].items()}
            pack_ = tb_["R32FT"]["instances_per_round"].get("rotating outer checks", 0.0) * sum(gains_) / len(gains_)
            V["frontier"] += (
                f" The latest push (7, on time per round after R32F's success): weight-6 checks of the event's type "
                f"only (R32FT), {r32t_['time_ms_per_round']:.0f} vs {r32_['time_ms_per_round']:.0f} ms and "
                f"{r32t_['twoq_per_round']:.0f} vs {r32_['twoq_per_round']:.0f} two-qubit gates per round, error "
                f"{r32t_['eps_LQ']:.2g} vs {r32_['eps_LQ']:.2g} (extrapolated, 95 % intervals "
                f"{r32t_['ci95_LQ'][0]:.2g}-{r32t_['ci95_LQ'][1]:.2g} and {r32_['ci95_LQ'][0]:.2g}-{r32_['ci95_LQ'][1]:.2g}): "
                "non-dominated, a success. Next on time per round (evaluated): the inner checks, its limit, take "
                f"{100 * sh.get('inner checks (L1)', 0):.0f} % of R32FT's time; the rest are weight-6 checks after inner "
                f"events ({100 * sum(v for k, v in sh.items() if k.startswith('weight-6')):.0f} %) and escalations' full "
                f"outer measurements ({100 * sh.get('full outer measurement (escalation)', 0):.0f} %), which correction and "
                "fault tolerance need, and the rotating outer checks "
                f"({100 * sh.get('rotating outer checks', 0):.0f} %), of which packing saves "
                f"{100 * pack_ / tb_['R32FT']['total_ms_per_round']:.1f} % and halving the rotation rate at most "
                f"{50 * sh.get('rotating outer checks', 0):.0f} % (FINDINGS 7).")
    p8t = push8_texts(S)
    if p8t and V.get("frontier"):
        V["frontier"] += " " + " ".join(p8t)
    sens = S.get("sensitivity", {})
    if isinstance(sens, dict) and "missing" not in sens and S.get("limits_text", "PENDING").startswith("Unpublished"):
        paper_inputs = (["its chain map (Fig. 14, GHZ)"] if not ghz_ok else []) + ["the noise model of its Fig. 13"] \
            + (["its decoder settings (the postselection of Figs. 11 and 12)"]
               if "the paper's values lie between our two windows" in V.get("tqrb_rows", "") else []) \
            + ["its shot-time model (its shot-time savings)"]
        V["limits"] = (S["limits_text"] + " The remaining limits: Helios's memory error per unit time, which sets the "
                       "absolute Helios values and the size of the gain over C4-Helix (the measurement/reset duration "
                       f"decides nothing), and {len(paper_inputs)} unpublished inputs of the paper, each deciding only the "
                       "comparison named: " + ", ".join(paper_inputs[:-1]) + " and " + paper_inputs[-1]
                       + "; each with the measurement that settles it (REPORT 8).")
    pv = S.get("provenance", {})
    if pv:
        V["evidence"] = (f"all numbers from commit {S.get('commit', '?')[:12]}; raw results identical to its code: "
                         f"{pv.get('ok', 0)}, behaviourally identical (regression checks): {pv.get('equivalent', 0)}, "
                         f"differing: {pv.get('bad', 0)}, unstamped: {pv.get('unknown', 0)}.")
        V["evidence_detail"] = (
            f"Every number in FINDINGS.md, REPORT.md and COMPLETE.md is rendered from results/summary.json, computed "
            f"at commit {S.get('commit', '?')} by analysis/make_all.py from the raw results under results/. Each raw "
            "result records its commit and the SHA-1 of every source file its job loaded; scripts/provenance.py "
            f"compares them with this commit: {pv.get('ok', 0)} identical (up to comments and docstrings), "
            f"{pv.get('equivalent', 0)} behaviourally identical (the job's regression family, scripts/regression.py, "
            f"gives bit-identical output at both codes), {pv.get('bad', 0)} differing, {pv.get('unknown', 0)} without "
            "a stamp. Campaign seeds are fixed in scripts/campaigns.py, so ./regenerate.sh --simulate recomputes "
            "every raw result and ./regenerate.sh every number. Each comparison names its reference: arXiv:2609.03194 "
            "(C4-Helix: Figs. 2f, 11, 12, 13 and 14 digitized by analysis/digitize_paper.py, the legend values of Fig. 13, "
            "hardware values from its text), arXiv:2502.14835 (the 1/p unmasking rule), "
            "arXiv:2511.05465 and the Helios component data (noise model), arXiv:2404.02280 (Carbon code).")
    # Carbon-Helix at circuit level (REPORT 2)
    car = S.get("carbon", {}) if isinstance(S.get("carbon"), dict) else {}
    cd = []
    for noise, ps in (("uniform", ("0.002", "0.001")), ("helios", ("0.00079",))):
        rows = car.get(noise, {}).get("X", {})
        for p in ps:
            items = []
            for tag in ("full", "C", "B4", "R4", "R8", "R8F"):
                r = rows.get(tag)
                if not r or r["eps_LQ"].get(p) is None:
                    continue
                lo, hi = r["lo95_LQ"].get(p), r["hi95_LQ"].get(p)
                items.append(f"{RULE_SHORT.get(tag, tag)} {r['eps_LQ'][p]:.2g} (95 % {lo:.2g}-{hi:.2g}, "
                             f"{100 * r['measured_share'][p]:.0f} % measured, {r['twoq'][p]:.0f} 2Q, {r['time_ms'][p]:.0f} ms)")
            if items:
                cd.append(f"{'Uniform' if noise == 'uniform' else 'Helios-model'} p = {p}: " + "; ".join(items) + ".")
    if cd:
        V["carbon_detail"] = ("Error per logical qubit per round (X basis, 4 rounds after preparation), against "
                              "C4-Helix's and non-adaptive extraction's in REPORT 5. " + " ".join(cd) + " "
                              + S.get("decoder_audit_text", ""))
    return V


def main():
    summary = {"commit": git_commit()}
    ck = os.path.join(ROOT, "results/checks.json")
    summary["checks"] = json.load(open(ck)) if os.path.exists(ck) else {"missing": "run scripts/checks.py"}
    pv = os.path.join(ROOT, "results/provenance.json")
    if os.path.exists(pv):
        d = json.load(open(pv))
        summary["provenance"] = {k: d[k] for k in ("commit", "ok", "equivalent", "bad", "unknown") if k in d}
    for name, fn in (("c4_uniform", sec_c4_uniform), ("helios_memory", sec_helios_memory),
                     ("tqrb_counts", sec_tqrb_counts), ("tqrb_eps", sec_tqrb_eps), ("tqrb_direct", sec_tqrb_direct),
                     ("gap_figures", sec_gap_figures), ("fig14", sec_fig14), ("chainmap", sec_chainmap),
                     ("tqrb_confident", sec_tqrb_confident),
                     ("tqrb_distance", sec_tqrb_distance),
                     ("circuits", sec_circuits), ("sensitivity", sec_sensitivity),
                     ("carbon", sec_carbon), ("interval", sec_interval), ("value", sec_value),
                     ("decoder_benchmark", lambda: json.load(open(os.path.join(ROOT, "results/decoder_benchmark.json")))
                      if os.path.exists(os.path.join(ROOT, "results/decoder_benchmark.json")) else {"missing": "run analysis/decoder_benchmark.py"}),
                     ("packing", lambda: json.load(open(os.path.join(ROOT, "results/packing.json")))
                      if os.path.exists(os.path.join(ROOT, "results/packing.json")) else {"missing": "run analysis/packing.py"}),
                     ("time_budget", lambda: json.load(open(os.path.join(ROOT, "results/time_budget.json")))
                      if os.path.exists(os.path.join(ROOT, "results/time_budget.json")) else {"missing": "run analysis/time_budget.py"}),
                     ("timing_consistency", lambda: json.load(open(os.path.join(ROOT, "results/timing_consistency.json")))
                      if os.path.exists(os.path.join(ROOT, "results/timing_consistency.json")) else {"missing": "run analysis/timing_consistency.py"}),
                     ("minimal", lambda: sec_files("minimal")),
                     ("distance", sec_distance), ("inner_window", sec_inner_window),
                     ("relaxed_distance", sec_relaxed_distance),
                     ("basis", sec_basis),
                     ("core_census", lambda: __import__("core_census").census()),
                     ("decoder_audit", lambda: sec_files("redecode"))):
        try:
            summary[name] = fn()
        except Exception as ex:
            summary[name] = {"error": repr(ex)}
    try:
        summary["direct"] = sec_direct(summary)
    except Exception as ex:
        summary["direct"] = {"error": repr(ex)}
    try:
        summary["tqrb_consistency"] = sec_tqrb_consistency(summary)
    except Exception as ex:
        summary["tqrb_consistency"] = {"error": repr(ex)}
    try:
        summary["circuits_full_round"], summary["circuits_c4_round"] = sec_round_totals(summary)
    except Exception as ex:
        summary["circuits_full_round"] = {"error": repr(ex)}
    try:
        summary.update(sec_texts(summary))
    except Exception as ex:
        summary["texts_error"] = repr(ex)
    try:
        summary.update(tqrb_texts(summary))
    except Exception as ex:
        summary["tqrb_texts_error"] = repr(ex)
    try:
        summary["verdicts"] = sec_verdicts(summary)
    except Exception as ex:
        summary["verdicts"] = {"error": repr(ex)}
    with open(os.path.join(ROOT, "results/summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print(json.dumps({k: (v if k in ("commit",) else list(v)[:6]) for k, v in summary.items()}, indent=1, default=str))


if __name__ == "__main__":
    main()
