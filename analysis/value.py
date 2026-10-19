"""Value of adaptive extraction: logical error, two-qubit gates, time and qubits per round for
every configuration tried, at fixed noise points, against non-adaptive extraction of the
same code and against C4-Helix; and the non-dominated set over all of them.

Logical error per round is the X-basis block error per round divided by the number of
logical qubits (k = 2), the convention of arXiv:2609.03194 (its hardware memory experiment
samples only the X basis, the worse one under dephasing). Each configuration is one
stratified data set (analysis/strat.py); values are 'measured' when at least 90 % of the
estimate comes from strata with >= 3 failures, otherwise the polynomial model supplies the
rest and they are labelled 'extrapolated' (model checked on held-out strata).

Resources per round are expectations over the executed paths at that noise point (direct
path sampling, strat.path_resources), after the preparation round: two-qubit gates,
Helios-model wall-clock time; peak simultaneously live qubits is the maximum over every
path the campaign executed.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from strat import load, StratModel, PolyModel, hybrid_boot, boot_summary, path_resources  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MIN_FAILS = 5        # fewer failures in a configuration: no estimate (the fit is unconstrained)

# (campaign directory, code) -> label prefix
SOURCES = [("c4_strat_uniform", "C4-Helix"), ("c4_strat_helios", "C4-Helix"),
           ("ch_main", "Carbon-Helix"), ("ch_helios", "Carbon-Helix"), ("ch_interval", "Carbon-Helix"),
           ("ch_helios_long", "Carbon-Helix"), ("ch_helios_long32", "Carbon-Helix"),
           ("ch_periodic_fast", "Carbon-Helix")]

RULE_NAMES = {"full": "non-adaptive", "mode=ase": "ASE (paper rule)", "mode=ase,T=4": "ASE, forced T=4",
              "l2=on_event,l3=any": "ASE any-event (C)", "l2=on_event,l3=ambiguous,T3=4": "ASE ambiguous, T=4 (B4)",
              "l2=on_event,l3=ambiguous,rot=4": "ASE rotation R4", "l2=on_event,l3=ambiguous,rot=8": "ASE rotation R8",
              "l2=on_event,l3=ambiguous,rot=8,l2fast=1": "ASE rotation R8F (parallel weight-6)",
              "l2=on_event,l3=ambiguous,rot=16,l2fast=1": "ASE rotation R16F (outer check every other round)",
              "l2=on_event,l3=ambiguous,rot=32,l2fast=1": "ASE rotation R32F (outer check every fourth round)",
              "l2=on_event,l3=ambiguous,rot=32,l2fast=1,l2type=1":
                  "ASE rotation R32FT (R32F, weight-6 checks of the event's type only)"}


def rule_label(rule):
    if rule in RULE_NAMES:
        return RULE_NAMES[rule]
    if "rot=" in rule:
        rot = int(rule.split('rot=')[1].split(',')[0])
        per = f"{8 // rot} outer check(s) per round" if rot <= 8 else f"one outer check every {rot // 8} rounds"
        return f"ASE rotation, {per}" + (", parallel weight-6" if "l2fast=1" in rule else "")
    if "ambiguous" in rule:
        T = rule.split("T3=")[1].split(",")[0] if "T3=" in rule else "inf"
        return f"ASE ambiguous, T={T}"
    if "any" in rule:
        T = rule.split("T3=")[1].split(",")[0] if "T3=" in rule else "inf"
        return f"ASE any-event, T={T}"
    if "l3=never" in rule and "T3=" in rule:
        T = rule.split("T3=")[1].split(",")[0]
        return f"ASE periodic, outer checks every {T} rounds (no escalation" + \
            (", parallel weight-6)" if "l2fast=1" in rule else ")")
    return rule


def cores_for(campaign, key):
    """Core sizes (w, k) from minimal-failing-set analyses of this configuration, if any."""
    out = []
    for f in glob.glob(os.path.join(ROOT, "results", "minimal", "*.json")):
        d = json.load(open(f))
        if d.get("campaign") == campaign and d.get("rule") == key[1] and d.get("basis") == key[3] \
                and d.get("noise") == key[2] and d.get("rounds") == key[4]:
            out += [(r["W"], r["size"]) for r in d.get("rows", []) if r.get("counted", True)]
    return out


def configurations(basis="X"):
    rows = []
    for camp, code in SOURCES:
        for key, d in load(os.path.join(ROOT, f"results/final/{camp}/*/*/*/*.json")).items():
            if key[3] != basis:
                continue
            rows.append(dict(campaign=camp, code=code, key=key, sm=StratModel(key, d)))
    return rows


def evaluate(row, p, k_logical=2, nboot=50):
    """Same estimator as FINDINGS section 3: measured f_w where a stratum has >= 3 failures,
    the polynomial model elsewhere (lowest order 6 for Carbon-Helix, 3 for C4-Helix);
    'measured' when >= 90 % of the value is measured; statistical 95 % interval from the
    smoothed parametric bootstrap. Resources per round from direct path sampling at p."""
    sm = row["sm"]
    rec = dict(code=row["code"], rule=sm.key[1], label=f"{row['code']} {rule_label(sm.key[1])}",
               noise=sm.key[2], rounds=sm.rounds, p=p, campaign=row["campaign"],
               shots=int(sm.shots), fails=int(sm.f.sum()))
    k0 = 6 if row["code"] == "Carbon-Helix" else 3
    make = lambda s, x0=None: PolyModel(s, k0=k0, J=3, x0=x0)
    if sm.f.sum() < MIN_FAILS:
        rec.update(eps_LQ=None, status=f"insufficient data ({int(sm.f.sum())} failures)")
    else:
        try:
            est, reps, mdl = hybrid_boot(sm, make, [p], nboot=nboot, seed=3)
            e, share = est[0]
            lo, hi = boot_summary(reps, 0)
            # where the fit holds a low-order coefficient at its bound, every refitted replica can
            # lie above the point estimate; the interval is then widened to contain it
            ext = not (lo <= e <= hi)
            lo, hi = min(lo, e), max(hi, e)
            rec.update(eps_LQ=e / k_logical, ci95_LQ=[lo / k_logical, hi / k_logical], measured_share=share,
                       status="measured" if share >= 0.9 else "extrapolated", model=type(mdl).__name__,
                       ci_widened_to_estimate=bool(ext))
        except Exception as ex:
            rec.update(eps_LQ=None, status=f"insufficient data ({ex})")
    res = path_resources(sm, [p])[p]
    rec["twoq_per_round"] = res["twoq"]
    rec["time_ms_per_round"] = res["time_ms"]
    rec["outer_checks_per_round"] = res["outer_checks"]
    rec["full_outer_fraction"] = res["full_outer_fraction"]
    rec["twoq_se"], rec["time_ms_se"] = res["twoq_se"], res["time_ms_se"]
    rec["peak_qubits"] = sm.peak_qubits()
    return rec


def pareto(points, keys=("eps_LQ", "twoq_per_round", "time_ms_per_round", "peak_qubits")):
    ok = [x for x in points if all(x.get(k) is not None for k in keys)]
    nd = []
    for a in ok:
        dom = any(all(b[k] <= a[k] for k in keys) and any(b[k] < a[k] for k in keys)
                  for b in ok if b is not a)
        if not dom:
            nd.append(a)
    return nd


def limits(code):
    """Best conceivable value of each resource for a code, per round after preparation:
    two-qubit gates and Helios time of the inner-check part alone (the cheapest round that
    still checks every inner block), and the data qubits plus the fewest ancillas one outer
    check can be measured with at full circuit distance (outer checks one at a time): a
    4-qubit cat, plus (Carbon-Helix) one ancilla reading the cat's parities in turn - a cat
    alone leaves the circuit distance at 9 (FINDINGS 2.2)."""
    from hase.protocols import make_protocol
    from hase.timing import apart_duration
    name = "carbonhelix" if code == "Carbon-Helix" else "c4helix"
    pr = make_protocol(name, 7.9e-4, "X", 2, "full", "helios")
    l1 = "L1_a" if name == "carbonhelix" else "inner_a"
    return dict(twoq_per_round=pr.parts[l1].two_q, time_ms_per_round=pr.parts[l1].duration,
                peak_qubits=pr.n_data + (5 if name == "carbonhelix" else 4))


def table(points=(("uniform", 1e-3), ("uniform", 5e-4), ("helios", 7.9e-4), ("helios", 3e-4))):
    rows = configurations("X")
    out = {}
    for noise, p in points:
        recs = [evaluate(r, p) for r in rows if r["key"][2] == noise]
        # comparisons: each configuration vs non-adaptive of its code and vs C4-Helix non-adaptive
        base = {}
        for r in recs:
            if r["rule"] == "full":
                base[(r["code"], r["rounds"])] = r
        c4 = next((r for r in recs if r["code"] == "C4-Helix" and r["rule"] == "full"), None)
        for r in recs:
            b = base.get((r["code"], r["rounds"]))
            for ref, tag in ((b, "vs_nonadaptive"), (c4, "vs_C4Helix")):
                if ref is None or ref is r:
                    continue
                cmp = {}
                for k in ("eps_LQ", "twoq_per_round", "time_ms_per_round", "peak_qubits"):
                    if r.get(k) is not None and ref.get(k):
                        cmp[k] = r[k] / ref[k]
                r[tag] = cmp
        nd = pareto(recs)
        lim = {c: limits(c) for c in {r["code"] for r in recs}}
        best = {}
        for r in recs:
            if r.get("eps_LQ") is not None:
                best[r["code"]] = min(best.get(r["code"], np.inf), r["eps_LQ"])
        for r in recs:
            L = lim[r["code"]]
            dist = {k: (r[k] / L[k] if r.get(k) is not None and L.get(k) else None)
                    for k in ("twoq_per_round", "time_ms_per_round", "peak_qubits")}
            if r.get("eps_LQ") is not None and best.get(r["code"]):
                dist["eps_LQ"] = r["eps_LQ"] / best[r["code"]]
            r["distance_from_limit"] = dist
            known = {k: v for k, v in dist.items() if v is not None}
            r["furthest_criterion"] = max(known, key=known.get) if known else None
        keyed = {f"{r['code']}|{r['rule']}|r{r['rounds']}": r for r in recs}
        out[f"{noise}|{p:g}"] = dict(rows=keyed, limits=lim,
                                     nondominated=[f"{r['code']}|{r['rule']}|r{r['rounds']}" for r in nd])
    return out


if __name__ == "__main__":
    t = table()
    for k, v in t.items():
        print(k)
        for r in v["rows"].values():
            e = r.get("eps_LQ")
            print(f"  {r['label']:45s} r={r['rounds']} eps/LQ={e if e is None else f'{e:.2e}'} [{r['status'][:12]}] "
                  f"2q={r['twoq_per_round']:.0f} t={r['time_ms_per_round'] or 0:.0f}ms q={r['peak_qubits']}")
        print("  non-dominated:", v["nondominated"])
