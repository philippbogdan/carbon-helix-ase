#!/usr/bin/env python3
"""Build the Carbon-Helix README figures from data/ (headline.csv,
distance.json, interval.json, circuits.json). Nothing else is read.

    python make_figures.py            # refuses to overwrite existing outputs
    python make_figures.py --force    # rebuild in place

Every image is a 10 in wide canvas meant to be shown full width in a GitHub
README (about 1000 px; still legible at 700 px). PNGs are 2000 px wide
(200 dpi) and SVGs draw text as paths so they look the same on any machine.
Each image gets an .export.json provenance manifest from the
scientific-visualization skill's exporter.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
sys.path.insert(0, str(HERE / "_export"))
from figure_export import export_figure  # noqa: E402

W = 10.0   # canvas width in inches; README shows it at ~1000 px (~100 px/in)
DPI = 200  # 2000 px wide PNGs: twice the display width, sharp on HiDPI

# Same palette and roles as v1. Audited with the skill's palette_audit.py:
# all >= 3:1 on white; vermillion and grey share lightness, so they never
# encode competing series.
C4_COL = "#000000"
FULL_COL = "#004488"
R8_COL = "#D55E00"
GREY = "#808080"
INK = "#222222"
NOTE = "#555555"
ZONE = "#EDEDED"

# Scaled from the skill's presentation.mplstyle for a 10 in canvas.
STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 14,
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "xtick.labelsize": 13.5,
    "ytick.labelsize": 13.5,
    "legend.fontsize": 13.5,
    "axes.linewidth": 1.2,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "text.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.minor.width": 0.9,
    "ytick.minor.width": 0.9,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.minor.size": 3.5,
    "ytick.minor.size": 3.5,
    "xtick.major.pad": 5,
    "ytick.major.pad": 5,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    # Text as paths: a public README is viewed on machines without Arial,
    # where font substitution would reflow labels into each other.
    "svg.fonttype": "path",
    "mathtext.default": "regular",
}

P_TODAY, P_FUTURE = 0.00079, 0.0003
SCHEMES = [
    # label in data, display name, colour, marker, linestyle, dodge factor
    ("C4 adaptive", "C4-Helix, adaptive", C4_COL, "o", "-", 1 / 1.075),
    ("Carbon full", "Carbon-Helix, full", FULL_COL, "s", "--", 1.0),
    ("Carbon R8", "Carbon-Helix, R8 adaptive", R8_COL, "D", "--", 1.075),
]


def sci(v: float) -> str:
    """7.9e-4 -> 7.9×10⁻⁴ in mathtext."""
    exp = math.floor(math.log10(v))
    man = f"{v / 10**exp:.1f}".rstrip("0").rstrip(".")
    return rf"$10^{{{exp}}}$" if man == "1" else rf"${man}\times10^{{{exp}}}$"


def load_headline():
    with open(DATA / "headline.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {(r["label"], float(r["p"])): {
        "eps": float(r["eps_LQ"]), "lo": float(r["ci_low"]),
        "hi": float(r["ci_high"]), "status": r["status"],
        "gates": float(r["twoq_per_round"]), "qubits": int(r["peak_qubits"]),
    } for r in rows}


def load_json(name):
    with open(DATA / name) as fh:
        return json.load(fh)


def status_key(threshold):
    handles = [Line2D([], [], color=INK, marker="o", ms=10, ls="none"),
               Line2D([], [], color=INK, marker="o", ms=10, mfc="white",
                      mew=2, ls="none"),
               Line2D([], [], color=INK, marker="|", ms=18, mew=2,
                      ls="none")]
    labels = [f"measured (≥ {threshold}% from sampled failures)",
              "extrapolated (fitted)", "95% interval"]
    return handles, labels


# ------------------------------------------------------- 1: logical error
def fig_error(force):
    H = load_headline()
    fig, ax = plt.subplots(figsize=(W, 6.6), layout="constrained")

    for lab, _, col, mk, ls, dodge in SCHEMES:
        a, b = H[(lab, P_FUTURE)], H[(lab, P_TODAY)]
        ax.plot([P_FUTURE * dodge, P_TODAY * dodge], [a["eps"], b["eps"]],
                color=col, ls=ls, lw=2.2, zorder=2)
        for p in (P_FUTURE, P_TODAY):
            d = H[(lab, p)]
            filled = d["status"] == "measured"
            ax.errorbar(p * dodge, d["eps"],
                        yerr=[[d["eps"] - d["lo"]], [d["hi"] - d["eps"]]],
                        fmt="none", ecolor=col, elinewidth=2, capsize=6,
                        capthick=2, zorder=3)
            ax.plot(p * dodge, d["eps"], marker=mk, ms=12, mec=col, mew=2.2,
                    mfc=col if filled else "white", ls="none", zorder=4)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(2.45e-4, 1.34e-3)
    ax.set_ylim(1.5e-9, 2e-4)
    assert all(1.5e-9 < d["lo"] and d["hi"] < 2e-4 for d in H.values())
    ax.xaxis.set_major_locator(FixedLocator([P_FUTURE, P_TODAY]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{sci(P_FUTURE)}\nfuture device",
                        f"{sci(P_TODAY)}\nHelios today"])
    ax.set_xlabel("physical two-qubit error rate $p$ (log scale)")
    ax.set_ylabel("logical error per logical qubit per round\n(log scale)")
    ax.grid(axis="y", which="major", color="#E6E6E6", lw=1, zorder=0)

    for p in (P_FUTURE, P_TODAY):
        top = H[("C4 adaptive", p)]["eps"]
        bot = H[("Carbon R8", p)]["eps"]
        fold = top / bot
        label = (f"{round(fold, -1):.0f}×" if fold >= 100 else
                 f"{fold:.0f}×" if fold >= 20 else f"{fold:.1f}×")
        xb = p * 1.21
        ax.annotate("", xy=(xb, bot), xytext=(xb, top),
                    arrowprops=dict(arrowstyle="<->", color=INK, lw=1.6,
                                    mutation_scale=20, shrinkA=0, shrinkB=0))
        right = p == P_TODAY
        ax.text(xb * (1.04 if right else 1 / 1.04), math.sqrt(top * bot),
                label, fontsize=26, fontweight="bold", va="center",
                ha="left" if right else "right", color=INK)

    handles = [Line2D([], [], color=c, ls=ls, lw=2.2, marker=m, ms=11,
                      mfc=c if lab == "C4 adaptive" else "white", mec=c,
                      mew=2.2) for lab, _, c, m, ls, _ in SCHEMES]
    leg = ax.legend(handles, [s[1] for s in SCHEMES], loc="upper left",
                    handlelength=3.2, borderaxespad=0.3, labelspacing=0.6)
    ax.add_artist(leg)
    h, lab = status_key(90)
    ax.legend(h, lab, loc="lower right", handletextpad=0.4,
              borderaxespad=0.3, labelspacing=0.5)

    export(fig, "fig1_error", force, {
        "raw_data": ["data/headline.csv"],
        "transformations": [
            "none to plotted values; fold labels = C4 eps_LQ / Carbon R8 "
            "eps_LQ, rounded to two significant figures",
            "points nudged sideways by a factor 1.075 in p for legibility"],
        "uncertainty": "95% interval from headline.csv ci_low/ci_high",
        "measured_vs_extrapolated": "headline.csv status column",
    })


# --------------------------------------------------- 2: cost and qubits
def fig_cost(force):
    H = load_headline()
    circ = load_json("circuits.json")["carbonhelix"]
    fig = plt.figure(figsize=(W, 6.8), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.12, hspace=0.08)
    axg, axq = fig.subplots(2, 1, height_ratios=[2.1, 1.25])

    order = ["C4 adaptive", "Carbon full", "Carbon R8"]
    cols = {s[0]: s[2] for s in SCHEMES}
    names = {s[0]: s[1] for s in SCHEMES}
    h = 0.38
    for i, lab in enumerate(order):
        y = len(order) - 1 - i
        t, f = H[(lab, P_TODAY)], H[(lab, P_FUTURE)]
        axg.barh(y + h / 2, t["gates"], height=h, color=cols[lab], lw=0)
        axg.barh(y - h / 2, f["gates"], height=h, color=cols[lab],
                 alpha=0.38, lw=0)
        for yy, v in ((y + h / 2, t["gates"]), (y - h / 2, f["gates"])):
            axg.text(v + 7, yy, f"{v:.0f}", va="center", ha="left",
                     fontsize=13.5)
    y_full = len(order) - 1 - order.index("Carbon full") + h / 2
    x0 = 0
    for key_, nm in (("L1_a", "level 1"), ("L2_all", "level 2"),
                     ("L3", "level 3 (outer)")):
        w = circ[key_]["two_q"]
        if x0:
            axg.plot([x0, x0], [y_full - h / 2, y_full + h / 2],
                     color="white", lw=2, zorder=3)
        axg.text(x0 + w / 2, y_full, nm, color="white", fontsize=13,
                 ha="center", va="center", fontweight="bold", zorder=4)
        x0 += w
    assert x0 == H[("Carbon full", P_TODAY)]["gates"]
    axg.set_yticks(range(len(order)))
    axg.set_yticklabels([names[k] for k in reversed(order)])
    axg.tick_params(axis="y", length=0)
    axg.set_xlim(0, 640)
    axg.set_ylim(-0.62, len(order) - 0.38)
    axg.set_xlabel("two-qubit gates per round")
    axg.spines["left"].set_visible(False)
    axg.legend([Patch(color=GREY), Patch(color=GREY, alpha=0.38)],
               ["Helios today", "future device"], loc="lower right",
               handlelength=1.6, handleheight=1.0, borderaxespad=0.1)

    for i, lab in enumerate(order):
        y = len(order) - 1 - i
        q = H[(lab, P_TODAY)]["qubits"]
        assert q == H[(lab, P_FUTURE)]["qubits"]
        axq.barh(y, q, height=0.6, color=cols[lab], lw=0)
        axq.text(q + 1.0, y, f"{q}", va="center", ha="left", fontsize=13.5)
    axq.axvline(98, color=INK, lw=1.8, ls=(0, (3, 2)))
    axq.text(96.8, len(order) - 0.42, "Helios: 98 qubits", ha="right",
             va="bottom", fontsize=13.5)
    axq.set_yticks(range(len(order)))
    axq.set_yticklabels([names[k] for k in reversed(order)])
    axq.tick_params(axis="y", length=0)
    axq.set_xlim(0, 108)
    axq.set_ylim(-0.5, len(order) - 0.02)
    axq.set_xlabel("physical qubits used (peak)")
    axq.spines["left"].set_visible(False)

    export(fig, "fig2_cost", force, {
        "raw_data": ["data/headline.csv", "data/circuits.json"],
        "transformations": [
            "none to plotted values",
            "full-round split = circuits.json L1_a + L2_all + L3 two_q "
            "(sums to the headline 560)"],
        "uncertainty": "none in data for these counts",
    })


# ------------------------------------------------------- 3: distance
def fig_distance(force):
    """One bead per fault: a column of d beads is a case of fault distance d.
    Filled beads are proven, open beads are found but not proven."""
    dist = load_json("distance.json")
    ends, designs, whole = dist["ends"], dist["designs"], dist["whole"]
    cols, groups = [], []
    x = 0.0
    gap = 1.7

    def group(title, items, extra=0.0):
        nonlocal x
        x += extra
        start = x
        for it in items:
            it["x"] = x
            cols.append(it)
            x += 1
        groups.append((title, start, x - 1))
        x += gap

    mk = {"X": "o", "Z": "s"}
    r8 = []
    for i in range(8):
        for b in ("X", "Z"):
            e = ends[f"k4cyc|R8_{i}|{b}"]
            assert e["all_optimal"] and math.isclose(
                e["min_lower_bound"], e["min_weight"], abs_tol=1e-6)
            assert set(e["combos"]) == {"0", "01", "1"}
            r8.append(dict(proven=e["min_weight"], total=e["min_weight"],
                           colour=R8_COL, marker=mk[b]))
    group("R8 adaptive rule:\n8 cases, each in X and Z", r8)
    other = []
    for k in ("XZ|X", "XZ|Z", "ZX|Z", "quiet|Z", "quiet|X"):
        rule, b = k.split("|")
        e = ends[f"k4cyc|{rule}|{b}"]
        assert e["all_optimal"]
        complete = set(e["combos"]) == {"0", "01", "1"}
        other.append(dict(proven=e["min_weight"] if complete else 0,
                          total=e["min_weight"], colour=INK, marker=mk[b]))
    group("other\nadaptive\nrules", other)
    des = []
    for name in sorted(designs, key=lambda n: (designs[n]["min_weight"], n)):
        e = designs[name]
        assert e["all_optimal"]
        des.append(dict(proven=e["min_weight"], total=e["min_weight"],
                        colour=GREY, marker="o"))
    group("alternative\nouter-check\ncircuits", des)
    wh = []
    for k in ("XZ|X", "ZX|Z"):
        w = whole[k]
        assert not w["optimal"]
        wh.append(dict(proven=int(round(w["lower_bound"])), total=w["best"],
                       colour=INK, marker=mk[k.split("|")[1]]))
    hours = max(w["seconds"] for w in whole.values()) / 3600
    group(f"whole circuit:\n{hours:.0f} h search,\nunfinished", wh, extra=0.6)
    xmax = x - gap

    fig, ax = plt.subplots(figsize=(W, 6.9), layout="constrained")
    ax.axhspan(10.5, 12.5, color=ZONE, lw=0, zorder=0)
    ax.text(-0.6, 13.05, "shaded: distance ≥ 11, so at least 6 faults are "
            "needed for a logical error, which falls like $p^6$",
            ha="left", va="bottom", fontsize=13.5)
    for c in cols:
        for k in range(1, int(c["total"]) + 1):
            proven = k <= c["proven"]
            ax.plot(c["x"], k, marker=c["marker"], ms=10.5, ls="none",
                    mec=c["colour"], mew=1.8,
                    mfc=c["colour"] if proven else "white", zorder=3)
    for title, x0, x1 in groups:
        ax.plot([x0 - 0.4, x1 + 0.4], [0.3, 0.3], color=INK, lw=1.2)
        ax.text((x0 + x1) / 2, 0.05, title, ha="center", va="top",
                fontsize=13.5, linespacing=1.15)
    ax.set_xlim(-0.8, xmax + 0.8)
    ax.set_ylim(-2.6, 13.9)
    ax.set_yticks([1, 5, 9, 10, 11, 12])
    ax.yaxis.set_minor_locator(FixedLocator(range(1, 13)))
    ax.set_xticks([])
    for sp in ("bottom", "top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_bounds(0.5, 12.5)
    ax.set_ylabel("faults needed for an undetected logical failure\n"
                  "(fault distance; one bead per fault)")
    key = [Line2D([], [], color=INK, marker="o", ms=10, ls="none"),
           Line2D([], [], color=INK, marker="s", ms=10, ls="none"),
           Line2D([], [], color=INK, marker="o", ms=10, mfc="white", mew=1.8,
                  ls="none")]
    fig.legend(key, ["X basis", "Z basis", "found, not proven minimal"],
               loc="outside upper center", ncol=3, handletextpad=0.3,
               columnspacing=2.0)

    export(fig, "fig3_distance", force, {
        "raw_data": ["data/distance.json"],
        "transformations": [
            "ends and designs: min_weight beads, all filled (all_optimal "
            "true, min_lower_bound equals min_weight)",
            "quiet|X: combos lacks '01', so its 11 beads are drawn open",
            "whole: filled to lower_bound, open from there to best "
            "(optimal false)",
            "designs sorted by min_weight then name"],
        "uncertainty": "solver bounds, not statistical intervals",
    })


# ------------------------------------------------------- 4: escalation
def fig_escalation(force):
    iv = load_json("interval.json")
    fig, ax = plt.subplots(figsize=(W, 6.0), layout="constrained")
    styles = {"B": ("o", "-"), "C": ("^", "--")}
    for fam, (m, ls) in styles.items():
        s = iv[fam]["escalation_scan"]
        p, f, n = s["p"], s["per_round"], s["shots"]
        half = [1.96 * math.sqrt(v * (1 - v) / n) for v in f]
        assert all(5e-3 < v - hh and v + hh < 1.4 for v, hh in zip(f, half))
        ax.plot(p, f, color=INK, ls=ls, lw=2.2, zorder=2)
        ax.errorbar(p, f, yerr=half, fmt="none", ecolor=INK, elinewidth=1.8,
                    capsize=5, capthick=1.8, zorder=3)
        ax.plot(p, f, marker=m, ms=11, mfc=INK, mec=INK, ls="none", zorder=4)
        ax.text(p[-1] / 1.22, f[-1], f"rule {fam}", va="center", ha="right",
                fontsize=15, fontweight="bold")
    ax.axvline(P_TODAY, color=GREY, lw=1.6, ls=(0, (4, 3)), zorder=1)
    ax.text(P_TODAY * 1.06, 0.0056, "Helios today", rotation=90, ha="left",
            va="bottom", fontsize=13.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(4.2e-5, 8e-3)
    ax.set_ylim(5e-3, 1.4)
    ax.set_yticks([0.01, 0.03, 0.1, 0.3, 1])
    ax.set_yticklabels(["1%", "3%", "10%", "30%", "100%"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("physical error rate $p$ (log scale)")
    ax.set_ylabel("rounds in which the rule calls for the\n"
                  "full outer measurement (log scale)")
    ax.legend([Line2D([], [], color=INK, marker="o", ms=10, ls="none"),
               Line2D([], [], color=INK, marker="|", ms=18, mew=2,
                      ls="none")],
              ["measured by direct sampling, 8000 shots",
               "95% interval (shots as units)"], loc="upper left",
              handletextpad=0.4, labelspacing=0.5)

    export(fig, "fig4_escalation", force, {
        "raw_data": ["data/interval.json"],
        "transformations": [
            "escalation_scan.per_round for rules B and C",
            "interval = per_round +- 1.96 sqrt(f(1-f)/shots), conservative "
            "(each shot treated as one unit)"],
        "uncertainty": "conservative normal 95%",
    })


# ------------------------------------------------------- 5: schedule
def fig_schedule(force):
    iv = load_json("interval.json")
    rules = ["B", "C", "R", "P"]
    slots_all = ["2", "4", "8", "16", "inf"]
    ylo, yhi = 0.45, 2.75
    fig, axs = plt.subplots(len(rules), 1, figsize=(W, 12.0), sharex=True,
                            sharey=True, layout="constrained")
    fig.get_layout_engine().set(h_pad=0.1, hspace=0.06)
    for axf, fam in zip(axs, rules):
        v = iv[fam]
        ps = v["ps"]
        k = len(ps)
        offs = [(-0.4 + 0.8 * i / (k - 1)) for i in range(k)]
        axf.axhline(1.0, color=FULL_COL, lw=2.2, zorder=1)
        for j, t in enumerate(slots_all):
            if j:
                axf.axvline(j - 0.5, color="#E3E3E3", lw=1, zorder=0)
            if t not in v["ratio"]:
                axf.text(j, 1.6, "not run", ha="center", va="center",
                         fontsize=13.5, color=NOTE)
                continue
            driven = set(v["ratio_driven"].get(t, []))
            lo_hi = v["ratio_ci95"][t]
            for i, p in enumerate(ps):
                rv = v["ratio"][t][i]
                lo, hi = lo_hi[0][i], lo_hi[1][i]
                assert ylo < lo <= rv <= hi < yhi, (fam, t, p)
                meas = p in driven
                col = INK if meas else GREY
                axf.plot([j + offs[i]] * 2, [lo, hi], color=col,
                         lw=2.0 if meas else 1.3, zorder=2,
                         solid_capstyle="butt")
                axf.plot(j + offs[i], rv, marker="o", ms=8.5 if meas else 7.5,
                         mfc=col if meas else "white", mec=col,
                         mew=1.8 if meas else 1.5, ls="none", zorder=3)
        axf.set_yscale("log")
        axf.set_ylim(ylo, yhi)
        axf.yaxis.set_major_locator(FixedLocator([0.5, 0.7, 1, 1.4, 2]))
        axf.yaxis.set_minor_locator(NullLocator())
        axf.set_yticklabels(["0.5", "0.7", "1", "1.4", "2"])
        axf.text(0.008, 0.96, f"rule {fam}", transform=axf.transAxes,
                 ha="left", va="top", fontsize=16, fontweight="bold")
        axf.tick_params(axis="x", length=0, labelbottom=True)
        axf.set_xticks(range(len(slots_all)))
        axf.set_xticklabels(["∞" if t == "inf" else t for t in slots_all],
                            fontsize=15)
        axf.set_xlim(-0.55, len(slots_all) - 0.45)
    fig.supylabel("logical error relative to full extraction every round "
                  "(log scale)", fontsize=15)
    fig.supxlabel("full outer measurement forced every T rounds "
                  "(∞ = never forced)\n"
                  f"within each T, $p$ falls left to right from {sci(5e-3)} "
                  f"to {sci(5e-4)}", fontsize=15, linespacing=1.4)
    h, lab = status_key(50)
    h.append(Line2D([], [], color=FULL_COL, lw=2.2))
    lab.append("equal to full extraction every round")
    fig.legend(h, lab, loc="outside upper center", ncol=2,
               handletextpad=0.4, columnspacing=2.0)

    export(fig, "fig5_schedule", force, {
        "raw_data": ["data/interval.json"],
        "transformations": [
            "ratio[T] and ratio_ci95[T] at every p, p falling left to right "
            "within each T",
            "filled where p is in ratio_driven[T] (gate: every compared "
            "configuration >= 50% measured)",
            "T values a rule was not run at are marked 'not run'"],
        "uncertainty": "interval.json ratio_ci95",
    })


# ------------------------------------------------------- export helpers
ALT = {
    "fig1_error": (
        "Logical error per round: Carbon-Helix against C4-Helix",
        "Log-log plot of logical error per logical qubit per round against "
        "physical error rate at Helios today (p = 7.9e-4) and a future "
        "device (p = 3e-4). C4-Helix (measured) sits 6.7 times above "
        "Carbon-Helix R8 today and about 120 times above it at p = 3e-4. "
        "Carbon-Helix values, full and R8, are extrapolated; all points "
        "carry 95% intervals."),
    "fig2_cost": (
        "Cost per round and qubits used",
        "Bars of two-qubit gates per round at Helios today and a future "
        "device: C4-Helix adaptive 54 and 45, Carbon-Helix full 560 (120 "
        "level 1, 200 level 2, 240 level 3 outer), Carbon-Helix R8 216 and "
        "170. Below, peak physical qubits: 52 for C4-Helix and 92 for "
        "Carbon-Helix, under Helios's 98."),
    "fig3_distance": (
        "Fault distance of Carbon-Helix adaptive rules",
        "Unit chart, one bead per fault. All 16 R8 cases are proven at "
        "distance 11 or 12; four other rule cases are proven at 11 or 12 "
        "and one (quiet, X) is 11 but incomplete. Alternative outer-check "
        "circuits reach 9 to 11. The whole-circuit search proved only "
        "lower bounds of 4 and 5 against best found 12 and 11. Distance 11 "
        "or more means at least six faults per logical error."),
    "fig4_escalation": (
        "How often adaptive rules call for the full outer measurement",
        "Log-log plot: the share of rounds in which rules B and C call for "
        "the full outer measurement falls roughly in proportion to p, from "
        "82% and 97% at p = 5e-3 to 0.9% and 3.6% at p = 1e-4; about 10% "
        "and 30% at Helios today."),
    "fig5_schedule": (
        "Error when the full outer measurement is forced only every T rounds",
        "Four stacked panels, rules B, C, R and P: logical error relative to "
        "full extraction every round, at each p, with 95% intervals. Rule B "
        "stays at about 1 for T = 2, 4, 8 and never; rule P sits at 0.64 to "
        "0.82 up to T = 8 and 1.0 to 1.2 at T = 16; rules C and R are 1.1 "
        "to 1.5."),
}


def export(fig, stem, force, provenance):
    title, desc = ALT[stem]
    export_figure(fig, HERE / stem, formats=["png", "svg"], dpi=DPI,
                  overwrite=force, provenance=provenance, write_manifest=True,
                  metadata={"Title": title, "Description": desc},
                  font_mode="current")  # keep STYLE's svg.fonttype "path"
    plt.close(fig)
    flatten_png(HERE / f"{stem}.png", HERE / f"{stem}.export.json")
    print(stem)


def flatten_png(png, manifest):
    """Agg writes RGBA even on an opaque white page. Drop the alpha channel
    only after checking that every pixel is fully opaque, keep the DPI and
    text metadata, and record the step in the export manifest."""
    from PIL import Image, PngImagePlugin

    with Image.open(png) as im:
        im.load()
        assert im.mode == "RGBA"
        assert im.getchannel("A").getextrema() == (255, 255), "not opaque"
        info = PngImagePlugin.PngInfo()
        for k, v in im.text.items():
            info.add_text(k, v)
        rgb = im.convert("RGB")
        dpi = im.info.get("dpi", (DPI, DPI))
    tmp = png.with_suffix(".tmp.png")
    rgb.save(tmp, format="PNG", pnginfo=info, dpi=dpi, optimize=True)
    tmp.replace(png)
    m = json.loads(manifest.read_text())
    for o in m["outputs"]:
        if o["format"] == "png":
            o["size_bytes"] = png.stat().st_size
            o["post_processing"] = ("alpha channel dropped after checking "
                                    "all pixels opaque; mode RGB")
    manifest.write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")


FIGS = {"1": fig_error, "2": fig_cost, "3": fig_distance,
        "4": fig_escalation, "5": fig_schedule}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing outputs")
    ap.add_argument("--only", choices=sorted(FIGS), help="one image")
    a = ap.parse_args()
    with plt.rc_context(STYLE):
        for n, fn in FIGS.items():
            if a.only in (None, n):
                fn(a.force)


if __name__ == "__main__":
    main()
