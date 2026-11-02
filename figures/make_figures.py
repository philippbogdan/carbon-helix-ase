#!/usr/bin/env python3
"""Render every figure in figures/ from the saved data in figures/data/.

Run from the repository root: ./run.sh figures
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter  # noqa: E402
from PIL import Image, ImageSequence  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "figures" / "data"
OUT = ROOT / "figures"
FRAMES = ROOT / "build" / "r8_frames"

DPI = 100
PNG_SCALE = 2
STATIC = ("memory_rate", "extraction_cost", "fault_order", "interval_cost")
BYTE_LIMIT = 10_000_000
P_HEADLINE = 0.00079

BG = "#FAF7F0"
INK = "#23211D"
MUTED = "#6B665D"
RULE = "#E2DBCE"
GREY = "#8C877E"
TEAL = "#0B8A8C"
AMBER = "#C27C0E"
VIOLET = "#6B4FBB"
SLATE = "#3E5C76"
DATA_EDGE = "#6F695F"
ANCILLA_FILL = "#B4AC9E"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12.5,
            "text.color": INK,
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
            "axes.edgecolor": MUTED,
            "axes.linewidth": 0.9,
            "axes.labelcolor": INK,
            "axes.labelsize": 13,
            "axes.labelpad": 8,
            "axes.unicode_minus": True,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK,
            "ytick.labelcolor": INK,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12.5,
            "xtick.major.size": 5,
            "xtick.minor.size": 3,
            "xtick.major.width": 0.9,
            "xtick.minor.width": 0.7,
            "legend.frameon": False,
            "legend.fontsize": 12,
            "legend.handletextpad": 0.5,
            "svg.fonttype": "path",
            "svg.hashsalt": "carbon-helix",
            "path.simplify": False,
        }
    )


def read_csv(name: str) -> list[dict[str, str]]:
    with open(DATA / name, newline="") as handle:
        return list(csv.DictReader(handle))


def rgb255(colour: str) -> np.ndarray:
    return np.array(to_rgb(colour)) * 255.0


def px(value: float) -> float:
    """Convert pixels at the export resolution to points."""
    return value * 72.0 / DPI


def new_figure(width: int, height: int):
    return plt.figure(figsize=(width / DPI, height / DPI), dpi=DPI)


def axes_px(fig, left: float, bottom: float, width: float, height: float):
    w, h = fig.get_size_inches() * DPI
    return fig.add_axes((left / w, bottom / h, width / w, height / h))


def save_static(fig, stem: str, size: tuple[int, int]) -> None:
    png, svg = OUT / f"{stem}.png", OUT / f"{stem}.svg"
    fig.savefig(png, dpi=DPI * PNG_SCALE, facecolor=BG)
    fig.savefig(svg, facecolor=BG, metadata={"Date": None, "Creator": None})
    plt.close(fig)
    with Image.open(png) as image:
        rgb = image.convert("RGB")
    rgb.save(png, optimize=True)
    expected = (size[0] * PNG_SCALE, size[1] * PNG_SCALE)
    if rgb.size != expected:
        raise SystemExit(f"{png.name}: expected {expected}, got {rgb.size}")


def clean_row_axis(ax) -> None:
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=10)


# Memory error comparison


def memory_rate() -> None:
    rows = {r["label"]: r for r in read_csv("headline.csv") if float(r["p"]) == P_HEADLINE}
    order = ["C4 adaptive", "Carbon full", "Carbon R8"]
    colour = {"C4 adaptive": GREY, "Carbon full": VIOLET, "Carbon R8": TEAL}
    expected = {"C4 adaptive": "measured", "Carbon full": "extrapolated", "Carbon R8": "extrapolated"}
    for label in order:
        row = rows[label]
        share = float(row["measured_share"])
        if row["status"] != expected[label]:
            raise SystemExit(f"headline.csv: unexpected status for {label}: {row['status']}")
        if (row["status"] == "measured") != (share >= 0.5):
            raise SystemExit(f"headline.csv: status and measured_share disagree for {label}")
        lo, mid, hi = float(row["rate_low"]), float(row["rate"]), float(row["rate_high"])
        if not lo <= mid <= hi:
            raise SystemExit(f"headline.csv: rate outside its interval for {label}")

    fig = new_figure(1000, 420)
    ax = axes_px(fig, 140, 82, 830, 286)
    ys = {label: float(len(order) - 1 - i) for i, label in enumerate(order)}
    lows, highs = [], []
    for label in order:
        row, y, c = rows[label], ys[label], colour[label]
        lo, mid, hi = float(row["rate_low"]), float(row["rate"]), float(row["rate_high"])
        lows.append(lo)
        highs.append(hi)
        ax.hlines(y, lo, hi, color=c, lw=2.4, zorder=2, capstyle="butt")
        ax.vlines([lo, hi], y - 0.11, y + 0.11, color=c, lw=2.0, zorder=2)
        filled = row["status"] == "measured"
        ax.plot(
            mid, y, "o", ms=11, mew=2.4, mec=c, mfc=c if filled else BG, zorder=3,
        )

    pad = 0.13
    lo_all, hi_all = math.log10(min(lows)), math.log10(max(highs))
    ax.set_xscale("log")
    ax.set_xlim(10 ** (lo_all - pad), 10 ** (hi_all + pad))
    ax.set_ylim(-0.55, len(order) - 0.45)
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    minor_labels = {2, 5}

    def minor_format(value: float, _pos: int) -> str:
        exponent = math.floor(math.log10(value) + 1e-9)
        mantissa = round(value / 10**exponent)
        if mantissa in minor_labels:
            return rf"${mantissa}{{\times}}10^{{{exponent}}}$"
        return ""

    ax.xaxis.set_minor_formatter(FuncFormatter(minor_format))
    ax.tick_params(axis="x", which="minor", labelsize=11, labelcolor=MUTED)
    ax.grid(axis="x", which="major", color=RULE, lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.set_yticks([ys[label] for label in order], order)
    clean_row_axis(ax)
    ax.set_xlabel("Estimated memory error (logical qubit⁻¹ s⁻¹)")
    handles = [
        Line2D([], [], ls="none", marker="o", ms=10, mew=2.0, mec=INK, mfc=INK),
        Line2D([], [], ls="none", marker="o", ms=10, mew=2.0, mec=INK, mfc=BG),
    ]
    ax.legend(
        handles,
        ["Observed-stratum dominated", "Partly extrapolated"],
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        ncol=2,
        borderaxespad=0.3,
        columnspacing=1.6,
    )
    save_static(fig, "memory_rate", (1000, 420))


# Extraction cost


def extraction_cost() -> None:
    rows = {r["label"]: r for r in read_csv("headline.csv") if float(r["p"]) == P_HEADLINE}
    full, r8 = rows["Carbon full"], rows["Carbon R8"]
    panels = [
        ("twoq_per_round", "Two-qubit gates per round (count)", 100.0),
        ("time_ms_per_round", "Modelled round time (ms)", 200.0),
    ]
    fig = new_figure(1000, 560)
    bottoms = [352, 84]
    for (field, xlabel, step), bottom in zip(panels, bottoms):
        ax = axes_px(fig, 140, bottom, 830, 176)
        values = [float(full[field]), float(r8[field])]
        ax.barh([1, 0], values, height=0.56, color=[VIOLET, TEAL], zorder=2)
        top = values[0] * 1.04
        ax.set_xlim(0, top)
        ax.set_ylim(-0.6, 1.6)
        ax.xaxis.set_major_locator(FixedLocator(np.arange(0, top + 1e-9, step)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
        ax.grid(axis="x", color=RULE, lw=0.9, zorder=0)
        ax.set_axisbelow(True)
        ax.set_yticks([1, 0], ["Carbon full", "Carbon R8"])
        clean_row_axis(ax)
        ax.set_xlabel(xlabel)
    save_static(fig, "extraction_cost", (1000, 560))


# Free-order fits


def fault_order() -> None:
    rows = read_csv("exponents.csv")
    rules = ["full", "C", "B4", "R4", "R8", "R8F"]
    names = ["Full", "Any event", "Ambiguous, T=4", "R4", "R8", "R8F"]
    noises = [("uniform", "o", SLATE, 9.5, 0.16), ("helios", "D", AMBER, 8.5, -0.16)]
    by_key = {(r["rule"], r["noise"]): r for r in rows}
    if len(by_key) != len(rows) or set(by_key) != {(r, n[0]) for r in rules for n in noises}:
        raise SystemExit("exponents.csv: expected one row per rule and noise")
    for rule, name in zip(rules, names):
        for noise, *_ in noises:
            if by_key[(rule, noise)]["label"] != name:
                raise SystemExit(f"exponents.csv: unexpected label for {rule}")

    fig = new_figure(1000, 620)
    ax = axes_px(fig, 168, 84, 802, 476)
    ends = []
    for i, rule in enumerate(rules):
        y = float(len(rules) - 1 - i)
        for noise, marker, c, size, dy in noises:
            row = by_key[(rule, noise)]
            lo, mid, hi = float(row["low"]), float(row["exponent"]), float(row["high"])
            ends += [lo, hi]
            ax.hlines(y + dy, lo, hi, color=c, lw=2.0, zorder=2, capstyle="butt")
            ax.vlines([lo, hi], y + dy - 0.07, y + dy + 0.07, color=c, lw=1.6, zorder=2)
            ax.plot(mid, y + dy, marker=marker, ms=size, color=c, mec=c, zorder=3)
    ax.axvline(6.0, color=GREY, lw=1.0, zorder=1)
    span = max(ends) - min(ends)
    ax.set_xlim(min(ends) - 0.06 * span, max(ends) + 0.06 * span)
    ax.set_ylim(-0.6, len(rules) - 0.4)
    ticks = [t for t in np.arange(5.0, 7.51, 0.5) if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.set_yticks(range(len(rules) - 1, -1, -1), names)
    clean_row_axis(ax)
    ax.set_xlabel("Local error exponent (dimensionless)")
    handles = [
        Line2D([], [], color=SLATE, lw=2.0, marker="o", ms=9.5),
        Line2D([], [], color=AMBER, lw=2.0, marker="D", ms=8.5),
    ]
    ax.legend(
        handles,
        ["Uniform noise", "Helios-like noise"],
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        ncol=2,
        borderaxespad=0.3,
        columnspacing=1.6,
        handlelength=2.2,
    )
    save_static(fig, "fault_order", (1000, 620))


# Periodic interval cost


def interval_cost() -> None:
    selected_p = ["0.004", "0.003", "0.0025"]
    rows = [r for r in read_csv("intervals.csv") if r["p"] in selected_p]
    for r in rows:
        if r["data_driven"] != "True" or float(r["sampled_share_min"]) < 0.5:
            raise SystemExit(f"intervals.csv: row T={r['T']}, p={r['p']} is not observation supported")
    styles = {"0.004": (VIOLET, "o", 9.0, -0.07), "0.003": (AMBER, "s", 8.0, 0.0), "0.0025": (TEAL, "^", 9.5, 0.07)}
    fig = new_figure(1000, 460)
    ax = axes_px(fig, 104, 82, 866, 358)
    ends = []
    t_values = sorted({int(r["T"]) for r in rows})
    if t_values != [2, 4, 8, 16]:
        raise SystemExit(f"intervals.csv: unexpected T values {t_values}")
    for p in selected_p:
        c, marker, size, offset = styles[p]
        series = sorted((r for r in rows if r["p"] == p), key=lambda r: int(r["T"]))
        if [int(r["T"]) for r in series] != t_values:
            raise SystemExit(f"intervals.csv: missing T values for p={p}")
        xs = np.array([int(r["T"]) * 2.0**offset for r in series])
        mid = np.array([float(r["ratio"]) for r in series])
        lo = np.array([float(r["low"]) for r in series])
        hi = np.array([float(r["high"]) for r in series])
        ends += list(lo) + list(hi)
        ax.plot(xs, mid, color=c, lw=1.5, alpha=0.75, zorder=2)
        ax.vlines(xs, lo, hi, color=c, lw=1.8, zorder=2)
        cap = 2.0**0.022
        ax.hlines(np.r_[lo, hi], np.r_[xs, xs] / cap, np.r_[xs, xs] * cap, color=c, lw=1.6, zorder=2)
        ax.plot(xs, mid, ls="none", marker=marker, ms=size, color=c, mec=c, zorder=3, label=f"$p$ = {p}")
    ax.axhline(1.0, color=GREY, lw=1.0, zorder=1)
    ax.set_xscale("log", base=2)
    ax.set_xlim(2 / 2**0.32, 16 * 2**0.32)
    ax.xaxis.set_major_locator(FixedLocator(t_values))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    span = max(ends) - min(ends)
    ax.set_ylim(min(ends) - 0.08 * span, max(ends) + 0.08 * span)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.1f}"))
    ax.grid(axis="y", color=RULE, lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    for side in ("right", "top"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("Full-measurement interval T (rounds)")
    ax.set_ylabel("Error per round / full extraction (ratio)")
    ax.legend(loc="upper left", borderaxespad=0.3, labelspacing=0.6)
    save_static(fig, "interval_cost", (1000, 460))


# Hero animation

HERO_W, HERO_H = 1000, 640
FPS = 20
CLEAR_S = 0.30
PHASE_S = 0.55
HOLD_S = 0.30
PULSE_S = 0.125
CENTRES = np.array([[250.0, 326.0], [750.0, 326.0]])
R_DATA = 222.0
PHASE_COLOURS = (TEAL, AMBER, VIOLET)
PHASE_OF_PART = {"L1": 0, "L2": 1, "W2": 1, "L3": 2}

# Edge states: (alpha, width in px) for pulse, phase trace and round trace.
EDGE_STATES = np.array([[1.0, 2.1], [0.40, 1.05], [0.16, 0.95]])


def part_phase(name: str) -> int:
    phase = PHASE_OF_PART.get(name[:2])
    if phase is None:
        raise SystemExit(f"r8_paths.json: unknown part {name}")
    return phase


def qubit_layout(data: dict) -> dict[str, np.ndarray]:
    """Virtual data slots around a polar ancilla grid, in pixels.

    Ancilla names are local to each part. Their position near a data block does
    not assign hardware ancillas to that block, even for single-block checks.
    """
    blocks, pool = data["blocks"], data["ancilla_pool"]
    if len(blocks) != 5 or any(len(b) != 12 for b in blocks) or pool != 32:
        raise SystemExit("r8_paths.json: expected five blocks of twelve and a pool of 32")
    if sorted(q for b in blocks for q in b) != list(range(data["data_qubits"])):
        raise SystemExit("r8_paths.json: blocks do not cover the data qubits")
    step, gap = 4.0, 7.0
    offsets, angle = [], 0.0
    for i in range(12):
        offsets.append(angle)
        angle += gap if i in (3, 7) else step
    span = offsets[-1]
    pos: dict[str, np.ndarray] = {}
    for b, members in enumerate(blocks):
        centre = 90.0 - 72.0 * b
        for i, q in enumerate(members):
            a = math.radians(centre + span / 2 - offsets[i])
            pos[f"d{q}"] = R_DATA * np.array([math.cos(a), math.sin(a)])
    r_out, r_in, r_mid = 0.37 * R_DATA, 0.20 * R_DATA, 0.068 * R_DATA
    sector = [(r_out, 27.0), (r_out, 9.0), (r_in, 18.0), (r_in, -18.0), (r_out, -9.0), (r_out, -27.0)]
    for n in range(pool):
        if n < 30:
            b, j = divmod(n, 6)
            radius, offset = sector[j]
            a = math.radians(90.0 - 72.0 * b + offset)
            pos[f"a{n}"] = radius * np.array([math.cos(a), math.sin(a)])
        else:
            pos[f"a{n}"] = np.array([(-1.0 if n == 30 else 1.0) * r_mid, 0.0])
    return pos


def bezier(p: np.ndarray, q: np.ndarray, bend: float, n: int = 20) -> np.ndarray:
    d = q - p
    control = (p + q) / 2.0 + bend * np.array([-d[1], d[0]])
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - t) ** 2 * p + 2 * (1 - t) * t * control + t**2 * q


def check_paths(data: dict) -> tuple[int, int]:
    parts = data["parts"]
    names = {f"d{q}" for q in range(data["data_qubits"])} | {f"a{n}" for n in range(data["ancilla_pool"])}
    for name, part in parts.items():
        layers = part["gate_layers"]
        if sum(len(layer) for layer in layers) != part["two_qubit_gates"]:
            raise SystemExit(f"r8_paths.json: {name} gate layers do not sum to two_qubit_gates")
        if name.startswith("W2_") and layers:
            raise SystemExit(f"r8_paths.json: {name} unexpectedly records gates")
        for layer in layers:
            for pair in layer:
                if len(pair) != 2 or not set(pair) <= names or pair[0] == pair[1]:
                    raise SystemExit(f"r8_paths.json: {name} has an unresolved pair {pair}")
                if pair[0][0] == "d" and pair[1][0] == "d":
                    raise SystemExit(f"r8_paths.json: {name} has a data-data pair {pair}")
    paths = data["paths"]
    rounds = data["rounds_after_preparation"] + 1
    if len(paths) != data["shots"] or any(len(p) != rounds for p in paths):
        raise SystemExit("r8_paths.json: paths do not match shots and rounds")
    for path in paths:
        for parts_in_round in path:
            phases = [part_phase(p) for p in parts_in_round]
            if phases != sorted(phases) or any(p not in parts for p in parts_in_round):
                raise SystemExit("r8_paths.json: a round lists unknown or unordered parts")

    def gates(parts_in_round: list[str]) -> int:
        return sum(parts[p]["two_qubit_gates"] for p in parts_in_round)

    full_round = data["full_round_parts"]
    if gates(full_round) != data["full_round_two_qubit_gates"] or data["full_round_two_qubit_gates"] != 560:
        raise SystemExit("r8_paths.json: the full reference round does not contain 560 gates")
    memory = [r for p in paths for r in p[1:]]
    mean = sum(gates(r) for r in memory) / len(memory)
    escalated = sum("L3" in r for r in memory) / len(memory)
    if not math.isclose(mean, data["mean_two_qubit_gates_per_round"], rel_tol=1e-12):
        raise SystemExit("r8_paths.json: path gate counts disagree with the stored mean")
    if not math.isclose(escalated, data["escalated_round_fraction"], rel_tol=1e-12):
        raise SystemExit("r8_paths.json: path escalations disagree with the stored fraction")
    first = next(i for i, p in enumerate(paths) if all("L3" not in r for r in p[1:]))
    second = next(
        i for i, p in enumerate(paths) if "L3" not in p[1] and any("L3" in r for r in p[2:])
    )
    if (first, second) != (0, 13):
        raise SystemExit(f"r8_paths.json: selected shots {first} and {second}, expected 0 and 13")
    for shot in (first, second):
        path = paths[shot]
        if path[0] != ["L1_a", "L2_all", "L3"] or gates(path[0]) != 560:
            raise SystemExit(f"r8_paths.json: shot {shot} does not start with full preparation")
        if "L3r_1" not in path[1]:
            raise SystemExit(f"r8_paths.json: shot {shot} does not begin memory with L3r_1")
    return first, second


def full_reference_round(data: dict, r: int) -> list[str]:
    l1 = "L1_a" if r % 2 == 0 else "L1_b"
    parts = [l1 if p.startswith("L1") else p for p in data["full_round_parts"]]
    if sum(data["parts"][p]["two_qubit_gates"] for p in parts) != 560:
        raise SystemExit("r8_paths.json: a full reference round does not contain 560 gates")
    return parts


def build_schedule(data: dict, shots: tuple[int, int]):
    """Edge records with display windows for both sides and both episodes."""
    pos = qubit_layout(data)
    rounds = data["rounds_after_preparation"] + 1
    round_s = 3 * PHASE_S + HOLD_S
    episode_s = CLEAR_S + rounds * round_s
    curves: dict[tuple[str, str], np.ndarray] = {}
    vertex_id = {}
    for side in range(2):
        for q in range(data["data_qubits"]):
            vertex_id[(side, f"d{q}")] = side * 92 + q
        for n in range(data["ancilla_pool"]):
            vertex_id[(side, f"a{n}")] = side * 92 + 60 + n
    rec = {k: [] for k in ("poly", "phase", "start", "pend", "rend", "n0", "n1")}
    labels = []
    for e, shot in enumerate(shots):
        t_episode = e * episode_s
        labels.append((t_episode, t_episode + CLEAR_S, "Preparation"))
        for r in range(rounds):
            t_round = t_episode + CLEAR_S + r * round_s
            labels.append((t_round, t_round + round_s, "Preparation" if r == 0 else f"Round {r}"))
            side_parts = [
                data["paths"][shot][0] if r == 0 else full_reference_round(data, r),
                data["paths"][shot][r],
            ]
            for side, parts in enumerate(side_parts):
                drawn = len(rec["start"])
                for phase in range(3):
                    layers = [
                        layer
                        for p in parts
                        if part_phase(p) == phase
                        for layer in data["parts"][p]["gate_layers"]
                        if layer
                    ]
                    t_phase = t_round + phase * PHASE_S
                    for j, layer in enumerate(layers):
                        start = t_phase + j * PHASE_S / len(layers)
                        for a, b in layer:
                            if a[0] == "a" and b[0] == "d":
                                a, b = b, a
                            if a[0] == "a" and b[0] == "a" and int(a[1:]) > int(b[1:]):
                                a, b = b, a
                            key = (a, b)
                            if key not in curves:
                                bend = 0.30 if a[0] == "a" else 0.13
                                curves[key] = bezier(pos[a], pos[b], bend)
                            rec["poly"].append(curves[key] + CENTRES[side])
                            rec["phase"].append(phase)
                            rec["start"].append(start)
                            rec["pend"].append(t_phase + PHASE_S)
                            rec["rend"].append(t_round + round_s)
                            rec["n0"].append(vertex_id[(side, a)])
                            rec["n1"].append(vertex_id[(side, b)])
                drawn = len(rec["start"]) - drawn
                if drawn != sum(data["parts"][p]["two_qubit_gates"] for p in parts) or (side == 0 and drawn != 560):
                    raise SystemExit(f"r8_paths.json: shot {shot} round {r} draws {drawn} edges on side {side}")
    arrays = {k: np.asarray(v) for k, v in rec.items()}
    total_s = len(shots) * episode_s
    return pos, arrays, labels, total_s


def place_legend(ax, renderer, y: float, data_style: dict, anc_style: dict) -> None:
    items = [("line", c, name) for c, name in zip(PHASE_COLOURS, ("Inner checks", "Block checks", "Outer checks"))]
    items += [("data", None, "Data"), ("anc", None, "Ancilla")]
    sample_w, text_gap, item_gap = 24.0, 8.0, 30.0
    texts = [ax.text(0, y, name, fontsize=12.5, color=INK, va="center", ha="left") for *_, name in items]
    widths = [t.get_window_extent(renderer).width for t in texts]
    total = sum(sample_w + text_gap + w for w in widths) + item_gap * (len(items) - 1)
    x = (HERO_W - total) / 2.0
    for (kind, c, _), text, w in zip(items, texts, widths):
        if kind == "line":
            ax.plot([x, x + sample_w], [y, y], color=c, lw=px(2.6), solid_capstyle="round")
        elif kind == "data":
            ax.scatter([x + sample_w / 2], [y], marker="o", **data_style)
        else:
            ax.scatter([x + sample_w / 2], [y], marker="s", **anc_style)
        text.set_x(x + sample_w + text_gap)
        x += sample_w + text_gap + w + item_gap


def hero_canvas(data: dict, shots: tuple[int, int]):
    """Return a function drawing the animation at time t as an RGB array, the loop length and the figure."""
    pos, rec, labels, total_s = build_schedule(data, shots)
    phase_rgb = np.array([to_rgb(c) for c in PHASE_COLOURS])

    fig = new_figure(HERO_W, HERO_H)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, HERO_W)
    ax.set_ylim(0, HERO_H)
    ax.axis("off")
    lines = LineCollection([], capstyle="round", zorder=2)
    ax.add_collection(lines)

    vertex_xy = np.zeros((184, 2))
    for side in range(2):
        for q in range(data["data_qubits"]):
            vertex_xy[side * 92 + q] = pos[f"d{q}"] + CENTRES[side]
        for n in range(data["ancilla_pool"]):
            vertex_xy[side * 92 + 60 + n] = pos[f"a{n}"] + CENTRES[side]
    is_data = np.zeros(184, bool)
    is_data[np.r_[0:60, 92:152]] = True
    base_face = np.where(is_data[:, None], np.array(to_rgb(BG)), np.array(to_rgb(ANCILLA_FILL)))
    data_style = dict(s=px(10.5) ** 2, facecolors=BG, edgecolors=DATA_EDGE, linewidths=px(1.2), zorder=3)
    anc_style = dict(s=px(7.5) ** 2, facecolors=ANCILLA_FILL, edgecolors="none", linewidths=0, zorder=3)
    data_vertices = ax.scatter(vertex_xy[is_data, 0], vertex_xy[is_data, 1], marker="o", **data_style)
    anc_vertices = ax.scatter(vertex_xy[~is_data, 0], vertex_xy[~is_data, 1], marker="s", **anc_style)

    for side, name in enumerate(("Full Carbon", "R8")):
        ax.text(CENTRES[side, 0], 606, name, fontsize=16, color=INK, ha="center", va="center")
    state_label = ax.text(500, 606, "", fontsize=13.5, color=MUTED, ha="center", va="center")
    fig.canvas.draw()
    place_legend(ax, fig.canvas.get_renderer(), 30.0, data_style, anc_style)

    def draw(t: float) -> np.ndarray:
        state_label.set_text(next((text for a, b, text in labels if a <= t < b), labels[-1][2]))
        idx = np.nonzero((rec["start"] <= t) & (t < rec["rend"]))[0]
        lit = t < rec["start"][idx] + PULSE_S
        in_phase = t < rec["pend"][idx]
        state = np.where(lit & in_phase, 0, np.where(in_phase, 1, 2))
        order = np.argsort(-state, kind="stable")
        idx, state, in_phase = idx[order], state[order], in_phase[order]
        colour_of = phase_rgb[rec["phase"][idx]]
        lines.set_segments(rec["poly"][idx])
        lines.set_color(np.column_stack([colour_of, EDGE_STATES[state, 0]]))
        lines.set_linewidth(px(1.0) * EDGE_STATES[state, 1])

        tint = np.zeros(184)
        active = idx[in_phase]
        colour = phase_rgb[0]
        if active.size:
            level = np.where(state[in_phase] == 0, 0.85, 0.22)
            np.maximum.at(tint, rec["n0"][active], level)
            np.maximum.at(tint, rec["n1"][active], level)
            colour = phase_rgb[rec["phase"][active[0]]]
        face = base_face * (1 - tint[:, None]) + colour * tint[:, None]
        data_vertices.set_facecolors(face[is_data])
        anc_vertices.set_facecolors(face[~is_data])
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()

    return draw, total_s, fig


def render_frames(data: dict, shots: tuple[int, int]) -> int:
    draw, total_s, fig = hero_canvas(data, shots)
    n_frames = int(round(total_s * FPS))
    FRAMES.mkdir(parents=True, exist_ok=True)
    for old in FRAMES.glob("*.png"):
        old.unlink()
    for i in range(n_frames):
        Image.fromarray(draw(i / FPS)).save(FRAMES / f"{i:04d}.png", compress_level=1)
    plt.close(fig)
    return n_frames


def frame_paths() -> list[Path]:
    return sorted(FRAMES.glob("*.png"))


GIF_COLOURS = 255  # one index stays free to mark unchanged pixels between frames


def gif_palette(paths: list[Path]) -> np.ndarray:
    """One palette for every frame: explicit ramps for the drawn colours, filled by sampled residuals."""
    bg = rgb255(BG)
    explicit = []
    hue_levels = np.r_[np.linspace(0.03, 0.30, 16), np.linspace(0.34, 1.0, 20)]
    for c in PHASE_COLOURS:
        explicit += [bg + (rgb255(c) - bg) * a for a in hue_levels]
    for c, n in ((INK, 28), (DATA_EDGE, 10), (ANCILLA_FILL, 8), (MUTED, 10)):
        explicit += [bg + (rgb255(c) - bg) * a for a in np.linspace(0.0, 1.0, n)]
    explicit = np.unique(np.round(np.array(explicit)).astype(np.int64), axis=0)
    samples = [np.asarray(Image.open(p).convert("RGB")).reshape(-1, 3) for p in paths[:: max(1, len(paths) // 40)]]
    pixels = np.concatenate(samples).astype(np.int64)
    packed = (pixels[:, 0] << 16) | (pixels[:, 1] << 8) | pixels[:, 2]
    colours, counts = np.unique(packed, return_counts=True)
    rgb = np.column_stack([(colours >> 16) & 255, (colours >> 8) & 255, colours & 255])
    nearest = np.full(len(rgb), np.inf)
    for chunk in range(0, len(explicit), 32):
        d = ((rgb[:, None, :] - explicit[None, chunk : chunk + 32, :]) ** 2).sum(-1)
        nearest = np.minimum(nearest, d.min(1))
    residual = nearest > 9.0
    free = GIF_COLOURS - len(explicit)
    extra = np.zeros((0, 3), np.int64)
    if residual.any() and free > 0:
        reps = np.minimum(counts[residual], 400)
        res_pixels = np.repeat(rgb[residual], reps, axis=0).astype(np.uint8)
        side = int(math.ceil(math.sqrt(len(res_pixels))))
        canvas = np.zeros((side * side, 3), np.uint8)
        canvas[: len(res_pixels)] = res_pixels
        canvas[len(res_pixels) :] = res_pixels[0]
        quant = Image.fromarray(canvas.reshape(side, side, 3)).quantize(colors=free, method=Image.Quantize.MEDIANCUT)
        extra = np.array(quant.getpalette()[: 3 * free], np.int64).reshape(-1, 3)
    return np.unique(np.concatenate([explicit, extra]), axis=0)[:GIF_COLOURS].astype(np.uint8)


def nearest_index(colours: np.ndarray, palette: np.ndarray) -> np.ndarray:
    pal = palette.astype(np.int32)[None, :, :]
    best = np.empty(len(colours), np.int16)
    for start in range(0, len(colours), 4096):
        chunk = colours[start : start + 4096].astype(np.int32)[:, None, :]
        best[start : start + 4096] = ((chunk - pal) ** 2).sum(-1).argmin(1)
    return best


def write_gif(paths: list[Path], target: Path) -> list[Image.Image]:
    """Quantise frames without dithering and encode unchanged pixels compactly.

    The GIF may use transparent differences between frames. check_gif verifies
    opaque decoded frames against the palette-mapped images, not the original RGB.
    """
    palette = gif_palette(paths)
    flat = palette.reshape(-1).tolist()
    lookup = np.full(1 << 24, -1, np.int16)
    frames = []
    for path in paths:
        rgb = np.asarray(Image.open(path).convert("RGB")).reshape(-1, 3).astype(np.int64)
        packed = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]
        colours, inverse = np.unique(packed, return_inverse=True)
        todo = colours[lookup[colours] < 0]
        if todo.size:
            lookup[todo] = nearest_index(np.column_stack([todo >> 16, (todo >> 8) & 255, todo & 255]), palette)
        frame = Image.new("P", (HERO_W, HERO_H))
        frame.frombytes(lookup[colours][inverse].astype(np.uint8).tobytes())
        frame.putpalette(flat)
        frames.append(frame)
    frames[0].save(
        target,
        save_all=True,
        append_images=frames[1:],
        duration=int(round(1000 / FPS)),
        loop=0,
        optimize=True,
        disposal=1,
    )
    return frames


def check_gif(target: Path, frames: list[Image.Image]) -> None:
    expected = [np.asarray(frames[0].convert("RGB"))]
    for frame in frames[1:]:
        current = np.asarray(frame.convert("RGB"))
        if not np.array_equal(current, expected[-1]):
            expected.append(current)
    with Image.open(target) as gif:
        if gif.size != (HERO_W, HERO_H) or gif.info.get("loop") != 0:
            raise SystemExit("r8_paths.gif: wrong size or not looping")
        decoded = 0
        for frame, want in zip(ImageSequence.Iterator(gif), expected):
            if frame.mode == "RGBA" and np.asarray(frame.getchannel("A")).min() < 255:
                raise SystemExit(f"r8_paths.gif: decoded frame {decoded} is not opaque")
            if not np.array_equal(np.asarray(frame.convert("RGB")), want):
                raise SystemExit(f"r8_paths.gif: decoded frame {decoded} differs from its source")
            decoded += 1
        if decoded != len(expected) or gif.n_frames != len(expected):
            raise SystemExit("r8_paths.gif: decoded frame count differs from the source")


def write_mp4(ffmpeg: str, target: Path) -> None:
    command = [
        ffmpeg, "-y", "-v", "error",
        "-framerate", str(FPS), "-i", str(FRAMES / "%04d.png"),
        "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
        "-c:v", "libx264", "-preset", "slow", "-crf", "19", "-tune", "animation",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        "-movflags", "+faststart", "-an", str(target),
    ]
    subprocess.run(command, check=True)


def check_mp4(ffmpeg: str, target: Path) -> None:
    probe = subprocess.run([ffmpeg, "-hide_banner", "-i", str(target)], capture_output=True, text=True)
    streams = [line for line in probe.stderr.splitlines() if "Stream #" in line]
    video = [s for s in streams if "Video:" in s]
    if len(streams) != 1 or not video or "Video: h264" not in video[0] or "yuv420p" not in video[0]:
        raise SystemExit("r8_paths.mp4: expected one H.264 yuv420p video stream and no audio")
    if f"{HERO_W}x{HERO_H}" not in video[0]:
        raise SystemExit("r8_paths.mp4: wrong frame size")
    atoms, raw = [], target.read_bytes()
    offset = 0
    while offset + 8 <= len(raw):
        size = int.from_bytes(raw[offset : offset + 4], "big")
        atoms.append(raw[offset + 4 : offset + 8].decode("latin-1"))
        if size == 1:
            size = int.from_bytes(raw[offset + 8 : offset + 16], "big")
        if size < 8:
            break
        offset += size
    if "moov" not in atoms or "mdat" not in atoms or atoms.index("moov") > atoms.index("mdat"):
        raise SystemExit("r8_paths.mp4: missing fast-start metadata")


def hero(ffmpeg: str) -> None:
    with open(DATA / "r8_paths.json") as handle:
        data = json.load(handle)
    shots = check_paths(data)
    if not any("L3" in r for r in data["paths"][shots[1]][2:]):
        raise SystemExit("r8_paths.json: the later selected shot does not escalate")
    mp4, gif = OUT / "r8_paths.mp4", OUT / "r8_paths.gif"
    try:
        render_frames(data, shots)
        paths = frame_paths()
        write_mp4(ffmpeg, mp4)
        check_mp4(ffmpeg, mp4)
        frames = write_gif(paths, gif)
        check_gif(gif, frames)
    finally:
        shutil.rmtree(FRAMES, ignore_errors=True)
    for target in (mp4, gif):
        if target.stat().st_size >= BYTE_LIMIT:
            raise SystemExit(f"{target.name}: {target.stat().st_size} bytes exceeds the limit")


def main() -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("make_figures.py: ffmpeg is required to write figures/r8_paths.mp4; install it and rerun.")
    style()
    memory_rate()
    extraction_cost()
    fault_order()
    interval_cost()
    hero(ffmpeg)
    outputs = [f"{stem}.{ext}" for stem in STATIC for ext in ("png", "svg")] + ["r8_paths.gif", "r8_paths.mp4"]
    for name in outputs:
        if not (OUT / name).is_file():
            raise SystemExit(f"figures/{name} was not written")
        print(f"figures/{name}: {(OUT / name).stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
