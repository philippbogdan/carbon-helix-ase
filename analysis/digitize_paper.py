"""Digitize the TQRB simulation figures of arXiv:2609.03194 (given material, data/papers):

  Fig. 12a (frontier_extrapolation_logic.png): non-adaptive TQRB, m = 6 random Cliffords
      and the inverse, error per Clifford 1 - (1 - p_L)^(1/7) against p; the gamma = 0
      markers (no postselection, black) at p = 7e-4, 9e-4, 1.1e-3.
  Fig. 13 (logic_sims.png): p_L(m) at m = 3, 5, 7, adaptive (filled circles) and
      non-adaptive (open squares), p = 1e-3 (blue) and 5e-4 (red); legend values of eps_C.

Axes are calibrated on the decade gridlines found in the image; a marker's centre is the
best match of a template (filled disc or square outline of the marker's size) to the
pixels of its colour, which is insensitive to the error bars and fit lines drawn through
it. Reading uncertainty: a pixel row is 1/701 (Fig. 12a) and 1/1373 (Fig. 13) of a
decade, i.e. 0.3 % and 0.2 % in value; the markers' own drawing (line widths, overlaps)
dominates and is taken as +-3 % (as for the other digitized values of this work).

Usage: python analysis/digitize_paper.py [--out results/paper_digitized.json]
"""
import json
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG = os.path.join(ROOT, "data/papers/2609.03194-source/figures")


def load(name):
    return np.array(Image.open(os.path.join(FIG, name)).convert("RGB")).astype(int)


def frame(im, rows_frac=0.5, cols_frac=0.25, top_half=False, y_range=None):
    """Bounding frame of the axes (in the top half, or within y_range): rows/columns that
    are mostly dark."""
    H, W, _ = im.shape
    dark = im.sum(axis=2) < 150
    ya, yb = y_range if y_range else (0, H // 2 if top_half else H)
    rows = [y for y in range(ya, yb) if dark[y].sum() > W * rows_frac]
    cols = [x for x in range(W) if dark[ya:yb, x].sum() > (yb - ya) * cols_frac]
    return rows[0], rows[-1], cols[0], cols[-1]


def gridlines(im, y0, y1, x0, x1, level):
    """Centres of the light-grey gridlines of grey value `level` (+-8) inside the frame."""
    sub = im[y0:y1, x0:x1]
    grey = ((np.abs(sub[..., 0] - sub[..., 1]) < 6) & (np.abs(sub[..., 1] - sub[..., 2]) < 6)
            & (np.abs(sub[..., 0] - level) <= 8))

    def runs(arr, thr, off):
        out = []
        for i in np.flatnonzero(arr > thr):
            if out and i - out[-1][-1] <= 1:
                out[-1].append(i)
            else:
                out.append([i])
        return [off + (r[0] + r[-1]) / 2 for r in out]
    return (runs(grey.sum(axis=0), (y1 - y0) * 0.3, x0), runs(grey.sum(axis=1), (x1 - x0) * 0.3, y0))


def _template(shape, r_in, r_out):
    R = int(r_out) + 1
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    if shape == "disc":
        tmpl = (xx ** 2 + yy ** 2) <= r_out ** 2
    elif shape == "ring":
        rr = xx ** 2 + yy ** 2
        tmpl = (rr <= r_out ** 2) & (rr >= r_in ** 2)
    else:
        m = np.maximum(np.abs(xx), np.abs(yy))
        tmpl = (m <= r_out) & (m >= r_in)
    ty, tx = np.nonzero(tmpl)
    return ty - R, tx - R, R


def progression(lines, tol=4.0):
    """The longest subset of `lines` that is equally spaced (decade gridlines; other light
    lines, e.g. a legend frame, are not)."""
    lines = sorted(lines)
    best = lines[:1]
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            d = lines[j] - lines[i]
            if d < 50:
                continue
            sel = [x for x in lines if abs(((x - lines[i]) / d) - round((x - lines[i]) / d)) * d <= tol]
            if len(sel) > len(best):
                best = sel
    return best


def match(parts, xc, ylo, yhi, dx=25):
    """Centre (x, y, score) of a marker template on boolean masks. parts: list of (mask,
    shape, r_in, r_out) with shape 'disc' (radius r_out), 'ring' (radii r_in..r_out) or
    'square' (outline, half-sizes r_in..r_out); the score is the mean over the parts of the
    fraction of their template pixels set in their mask. x within xc +- dx, y in [ylo, yhi].
    The centre is the centroid of the positions within 0.002 of the best score (a disc
    slightly smaller than its marker fits on a small plateau around the centre)."""
    tm = [_template(*pt[1:]) for pt in parts]
    R = max(t[2] for t in tm)
    H = parts[0][0].shape[0]
    ys = np.arange(max(int(ylo), R), min(int(yhi), H - R))
    xs = np.arange(int(xc) - dx, int(xc) + dx + 1)
    S = np.zeros((len(ys), len(xs)))
    for (mask, *_), (ty, tx, _) in zip(parts, tm):
        Sp = np.zeros_like(S)
        for oy, ox in zip(ty, tx):
            Sp += mask[ys[0] + oy:ys[-1] + oy + 1, xs[0] + ox:xs[-1] + ox + 1]
        S += Sp / len(ty) / len(parts)
    best = S.max()
    iy, ix = np.nonzero(S >= best - 0.002)
    return float(xs[ix].mean()), float(ys[iy].mean()), float(best)


def fig13():
    im = load("logic_sims.png")
    top, bot, left, right = frame(im)
    gx, gy = gridlines(im, top + 15, bot - 15, left + 15, right - 15, level=219)
    # x: gridlines at m = 2, 4, 6, 8; y: decades 1e-3 and 1e-4 (the legend box is excluded:
    # only lines spanning most of the width count, see gridlines())
    gx = sorted(gx)
    per_m = (gx[-1] - gx[0]) / 6.0
    x_m0 = gx[0] - 2 * per_m
    gy = sorted(gy)
    dec_y = gy[1] - gy[0]
    y_m3 = gy[0]                          # 1e-3

    def val(y):
        return 10 ** (-3 - (y - y_m3) / dec_y)
    blue = (im[..., 2] > 120) & (im[..., 0] < 90) & (im[..., 1] < 120)
    red = (im[..., 0] > 150) & (im[..., 1] < 60) & (im[..., 2] < 80)
    black = im.max(axis=2) < 60
    pts = {}
    for p, col in ((1e-3, blue), (5e-4, red)):
        for ad, shape in ((1, "disc"), (0, "square")):
            for m in (3, 5, 7):
                xc = x_m0 + m * per_m
                if shape == "disc":
                    # the black edge ring of the filled circle (the fill may be covered by a
                    # square drawn on top of it)
                    x, y, s = match([(black, "ring", 50, 60), (col, "disc", 0, 42)], xc,
                                    y_m3 - 1.2 * dec_y, y_m3 + 1.9 * dec_y)
                else:
                    x, y, s = match([(col, "square", 50, 60)], xc, y_m3 - 1.2 * dec_y, y_m3 + 1.9 * dec_y)
                # the error bar (the paper's 95 % interval): the series-colour line above and below the
                # marker, read in a 3-px column through its centre; when it does not leave the marker,
                # the marker's own extent bounds it
                xi, yi = int(round(x)), int(round(y))
                colm = col[:, xi - 1:xi + 2].any(axis=1)
                R_m = 64

                def reach(step):
                    yy = yi + step * R_m
                    last = None
                    while 0 < yy < im.shape[0] - 1 and colm[yy]:
                        last = yy
                        yy += step
                    return last
                up, dn = reach(-1), reach(1)
                lo_y = dn if dn is not None else yi + R_m
                hi_y = up if up is not None else yi - R_m
                pts[f"p{p:g}_a{ad}_m{m}"] = dict(pL=float(val(y)), x_px=round(x, 1), y_px=round(y, 1), score=round(float(s), 3),
                                                 ci95=[float(val(lo_y)), float(val(hi_y))],
                                                 bar_visible=[dn is not None, up is not None])
    return dict(figure="arXiv:2609.03194 Fig. 13 (logic_sims.png)",
                legend_eps_C={"0.001": dict(adaptive=2.8e-4, nonadaptive=4.2e-4),
                              "0.0005": dict(adaptive=2.6e-5, nonadaptive=4.7e-5)},
                fit="p_L(m) = (eps0 - 1/2)(1 - 2 eps_C)^m + 1/2, eps0 shared by adaptive and non-adaptive",
                calibration={"x_m0": x_m0, "px_per_m": per_m, "y_1e-3": y_m3, "px_per_decade_y": dec_y},
                points=pts)


def colorbar(im):
    """Colour -> gamma table of the colorbar of Figs. 11/12 (ticks every 0.5 inside the bar,
    the middle one of the seven inner ticks labelled 2.0)."""
    H, W, _ = im.shape
    row = im[3500]
    blk = np.flatnonzero(row[int(W * 0.75):].max(axis=1) < 30) + int(W * 0.75)
    runs = []
    for x in blk:
        if runs and x - runs[-1][1] <= 1:
            runs[-1][1] = x
        else:
            runs.append([x, x])
    xl, xr = runs[0][1] + 1, runs[1][0] - 1          # interior between the frame edges
    col = im[:, xr - 10]
    rr = []
    for y in np.flatnonzero(col.max(axis=1) < 30):
        if rr and y - rr[-1][1] <= 1:
            rr[-1][1] = y
        else:
            rr.append([y, y])
    ticks = [(a + b) / 2 for a, b in rr if 5 <= b - a <= 15]
    assert len(ticks) == 7, ticks
    per = (ticks[-1] - ticks[0]) / 6 * 2              # pixels per unit of gamma
    y2 = ticks[3]                                     # gamma = 2.0

    def gamma(y):
        return 2.0 + (y2 - y) / per
    top = [b for a, b in rr if b - a > 15][0] + 1     # inner edge of the top frame line
    ys = np.arange(top + 2, int(y2 + 2 * per) - 12)
    ys = np.array([y for y in ys if min(abs(y - t) for t in ticks) > 9])
    cols = im[ys][:, xl + 5:xr - 20].mean(axis=1)
    return np.array([gamma(y) for y in ys]), cols


def _runs(mask):
    out = []
    for i in np.flatnonzero(mask):
        if out and i - out[-1][1] <= 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return out


def gap_figure(name, top_decade):
    """Figs. 11/12 (forced-gap postselection): for panel a (error per round or per Clifford)
    and panel b (rejection) and each p = 7e-4, 9e-4, 1.1e-3, the value of every marker by
    its gamma. gamma comes from the marker's fill colour (the nearest colorbar colour; the
    fills match it to a few RGB units, alpha-blended lines do not); the black gamma = 0
    markers (panel a) by a disc template. A column 20 px right of the marker centres is
    scanned (clear of the error bars); a fill whose chord is shorter than an unobstructed
    one is placed from its unobstructed end. top_decade: log10 of the topmost major
    gridline of each panel (read off the tick labels)."""
    im = load(name)
    H, W, _ = im.shape
    gam, cb = colorbar(im)
    black = im.max(axis=2) < 40
    out = {}
    for panel, yr in (("a", (0, H // 2)), ("b", (H // 2, H))):
        top, bot, left, right = frame(im, y_range=yr)
        gx, gy = gridlines(im, top + 15, bot - 15, left + 15, right - 15, level=223)
        x4, x3 = gx[0], gx[-1]
        ys = sorted(gy)
        dec = (ys[-1] - ys[0]) / (len(ys) - 1)

        def val(y):
            return 10 ** (top_decade[panel] - (y - ys[0]) / dec)
        pts = {}
        for p in (7e-4, 9e-4, 1.1e-3):
            xc = x4 + (np.log10(p) + 4) * (x3 - x4)
            recs = {}
            if panel == "a":
                x, y, sc = match([(black, "disc", 0, 44)], xc, top + 20, bot - 60)
                recs["0"] = dict(value=float(val(y)), y_px=round(y, 1), x_px=round(x, 1), score=round(sc, 3))
                xc = x
            xs = int(round(xc)) + 20
            col = im[top + 10:bot - 10, xs - 1:xs + 2].mean(axis=1)
            mx, mn = col.max(axis=1), col.min(axis=1)
            fill = (mx >= 25) & (mn < 235) & ~((mx - mn < 8) & (mx > 150))
            cand = []
            for a, b in _runs(fill):
                a, b = a + top + 10, b + top + 10
                if b - a < 30:
                    continue
                c = np.median(im[a + 5:b - 4, xs - 1:xs + 2].reshape(-1, 3), axis=0)
                d = np.sqrt(((cb - c) ** 2).sum(axis=1))
                if d.min() > 6:                      # an alpha-blended line, not a marker
                    continue
                cand.append((a, b, float(gam[int(np.argmin(d))]), float(d.min())))
            L = max(b - a + 1 for a, b, _, _ in cand)
            for a, b, g, dist in cand:
                if b - a + 1 >= L - 6:
                    yc = (a + b) / 2
                else:
                    below = im[min(b + 14, H - 1), xs]
                    obstructed_bottom = below.max() >= 25 and not (below.max() - below.min() < 8 and below.max() > 150) \
                        and below.min() < 235
                    yc = a + (L - 1) / 2 if obstructed_bottom else b - (L - 1) / 2
                recs[f"{round(g * 2) / 2:g}"] = dict(value=float(val(yc)), y_px=round(float(yc), 1), gamma_read=round(g, 2),
                                                     colour_dist=round(dist, 1), chord=int(b - a + 1))
            pts[f"{p:g}"] = recs
        out[panel] = dict(calibration={"x_1e-4": x4, "x_1e-3": x3, "y_top_major": ys[0], "top_decade": top_decade[panel],
                                       "px_per_decade_y": dec}, points=pts)
    return out


def fig14():
    """Fig. 14 (cnot_compare.png): logical error per CNOT of the C4-Helix transversal CNOT
    (blue filled circles) at p = 7e-4, 9e-4, 1.1e-3."""
    im = load("cnot_compare.png")
    top, bot, left, right = frame(im)
    gx, gy = gridlines(im, top + 15, bot - 15, left + 15, right - 15, level=219)
    x4, x3 = gx[0], gx[-1]                    # major gridlines at 1e-4 and 1e-3
    ys = progression(gy)                      # decades 1e-4 (top) ... 1e-8 (bottom)
    dec = (ys[-1] - ys[0]) / (len(ys) - 1)

    def val(y):
        return 10 ** (-4 - (y - ys[0]) / dec)
    blue = (np.abs(im[..., 0] - 24) < 30) & (np.abs(im[..., 1] - 70) < 30) & (np.abs(im[..., 2] - 178) < 40)
    red = (np.abs(im[..., 0] - 214) < 35) & (np.abs(im[..., 1] - 40) < 35) & (np.abs(im[..., 2] - 40) < 35)
    black = im.max(axis=2) < 60
    pts, pts_s = {}, {}
    for p in (7e-4, 9e-4, 1.1e-3):
        xc = x4 + (np.log10(p) + 4) * (x3 - x4)
        x, y, sc = match([(black, "ring", 38, 46), (blue, "disc", 0, 30)], xc, ys[0] - 0.3 * dec, ys[0] + 2 * dec)
        pts[f"{p:g}"] = dict(value=float(val(y)), x_px=round(x, 1), y_px=round(y, 1), score=round(sc, 3))
        x, y, sc = match([(black, "ring", 38, 46), (red, "disc", 0, 30)], xc, ys[0] - 0.3 * dec, ys[0] + 2 * dec)
        pts_s[f"{p:g}"] = dict(value=float(val(y)), x_px=round(x, 1), y_px=round(y, 1), score=round(sc, 3))
    return dict(figure="Fig. 14 (cnot_compare.png): two C4-Helix blocks, 6 transversal CNOTs interleaved with QEC "
                       "cycles on both, logical error per CNOT ('C4-Helix transv.', points), and 6 chain-map CNOTs "
                       "from the distance-5 surface code to C4-Helix's first logical qubit ('Surface-C4-Helix', "
                       "points_surface), uniform noise",
                calibration={"x_1e-4": x4, "x_1e-3": x3, "y_major": ys, "px_per_decade_y": dec},
                points=pts, points_surface=pts_s)


def main():
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else os.path.join(ROOT, "results/paper_digitized.json")
    res = dict(source="arXiv:2609.03194 v1 (data/papers/2609.03194-source/figures)",
               method="analysis/digitize_paper.py: gridline and colorbar calibration, template-matched or "
                      "colour-scanned marker centres; +-3 % reading",
               fig11=dict(figure="Fig. 11 (frontier_extrapolation_memory.png): C4-Helix memory, r = 12, error per "
                                 "round 1 - (1 - p_L)^(1/12) (a) and rejection per round (b), mean of X and Z bases, "
                                 "forced-gap threshold gamma (nats)",
                          panels=gap_figure("frontier_extrapolation_memory.png", {"a": -5, "b": -4})),
               fig12=dict(figure="Fig. 12 (frontier_extrapolation_logic.png): non-adaptive TQRB, m = 6 random "
                                 "Cliffords + the inverse, error per Clifford 1 - (1 - p_L)^(1/7) (a) and rejection "
                                 "(b), forced-gap threshold gamma (nats)",
                          panels=gap_figure("frontier_extrapolation_logic.png", {"a": -4, "b": -3})),
               fig13=fig13(), fig14=fig14())
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    for k in ("fig11", "fig12"):
        for panel, v in res[k]["panels"].items():
            for p, recs in v["points"].items():
                print(k, panel, p, {g: f"{r['value']:.3g}" for g, r in recs.items()})
    for nm, v in res["fig13"]["points"].items():
        print("fig13", nm, f"{v['pL']:.3g}")
    for nm, v in res["fig14"]["points"].items():
        print("fig14", nm, f"{v['value']:.3g}", v)
    for nm, v in res["fig14"]["points_surface"].items():
        print("fig14 surface", nm, f"{v['value']:.3g}", v)


if __name__ == "__main__":
    main()
