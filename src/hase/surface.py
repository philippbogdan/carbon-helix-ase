"""The rotated surface code [[d^2,1,d]] and its syndrome-extraction circuit, for the
surface-code interface of arXiv:2609.03194 (chain-map CNOT into C4-Helix: App. Fig. 14,
'Surface-C4-Helix', and the inter-code GHZ state of its Sec. 4).

Layout: data qubit (i, j), i, j = 0..d-1, slot d*i + j. Faces (i, j), i, j = -1..d-1, act on
the data corners (i, j), (i, j+1), (i+1, j), (i+1, j+1) that exist; a face is X-type when
i + j is even. Bulk faces (weight 4) are all kept; of the weight-2 boundary faces the
X-type ones on the top and bottom rows and the Z-type ones on the left and right columns.
Logical X: column 0; logical Z: row 0.

Extraction: one ancilla per face, all faces at once in four CX layers; X faces visit their
corners in the order NW, NE, SW, SE and Z faces NW, SW, NE, SE, so that the two-qubit
'hook' error an ancilla fault leaves runs across the logical of its own type (X hooks
horizontal, logical X vertical; Z hooks vertical, logical Z horizontal) and does not
lower the distance.
"""
from __future__ import annotations

import numpy as np

from .codes import CSSCode
from .engine import APart

ORDER = {"X": ((0, 0), (0, 1), (1, 0), (1, 1)), "Z": ((0, 0), (1, 0), (0, 1), (1, 1))}
# the mirrored orders: valid as well, but each check's hook then runs along the logical of
# its own type (a sensitivity variant, not the circuit used)
ORDER_ALIGNED = {"X": ORDER["Z"], "Z": ORDER["X"]}


def surface_faces(d=5, order=None):
    """[(kind, (i, j), [corner slots in the face's CX order, None where absent])]."""
    order = order or ORDER
    faces = []
    for i in range(-1, d):
        for j in range(-1, d):
            kind = "X" if (i + j) % 2 == 0 else "Z"
            corners = []
            for di, dj in order[kind]:
                a, b = i + di, j + dj
                corners.append(d * a + b if 0 <= a < d and 0 <= b < d else None)
            w = sum(c is not None for c in corners)
            if w == 4:
                faces.append((kind, (i, j), corners))
            elif w == 2:
                top_bottom = i in (-1, d - 1)
                if (kind == "X" and top_bottom) or (kind == "Z" and not top_bottom):
                    faces.append((kind, (i, j), corners))
    return faces


def rotated_surface_code(d=5) -> CSSCode:
    n = d * d
    faces = surface_faces(d)
    hx = np.array([[1 if q in c else 0 for q in range(n)] for k, _, c in faces if k == "X"], dtype=np.uint8)
    hz = np.array([[1 if q in c else 0 for q in range(n)] for k, _, c in faces if k == "Z"], dtype=np.uint8)
    lx = np.zeros((1, n), dtype=np.uint8)
    lx[0, [d * i for i in range(d)]] = 1
    lz = np.zeros((1, n), dtype=np.uint8)
    lz[0, list(range(d))] = 1
    c = CSSCode(f"rotated surface d={d}", hx, hz, lx, lz)
    c.check()
    return c


def surface_apart(d=5, d_off=0, a_off=0, name="SQ", check_prefix="s", order=None) -> APart:
    """Every face measured once (ancilla per face), data slots shifted by d_off and ancillas
    by a_off; check names (check_prefix + 'x'|'z', index in hx|hz order)."""
    faces = surface_faces(d, order)
    part = APart(name, n_anc=a_off + len(faces))
    resets, cx, meas = [], [[], [], [], []], []
    ix = {"X": 0, "Z": 0}
    for f, (kind, _, corners) in enumerate(faces):
        a = ("a", a_off + f)
        resets.append(("RX" if kind == "X" else "R", (a,)))
        for layer, q in enumerate(corners):
            if q is None:
                continue
            dq = ("d", d_off + q)
            cx[layer].append(("CX", (a, dq)) if kind == "X" else ("CX", (dq, a)))
        key = (check_prefix + kind.lower(), ix[kind])
        ix[kind] += 1
        meas.append(("MX" if kind == "X" else "M", (a,), key))
        part.meas[key] = ("chk", key)
    part.add(resets)
    for lay in cx:
        part.add(lay)
    part.add(meas)
    return part
