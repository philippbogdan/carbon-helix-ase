"""Small GF(2) linear-algebra helpers on numpy uint8 arrays."""
from __future__ import annotations

import numpy as np


def as_gf2(a) -> np.ndarray:
    return (np.asarray(a, dtype=np.int64) % 2).astype(np.uint8)


def rref(a: np.ndarray):
    """Row-reduced echelon form over GF(2). Returns (R, pivot_columns)."""
    m = as_gf2(a).copy()
    rows, cols = m.shape
    pivots = []
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        nz = np.nonzero(m[r:, c])[0]
        if len(nz) == 0:
            continue
        p = r + nz[0]
        if p != r:
            m[[r, p]] = m[[p, r]]
        hit = np.nonzero(m[:, c])[0]
        hit = hit[hit != r]
        if len(hit):
            m[hit] ^= m[r]
        pivots.append(c)
        r += 1
    return m[:r], pivots


def rank(a) -> int:
    a = as_gf2(a)
    if a.size == 0:
        return 0
    return len(rref(a)[1])


def nullspace(a) -> np.ndarray:
    """Basis (rows) of {x : a x = 0}."""
    a = as_gf2(a)
    cols = a.shape[1]
    r, piv = rref(a)
    free = [c for c in range(cols) if c not in set(piv)]
    basis = []
    for f in free:
        x = np.zeros(cols, dtype=np.uint8)
        x[f] = 1
        for i, pc in enumerate(piv):
            if r[i, f]:
                x[pc] = 1
        basis.append(x)
    if not basis:
        return np.zeros((0, cols), dtype=np.uint8)
    return np.array(basis, dtype=np.uint8)


def in_rowspace(v, a) -> bool:
    return rank(np.vstack([a, v])) == rank(a)


def solve(a, b):
    """Find x with x @ a = b (b in rowspace of a). Returns coefficient vector or None."""
    a = as_gf2(a)
    b = as_gf2(b)
    m, n = a.shape
    aug = np.hstack([a.T, b.reshape(-1, 1)])
    r, piv = rref(aug)
    if m in piv:
        return None
    x = np.zeros(m, dtype=np.uint8)
    for i, pc in enumerate(piv):
        x[pc] = r[i, m]
    return x
