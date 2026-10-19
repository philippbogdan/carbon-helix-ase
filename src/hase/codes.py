"""Code constructions: [[10,2,3]] twisted toric code, C4 and Carbon inner codes,
and their concatenations C4-Helix [[20,2,6]] and Carbon-Helix [[60,2,12]].

Conventions: a CSS code is given by X-type stabilizer rows ``hx`` (support of X
operators), Z-type rows ``hz``, and logical operators ``lx``/``lz`` (k rows each,
paired so that lx[i] anticommutes with lz[i] only).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np

from .gf2 import as_gf2, rank, nullspace, in_rowspace


def _rows(n, supports):
    m = np.zeros((len(supports), n), dtype=np.uint8)
    for i, s in enumerate(supports):
        m[i, list(s)] = 1
    return m


@dataclass
class CSSCode:
    name: str
    hx: np.ndarray
    hz: np.ndarray
    lx: np.ndarray
    lz: np.ndarray
    # optional structure for concatenated codes
    blocks: list = field(default_factory=list)          # list of lists of physical qubits
    hx_level: list = field(default_factory=list)        # 'inner'/'outer' for each hx row
    hz_level: list = field(default_factory=list)
    hx_block: list = field(default_factory=list)        # block index of inner rows (-1 outer)
    hz_block: list = field(default_factory=list)

    @property
    def n(self):
        return self.hx.shape[1]

    @property
    def k(self):
        return self.lx.shape[0]

    def check(self):
        hx, hz, lx, lz = self.hx, self.hz, self.lx, self.lz
        assert not ((hx.astype(int) @ hz.T.astype(int)) % 2).any(), "stabilizers anticommute"
        assert not ((hx.astype(int) @ lz.T.astype(int)) % 2).any(), "lz anticommutes with hx"
        assert not ((hz.astype(int) @ lx.T.astype(int)) % 2).any(), "lx anticommutes with hz"
        sym = (lx.astype(int) @ lz.T.astype(int)) % 2
        assert (sym == np.eye(self.k, dtype=int)).all(), f"logical pairing broken {sym}"
        k = self.n - rank(hx) - rank(hz)
        assert k == self.k, f"k mismatch {k} vs {self.k}"
        return True


# ---------------------------------------------------------------- outer code
def twisted_toric_10() -> CSSCode:
    """[[10,2,3]] twisted toric (symplectic double of [[5,1,3]]), rows as in
    Table I (left) of arXiv:2609.03194 (5 X rows and 5 Z rows, one redundant each)."""
    n = 10
    hx = _rows(n, [{0, 3, 6, 7}, {1, 4, 7, 8}, {0, 2, 8, 9}, {1, 3, 5, 9}, {2, 4, 5, 6}])
    hz = _rows(n, [{1, 2, 5, 8}, {2, 3, 6, 9}, {3, 4, 5, 7}, {0, 4, 6, 8}, {0, 1, 7, 9}])
    lx = _rows(n, [{2, 3, 5}, {0, 6, 9}])
    lz = _rows(n, [{1, 4, 5}, {0, 7, 8}])
    c = CSSCode("[[10,2,3]]", hx, hz, lx, lz)
    c.check()
    return c


# ---------------------------------------------------------------- inner codes
def c4_code() -> CSSCode:
    """[[4,2,2]] with logicals X0=XIIX, Z0=IIZZ, X1=IIXX, Z1=ZIIZ (arXiv:2609.03194)."""
    n = 4
    hx = _rows(n, [{0, 1, 2, 3}])
    hz = _rows(n, [{0, 1, 2, 3}])
    lx = _rows(n, [{0, 3}, {2, 3}])
    lz = _rows(n, [{2, 3}, {0, 3}])
    c = CSSCode("C4", hx, hz, lx, lz)
    c.check()
    return c


def carbon_code() -> CSSCode:
    """[[12,2,4]] Carbon code, stabilizers and logicals of Table IV of arXiv:2404.02280."""
    n = 12
    hx = _rows(n, [{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11},
                   {0, 1, 5, 7, 8, 11}, {0, 3, 4, 5, 9, 11}])
    hz = _rows(n, [{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11},
                   {0, 2, 6, 7, 8, 11}, {0, 3, 4, 6, 10, 11}])
    lx = _rows(n, [{0, 3, 10, 11}, {1, 3, 9, 10}])
    lz = _rows(n, [{0, 3, 9, 11}, {0, 1, 9, 10}])
    c = CSSCode("Carbon", hx, hz, lx, lz)
    c.check()
    return c


# ---------------------------------------------------------------- concatenation
def min_weight_rep(v: np.ndarray, stab: np.ndarray) -> np.ndarray:
    """Minimum-weight element of the coset v + rowspace(stab) (brute force, small)."""
    stab = as_gf2(stab)
    m = stab.shape[0]
    best = v.copy()
    for coeffs in product([0, 1], repeat=m):
        w = v.copy()
        for c, row in zip(coeffs, stab):
            if c:
                w ^= row
        if w.sum() < best.sum():
            best = w
    return best


def helix_pairing(i: int):
    """Outer qubit i -> (block, inner logical index): pairs {i, i+5} share a block."""
    return i % 5, (0 if i < 5 else 1)


def concatenate(outer: CSSCode, inner: CSSCode, name: str, reduce_weight=True) -> CSSCode:
    nb = outer.n // inner.k
    ni = inner.n
    n = nb * ni
    blocks = [list(range(b * ni, (b + 1) * ni)) for b in range(nb)]

    def embed(vec_inner, b):
        v = np.zeros(n, dtype=np.uint8)
        v[blocks[b]] = vec_inner
        return v

    def lift(outer_row, inner_logicals, inner_stabs):
        v = np.zeros(n, dtype=np.uint8)
        per_block = [np.zeros(ni, dtype=np.uint8) for _ in range(nb)]
        for i in np.nonzero(outer_row)[0]:
            b, j = helix_pairing(int(i))
            per_block[b] ^= inner_logicals[j]
        for b in range(nb):
            pb = per_block[b]
            if reduce_weight and pb.any():
                pb = min_weight_rep(pb, inner_stabs)
            v[blocks[b]] = pb
        return v

    hx_rows, hz_rows, hx_level, hz_level, hx_block, hz_block = [], [], [], [], [], []
    for b in range(nb):
        for r in inner.hx:
            hx_rows.append(embed(r, b)); hx_level.append("inner"); hx_block.append(b)
        for r in inner.hz:
            hz_rows.append(embed(r, b)); hz_level.append("inner"); hz_block.append(b)
    for r in outer.hx:
        hx_rows.append(lift(r, inner.lx, inner.hx)); hx_level.append("outer"); hx_block.append(-1)
    for r in outer.hz:
        hz_rows.append(lift(r, inner.lz, inner.hz)); hz_level.append("outer"); hz_block.append(-1)
    lx = np.array([lift(r, inner.lx, inner.hx) for r in outer.lx], dtype=np.uint8)
    lz = np.array([lift(r, inner.lz, inner.hz) for r in outer.lz], dtype=np.uint8)
    c = CSSCode(name, np.array(hx_rows, dtype=np.uint8), np.array(hz_rows, dtype=np.uint8),
                lx, lz, blocks, hx_level, hz_level, hx_block, hz_block)
    c.check()
    return c


def c4_helix() -> CSSCode:
    return concatenate(twisted_toric_10(), c4_code(), "C4-Helix [[20,2,6]]")


def carbon_helix() -> CSSCode:
    return concatenate(twisted_toric_10(), carbon_code(), "Carbon-Helix [[60,2,12]]")


def c4_helix_paper_table() -> CSSCode:
    """The [[20,2,6]] generators exactly as printed in Table I (right) of arXiv:2609.03194."""
    n = 20
    inner = [set(range(4 * b, 4 * b + 4)) for b in range(5)]
    outer = [{0, 3, 6, 7, 10, 11, 12, 15}, {4, 7, 8, 9, 14, 15, 16, 19},
             {1, 2, 8, 11, 12, 13, 18, 19}, {2, 3, 5, 6, 13, 14, 16, 17},
             {0, 1, 4, 5, 9, 10, 17, 18}]
    h = _rows(n, inner + outer)
    lx = _rows(n, [{2, 3, 8, 11, 12, 15}, {0, 3, 6, 7, 18, 19}])
    lz = _rows(n, [{0, 3, 6, 7, 18, 19}, {2, 3, 8, 11, 12, 15}])
    c = CSSCode("C4-Helix (paper table)", h.copy(), h.copy(), lx, lz,
                [list(range(4 * b, 4 * b + 4)) for b in range(5)],
                ["inner"] * 5 + ["outer"] * 5, ["inner"] * 5 + ["outer"] * 5,
                list(range(5)) + [-1] * 5, list(range(5)) + [-1] * 5)
    c.check()
    return c


# ---------------------------------------------------------------- distances
def css_distance_bruteforce(code: CSSCode, max_weight=None):
    """Exact (dX, dZ) by enumerating cosets: logical Z ops are elements of ker(hx)
    outside rowspace(hz). Feasible for small n only (enumerates rowspace(hz))."""
    def dist(h_commute, h_stab, logicals):
        # enumerate all vectors in span(h_stab + logicals) that contain a logical
        sb = rowspace_basis(as_gf2(h_stab))
        best = None
        for coeffs in product([0, 1], repeat=logicals.shape[0]):
            if not any(coeffs):
                continue
            base = np.zeros(code.n, dtype=np.uint8)
            for c, row in zip(coeffs, logicals):
                if c:
                    base ^= row
            w = min_weight_coset_gray(base, sb)
            best = w if best is None else min(best, w)
        return best
    dz = dist(code.hx, code.hz, code.lz)
    dx = dist(code.hz, code.hx, code.lx)
    return dx, dz


def rowspace_basis(a):
    from .gf2 import rref
    r, piv = rref(a)
    return r


def min_weight_coset_gray(base: np.ndarray, basis: np.ndarray) -> int:
    """Min weight over base + span(basis) using Gray-code enumeration (2^m steps)."""
    m = basis.shape[0]
    # pack rows into python ints for speed
    def to_int(v):
        return int("".join(str(int(x)) for x in v[::-1]), 2) if v.size else 0
    b_ints = [to_int(r) for r in basis]
    cur = to_int(base)
    best = bin(cur).count("1")
    for i in range(1, 1 << m):
        # bit that flips in gray code
        j = (i & -i).bit_length() - 1
        cur ^= b_ints[j]
        w = cur.bit_count()
        if w < best:
            best = w
    return best


def concatenated_distance(outer: CSSCode, inner: CSSCode):
    """Exact distance of the Helix concatenation via block decomposition.

    A logical Z of the concatenated code restricts on each block to an inner coset
    (a,b) (inner logical class); the outer vector of classes must be a nontrivial
    outer Z logical. Its minimum weight is sum over blocks of the minimum weight of
    the inner coset. Returns (dX, dZ, details)."""
    nb = outer.n // inner.k

    def inner_coset_weights(logicals, stabs):
        sb = rowspace_basis(stabs)
        w = {}
        for a, b in product([0, 1], repeat=2):
            v = np.zeros(inner.n, dtype=np.uint8)
            if a:
                v ^= logicals[0]
            if b:
                v ^= logicals[1]
            w[(a, b)] = min_weight_coset_gray(v, sb) if (a or b) else 0
        return w

    def outer_min(h_commute, h_stab, logicals, cw):
        # enumerate all outer vectors in ker(h_commute) (2^dim, tiny)
        ker = nullspace(h_commute)
        best = None
        arg = None
        for coeffs in product([0, 1], repeat=ker.shape[0]):
            v = np.zeros(outer.n, dtype=np.uint8)
            for c, row in zip(coeffs, ker):
                if c:
                    v ^= row
            if not v.any() or in_rowspace(v, h_stab):
                continue
            tot = 0
            for blk in range(nb):
                cls = [0, 0]
                for i in range(outer.n):
                    bb, j = helix_pairing(i)
                    if bb == blk and v[i]:
                        cls[j] = 1
                tot += cw[tuple(cls)]
            if best is None or tot < best:
                best, arg = tot, v
        return best, arg

    cwz = inner_coset_weights(inner.lz, inner.hz)
    cwx = inner_coset_weights(inner.lx, inner.hx)
    dz, vz = outer_min(outer.hx, outer.hz, outer.lz, cwz)
    dx, vx = outer_min(outer.hz, outer.hx, outer.lx, cwx)
    return dx, dz, {"inner_coset_weights_X": cwx, "inner_coset_weights_Z": cwz,
                    "outer_argmin_X": vx, "outer_argmin_Z": vz}
