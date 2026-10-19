"""Exact lower bound on the circuit fault distance of Carbon-Helix paths by decomposition
(scripts/relaxed_distance.py; docs/methods.md, "Circuit distance by decomposition").

Relaxation (fewer constraints, so its optimum is a lower bound on the true distance): keep
every flag, every detector of the inner checks of the memory basis, and of each outer check of
the memory basis only the sum of its detector chain (its final syndrome); drop the rest.

In the relaxed problem a silent history has every memory-basis outcome unflipped, so the
state of a Carbon block between parts is its residual error modulo the block's invisible
stabilizers: 5 syndrome bits and 2 class bits (128 states). Faults are grouped as
  * block terms: residual on one block, flipping only that block's checks and level-2 flags;
  * gadget terms: in an outer-check part, flipping that gadget's flags, or unflagged with a
    residual on several blocks.
Every gadget's flag-silent combinations are tabulated exactly by breadth-first search over
(flags, residual on its support); every block's cost as a function of the syndromes injected
at the outer-check instances by dynamic programming over the parts (a block's own level-2
part may be optional: min(part, skip) covers every per-block schedule, hence every joint
path). A gadget effect's class part only relabels the final outer pattern, so effects reduce
to syndrome injections per block and a 6-bit outer invariant (outer syndrome, logical flips).

Counting bound (every nonzero reduced gadget effect costs >= 2, a pure-class one >= 6; every
nonzero syndrome injection on a block needs a block fault on that block between the check's
last measurement before the instance and its first one after; such intervals of different
instances meet only for adjacent instances, a fault lying in at most two of them, so block
faults >= sum_b ceil(n_b / 2)):
  r >= 5 active instances: cost >= 2 r + 1 >= 11;
  r = 4: cost <= 10 needs every instance at cost exactly 2;
  r = 3: cost <= 10 needs every instance at cost <= 4 (no pure-class effect);
  r <= 2: any entry.
LongPath.search(A) searches the subset A of active instances exactly under these
restrictions; every subset of at most four instances of a path, with the bound for more,
covers every undetected logical of at most 10 faults on the path.
"""
import itertools
import time
from collections import defaultdict

import numpy as np

from .protocols import make_protocol

INF = 255
BIG = 10 ** 6


def gf2_rank(M):
    M = (np.array(M, dtype=np.uint8) % 2).copy()
    r = 0
    rows, cols = M.shape
    for c in range(cols):
        piv = None
        for i in range(r, rows):
            if M[i, c]:
                piv = i
                break
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        for i in range(rows):
            if i != r and M[i, c]:
                M[i] ^= M[r]
        r += 1
        if r == rows:
            break
    return r


def gf2_solve(A, b):
    """x with x @ A = b (A rows are basis vectors); None if b not in the row space."""
    A = np.array(A, dtype=np.uint8) % 2
    k, n = A.shape
    aug = np.concatenate([A.T, np.array(b, dtype=np.uint8).reshape(-1, 1) % 2], axis=1)   # n x (k+1)
    r = 0
    piv_cols = []
    for c in range(k):
        piv = None
        for i in range(r, n):
            if aug[i, c]:
                piv = i
                break
        if piv is None:
            continue
        aug[[r, piv]] = aug[[piv, r]]
        for i in range(n):
            if i != r and aug[i, c]:
                aug[i] ^= aug[r]
        piv_cols.append(c)
        r += 1
    if aug[r:, k].any():
        return None
    x = np.zeros(k, dtype=np.uint8)
    for i, c in enumerate(piv_cols):
        x[c] = aug[i, k]
    return x


def bfs_table(gens, nbits, max_depth):
    """dist[state] for states of nbits bits reachable as XOR of <= max_depth generators."""
    dist = np.full(1 << nbits, INF, dtype=np.uint8)
    gens = np.unique(np.array([g for g in gens if g], dtype=np.int64))
    dist[0] = 0
    frontier = np.array([0], dtype=np.int64)
    for d in range(1, max_depth + 1):
        if len(frontier) == 0 or len(gens) == 0:
            break
        nxt = []
        for g in gens:
            cand = frontier ^ g
            cand = cand[dist[cand] == INF]
            if len(cand):
                dist[cand] = d
                nxt.append(cand)
        frontier = np.unique(np.concatenate(nxt)) if nxt else np.array([], dtype=np.int64)
    return dist


class Decomp:
    def __init__(self, basis, rounds=1, rule=None, path=None, max_depth=11, verbose=True):
        self.basis = basis
        rule = rule or ("g3=k4cyc,g3order=ZX" if basis == "Z" else "g3=k4cyc,g3order=XZ")
        self.proto = proto = make_protocol("carbonhelix", 1e-3, basis, rounds, rule, "uniform")
        if path is None:
            path = [(r, pt) for r in range(proto.n_rounds) for pt, _ in proto.plan(r)]
        self.path = [tuple(p) for p in path]
        self.max_depth = max_depth
        self.verbose = verbose
        n = self.n = proto.n_data
        self.off = 0 if basis == "Z" else n          # error component the memory-basis checks see
        self.blocks = [list(b) for b in proto.code.blocks]
        self.nb = len(self.blocks)
        self.blk_of = {q: b for b, bl in enumerate(self.blocks) for q in bl}
        self.pos_in_blk = {q: i for bl in self.blocks for i, q in enumerate(bl)}
        self._inner_checks()
        self._outer_functionals()
        self._classify_terms()

    # ------------------------------------------------------------------ code structure
    def _typ(self, c):
        return c[3] if c[0] == "c4" else c[2]

    def _vec(self, supp, b):
        v = np.zeros(len(self.blocks[b]), dtype=np.uint8)
        for q in supp:
            q = int(q)
            if self.blk_of.get(q) == b:
                v[self.pos_in_blk[q]] = 1
        return v

    def _inner_checks(self):
        proto = self.proto
        self.inner = {}
        self.Hb, self.Hnb, self.Lrep = {}, {}, {}
        for b in range(self.nb):
            cs = [c for c in proto.check_names if c[0] in ("c4", "c6") and c[1] == b]
            mine = [c for c in cs if self._typ(c) == self.basis]
            other = [c for c in cs if self._typ(c) != self.basis]
            self.inner[b] = mine
            Hb = np.array([self._vec(proto.checks[c].support, b) for c in mine])
            Hnb = np.array([self._vec(proto.checks[c].support, b) for c in other])
            assert len(mine) == 5 and gf2_rank(Hb) == 5 and gf2_rank(Hnb) == 5
            # two class representatives: vectors orthogonal to the other type, independent of Hb
            kern = []
            for v in itertools.product((0, 1), repeat=12):
                v = np.array(v, dtype=np.uint8)
                if not ((Hnb @ v) % 2).any():
                    kern.append(v)
            reps = []
            base = list(Hb)
            for v in kern:
                if gf2_rank(np.array(base + reps + [v])) > len(base) + len(reps):
                    reps.append(v)
                if len(reps) == 2:
                    break
            self.Hb[b], self.Hnb[b], self.Lrep[b] = Hb, Hnb, np.array(reps)
        self.chk_idx = {c: (b, i) for b in range(self.nb) for i, c in enumerate(self.inner[b])}

    def q_of(self, b, v):
        """quotient (5 syndrome bits, 2 class bits) of an error vector v on block b."""
        s = (self.Hb[b] @ v) % 2
        c = (self.Lrep[b] @ v) % 2
        return int(sum(int(x) << i for i, x in enumerate(s)) + (int(c[0]) << 5) + (int(c[1]) << 6))

    def _outer_functionals(self):
        """alpha[name][b] in GF(2)^2: contribution of block b's final class to each outer check
        of the memory basis and to each logical observable."""
        proto = self.proto
        self.outer = [c for c in proto.check_names if c[0] == "out" and c[1] == self.basis]
        self.alpha = {}
        items = [(c, proto.checks[c].support) for c in self.outer] + \
                [(("L", i), s) for i, s in enumerate(proto.logical_supports)]
        for name, supp in items:
            al = []
            for b in range(self.nb):
                P = self._vec(supp, b)
                A = np.concatenate([self.Lrep[b], self.Hb[b]])
                x = gf2_solve(A, P)
                assert x is not None, (name, b)
                al.append((int(x[0]), int(x[1])))
            self.alpha[name] = al

    def outer_patterns(self, combo):
        """class patterns (c_b) of the five blocks with zero outer syndrome flipping the combo."""
        pats = []
        for bits in itertools.product(range(4), repeat=self.nb):
            ok = True
            for c in self.outer:
                par = 0
                for b, cb in enumerate(bits):
                    a = self.alpha[c][b]
                    par ^= (a[0] & cb) ^ (a[1] & (cb >> 1))
                if par:
                    ok = False
                    break
            if not ok:
                continue
            par = 0
            for li in combo:
                for b, cb in enumerate(bits):
                    a = self.alpha[("L", li)][b]
                    par ^= (a[0] & cb) ^ (a[1] & (cb >> 1))
            if par:
                pats.append(bits)
        return pats

    # ------------------------------------------------------------------ fault terms
    def _classify_terms(self):
        proto, n, off = self.proto, self.n, self.off
        self.block_terms = defaultdict(list)     # (k, b) -> list of (dq, mu_checks dict, flags list)
        self.gadget_terms = defaultdict(list)    # (k, gadget) -> list of (flag idx list, residual qubits)
        self.l3_instances = [k for k, (r, pt) in enumerate(self.path) if pt == "L3" or pt.startswith("L3")]
        self.anomalies = defaultdict(int)
        self.l2flags = {}                        # (pt, j) -> (block, local index)
        self.gadget_flags = {}                   # (pt, j) -> (gadget, local index)
        self.gadget_supp = {}
        for k, (r, pt) in enumerate(self.path):
            cp = proto.parts[pt]
            gflag_local = defaultdict(list)
            bflag_local = defaultdict(list)
            for j, (kd, key) in enumerate(zip(cp.meas_kind, cp.meas_keys)):
                if kd != "flag":
                    continue
                if key[0] == "f16":
                    g = (key[1], key[2])
                    gflag_local[g].append(j)
                elif key[0] == "f6":
                    bflag_local[key[1]].append(j)
                else:
                    raise ValueError(key)
            fl_of = {}
            for g, js in gflag_local.items():
                for t, j in enumerate(js):
                    fl_of[j] = ("g", g, t)
                self.gadget_supp[g] = [int(q) for q in proto.checks[("out",) + g].support]
            for b, js in bflag_local.items():
                for t, j in enumerate(js):
                    fl_of[j] = ("b", b, t)
            self.nflags_g = max([len(v) for v in gflag_local.values()], default=getattr(self, "nflags_g", 0))
            self.nflags_b = {b: len(v) for b, v in bflag_local.items()} or getattr(self, "nflags_b", {})
            TM = cp.term_meas.tocsr()
            TO = cp.term_out.tocsr()
            for t in range(TM.shape[0]):
                ms = TM.indices[TM.indptr[t]:TM.indptr[t + 1]][TM.data[TM.indptr[t]:TM.indptr[t + 1]] % 2 == 1]
                os_ = TO.indices[TO.indptr[t]:TO.indptr[t + 1]][TO.data[TO.indptr[t]:TO.indptr[t + 1]] % 2 == 1]
                res = [int(c - off) for c in os_ if off <= c < off + n]
                gfl = defaultdict(list)
                bfl = defaultdict(list)
                chk = defaultdict(list)
                for j in ms:
                    if cp.meas_kind[j] == "flag":
                        kind, owner, loc = fl_of[j]
                        (gfl if kind == "g" else bfl)[owner].append(loc)
                    else:
                        c = cp.meas_check[j]
                        if c in self.chk_idx:
                            bb, i = self.chk_idx[c]
                            chk[bb].append(i)
                rb = {self.blk_of[q] for q in res}
                if gfl:
                    if len(gfl) > 1 or bfl or chk:
                        self.anomalies["multi-gadget or mixed flagged term"] += 1
                    g = next(iter(gfl))
                    if not set(res) <= set(self.gadget_supp[g]):
                        self.anomalies["flagged residual outside support"] += 1
                    self.gadget_terms[(k, g)].append((tuple(sorted(gfl[g])), tuple(sorted(res))))
                    continue
                touched = rb | set(bfl) | set(chk)
                if len(touched) == 0:
                    continue
                if len(touched) > 1:
                    # unflagged multi-block: give it to a gadget of this part whose support holds it
                    gs = [g for g in gflag_local if set(res) <= set(self.gadget_supp[g])]
                    if not gs or bfl or chk:
                        self.anomalies["unflagged multi-block term not in a gadget"] += 1
                        continue
                    for g in gs:
                        self.gadget_terms[(k, g)].append((tuple(), tuple(sorted(res))))
                    self.anomalies["unflagged multi-block term (as gadget)"] += 1
                    continue
                b = next(iter(touched))
                v = self._vec(res, b)
                dq = self.q_of(b, v)
                self.block_terms[(k, b)].append((dq, tuple(sorted(chk[b])), tuple(sorted(bfl[b]))))
        if self.verbose:
            print("anomalies", dict(self.anomalies), flush=True)

    # ------------------------------------------------------------------ gadget tables
    def gadget_table(self, k, g):
        """effects (per block of the support: quotient of the residual) -> minimal flag-silent
        weight, as a dict {effect tuple: weight}."""
        supp = self.gadget_supp[g]
        pos = {q: i for i, q in enumerate(supp)}
        nf = self.nflags_g
        gens = []
        for fl, res in self.gadget_terms[(k, g)]:
            s = 0
            for t in fl:
                s ^= 1 << t
            for q in res:
                s ^= 1 << (nf + pos[q])
            gens.append(s)
        nbits = nf + len(supp)
        dist = bfs_table(gens, nbits, self.max_depth)
        # flag-silent states: low nf bits zero
        Rs = np.arange(1 << len(supp), dtype=np.int64)
        d = dist[Rs << nf]
        blocks = sorted({self.blk_of[q] for q in supp})
        table = {}
        for R, w in zip(Rs, d):
            if w == INF or R == 0:
                continue
            eff = []
            for b in blocks:
                v = np.zeros(12, dtype=np.uint8)
                for q in supp:
                    if self.blk_of[q] == b and (R >> pos[q]) & 1:
                        v[self.pos_in_blk[q]] = 1
                eff.append(self.q_of(b, v))
            eff = tuple(eff)
            if table.get(eff, INF) > w:
                table[eff] = int(w)
        return blocks, table

    # ------------------------------------------------------------------ block dynamic programme
    def _part_table(self, k, b):
        """P[q_in, dq_out]: minimal weight of block-b faults of part k that cancel the outcome
        flips the incoming state q_in causes and leave flags silent, with net change dq."""
        terms = self.block_terms.get((k, b), [])
        r, pt = self.path[k]
        cp = self.proto.parts[pt]
        measured = sorted({self.chk_idx[c][1] for c in cp.meas_check if c in self.chk_idx and self.chk_idx[c][0] == b})
        mpos = {i: t for t, i in enumerate(measured)}
        nfl = sum(1 for kd, key in zip(cp.meas_kind, cp.meas_keys) if kd == "flag" and key[0] == "f6" and key[1] == b)
        nm = len(measured) + nfl
        gens = []
        for dq, chk, fl in terms:
            s = dq
            for i in chk:
                s ^= 1 << (7 + mpos[i])
            for t in fl:
                s ^= 1 << (7 + len(measured) + t)
            gens.append(s)
        dist = bfs_table(gens, 7 + nm, self.max_depth)
        P = np.full((128, 128), INF, dtype=np.int32)
        for qin in range(128):
            syn = qin & 31
            mu = 0
            for i in measured:
                if (syn >> i) & 1:
                    mu ^= 1 << mpos[i]
            row = dist[(mu << 7) + np.arange(128)]
            P[qin] = row
        P[P == INF] = 10 ** 6
        return P

    @staticmethod
    def _minplus_vec(f, P):
        """g[q ^ dq] = min f[q] + P[q, dq]."""
        g = np.full(128, 10 ** 6, dtype=np.int64)
        idx = np.arange(128)
        for q in range(128):
            if f[q] >= 10 ** 6:
                continue
            cand = f[q] + P[q]
            tgt = q ^ idx
            np.minimum.at(g, tgt, cand)
        return g

    def _single_flip_table(self, b):
        """cost of the init / final data flips (one per qubit of the block) as quotient changes."""
        gens = []
        for i in range(12):
            v = np.zeros(12, dtype=np.uint8)
            v[i] = 1
            gens.append(self.q_of(b, v))
        dist = bfs_table(gens, 7, 12)
        out = dist.astype(np.int64)
        out[dist == INF] = 10 ** 6
        return out

    def block_cost(self, b):
        """C[e_1, ..., e_m, lam] for the m outer-check instances of the path: minimal weight of
        block-b faults given residues e_i injected at the end of instance i, ending with class
        lam (and zero syndrome). Returned as an array of shape (128,)*m + (4,)."""
        flips_init = self._single_flip_table(b) if self.proto.noise.pr else None
        flips_final = self._single_flip_table(b) if self.proto.noise.pm else None
        f0 = np.full(128, 10 ** 6, dtype=np.int64)
        if flips_init is not None:
            f0 = flips_init.copy()
        else:
            f0[0] = 0
        segs = []           # list of lists of part indices between injections
        cur = []
        for k in range(len(self.path)):
            cur.append(k)
            if k in self.l3_instances:
                segs.append(cur)
                cur = []
        segs.append(cur)
        Ps = {k: self._part_table(k, b) for k in range(len(self.path))}

        def run(f, seg):
            for k in seg:
                f = self._minplus_vec(f, Ps[k])
            return f

        # final: cost to end in (syndrome 0, class lam) from state q
        fin = np.full((128, 4), 10 ** 6, dtype=np.int64)
        for q in range(128):
            for lam in range(4):
                target = lam << 5
                dq = q ^ target
                fin[q, lam] = (flips_final[dq] if flips_final is not None else (0 if dq == 0 else 10 ** 6))
        # transfer over segments: state vectors per injection history
        m = len(self.l3_instances)
        # first segment
        fA = run(f0, segs[0])
        # matrices for middle segments: M[q_start, q_end]
        mats = []
        for seg in segs[1:-1] + ([segs[-1]] if m >= 1 else []):
            M = np.zeros((128, 128), dtype=np.int64)
            for q in range(128):
                e = np.full(128, 10 ** 6, dtype=np.int64)
                e[q] = 0
                M[q] = run(e, seg)
            mats.append(M)
        # combine: C(e_1..e_m, lam)
        # after injection i the state shifts by e_i
        cur = fA[None, :]                     # shape (prod of e's so far, 128)
        idx = np.arange(128)
        for i in range(m):
            M = mats[i]
            # new[(..., e_i), q'] = min_q cur[..., q ^ e_i] + M[q, q']   (shifted start q ^ e_i)
            nxt = np.full((cur.shape[0] * 128, 128), 10 ** 6, dtype=np.int64)
            for e in range(128):
                shifted = cur[:, idx ^ e]       # cur[..., q ^ e] indexed by q
                # min-plus product shifted (A x 128) with M (128 x 128)
                prod = np.min(shifted[:, :, None] + M[None, :, :], axis=1)
                nxt[np.arange(cur.shape[0]) * 128 + e] = prod
            cur = np.minimum(nxt, 10 ** 6)
        # final
        out = np.min(cur[:, :, None] + fin[None, :, :], axis=1)      # (prod e's, 4)
        return np.minimum(out, 10 ** 6).reshape((128,) * m + (4,))


def invariant_table(D):
    inv = np.zeros((D.nb, 4), dtype=np.int64)
    for b in range(D.nb):
        for c in range(4):
            v = 0
            for i, o in enumerate(D.outer):
                a = D.alpha[o][b]
                v |= ((a[0] & c) ^ (a[1] & (c >> 1))) << i
            for li in range(2):
                a = D.alpha[("L", li)][b]
                v |= ((a[0] & c) ^ (a[1] & (c >> 1))) << (4 + li)
            inv[b, c] = v
    return inv


def reduce_gadget(D, blocks, tab, inv):
    keys, costs = [], []
    for E, w in tab.items():
        sig, kap = 0, 0
        for b, e in zip(blocks, E):
            sig |= (e & 31) << (5 * b)
            kap ^= int(inv[b, e >> 5])
        keys.append(sig | (kap << 25))
        costs.append(w)
    keys = np.array(keys + [0], dtype=np.int64)
    costs = np.array(costs + [0], dtype=np.int64)
    return dedup(keys, costs)


def dedup(keys, costs):
    if len(keys) == 0:
        return keys, costs
    o = np.lexsort((costs, keys))
    k, c = keys[o], costs[o]
    first = np.r_[True, k[1:] != k[:-1]]
    return k[first], c[first]


def conv(A, B, cap):
    ka, ca = A
    kb, cb = B
    outk, outc = [], []
    for a in np.unique(ca):
        if a > cap:
            continue
        ia = ka[ca == a]
        sel = cb <= cap - a
        jb, jc = kb[sel], cb[sel]
        if len(jb) == 0:
            continue
        # chunk to bound memory
        step = max(1, 20_000_000 // max(1, len(jb)))
        for s in range(0, len(ia), step):
            blk = ia[s:s + step]
            outk.append((blk[:, None] ^ jb[None, :]).ravel())
            outc.append((a + np.broadcast_to(jc, (len(blk), len(jc)))).ravel())
    k = np.concatenate(outk)
    c = np.concatenate(outc)
    return dedup(k, c)


class PathDecomp(Decomp):
    def __init__(self, basis, rule, items, max_depth=11, verbose=False):
        self.optional = [bool(o) for _, _, o in items]
        path = [(r, pt) for r, pt, _ in items]
        super().__init__(basis, rounds=1, rule=rule, path=path, max_depth=max_depth, verbose=verbose)

    def part_matrix(self, k, b):
        P = self._part_table(k, b).astype(np.int64)
        if self.optional[k]:
            skip = np.full((128, 128), BIG, dtype=np.int64)
            skip[:, 0] = 0
            P = np.minimum(P, skip)
        return P

    def block_chain(self, b):
        """fA (start to the first instance, (128,)), transfers between instances, and the
        final table (after the last instance, (128, 4))."""
        flips_init = self._single_flip_table(b) if self.proto.noise.pr else None
        flips_final = self._single_flip_table(b) if self.proto.noise.pm else None
        f0 = flips_init.copy() if flips_init is not None else np.where(np.arange(128) == 0, 0, BIG)
        segs, cur = [], []
        for k in range(len(self.path)):
            cur.append(k)
            if k in self.l3_instances:
                segs.append(cur)
                cur = []
        segs.append(cur)
        Ps = {k: self.part_matrix(k, b) for k in range(len(self.path))}

        def run(f, seg):
            for k in seg:
                f = self._minplus_vec(f, Ps[k])
            return f
        fA = run(f0, segs[0])
        mats = []
        for seg in segs[1:]:
            M = np.zeros((128, 128), dtype=np.int64)
            for q in range(128):
                e = np.full(128, BIG, dtype=np.int64)
                e[q] = 0
                M[q] = run(e, seg)
            mats.append(np.minimum(M, BIG))
        fin = np.full((128, 4), BIG, dtype=np.int64)
        for q in range(128):
            for lam in range(4):
                dq = q ^ (lam << 5)
                fin[q, lam] = flips_final[dq] if flips_final is not None else (0 if dq == 0 else BIG)
        last = mats.pop()
        fin2 = np.minimum(np.min(last[:, :, None] + fin[None, :, :], axis=1), BIG)
        return fA, mats, fin2

    def block_table(self, b):
        """class-reduced cost C[s_1, ..., s_m, lam] (syndrome injections only, 32 each)."""
        fA, mats, fin = self.block_chain(b)
        m = len(self.l3_instances)
        idx = np.arange(128)
        cur = fA[None, :]
        for i in range(m):
            nxt = np.full((cur.shape[0] * 32, 128), BIG, dtype=np.int64)
            for s in range(32):
                shifted = cur[:, idx ^ s]
                if i < m - 1:
                    prod = np.min(shifted[:, :, None] + mats[i][None, :, :], axis=1)
                else:
                    prod = np.full((cur.shape[0], 128), BIG, dtype=np.int64)
                    prod[:, :4] = np.min(shifted[:, :, None] + fin[None, :, :], axis=1)
                nxt[np.arange(cur.shape[0]) * 32 + s] = prod
            cur = np.minimum(nxt, BIG)
        return cur[:, :4].reshape((32,) * m + (4,))


def conv_pruned(A, B, cap, keep_fn, chunk=20_000_000):
    """XOR min-plus convolution of two (keys, costs) tables up to cost cap, keeping only
    entries with keep_fn(keys, costs); deduplicated as it goes."""
    ka, ca = A
    kb, cb = B
    accK, accC = [ka[keep_fn(ka, ca)]], [ca[keep_fn(ka, ca)]]
    for a in np.unique(ca):
        if a > cap:
            continue
        ia = ka[ca == a]
        sel = (cb <= cap - a) & (kb != 0)
        jb, jc = kb[sel], cb[sel]
        if len(jb) == 0:
            continue
        step = max(1, chunk // len(jb))
        for s in range(0, len(ia), step):
            blk = ia[s:s + step]
            k = (blk[:, None] ^ jb[None, :]).ravel()
            c = (a + np.broadcast_to(jc, (len(blk), len(jc)))).ravel()
            m = keep_fn(k, c)
            k, c = dedup(k[m], c[m])
            accK.append(k)
            accC.append(c)
        k, c = dedup(np.concatenate(accK), np.concatenate(accC))
        accK, accC = [k], [c]
    return dedup(np.concatenate(accK), np.concatenate(accC))


def full_items(rounds):
    items = []
    for r in range(rounds + 1):
        items += [(r, "L1_a" if r % 2 == 0 else "L1_b", False), (r, "L2_all", False), (r, "L3", False)]
    return items


def window_items(T, n_windows=1):
    """P-family path: L1, L2_all, L3 at multiples of T; in between L1 every round, each block's
    own level-2 part optional, the idle parts while other blocks measure included."""
    items = []
    for r in range(n_windows * T + 1):
        L1 = "L1_a" if r % 2 == 0 else "L1_b"
        if r % T == 0:
            items += [(r, L1, False), (r, "L2_all", False), (r, "L3", False)]
        else:
            items += [(r, L1, False)] + [(r, f"L2_{b}", True) for b in range(5)] + \
                     [(r, f"W2_{b}", False) for b in range(5)]
    return items


def minplus(A, B, chunk=4096):
    """(N, K) x (K, L) min-plus product."""
    out = np.empty((A.shape[0], B.shape[1]), dtype=np.int64)
    for s in range(0, A.shape[0], chunk):
        out[s:s + chunk] = np.min(A[s:s + chunk, :, None] + B[None, :, :], axis=1)
    return np.minimum(out, BIG)


class LongPath:
    def __init__(self, basis, order, items, W=10, log=print):
        t0 = time.time()
        self.W = W
        self.log = log
        self.D = D = PathDecomp(basis, f"g3=k4cyc,g3order={order}", items)
        # every fault term is a block term or a gadget term (none skipped)
        assert set(D.anomalies) <= {"unflagged multi-block term (as gadget)"}, dict(D.anomalies)
        self.inv = invariant_table(D)
        self.m = m = len(D.l3_instances)
        # gadget tables (identical parts at every instance: computed at the first, checked
        # for equal term lists at the others)
        k0 = D.l3_instances[0]
        gads = sorted(g for (k, g) in D.gadget_terms if k == k0)
        for k in D.l3_instances[1:]:
            for g in gads:
                assert sorted(D.gadget_terms[(k, g)]) == sorted(D.gadget_terms[(k0, g)])
        self.red = [reduce_gadget(D, *D.gadget_table(k0, g), self.inv) for g in gads]
        assert min(int(c[k != 0].min()) for k, c in self.red) >= 2
        # per-block chains
        self.chains = [D.block_chain(b) for b in range(D.nb)]
        log(f"setup {round(time.time() - t0, 1)} s: {m} instances")
        # combined table of one instance (all its gadgets), exact to cost W; entries above
        # W - 2 are final and kept only where their cost plus the weakest block bound of any
        # instance stays <= W
        lb_any = np.full((D.nb, 32), BIG, dtype=np.int64)
        for A in [(i,) for i in range(m)]:
            T = self.block_tables(A)
            for b in range(D.nb):
                lb_any[b] = np.minimum(lb_any[b], T[b].min(axis=-1))
        self.lb_any = lb_any

        def keep(kk, cc):
            s = (kk[:, None] >> (5 * np.arange(D.nb))) & 31
            return (cc <= W - 2) | (cc + lb_any[np.arange(D.nb)[None, :], s].sum(axis=1) <= W)
        acc = (np.array([0], dtype=np.int64), np.array([0], dtype=np.int64))
        for r in self.red:
            acc = conv_pruned(acc, r, W, keep)
        self.U = acc
        k, c = acc
        s = (k[:, None] >> (5 * np.arange(D.nb))) & 31
        self.Us = s
        self.Ukap = k >> 25
        pure = (s.sum(axis=1) == 0) & (k != 0)
        self.min_pure = int(c[pure].min()) if pure.any() else BIG
        assert self.min_pure >= 6, self.min_pure
        mus = np.array(list(itertools.product(range(4), repeat=D.nb)))
        mu_inv = np.zeros(len(mus), dtype=np.int64)
        for b in range(D.nb):
            mu_inv ^= self.inv[b, mus[:, b]]
        self.mus, self.mu_inv = mus, mu_inv
        # between consecutive instances, does every block's weight-6 check have a measurement
        # that always runs? If not, weight-6-only injections of different instances can cancel
        # without a block fault, and only injections with level-1 (C4) bits must each cost one
        cp_meas = {}
        for k_, (r_, pt) in enumerate(D.path):
            cp_meas[k_] = {D.chk_idx[c_][0] for c_ in D.proto.parts[pt].meas_check
                           if c_ in D.chk_idx and c_[0] == "c6"}
        gap = False
        for a_, b_ in zip(D.l3_instances[:-1], D.l3_instances[1:]):
            for blk in range(D.nb):
                if not any(blk in cp_meas[k_] and not D.optional[k_] for k_ in range(a_ + 1, b_)):
                    gap = True
        self.c6_gap = gap
        self.mask4 = np.array([sum(1 << i for i, c_ in enumerate(D.inner[b]) if c_[0] == "c4") for b in range(D.nb)])
        self.Un4 = ((s & self.mask4[None, :]) != 0).sum(axis=1)
        log(f"instance table {len(k)} entries, cheapest pure-class {self.min_pure}, weight-6 gap {gap}, "
            f"{round(time.time() - t0, 1)} s")

    def _prep_chains(self):
        """per block: H[j] (cost vector before instance j), S[(j1, j2)] (after instance j1 to
        before j2), Tail[j] (after instance j to the end, (128, 4))."""
        self.H, self.S, self.Tail = [], [], []
        for b in range(self.D.nb):
            fA, mats, fin = self.chains[b]
            H = [fA[None, :].astype(np.int64)]
            for j in range(self.m - 1):
                H.append(minplus(H[-1], mats[j]))
            S = {}
            for j1 in range(self.m):
                M = None
                for j2 in range(j1 + 1, self.m):
                    M = mats[j2 - 1] if M is None else minplus(M, mats[j2 - 1])
                    S[(j1, j2)] = M
            Tail = [None] * self.m
            Tail[self.m - 1] = fin
            for j in range(self.m - 2, -1, -1):
                Tail[j] = minplus(mats[j], Tail[j + 1])
            self.H.append(H)
            self.S.append(S)
            self.Tail.append(Tail)

    def block_tables(self, A, syn=None):
        """C_b[s_1, ..., s_r, lam] for active instances A (sorted), zero injections elsewhere;
        with syn (a list per block of syndrome values), the axes run over those values only."""
        if not hasattr(self, "H"):
            self._prep_chains()
        idx = np.arange(128)
        out = []
        for b in range(self.D.nb):
            if len(A) == 0:
                out.append(minplus(self.H[b][0], self.Tail[b][0])[0])
                continue
            vals = list(range(32)) if syn is None else list(syn[b])
            v = self.H[b][A[0]]
            for k in range(len(A)):
                nxt = self.S[b][(A[k], A[k + 1])] if k + 1 < len(A) else self.Tail[b][A[k]]
                parts = [minplus(v[:, idx ^ s_], nxt) for s_ in vals]
                v = np.stack(parts, axis=1).reshape(-1, nxt.shape[1])
            out.append(v.reshape((len(vals),) * len(A) + (4,)))
        return out

    def search(self, A, combos=((0,), (1,), (0, 1))):
        D, W = self.D, self.W
        r = len(A)
        nb = D.nb
        k, c = self.U
        s_raw, kap = self.Us, self.Ukap
        # r >= 3: total <= W needs sum of gadget costs <= W - 2 (>= 3 injections need >= 2
        # block faults; a pure-class effect costs >= 6), so each costs <= W - 2 - 2 (r - 1)
        allowed = k != 0
        if self.c6_gap:
            # every instance costs w = c + (blocks with level-1 bits injected) >= 2, additive
            # over instances (level-1 checks run every round: disjoint intervals)
            w = c + self.Un4
            if r >= 2:
                allowed = allowed & (w <= W - 2 * (r - 1))
        elif r >= 3:
            # derived for W = 10: a pure-class effect (>= 6) with two more instances (>= 4)
            # and a block fault already exceeds 10
            assert W == 10 and self.min_pure >= 6
            allowed = allowed & (c <= W - 2 - 2 * (r - 1))
        # the block tables only over the syndromes the allowed entries inject (s: their positions)
        syn = [sorted(set(s_raw[allowed, b].tolist()) | {0}) for b in range(nb)]
        pos = np.full((nb, 32), 0, dtype=np.int64)
        for b in range(nb):
            pos[b, syn[b]] = np.arange(len(syn[b]))
        s = np.stack([pos[b][s_raw[:, b]] for b in range(nb)], axis=1)
        T = self.block_tables(A, syn)
        # per-instance block bounds
        lbs = []
        for i in range(r):
            ax = tuple(j for j in range(r) if j != i) + (r,)
            lbs.append([T[b].min(axis=ax) for b in range(nb)])
        lbj = [T[b].min(axis=-1) for b in range(nb)]
        cand = []
        al = np.flatnonzero(allowed)
        for i in range(r):
            lbv = c[al] + sum(lbs[i][b][s[al, b]] for b in range(nb))
            cand.append(al[lbv + 2 * (r - 1) <= W])
        targets = {str(combo): np.array(sorted({l << 4 for l in range(4) if sum((l >> li) & 1 for li in combo) % 2}))
                   for combo in combos}
        best = {key: BIG for key in targets}
        wv = c + self.Un4 if self.c6_gap else c
        # prefix bounds: block cost minimised over the instances not yet chosen and the class
        pre = [[T[b].min(axis=tuple(range(j, r)) + (r,)) for b in range(nb)] for j in range(r + 1)]

        def leaf(hc, L):
            ch = int(sum(c[e] for e in hc))
            kv_all = kap[L].copy()
            for e in hc:
                kv_all ^= kap[e]
            tot0 = ch + c[L]
            Cb = []
            for b in range(nb):
                ix = tuple(np.full(len(L), s[e, b]) for e in hc) + (s[L, b],)
                Cb.append(T[b][ix])
            for key, target in targets.items():
                bestL = np.full(len(L), BIG, dtype=np.int64)
                for kv in np.unique(kv_all):
                    sel = np.flatnonzero(kv_all == kv)
                    for mi in np.flatnonzero(np.isin(self.mu_inv ^ kv, target)):
                        mu = self.mus[mi]
                        tot = tot0[sel] + sum(Cb[b][sel, mu[b]] for b in range(nb))
                        bestL[sel] = np.minimum(bestL[sel], tot)
                best[key] = min(best[key], int(bestL.min()))

        def dfs(j, hc, ch, wh):
            """choose instance j's entry given the entries hc of instances < j (cost ch, weight wh)."""
            C_ = cand[j]
            rest = 2 * (r - j - 1)
            ok = (ch + c[C_] + rest <= W) & (wh + wv[C_] + rest <= W)
            C_ = C_[ok]
            if len(C_) == 0:
                return
            bound = ch + c[C_] + rest
            for b in range(nb):
                ix = tuple(np.full(len(C_), s[e, b]) for e in hc) + (s[C_, b],)
                bound = bound + pre[j + 1][b][ix]
            C_ = C_[bound <= W]
            if len(C_) == 0:
                return
            if j == r - 1:
                leaf(hc, C_)
                return
            for e in C_:
                dfs(j + 1, hc + (int(e),), ch + int(c[e]), wh + int(wv[e]))
        dfs(0, (), 0, 0)
        res = {key: (v if v <= W else f"> {W}") for key, v in best.items()}
        return res, [len(x) for x in cand]

    def no_gadget(self, combos=((0,), (1,), (0, 1))):
        D = self.D
        T = self.block_tables(())
        out = {}
        for combo in combos:
            target = sorted({l << 4 for l in range(4) if sum((l >> li) & 1 for li in combo) % 2})
            best = BIG
            for mi in np.flatnonzero(np.isin(self.mu_inv, target)):
                mu = self.mus[mi]
                best = min(best, int(sum(T[b][mu[b]] for b in range(D.nb))))
            out[str(combo)] = best if best <= self.W else f"> {self.W}"
        return out


def any_items(rounds):
    """Superset of every path of the rules with inner checks every round: round 0 full; in
    rounds 1..R the level-2 parts (all blocks, or each block's own) optional, the idle parts
    and the outer-check part always included (in the relaxed problem an outer-check part only
    adds faults, so including it bounds every escalation pattern from below)."""
    items = [(0, "L1_a", False), (0, "L2_all", False), (0, "L3", False)]
    for r in range(1, rounds + 1):
        items += [(r, "L1_a" if r % 2 == 0 else "L1_b", False), (r, "L2_all", True)]
        items += [(r, f"L2_{b}", True) for b in range(5)] + [(r, f"W2_{b}", False) for b in range(5)]
        items += [(r, "L3", False)]
    return items


def items_of(kind):
    if kind.startswith("full"):
        return full_items(int(kind[4:]))
    if kind.startswith("any"):
        return any_items(int(kind[3:]))
    T, nw = kind[1:].split("x")
    return window_items(int(T), int(nw))


def subsets(m, W=10, c6_gap=False):
    # r active instances cost >= 2 r + 1 (each nonzero injection needs a block fault, shared at
    # most pairwise), or >= 2 r where weight-6 checks may be skipped between instances
    rmax = W // 2 if c6_gap else (W - 1) // 2
    out = []
    for r in range(1, rmax + 1):
        out += list(itertools.combinations(range(m), r))
    return out
