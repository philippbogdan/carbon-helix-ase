"""Decoders for stim detector error models.

FrontierDecoder wraps the Frontier decoder of Leverrier & Urbanke (arXiv:2606.20513,
github.com/aleverrier/frontier), the decoder used by the C4-Helix paper. It approximates
logical maximum-likelihood decoding and returns the confidence gap between the two most
likely logical classes (in nats).
"""
from __future__ import annotations

import os

import numpy as np
import stim

from .distance import dem_to_matrices


def _frontier_modules():
    from tools import frontier_progressive as fp
    from tools import frontier_decoder as fdec
    return fp, fdec


class FrontierDecoder:
    def __init__(self, dem: stim.DetectorErrorModel | None = None, *, H=None, L=None,
                 priors=None, K=512, Delta=12.0, committee=False, score_alpha=0.8):
        fp, fdec = _frontier_modules()
        self.fp, self.fdec = fp, fdec
        if dem is not None:
            H, L, priors = dem_to_matrices(dem)
        H = H.tocsc()
        L = L.tocsc()
        self.num_detectors = H.shape[0]
        self.num_observables = L.shape[0]
        self.K, self.Delta, self.committee, self.alpha = K, Delta, committee, score_alpha
        cols = []
        for j in range(H.shape[1]):
            p = float(priors[j])
            if p <= 0:
                continue
            d = H.indices[H.indptr[j]:H.indptr[j + 1]]
            o = L.indices[L.indptr[j]:L.indptr[j + 1]]
            dm = 0
            for i in d:
                dm |= 1 << int(i)
            om = 0
            for i in o:
                om |= 1 << int(i)
            cols.append(fp.ProgressiveColumn(
                family="dem", index=len(cols), label=f"e{j}", instruction_offset=len(cols),
                prior_probs=(1.0 - p, p), detector_response_masks=(0, dm),
                logical_response_masks=(0, om), detector_support_mask=dm,
                original_column_index=j))
        nd = max(self.num_detectors, 1)
        fwd, _ = fp.optimize_column_order(cols, num_detectors=nd)
        self.model_fwd = fdec.FrontierModel(
            columns=tuple(fwd), layout=fp.build_frontier_layout(list(fwd), num_detectors=nd),
            num_detectors=nd, num_observables=max(self.num_observables, 1))
        from .native import get_native
        hn = get_native()
        self._hn = hn
        self.native_fwd = hn.make_model(fdec._native_binary_model_spec(self.model_fwd))
        self.native_bwd = None
        if committee:
            import dataclasses
            rev = [dataclasses.replace(c, index=i) for i, c in enumerate(reversed(fwd))]
            bwd, _ = fp.optimize_column_order(rev, num_detectors=nd)
            self.model_bwd = fdec.FrontierModel(
                columns=tuple(bwd), layout=fp.build_frontier_layout(list(bwd), num_detectors=nd),
                num_detectors=nd, num_observables=max(self.num_observables, 1))
            self.native_bwd = hn.make_model(fdec._native_binary_model_spec(self.model_bwd))
        self.n_limbs = fdec._detector_limb_count(nd)
        self.max_active = self.model_fwd.layout.max_active_detectors

    def _limbs(self, row):
        idx = np.flatnonzero(row)
        limbs = [0] * self.n_limbs
        for i in idx:
            limbs[i >> 6] |= 1 << (int(i) & 63)
        return tuple(limbs)

    def _run(self, native, rows):
        payloads = self._hn.decode_many(native, tuple(self._limbs(r) for r in rows), int(self.K),
                                        float(self.Delta), float(self.alpha), "logsumexp_float", 1024)
        return [dict(p) for p in payloads]

    def decode_batch(self, dets: np.ndarray, return_details=False):
        """dets: bool array [shots, num_detectors]. Returns (pred [shots, num_obs] uint8,
        gap [shots] float in nats, status list)."""
        dets = np.asarray(dets, dtype=bool)
        n = dets.shape[0]
        pred = np.zeros((n, self.num_observables), dtype=np.uint8)
        gap = np.full(n, np.nan)
        logev = np.full(n, -np.inf)
        status = []
        fwd = self._run(self.native_fwd, dets)
        bwd = self._run(self.native_bwd, dets) if self.native_bwd is not None else None
        self.last_disagree = np.zeros(n, dtype=bool)
        for i in range(n):
            choice = fwd[i]
            if bwd is not None:
                h1 = fwd[i].get("logical_hat") if str(fwd[i].get("status")) == "ok" else None
                h2 = bwd[i].get("logical_hat") if str(bwd[i].get("status")) == "ok" else None
                self.last_disagree[i] = h1 != h2
                choice = max((fwd[i], bwd[i]), key=_committee_key)
            st = str(choice.get("status", "no_path"))
            status.append(st)
            lh = choice.get("logical_hat")
            if st == "ok" and lh is not None:
                for o in range(self.num_observables):
                    pred[i, o] = (int(lh) >> o) & 1
            gap[i] = float(choice.get("terminal_top_log_mass_gap", np.nan))
            logev[i] = float(choice.get("log_evidence", -np.inf))
        if return_details:
            return pred, gap, status, logev
        return pred, gap, status


def _committee_key(payload):
    st = str(payload.get("status", "no_path"))
    rank = 2.0 if st == "ok" else (1.0 if st == "no_path" else 0.0)
    le = float(payload.get("log_evidence", -np.inf)) if st == "ok" else -np.inf
    g = float(payload.get("terminal_top_log_mass_gap", np.nan))
    if np.isnan(g):
        g = -np.inf
    return (rank, le, g)


def sample_and_decode(circuit: stim.Circuit, shots: int, decoder: FrontierDecoder, seed=None,
                      batch=2000):
    """Sample a circuit with stim and decode. Returns (fails bool [shots], gaps)."""
    sampler = circuit.compile_detector_sampler(seed=seed)
    fails, gaps = [], []
    done = 0
    while done < shots:
        b = min(batch, shots - done)
        dets, obs = sampler.sample(b, separate_observables=True)
        pred, gap, status = decoder.decode_batch(dets)
        fails.append(np.any(pred != obs.astype(np.uint8), axis=1))
        gaps.append(gap)
        done += b
    return np.concatenate(fails), np.concatenate(gaps)


def _patched_available():
    try:
        from .native import get_native
        get_native()
        return True
    except Exception:
        return False


class PriorSwapFrontier:
    """Frontier model whose column structure (order, layout) is fixed, while the column
    priors - and the prior-dependent pruning weights - are replaced for every decode.
    Used with SupersetDEM to decode adaptive paths without rebuilding the model."""

    def __init__(self, H, L, base_priors, K=512, Delta=12.0, committee=True, score_alpha=0.8):
        import dataclasses
        self.K, self.Delta, self.committee, self.alpha = K, Delta, committee, score_alpha
        fp, fdec = _frontier_modules()
        self.fp, self.fdec = fp, fdec
        H = H.tocsc()
        L = L.tocsc()
        self.num_detectors = H.shape[0]
        self.num_observables = L.shape[0]
        cols = []
        for j in range(H.shape[1]):
            p = float(min(max(base_priors[j], 1e-30), 0.5 - 1e-12))
            d = H.indices[H.indptr[j]:H.indptr[j + 1]]
            o = L.indices[L.indptr[j]:L.indptr[j + 1]]
            dm = 0
            for i in d:
                dm |= 1 << int(i)
            om = 0
            for i in o:
                om |= 1 << int(i)
            cols.append(fp.ProgressiveColumn(
                family="dem", index=len(cols), label=f"e{j}", instruction_offset=len(cols),
                prior_probs=(1.0 - p, p), detector_response_masks=(0, dm),
                logical_response_masks=(0, om), detector_support_mask=dm,
                original_column_index=j))
        nd = max(self.num_detectors, 1)
        self.nd = nd
        self.n_limbs = fdec._detector_limb_count(nd)
        dirs = []
        fwd, _ = fp.optimize_column_order(cols, num_detectors=nd)
        dirs.append(fwd)
        if committee:
            rev = [dataclasses.replace(c, index=i) for i, c in enumerate(reversed(fwd))]
            bwd, _ = fp.optimize_column_order(rev, num_detectors=nd)
            dirs.append(bwd)
        self.models = []
        for colist in dirs:
            model = fdec.FrontierModel(columns=tuple(colist),
                                       layout=fp.build_frontier_layout(list(colist), num_detectors=nd),
                                       num_detectors=nd, num_observables=max(self.num_observables, 1))
            spec = fdec._native_binary_model_spec(model)
            order = np.array([c.original_column_index for c in colist], dtype=np.int64)
            # incidences (row, position in order) grouped by row in position order
            inc = []
            pos_of = {int(g): i for i, g in enumerate(order)}
            for i, g in enumerate(order):
                for r in H.indices[H.indptr[g]:H.indptr[g + 1]]:
                    inc.append((int(r), i))
            inc.sort()
            inc = np.array(inc, dtype=np.int64).reshape(-1, 2)
            rows, posn = inc[:, 0], inc[:, 1]
            # k = offset of the incidence within its row, total touches per row
            starts = np.r_[0, np.flatnonzero(np.diff(rows)) + 1]
            counts = np.diff(np.r_[starts, len(rows)])
            k = np.arange(len(rows)) - np.repeat(starts, counts)
            tot = np.repeat(counts, counts)
            # per column: before terms are rows with k>=1, after terms rows with k+1<tot,
            # both in increasing row order (as in Frontier's layout builder)
            before_idx = [[] for _ in range(len(order))]
            after_idx = [[] for _ in range(len(order))]
            for t in range(len(rows)):
                if k[t] >= 1:
                    before_idx[posn[t]].append(t)
                if k[t] + 1 < tot[t]:
                    after_idx[posn[t]].append(t)
            bi = [np.array(x, dtype=np.int64) for x in before_idx]
            ai = [np.array(x, dtype=np.int64) for x in after_idx]
            self.models.append(dict(spec=spec, order=order, rows=rows, posn=posn, k=k, tot=tot,
                                    starts=starts, counts=counts, before_idx=bi, after_idx=ai,
                                    before_all=np.concatenate(bi) if bi else np.zeros(0, dtype=np.int64),
                                    after_all=np.concatenate(ai) if ai else np.zeros(0, dtype=np.int64),
                                    capsule=None))

    def _spec_for(self, m, priors):
        import math
        order = m["order"]
        p = np.clip(priors[order], 1e-300, 0.5)
        no_err = np.log1p(-np.minimum(p, 0.5))
        tog = np.log(p) - no_err
        # suffix parity per incidence: S_t = prod over incidences of same row at offset>=k
        pinc = p[m["posn"]]
        f = np.log(np.clip(np.abs(1 - 2 * pinc), 1e-300, None))
        zero = (np.abs(1 - 2 * pinc) < 1e-300)
        # reverse cumulative sums within rows
        rs = np.zeros(len(f))
        rz = np.zeros(len(f), dtype=np.int64)
        for s, c in zip(m["starts"], m["counts"]):
            seg = f[s:s + c][::-1].cumsum()[::-1]
            rs[s:s + c] = seg
            rz[s:s + c] = zero[s:s + c][::-1].cumsum()[::-1]
        S = np.where(rz > 0, 0.0, np.exp(rs))
        odd = np.maximum((1 - S) / 2, 2.2250738585072014e-308)
        even = np.maximum((1 + S) / 2, 2.2250738585072014e-308)
        par_at = np.log(odd) - np.log(even)       # parity log-odds of suffix starting at t
        # 'after' value at incidence t is the suffix starting at t+1 (same row)
        par_next = np.r_[par_at[1:], 0.0]
        old = m["spec"]
        cols = []
        for i, c in enumerate(old["columns"]):
            nc = dict(c)
            nc["no_error_log_const"] = float(no_err[i])
            nc["toggle_logodds"] = float(tog[i])
            bi = m["before_idx"][i]
            if len(bi):
                nc["before_parity"] = tuple(par_at[bi].tolist())
            ai = m["after_idx"][i]
            if len(ai):
                nc["after_parity"] = tuple(par_next[ai].tolist())
            cols.append(nc)
        spec = dict(old)
        spec["columns"] = cols
        return spec

    def _prior_arrays(self, m, priors):
        order = m["order"]
        p = np.clip(priors[order], 1e-300, 0.5)
        no_err = np.log1p(-np.minimum(p, 0.5))
        tog = np.log(p) - no_err
        pinc = p[m["posn"]]
        a = np.abs(1 - 2 * pinc)
        zero = a < 1e-300
        f = np.log(np.clip(a, 1e-300, None))
        # reverse cumulative sums within row segments (vectorized)
        starts, counts = m["starts"], m["counts"]
        seg_id = np.repeat(np.arange(len(starts)), counts)
        csum = np.cumsum(f[::-1])[::-1]
        zsum = np.cumsum(zero[::-1].astype(np.int64))[::-1]
        ends = starts + counts
        after_end_c = np.where(ends < len(f), np.r_[csum, 0.0][ends], 0.0)
        after_end_z = np.where(ends < len(f), np.r_[zsum, 0][ends], 0)
        rs = csum - after_end_c[seg_id]
        rz = zsum - after_end_z[seg_id]
        S = np.where(rz > 0, 0.0, np.exp(rs))
        tiny = 2.2250738585072014e-308
        par_at = np.log(np.maximum((1 - S) / 2, tiny)) - np.log(np.maximum((1 + S) / 2, tiny))
        par_next = np.r_[par_at[1:], 0.0]
        return (np.ascontiguousarray(no_err, dtype=np.float64), np.ascontiguousarray(tog, dtype=np.float64),
                np.ascontiguousarray(par_at[m["before_all"]], dtype=np.float64),
                np.ascontiguousarray(par_next[m["after_all"]], dtype=np.float64))

    def _inplace_capsules(self, priors):
        from .native import get_native
        hn = get_native()
        caps = []
        for m in self.models:
            if m["capsule"] is None:
                m["capsule"] = hn.make_model(m["spec"])
                if getattr(self, "free_spec", True):
                    m["spec"] = None          # the native model holds everything it needs
            hn.update_priors(m["capsule"], *self._prior_arrays(m, np.asarray(priors, dtype=float)))
            caps.append(m["capsule"])
        return caps

    def native_models(self, priors, cache_key=None):
        import frontier_native
        cache = self.__dict__.setdefault("_cache", {})
        order = self.__dict__.setdefault("_cache_order", [])
        if cache_key is not None and cache_key in cache:
            return cache[cache_key]
        nms = [frontier_native.NativeBinaryFrontierModel(self._spec_for(m, np.asarray(priors, dtype=float)))
               for m in self.models]
        if cache_key is not None:
            cache[cache_key] = nms
            order.append(cache_key)
            if len(order) > 48:
                del cache[order.pop(0)]
        return nms

    def decode_batch(self, priors, dets, cache_key=None, inplace=None, K=None, Delta=None):
        dets = np.asarray(dets, dtype=bool)
        results = []
        K = int(self.K if K is None else K)
        Delta = float(self.Delta if Delta is None else Delta)
        if inplace is None:
            inplace = _patched_available()
        if inplace:
            from .native import get_native
            hn = get_native()
            rows = []
            for row in dets:
                limbs = [0] * self.n_limbs
                for i in np.flatnonzero(row):
                    limbs[i >> 6] |= 1 << (int(i) & 63)
                rows.append(tuple(limbs))
            for cap in self._inplace_capsules(priors):
                results.append(hn.decode_many(cap, tuple(rows), K, Delta,
                                              float(self.alpha), "logsumexp_float", 1024))
            return self._collect(results, dets.shape[0])
        for nm in self.native_models(priors, cache_key):
            rows = []
            for row in dets:
                idx = np.flatnonzero(row)
                limbs = [0] * self.n_limbs
                for i in idx:
                    limbs[i >> 6] |= 1 << (int(i) & 63)
                rows.append(tuple(limbs))
            results.append(nm.decode_many(rows, int(self.K), float(self.Delta), float(self.alpha),
                                          "logsumexp_float", 1024))
        return self._collect(results, dets.shape[0])

    def _collect(self, results, n):
        pred = np.zeros((n, self.num_observables), dtype=np.uint8)
        gap = np.full(n, np.nan)
        status = []
        self.last_disagree = np.zeros(n, dtype=bool)
        for i in range(n):
            members = [dict(r[i]) for r in results]
            if len(members) > 1:
                hats = {(m.get("logical_hat") if str(m.get("status")) == "ok" else None)
                        for m in members}
                self.last_disagree[i] = len(hats) > 1
            choice = max(members, key=_committee_key)
            st = str(choice.get("status", "no_path"))
            status.append(st)
            lh = choice.get("logical_hat")
            if st == "ok" and lh is not None:
                for o in range(self.num_observables):
                    pred[i, o] = (int(lh) >> o) & 1
            gap[i] = float(choice.get("terminal_top_log_mass_gap", np.nan))
        return pred, gap, status
