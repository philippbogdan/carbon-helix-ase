"""Cluster job: exact lower bound on the circuit fault distance of a Carbon-Helix path by
decomposition (src/hase/relaxed_distance.py). Each task searches its share of the subsets of
active outer-check instances (at most four; five or more cost at least 11 by the counting
bound); together the tasks decide whether an undetected logical of at most W faults exists
on every path the kind stands for, in the relaxed problem whose optimum bounds the true
distance from below.

Kinds: full<R>  (rounds 0..R, every round L1, L2_all, L3: non-adaptive extraction);
       P<T>x<n> (n windows of T rounds: L1, L2_all, L3 at multiples of T, in between L1 every
                 round, each block's own level-2 part optional, the idle parts included);
       any<R>   (rounds 0..R, the superset of every rule's paths: level-2 parts optional and the
                 outer-check part present in every round after the first).

Usage: python scripts/relaxed_distance.py --basis X --order XZ --kind full8 \
           --out results/relaxed_distance/X_XZ_full8/task{task}.json     (XPU_TASK / XPU_NTASKS)
       python scripts/relaxed_distance.py --basis Z --order ZX --kind full1 --W 11 --only 0 --out ...
       python scripts/relaxed_distance.py --validate results/distance/whole__k4cyc__ZX__Z__c1.json --out ...
           (a known undetected logical of the full detector model, mapped into the decomposition:
            its gadget effects and block costs must not exceed its weight)
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

from hase.version import commit, provenance  # noqa: E402
from hase.relaxed_distance import LongPath, items_of, subsets, full_items  # noqa: E402


def validate(path_json, W=12):
    """Evaluate a known solution of the full detector model in the decomposition."""
    from hase.engine import PathDEM
    d = json.load(open(path_json))
    order = d["rule"].split("g3order=")[1].split(",")[0]
    P = LongPath(d["basis"], order, full_items(d["rounds"]), W=W, log=lambda s: None)
    D = P.D
    assert [tuple(p) for p in d["path"]] == D.path
    pd = PathDEM(D.proto, D.path)
    H, L = pd.H.tocsc(), pd.L.tocsc()
    Hr, Lr = pd.Hraw.tocsr(), pd.Lraw.tocsr()
    key_of_raw = {}
    for i in range(Hr.shape[0]):
        k = (tuple(sorted(Hr.indices[Hr.indptr[i]:Hr.indptr[i + 1]].tolist())),
             tuple(sorted(Lr.indices[Lr.indptr[i]:Lr.indptr[i + 1]].tolist())))
        key_of_raw.setdefault(k, i)
    owner = [(k, t) for k, (r, pt) in enumerate(D.path) for t in range(D.proto.parts[pt].term_meas.shape[0])]
    n, off = D.n, D.off
    gad = defaultdict(lambda: [0, 0])          # (k, gadget) -> [flag xor bits, residual xor over its support]
    n_block, n_gadget = 0, 0
    for j in d["solution"]:
        key = (tuple(sorted(H.indices[H.indptr[j]:H.indptr[j + 1]].tolist())),
               tuple(sorted(L.indices[L.indptr[j]:L.indptr[j + 1]].tolist())))
        i = key_of_raw[key]
        if i >= len(owner):
            n_block += 1                         # init / final flip
            continue
        k, t = owner[i]
        cp = D.proto.parts[D.path[k][1]]
        TM, TO = cp.term_meas.tocsr(), cp.term_out.tocsr()
        ms = [int(x) for x, v in zip(TM.indices[TM.indptr[t]:TM.indptr[t + 1]], TM.data[TM.indptr[t]:TM.indptr[t + 1]]) if v % 2]
        res = [int(x - off) for x, v in zip(TO.indices[TO.indptr[t]:TO.indptr[t + 1]], TO.data[TO.indptr[t]:TO.indptr[t + 1]])
               if v % 2 and off <= x < off + n]
        gk = [cp.meas_keys[x] for x in ms if cp.meas_kind[x] == "flag" and cp.meas_keys[x][0] == "f16"]
        blocks = {D.blk_of[q] for q in res}
        if gk or len(blocks) > 1:
            g = (gk[0][1], gk[0][2]) if gk else next(gg for (kk, gg) in D.gadget_terms if kk == k
                                                     and set(res) <= set(D.gadget_supp[gg]))
            supp = D.gadget_supp[g]
            loc = {j2: tt for tt, j2 in enumerate([j2 for j2, key2 in enumerate(cp.meas_keys)
                                                   if key2[0] == "f16" and (key2[1], key2[2]) == g])}
            for x in ms:
                if x in loc:
                    gad[(k, g)][0] ^= 1 << loc[x]
            for q in res:
                gad[(k, g)][1] ^= 1 << supp.index(q)
            n_gadget += 1
        else:
            n_block += 1
    # the gadget clusters' effects and costs
    inst_of = {k: i for i, k in enumerate(D.l3_instances)}
    per_inst = defaultdict(lambda: [0, 0, 0])   # instance -> [cost, syndrome key, invariant]
    for (k, g), (fl, R) in gad.items():
        assert fl == 0, "flags of a gadget cluster must cancel"
        blocks, tab = D.gadget_table(k, g)
        supp = D.gadget_supp[g]
        eff = []
        for b in blocks:
            v = np.zeros(12, dtype=np.uint8)
            for qi, q in enumerate(supp):
                if D.blk_of[q] == b and (R >> qi) & 1:
                    v[D.pos_in_blk[q]] = 1
            eff.append(D.q_of(b, v))
        w = tab.get(tuple(eff), 0 if all(e == 0 for e in eff) else None)
        assert w is not None
        i = inst_of[k]
        per_inst[i][0] += w
        for b, e in zip(blocks, eff):
            per_inst[i][1] ^= (e & 31) << (5 * b)
            per_inst[i][2] ^= int(P.inv[b, e >> 5])
    A = tuple(sorted(i for i, v in per_inst.items() if v[1] or v[2]))
    T = P.block_tables(A)
    combo = tuple(d["combo"])
    target = sorted({l << 4 for l in range(4) if sum((l >> li) & 1 for li in combo) % 2})
    kap = 0
    for i in A:
        kap ^= per_inst[i][2]
    best = None
    for mi in np.flatnonzero(np.isin(P.mu_inv ^ kap, target)):
        mu = P.mus[mi]
        tot = 0
        for b in range(D.nb):
            ix = tuple((per_inst[i][1] >> (5 * b)) & 31 for i in A) + (mu[b],)
            tot += int(T[b][ix])
        best = tot if best is None else min(best, tot)
    gcost = sum(per_inst[i][0] for i in A)
    return dict(solution_weight=len(d["solution"]), gadget_faults=n_gadget, block_faults=n_block,
                active_instances=list(A), decomposition_gadget_cost=gcost, decomposition_block_cost=best,
                decomposition_total=gcost + best, represented=bool(gcost + best <= len(d["solution"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis")
    ap.add_argument("--order", default="XZ")
    ap.add_argument("--kind")
    ap.add_argument("--W", type=int, default=10)
    ap.add_argument("--only", default=None, help="comma-separated subset indices (default: this task's share)")
    ap.add_argument("--ntasks", type=int, default=None)
    ap.add_argument("--validate", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--task-offset", type=int, default=0, help="ignored (queue tooling)")
    a = ap.parse_args()
    task = int(os.environ.get("XPU_TASK", os.environ.get("GPURUN_TASK", "0")))
    ntasks = a.ntasks or int(os.environ.get("XPU_NTASKS", "1"))
    path = a.out.replace("{task}", str(task))
    if os.path.exists(path) and not os.environ.get("HASE_OVERWRITE"):
        print("result exists:", path)
        return
    t0 = time.time()
    if a.validate:
        out = dict(validate=a.validate, **validate(a.validate))
    else:
        P = LongPath(a.basis, a.order, items_of(a.kind), W=a.W, log=lambda s: print(s, flush=True))
        subs = subsets(P.m, P.W, P.c6_gap)
        mine = ([int(x) for x in a.only.split(",")] if a.only is not None
                else [i for i in range(len(subs)) if i % ntasks == task])
        out = dict(basis=a.basis, order=a.order, kind=a.kind, path=[list(p) for p in P.D.path],
                   optional=P.D.optional, instances=P.m, W=P.W, n_subsets=len(subs), task=task, ntasks=ntasks,
                   subsets=mine, cheapest_pure_class=P.min_pure, instance_table_entries=int(len(P.U[0])),
                   weight6_gap=bool(P.c6_gap),
                   results={})
        if task == 0 and a.only is None:
            out["no_gadget"] = P.no_gadget()
        for i in mine:
            t = time.time()
            res, nc = P.search(subs[i])
            out["results"][str(list(subs[i]))] = dict(min=res, candidates=nc, seconds=round(time.time() - t, 1))
            print(subs[i], res, nc, round(time.time() - t, 1), "s", flush=True)
    out["seconds"] = time.time() - t0
    out["commit"], out["provenance"] = commit(), provenance()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
