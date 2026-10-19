"""Cluster job: odds-scaled superset fault sampling for stratified (rare-event) estimation.

Per task: histogram over W (number of fired fault sites in the protocol's superset) of
shots, failures, and summed two-qubit gate counts / executed outer parts, decoded with
Frontier at priors of p_dec. Fault lists of failing shots with W <= --save-w are saved
for minimal-failing-set analysis.
"""
import argparse
import json
import os
import pickle
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "third_party", "frontier"))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("FRONTIER_NATIVE_BATCH_THREADS", "1")
# one thread per process: BLAS/OpenMP pools sized to the whole machine oversubscribe
# shared servers (results do not depend on it)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

from hase.version import commit, provenance  # noqa: E402

from hase.protocols import make_protocol  # noqa: E402
from hase.runner import SupersetDecoder, run_stratified  # noqa: E402
from hase.timing import path_time_ms, peak_qubits  # noqa: E402


def run_one(a, seed):
    proto = make_protocol(a.proto, a.p_dec, a.basis, a.rounds, a.rule, a.noise)
    dec = SupersetDecoder(proto, K=a.K, Delta=a.Delta, committee=bool(a.committee), K2=a.K2,
                          Delta2=a.Delta2, recheck_gap=a.recheck_gap,
                          gap_stages=a.gap_stages if a.gap_stages >= 0 else None)
    rng = np.random.default_rng(seed)
    rng_ok = np.random.default_rng(seed + 424242)     # sample of successful shots kept
    WMAX = 400
    n = np.zeros(WMAX, dtype=np.int64)
    f = np.zeros(WMAX, dtype=np.int64)
    g = np.zeros(WMAX)
    l3 = np.zeros(WMAX)
    tms = np.zeros(WMAX)
    qmax = np.zeros(WMAX)
    nstage = np.zeros((WMAX, 2 + len(dec.stages)), dtype=np.int64)   # stage -1 (trivial), 0, 1, ...
    unres = np.zeros(WMAX, dtype=np.int64)
    saved = []
    done = 0
    t0 = time.time()
    while done < a.shots and time.time() - t0 < a.max_seconds:
        b = min(a.batch, a.shots - done)
        o = run_stratified(proto, b, rng, dec, a.kappa)
        W = np.minimum(o["W"], WMAX - 1)
        np.add.at(n, W, 1)
        np.add.at(f, W, o["fails"].astype(np.int64))
        np.add.at(g, W, o["twoq"])
        nl3 = np.array([sum(1 for _, pt in p if pt == "L3" or pt == "outer") for p in o["state"].paths])
        np.add.at(l3, W, nl3)
        np.add.at(tms, W, np.array([path_time_ms(proto, p) for p in o["state"].paths]))
        pq = np.array([peak_qubits(proto, p) for p in o["state"].paths])
        np.maximum.at(qmax, W, pq)
        np.add.at(nstage, (W, o["stage"] + 1), 1)
        np.add.at(unres, W, o["unresolved"].astype(np.int64))
        # fault lists of every failing shot (w <= save_w) and of a fixed random fraction of
        # the others: re-decoding both with a stronger decoder measures the decoder's share
        # of the failures in both directions (analysis/redecode.py)
        keep_ok = rng_ok.random(len(o["W"])) < a.ok_fraction
        for i in np.flatnonzero((o["fails"] & (o["W"] <= a.save_w)) | (~o["fails"] & keep_ok)):
            fl = {}
            for key, val in o["faults"].items():
                if key in ("init", "final"):
                    m = np.flatnonzero(val[i])
                    if len(m):
                        fl[key] = m.tolist()
                    continue
                sh, tt = val
                sel = tt[sh == i]
                if len(sel):
                    fl[key] = sel.tolist()
            saved.append(dict(W=int(o["W"][i]), faults=fl, path=o["state"].paths[i],
                              failed=bool(o["fails"][i])))
        done += b
    return dict(n=n, f=f, g=g, l3=l3, t=tms, q=qmax, saved=saved, shots=done, stages=dec.stages,
                nstage=nstage, unres=unres)


def _worker(args):
    a, seed = args
    return run_one(a, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proto", required=True)
    ap.add_argument("--rule", default="full")
    ap.add_argument("--noise", default="uniform")
    ap.add_argument("--p-dec", type=float, required=True, help="decoder/noise reference p")
    ap.add_argument("--kappa", type=float, required=True, help="odds scale for sampling")
    ap.add_argument("--basis", default="X")
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--shots", type=int, required=True)
    ap.add_argument("--max-seconds", type=float, default=1e9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--Delta", type=float, default=10.0)
    ap.add_argument("--committee", type=int, default=1)
    ap.add_argument("--K2", type=int, default=4096)
    ap.add_argument("--Delta2", type=float, default=24.0)
    ap.add_argument("--recheck-gap", type=float, default=8.0)
    ap.add_argument("--gap-stages", type=int, default=-1,
                    help="re-decoding stages the low-gap criterion applies to (-1: all)")
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--save-w", type=int, default=40)
    ap.add_argument("--ok-fraction", type=float, default=0.005,
                    help="fraction of successful shots whose fault lists are kept")
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--task-offset", type=int, default=0,
                    help="added to GPURUN_TASK (for places that run one task per job)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    task = int(os.environ.get("GPURUN_TASK", "0")) + a.task_offset
    seed = a.seed * 100003 + task
    if os.path.exists(a.out.replace("{task}", str(task))) and not os.environ.get("HASE_OVERWRITE"):
        # computed elsewhere already (the same task can be queued on several clusters)
        print("result exists:", a.out.replace("{task}", str(task)))
        return
    t0 = time.time()
    if a.procs > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.procs) as pool:
            parts = pool.map(_worker, [(a, seed * 1000 + i) for i in range(a.procs)])
    else:
        parts = [run_one(a, seed)]
    n = sum(p["n"] for p in parts)
    f = sum(p["f"] for p in parts)
    g = sum(p["g"] for p in parts)
    l3 = sum(p["l3"] for p in parts)
    tms = sum(p["t"] for p in parts)
    qmax = np.max([p["q"] for p in parts], axis=0)
    nstage = sum(p["nstage"] for p in parts)
    unres = sum(p["unres"] for p in parts)
    top = int(np.max(np.flatnonzero(n))) + 1 if n.any() else 0
    out = dict(proto=a.proto, rule=a.rule, noise=a.noise, p_dec=a.p_dec, kappa=a.kappa,
               basis=a.basis, rounds=a.rounds, shots=int(sum(p["shots"] for p in parts)),
               n=n[:top].tolist(), f=f[:top].tolist(), g=g[:top].tolist(), l3=l3[:top].tolist(),
               t_ms=tms[:top].tolist(), peak_qubits=qmax[:top].tolist(),
               decode_stage_counts=nstage[:top].tolist(), unresolved=unres[:top].tolist(),
               K=a.K, Delta=a.Delta, committee=a.committee, K2=a.K2, Delta2=a.Delta2,
               recheck_gap=a.recheck_gap, stages=[list(x) for x in parts[0]["stages"]],
               gap_stages=a.gap_stages,
               seed=seed, task=task, seconds=time.time() - t0)
    out["commit"] = commit()
    out["provenance"] = provenance()
    path = a.out.replace("{task}", str(task))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh)
    saved = [s for p in parts for s in p["saved"]]
    if saved:
        with open(path.replace(".json", "_faults.pkl"), "wb") as fh:
            pickle.dump(saved, fh)
    print(json.dumps({k: out[k] for k in ("proto", "rule", "kappa", "basis", "shots", "seconds")}),
          "fails", int(f.sum()))


if __name__ == "__main__":
    main()
