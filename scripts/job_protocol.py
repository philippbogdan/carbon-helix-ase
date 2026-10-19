"""Cluster job: (adaptive) memory protocol on the engine, decoded with Frontier on the
protocol's superset DEM. Writes one JSON per task with failure counts and resources."""
import argparse
import json
import os
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
from hase.runner import run_protocol, SupersetDecoder  # noqa: E402


def run_one(a, seed):
    proto = make_protocol(a.proto, a.p, a.basis, a.rounds, a.rule, a.noise)
    dec = SupersetDecoder(proto, K=a.K, Delta=a.Delta, committee=bool(a.committee), K2=a.K2,
                          Delta2=a.Delta2, recheck_gap=a.recheck_gap)
    rng = np.random.default_rng(seed)
    done = fails = rechecked = changed = 0
    twoq = 0
    twoq_sq = 0.0
    parts = {}
    npaths = 0
    gap_hist = np.zeros(64, dtype=np.int64)
    fail_gap_hist = np.zeros(64, dtype=np.int64)
    while done < a.shots:
        b = min(a.batch, a.shots - done)
        out = run_protocol(proto, b, rng, dec)
        fails += int(out["fails"].sum())
        rechecked += out["rechecked"]
        changed += out["changed"]
        twoq += int(out["twoq"].sum())
        twoq_sq += float((out["twoq"].astype(float) ** 2).sum())
        for k, v in out["part_counts"].items():
            parts[k] = parts.get(k, 0) + v
        npaths += out["n_paths"]
        g = np.nan_to_num(out["gaps"], nan=0.0, posinf=31.75)
        bins = np.clip((g / 0.5).astype(int), 0, 63)
        np.add.at(gap_hist, bins, 1)
        np.add.at(fail_gap_hist, bins[out["fails"]], 1)
        done += b
    return dict(shots=done, fails=fails, rechecked=rechecked, changed=changed, twoq_sum=twoq,
                twoq_sq=twoq_sq, part_counts=parts, n_paths_sum=npaths, gap_hist=gap_hist,
                fail_gap_hist=fail_gap_hist,
                part_twoq={k: v.two_q for k, v in proto.parts.items()},
                part_layers={k: v.n_layers for k, v in proto.parts.items()},
                n_det_superset=dec.sup.n_det)


def _worker(args):
    a, seed = args
    return run_one(a, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proto", required=True)
    ap.add_argument("--rule", default="full", help="protocol rule string, see hase.protocols")
    ap.add_argument("--noise", default="uniform")
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--basis", default="X")
    ap.add_argument("--rounds", type=int, required=True, help="rounds after the prep round")
    ap.add_argument("--shots", type=int, required=True, help="shots per process")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--Delta", type=float, default=12.0)
    ap.add_argument("--committee", type=int, default=1)
    ap.add_argument("--K2", type=int, default=4096)
    ap.add_argument("--Delta2", type=float, default=20.0)
    ap.add_argument("--recheck-gap", type=float, default=8.0)
    ap.add_argument("--batch", type=int, default=500)
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
    out = {k: sum(p[k] for p in parts) for k in ("shots", "fails", "rechecked", "changed",
                                                   "twoq_sum", "twoq_sq", "n_paths_sum")}
    pc = {}
    for p in parts:
        for k, v in p["part_counts"].items():
            pc[k] = pc.get(k, 0) + v
    out.update(part_counts=pc, gap_hist=sum(p["gap_hist"] for p in parts).tolist(),
               fail_gap_hist=sum(p["fail_gap_hist"] for p in parts).tolist(),
               part_twoq=parts[0]["part_twoq"], part_layers=parts[0]["part_layers"],
               n_det_superset=parts[0]["n_det_superset"])
    out.update(proto=a.proto, rule=a.rule, noise=a.noise, p=a.p, basis=a.basis,
               rounds=a.rounds, K=a.K, Delta=a.Delta, committee=a.committee, K2=a.K2,
               Delta2=a.Delta2, recheck_gap=a.recheck_gap, seed=seed, task=task,
               procs=a.procs, seconds=time.time() - t0)
    out["commit"] = commit()
    out["provenance"] = provenance()
    path = a.out.replace("{task}", str(task))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f)
    print(json.dumps({k: out[k] for k in ("proto", "rule", "p", "basis", "shots", "fails",
                                          "seconds")}))


if __name__ == "__main__":
    main()
