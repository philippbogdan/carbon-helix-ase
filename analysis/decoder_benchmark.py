"""Decoder settings benchmark behind the choice of the Carbon-Helix campaign decoder (a
frontier push on decoding cost and accuracy): the same fixed sample of shots (non-adaptive
Carbon-Helix, uniform noise, 4 rounds, odds scales 2 and 3) decoded with primary beams
K = 256 (Delta 10) and K = 512 (Delta 12), both forward + backward with the staged
re-decoding (1024/16, then 4096/24). Records seconds per shot, how many shots needed each
stage, and whether the final decisions differ. Writes results/decoder_benchmark.json.
(Timings depend on the machine and its load; the decisions do not.)"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hase.protocols import make_protocol  # noqa: E402
from hase.runner import SupersetDecoder  # noqa: E402
from hase.engine import sample_superset_faults, simulate_with_faults  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main(shots=200):
    proto = make_protocol("carbonhelix", 1e-3, "X", 4, "full")
    plan_all = [[pt for pt, _ in proto.plan(r)] for r in range(proto.n_rounds)]
    out = {}
    for kappa in (2.0, 3.0):
        rng = np.random.default_rng(77)
        faults, W = sample_superset_faults(proto, plan_all, shots, rng, kappa)
        st = simulate_with_faults(proto, faults, shots)
        preds = {}
        for K, D in ((256, 10.0), (512, 12.0)):
            dec = SupersetDecoder(proto, K=K, Delta=D, committee=True, K2=4096, Delta2=24)
            sup = dec.sup
            path = tuple(st.paths[0])
            Dm = np.array([sup.syndrome(path, st.records[i], st.final[i])[0] for i in range(shots)])
            Om = np.array([sup.syndrome(path, st.records[i], st.final[i])[1] for i in range(shots)])
            dec.decode(path, Dm[:2])                    # build the native models first
            t = time.time()
            pred, gap, nre, nch = dec.decode(path, Dm)
            dt = time.time() - t
            preds[K] = pred
            out[f"kappa{kappa:g}_K{K}"] = dict(seconds_per_shot=dt / shots, mean_W=float(W.mean()),
                                               stage_use=np.bincount(dec.last_stage, minlength=3).tolist(),
                                               failures=int(np.any(pred != Om, axis=1).sum()))
        out[f"kappa{kappa:g}_decisions_differ"] = int(np.any(preds[256] != preds[512], axis=1).sum())
    out["shots"] = shots
    json.dump(out, open(os.path.join(ROOT, "results", "decoder_benchmark.json"), "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
