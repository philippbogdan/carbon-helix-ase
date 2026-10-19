"""Decoder audit of a stratified campaign: re-decode the stored shots with a much stronger
decoder (Frontier K = 2048, Delta = 20 primary, then 4096/24) and count, per stratum w,
  * campaign failures that the strong decoder also fails ('genuine'),
  * campaign failures that the strong decoder corrects ('decoder-induced'),
  * sampled campaign successes (a fixed random fraction of all shots) that the strong
    decoder fails (its own approximation in the other direction).
From these the strong decoder's failure probability per stratum is
    f_w(strong) = f_w(campaign) * g_w + (1 - f_w(campaign)) * r_w,
g_w the genuine fraction and r_w the success-to-failure rate. Writes
results/redecode/<campaign>__<tag>__<basis>.json.

Usage: python analysis/redecode.py <campaign> <tag> <basis> [limit]
"""
import glob
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hase.protocols import make_protocol  # noqa: E402
from hase.runner import SupersetDecoder, fails_with  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main(campaign, tag, basis, limit=100000, K=2048, Delta=20.0, out_path=None):
    files = sorted(glob.glob(os.path.join(ROOT, f"results/final/{campaign}/{tag}/{basis}/*/*_faults.pkl")))
    if not files:
        print("no stored shots")
        return None
    meta = json.load(open(files[0].replace("_faults.pkl", ".json")))
    proto = make_protocol(meta["proto"], meta["p_dec"], basis, meta["rounds"], meta["rule"], meta["noise"])
    strong = SupersetDecoder(proto, K=K, Delta=Delta, committee=True, K2=4096, Delta2=24, stages=[])
    cnt = defaultdict(lambda: [0, 0, 0, 0])     # w -> [fail, fail&strongfail, ok, ok&strongfail]
    n = 0
    for f in files:
        for it in pickle.load(open(f, "rb")):
            fs, _ = fails_with(proto, strong, it["faults"])
            c = cnt[int(it["W"])]
            if it.get("failed", True):
                c[0] += 1
                c[1] += int(fs)
            else:
                c[2] += 1
                c[3] += int(fs)
            n += 1
            if n >= limit:
                break
        if n >= limit:
            break
    out = dict(campaign=campaign, tag=tag, basis=basis, rule=meta["rule"], noise=meta["noise"],
               rounds=meta["rounds"], strong_decoder=dict(K=K, Delta=Delta, committee=True, final_stage=[4096, 24]),
               per_w={str(w): dict(fail=c[0], fail_strong=c[1], ok_sampled=c[2], ok_strong_fail=c[3])
                      for w, c in sorted(cnt.items())})
    tot = np.array(list(cnt.values())).sum(axis=0) if cnt else np.zeros(4)
    out["totals"] = dict(fail=int(tot[0]), genuine=int(tot[1]), ok_sampled=int(tot[2]), ok_strong_fail=int(tot[3]))
    from hase.version import commit, provenance
    out["commit"], out["provenance"] = commit(), provenance()
    os.makedirs(os.path.join(ROOT, "results", "redecode"), exist_ok=True)
    path = out_path or os.path.join(ROOT, "results", "redecode", f"{campaign}__{tag}__{basis}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps(out["totals"]))
    return out


if __name__ == "__main__":
    a = list(sys.argv)
    op = None
    if "--out" in a:                       # output override (regression runs)
        op = a[a.index("--out") + 1]
        del a[a.index("--out"):a.index("--out") + 2]
    main(a[1], a[2], a[3], limit=int(a[4]) if len(a) > 4 else 100000, out_path=op)
