"""Describe an undetectable logical error found by scripts/job_distance.py: for each mechanism
(merged column of the path's detector error model) the part instances (round, part) and
data-flip sites whose faults have that effect, and its detectors.

Usage: python analysis/distance_witness.py results/distance/<file>.json
"""
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from hase.protocols import make_protocol  # noqa: E402
from hase.engine import PathDEM  # noqa: E402


def describe(d):
    proto = make_protocol("carbonhelix", 1e-3, d["basis"], d["rounds"], d["rule"], "uniform")
    path = [tuple(p) for p in d["path"]]
    pd = PathDEM(proto, path)
    H, L = pd.Hraw.tocsr(), pd.Lraw.tocsr()
    key_rows = {}
    for i in range(H.shape[0]):
        key = (tuple(sorted(H.indices[H.indptr[i]:H.indptr[i + 1]].tolist())),
               tuple(sorted(L.indices[L.indptr[i]:L.indptr[i + 1]].tolist())))
        key_rows.setdefault(key, []).append(i)
    Hm, Lm = pd.H.tocsc(), pd.L.tocsc()
    out = []
    for j in d["solution"]:
        key = (tuple(sorted(Hm.indices[Hm.indptr[j]:Hm.indptr[j + 1]].tolist())),
               tuple(sorted(Lm.indices[Lm.indptr[j]:Lm.indptr[j + 1]].tolist())))
        rows = key_rows.get(key, [])
        owners = Counter()
        for i in rows:
            k = pd.owner[i][0]
            owners["data flip (init/final)" if k < 0 else f"round {path[k][0]} {path[k][1]}"] += 1
        out.append(dict(column=j, detectors=[pd.dets[x] for x in key[0]], observables=list(key[1]),
                        sources=dict(owners)))
    return out


if __name__ == "__main__":
    d = json.load(open(sys.argv[1]))
    for m in describe(d):
        print(json.dumps({k: (str(v) if k == "detectors" else v) for k, v in m.items()}))
