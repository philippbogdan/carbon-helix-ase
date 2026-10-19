"""Reduce saved failing fault configurations to locally minimal failing subsets and
classify each by exact most-likely-error decoding (integer program on the shot's own
path DEM) and by a strong Frontier setting. The smallest size among genuine minimal
sets is the minimal failing weight, i.e. the asymptotic exponent of the logical error.

Usage: python analysis/minimal_fails.py <glob of *_fails.pkl> <proto> <rule> <basis>
       <rounds> [limit] [out.json] [noise] [p_dec]
       python analysis/minimal_fails.py --campaign <campaign> <tag> <basis> [limit] [part/nparts]
"""
import glob
import json
import os
import pickle
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from hase.protocols import make_protocol  # noqa: E402
from hase.runner import SupersetDecoder, minimize_failure, fails_with  # noqa: E402
from decoder_audit import shot_from_faults, exact_mle  # noqa: E402
from hase.decoding import FrontierDecoder  # noqa: E402


def classify(proto, fl):
    pd, D, O = shot_from_faults(proto, fl)
    mle = exact_mle(pd, D, time_limit=300)
    fd = FrontierDecoder(H=pd.H, L=pd.L, priors=pd.priors, K=8192, Delta=30, committee=True)
    p, g, s = fd.decode_batch(D[None, :])
    return (None if mle is None else bool(np.any(mle != O))), bool(np.any(p[0] != O))


def failing_items(paths, order="file"):
    """Stored failing shots of the given files, in file order or by increasing number of
    faults (W; ties by file and position) - the fewest-fault failures reduce to the smallest
    cores, which set the asymptotic order."""
    items = []
    for fi, fpath in enumerate(sorted(paths)):
        for ii, item in enumerate(pickle.load(open(fpath, "rb"))):
            if item.get("failed", True):
                items.append((item["W"], fi, ii, item))
    if order == "lowest_w":
        items.sort(key=lambda t: t[:3])
    return [t[3] for t in items]


def main(pattern, proto_name, rule, basis, rounds, limit=200, out=None, noise="uniform",
         p_dec=1e-3, campaign=None, classify_exact=True, K=512, Delta=12.0, restarts=3,
         classify_max=8, order="file", part=None, nparts=None):
    proto = make_protocol(proto_name, p_dec, basis, rounds, rule, noise)
    # the campaign's decoder (scripts/campaigns.py)
    dec = SupersetDecoder(proto, K=K, Delta=Delta, committee=True, K2=4096, Delta2=24)
    rng = np.random.default_rng(0 if part is None else part)
    rows = []
    paths = sorted(pattern) if isinstance(pattern, (list, tuple)) else sorted(glob.glob(pattern))
    items = failing_items(paths, order)
    if part is not None:
        items = items[part::nparts]
    for item in items:
        f, _ = fails_with(proto, dec, item["faults"])
        if not f:
            continue
        # greedy removal finds a locally minimal failing subset; keep the smallest of
        # three random removal orders
        best = None
        for _ in range(restarts):
            fl_, k_ = minimize_failure(proto, dec, item["faults"], rng)
            if best is None or k_ < best[1]:
                best = (fl_, k_)
        fl, k = best
        # exact most-likely-error classification (integer program) only for the small
        # cores that set the asymptotic order
        mle_fails, strong_fails = classify(proto, fl) if (classify_exact and k <= classify_max) \
            else (None, None)
        print(json.dumps(dict(W=item["W"], size=k, mle_fails=mle_fails, strong_fails=strong_fails)),
              flush=True)
        rows.append(dict(W=item["W"], size=k, mle_fails=mle_fails, strong_fails=strong_fails,
                         faults={str(kk): v for kk, v in fl.items()}))
        if len(rows) >= limit:
            break
    sizes_all = Counter(r["size"] for r in rows)
    sizes_genuine = Counter(r["size"] for r in rows if r["mle_fails"] or r["strong_fails"])
    for r in rows:
        # a core counts as a failure of the decoder used for the estimates in any case; it
        # is 'genuine' when exact most-likely-error decoding (or a much wider beam) fails too
        r["counted"] = True
    res = dict(campaign=campaign, proto=proto_name, rule=rule, basis=basis, rounds=rounds, noise=noise,
               p_dec=p_dec, n=len(rows), rows=rows, selection=order, part=part, nparts=nparts,
               sizes_all=dict(sorted(sizes_all.items())),
               sizes_genuine=dict(sorted(sizes_genuine.items())),
               min_genuine=min(sizes_genuine) if sizes_genuine else None,
               examples=[r for r in rows if r["size"] == (min(sizes_genuine) if sizes_genuine else -1)][:10])
    from hase.version import commit, provenance
    res["commit"], res["provenance"] = commit(), provenance()
    print(json.dumps({k: v for k, v in res.items() if k not in ("examples", "rows", "provenance")}))
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(res, open(out, "w"), indent=1)
    return res


def campaign_main(campaign, tag, basis, limit=150, part=None, nparts=None, out_path=None):
    """Minimal failing sets of one campaign configuration -> results/minimal/: the stored
    failures with the fewest faults first. With part/nparts they are dealt round-robin so that
    several jobs share the work; each writes <campaign>__<tag>__<basis>__p<part>of<nparts>.json
    and the analyses merge the parts."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    files = sorted(glob.glob(os.path.join(root, f"results/final/{campaign}/{tag}/{basis}/*/*_faults.pkl")))
    if not files:
        print("no stored shots")
        return None
    meta = json.load(open(files[0].replace("_faults.pkl", ".json")))
    name = f"{campaign}__{tag.replace('/', '_')}__{basis}"
    if part is not None:
        name += f"__p{part}of{nparts}"
    out = out_path or os.path.join(root, "results", "minimal", f"{name}.json")
    # the failures with the fewest faults first (the smallest cores come from them), dealt
    # round-robin to the parts
    return main(files, meta["proto"], meta["rule"], basis, meta["rounds"], limit=limit, out=out,
                noise=meta["noise"], p_dec=meta["p_dec"], campaign=campaign,
                K=meta.get("K", 512), Delta=meta.get("Delta", 12.0), classify_max=7,
                order="lowest_w", part=part, nparts=nparts)


if __name__ == "__main__":
    a = list(sys.argv)
    op = None
    if "--out" in a:                        # output override (regression runs)
        op = a[a.index("--out") + 1]
        del a[a.index("--out"):a.index("--out") + 2]
    if len(a) > 1 and a[1] == "--campaign":
        part = nparts = None
        if len(a) > 6:                      # ... <limit> <part>/<nparts>
            part, nparts = (int(x) for x in a[6].split("/"))
        campaign_main(a[2], a[3], a[4], limit=int(a[5]) if len(a) > 5 else 150, part=part, nparts=nparts,
                      out_path=op)
        sys.exit(0)
    main(a[1], a[2], a[3], a[4], int(a[5]), limit=int(a[6]) if len(a) > 6 else 200,
         out=op or (a[7] if len(a) > 7 else None), noise=a[8] if len(a) > 8 else "uniform",
         p_dec=float(a[9]) if len(a) > 9 else 1e-3, campaign=a[10] if len(a) > 10 else None,
         classify_exact=(a[11] != "0") if len(a) > 11 else True)
