"""Time structure of the minimal failing sets (cores) of each configuration: how many
distinct rounds a core's faults occupy and the span between its first and last round.
In-round failures of non-adaptive extraction sit in one or two consecutive rounds (a fault
is detected at the latest one round later); window failures of adaptive rules can spread
over the rounds between two outer measurements (docs/interval_theory.md). Counted: confirmed
six-fault cores whose faults lie at least two rounds apart, and cores that only the
campaign decoder fails (exact decoding corrects them). Reads results/minimal/*.json."""
import ast
import glob
import json
import os
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def census():
    out = {}
    groups = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "minimal", "*__*__*.json"))):
        # parts of one configuration (<campaign>__<tag>__<basis>__p<i>of<n>) are merged
        key = "__".join(os.path.basename(f)[:-5].split("__")[:3])
        groups.setdefault(key, []).append(json.load(open(f)))
    for key, ds in groups.items():
        d = dict(ds[0])
        d["rows"] = [r for x in ds for r in x.get("rows", [])]
        d["n"] = len(d["rows"])
        spans, nrounds = Counter(), Counter()
        sizes, genuine, genuine_span = Counter(), Counter(), Counter()
        multi_small = decoder_only = 0
        for r in d.get("rows", []):
            fl = r.get("faults")
            if not fl:
                continue
            rounds = sorted({ast.literal_eval(k)[0] for k in fl if k not in ("init", "final")})
            if not rounds:
                continue
            spans[rounds[-1] - rounds[0]] += 1
            nrounds[len(rounds)] += 1
            sizes[r["size"]] += 1
            if r.get("mle_fails") or r.get("strong_fails"):
                genuine[r["size"]] += 1
                # confirmed six-fault cores whose faults lie at least two rounds apart: beyond
                # the adjacent-round reach of in-round failures (detection one round later)
                if r["size"] <= 6 and rounds[-1] - rounds[0] >= 2:
                    multi_small += 1
                genuine_span[rounds[-1] - rounds[0]] += 1
            decoder_only += int(r.get("mle_fails") is False and r.get("strong_fails") is False)
        out[key] = dict(rule=d.get("rule"), rounds=d.get("rounds"), cores=d.get("n"),
                        sizes={str(k): v for k, v in sorted(sizes.items())},
                        sizes_genuine={str(k): v for k, v in sorted(genuine.items())},
                        genuine_six_fault_cores_over_several_rounds=multi_small,
                        genuine_round_span={str(k): v for k, v in sorted(genuine_span.items())},
                        decoder_only_cores=decoder_only,
                        round_span={str(k): v for k, v in sorted(spans.items())},
                        distinct_rounds={str(k): v for k, v in sorted(nrounds.items())})
    return out


if __name__ == "__main__":
    print(json.dumps(census(), indent=1))
