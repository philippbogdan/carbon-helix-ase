"""Size of the timing/noise simplification for per-block weight-6 parts on the Helios model:
for each adaptive rule, the distribution of the number k of blocks measuring their weight-6
checks in the same round (2000 simulated shots, 4 rounds) and the memory error left out
because each block's part carries the noise of its own single-block schedule while k blocks
together take longer (src/hase/timing.py). Writes results/timing_consistency.json."""
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hase.protocols import make_protocol  # noqa: E402
from hase.engine import simulate  # noqa: E402
from hase.timing import l2_parallel_duration  # noqa: E402
from hase.helios import helios_noise  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main(shots=2000):
    nz = helios_noise("helios", 7.9e-4)
    rate = nz.mem_dep_per_ms + nz.mem_z_per_ms
    out = {}
    for tag, rule in (("B4", "l2=on_event,l3=ambiguous,T3=4"), ("C", "l2=on_event,l3=any"),
                      ("R4", "l2=on_event,l3=ambiguous,rot=4"), ("R8", "l2=on_event,l3=ambiguous,rot=8")):
        proto = make_protocol("carbonhelix", 7.9e-4, "X", 4, rule, "helios")
        st = simulate(proto, shots, np.random.default_rng(5))
        ks = Counter()
        for p in st.paths:
            by = {}
            for r, pt in p:
                if r:
                    by.setdefault(r, []).append(pt)
            for pts in by.values():
                ks[sum(1 for x in pts if x.startswith("L2_") and x != "L2_all")] += 1
        tot = sum(ks.values())
        miss = sum(c * (l2_parallel_duration(k) - l2_parallel_duration(1)) for k, c in ks.items() if k >= 2) / tot
        faults = float(np.mean([sum(proto.parts[pt].site_prob.sum() for r, pt in p if r) for p in st.paths])) / 4
        out[tag] = dict(blocks_per_round={str(k): c / tot for k, c in sorted(ks.items())},
                        missing_ms_per_round=miss, missing_faults_per_round=miss * rate * proto.n_data,
                        faults_per_round=faults)
    json.dump(out, open(os.path.join(ROOT, "results", "timing_consistency.json"), "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
