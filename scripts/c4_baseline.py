"""Check full-extraction C4-Helix against the authors' Fig. 11a values.

Refit the saved c4_repro samples under uniform noise: 12 memory rounds,
mean X/Z block error per round, no postselection (gamma = 0). Reference
values are our readings of arXiv:2609.03194, recorded in results/reference/.
The bootstrap intervals are conditional on the minimum-order-three fit and
exclude figure-reading uncertainty. This does not validate adaptive C4 or
the Helios-like noise model.

Require agreement with results/summary.json section c4_uniform, print the
comparison and write build/c4_baseline.json. No fresh circuit sampling or decoding.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'analysis'), str(ROOT / 'src')]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()
    from aggregate import sec_c4_uniform, same
    fresh = json.loads(json.dumps(sec_c4_uniform(), default=float))
    saved = json.loads((ROOT / 'results/summary.json').read_text())['c4_uniform']
    if not same(fresh, saved):
        raise SystemExit('differs from results/summary.json section c4_uniform')
    out = ROOT / 'build'
    out.mkdir(exist_ok=True)
    (out / 'c4_baseline.json').write_text(json.dumps(fresh, indent=2) + '\n')
    print(f"C4-Helix, uniform noise, {fresh['rounds']} rounds, mean of X and Z block error per round; "
          f"{fresh['shots']['X']} + {fresh['shots']['Z']} stratified shots")
    for p, r in fresh['points'].items():
        print(f"p = {p}: ours {r['eps_avg']:.3g} (95% {r['ci95'][0]:.3g} to {r['ci95'][1]:.3g}), "
              f"reference {r['paper']:.3g}, ratio {r['ratio']:.3f}, reference inside interval: {r['paper_inside_ci']}")
    print('matches results/summary.json; written to build/c4_baseline.json')


if __name__ == '__main__':
    main()
