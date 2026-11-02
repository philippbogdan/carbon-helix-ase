"""Export selected saved summary sections, including the c4_uniform comparison.

The C4 reference values are credited in results/reference/; this copies the
comparison without refitting or rendering. --check compares without writing.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'figures/data'
SECTIONS = ('circuits', 'distance', 'interval', 'sensitivity', 'direct', 'c4_uniform')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    summary = json.loads((ROOT / 'results/summary.json').read_text())
    DATA.mkdir(parents=True, exist_ok=True)
    for key in SECTIONS:
        path = DATA / f'{key}.json'
        if a.check:
            if json.loads(path.read_text()) != summary[key]:
                raise SystemExit(f'{path.relative_to(ROOT)} differs from results/summary.json')
        else:
            path.write_text(json.dumps(summary[key], indent=1) + '\n')
    print('figure data pack ' + ('matches' if a.check else 'written from') + ' results/summary.json')


if __name__ == '__main__':
    main()
