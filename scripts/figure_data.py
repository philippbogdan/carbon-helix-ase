"""Export the figure pack and record its source files. --check leaves the pack intact."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

from headline import rows

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'figures/data'
SOURCE = ROOT / 'results/summary.json'
sys.path.insert(0, str(ROOT / 'analysis'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    summary = json.loads(SOURCE.read_text())
    pack = {f'{key}.json': summary[key] for key in ('circuits', 'distance', 'interval')}
    table = rows()
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(table[0]))
    writer.writeheader(); writer.writerows(table)
    if args.check:
        for name, expected in pack.items():
            if json.loads((DATA / name).read_text()) != expected:
                raise RuntimeError(f'{name} differs from results/summary.json')
        if (DATA / 'headline.csv').read_bytes() != stream.getvalue().encode():
            raise RuntimeError('headline.csv differs from the six exported rows')
    else:
        DATA.mkdir(exist_ok=True)
        for name, value in pack.items():
            (DATA / name).write_text(json.dumps(value, indent=2) + '\n')
        (DATA / 'headline.csv').write_bytes(stream.getvalue().encode())
    # Check the saved solver aggregation and circuit accounting without rerunning searches.
    from make_all import sec_distance, sec_circuits
    if sec_distance() != pack['distance.json']:
        raise RuntimeError('Distance aggregation differs from saved solver tasks')
    if sec_circuits() != pack['circuits.json']:
        raise RuntimeError('Circuit accounting differs from the source protocols')
    patterns = {
        'headline.csv': ['results/final/c4_strat_helios/*/*/*/*.json',
                         'results/final/ch_helios/*/*/*/*.json'],
        'interval.json': ['results/final/ch_interval/*/*/*/*.json'],
        'distance.json': ['results/distance/*.json', 'results/gadget_gain.json'],
        'circuits.json': ['src/hase/*.py'],
    }
    report = {'summary_sha256': digest(SOURCE), 'source_commit': summary['commit'],
              'implementation_sha256': {str(p.relative_to(ROOT)): digest(p) for p in sorted([
                  ROOT/'scripts/figure_data.py', ROOT/'scripts/headline.py',
                  ROOT/'scripts/reproduce.py', ROOT/'scripts/evidence.py',
                  ROOT/'analysis/make_all.py', ROOT/'analysis/strat.py',
                  ROOT/'analysis/value.py', ROOT/'analysis/interval.py',
                  ROOT/'analysis/resources.py', ROOT/'figures/make_figures.py',
                  ROOT/'figures/_export/figure_export.py', ROOT/'figures/_export/_common.py',
                  ROOT/'requirements.txt'])}, 'files': {}}
    for name, globs in patterns.items():
        paths = sorted({p for pattern in globs for p in ROOT.glob(pattern)})
        if not paths:
            raise RuntimeError(f'No inputs for {name}')
        report['files'][name] = {'sha256': digest(DATA / name),
            'inputs': {str(p.relative_to(ROOT)): digest(p) for p in paths}}
    (DATA / 'provenance.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print('Checked all four figure inputs, saved solver aggregation and circuit accounting')


if __name__ == '__main__':
    main()
