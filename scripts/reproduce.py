"""Refit the displayed comparisons from saved fault-count samples."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'analysis'), str(ROOT/'src')]
from value import configurations, evaluate

KEYS = [('C4-Helix', 'mode=ase'), ('Carbon-Helix', 'full'),
        ('Carbon-Helix', 'l2=on_event,l3=ambiguous,rot=8')]

def main():
    source = json.loads((ROOT/'results/summary.json').read_text())
    selected = [r for r in configurations() if (r['code'], r['key'][1]) in KEYS
                and r['key'][2] == 'helios' and r['key'][4] == 4]
    assert len(selected) == 3
    report = []
    for row in selected:
        for p in (0.00079, 0.0003):
            rec = evaluate(row, p, nboot=50)
            key = f"{row['code']}|{row['key'][1]}|r4"
            saved = source['value'][f'helios|{p}']['rows'][key]
            for field in ('eps_LQ','ci95_LQ','measured_share','twoq_per_round','time_ms_per_round'):
                np.testing.assert_allclose(rec[field], saved[field], rtol=2e-7, atol=1e-14,
                                           err_msg=f'{key} p={p}: {field}')
            report.append({'key':key, 'p':p, 'recomputed':rec})
            print(f'Reproduced {key} at p={p}', flush=True)
    out = ROOT/'build';out.mkdir(exist_ok=True)
    (out/'reproduction.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__ == '__main__': main()
