"""Refit the headline comparison from the saved fault-count samples and print its numbers."""
import argparse
import csv
import io
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'analysis'), str(ROOT / 'src')]
from hase.protocols import make_protocol  # noqa: E402
from hase.timing import path_time_ms  # noqa: E402

SUMMARY = ROOT / 'results/summary.json'
TABLE = ROOT / 'figures/data/headline.csv'
SERIES = [('C4 adaptive', 'C4-Helix', 'mode=ase'),
          ('Carbon full', 'Carbon-Helix', 'full'),
          ('Carbon R8', 'Carbon-Helix', 'l2=on_event,l3=ambiguous,rot=8')]
POINTS = (0.00079, 0.0003)
CAPACITY = 98
FIELDS = ('eps_LQ', 'ci95_LQ', 'measured_share', 'twoq_per_round', 'time_ms_per_round')


def rate(e, rounds, prep_ms, round_ms):
    return -rounds * math.log1p(-2 * e) / (2 * (prep_ms + rounds * round_ms) / 1000)


def prep_ms(code, rule, p, noise='helios'):
    name = 'c4helix' if code == 'C4-Helix' else 'carbonhelix'
    pr = make_protocol(name, p, 'X', 4, rule, noise)
    return path_time_ms(pr, [(0, part) for part, _ in pr.plan(0)])


def table_rows(estimates):
    out = []
    for p in POINTS:
        for label, code, rule in SERIES:
            key = f'{code}|{rule}|r4'
            r = estimates[p][key]
            t0 = prep_ms(code, rule, p)
            row = {k: r[k] for k in ('p', 'rounds', 'eps_LQ', 'measured_share', 'status', 'twoq_per_round',
                                     'time_ms_per_round', 'peak_qubits', 'shots', 'fails')}
            row.update(label=label, key=key, prep_ms=t0, ci_low=r['ci95_LQ'][0], ci_high=r['ci95_LQ'][1])
            for field, e in (('rate', r['eps_LQ']), ('rate_low', r['ci95_LQ'][0]), ('rate_high', r['ci95_LQ'][1])):
                row[field] = rate(e, 4, t0, r['time_ms_per_round'])
            out.append(row)
    return out


def csv_bytes(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def refit(nboot):
    from value import configurations, evaluate
    wanted = {(code, rule) for _, code, rule in SERIES}
    selected = [r for r in configurations() if (r['code'], r['key'][1]) in wanted
                and r['key'][2] == 'helios' and r['key'][4] == 4]
    assert len(selected) == 3
    out = {p: {} for p in POINTS}
    for row in selected:
        for p in POINTS:
            rec = evaluate(row, p, nboot=nboot)
            out[p][f"{row['code']}|{row['key'][1]}|r4"] = rec
            print(f"refitted {row['code']} {row['key'][1]} at p = {p}", flush=True)
    return out


def numbers(rows):
    at = {(r['label'], r['p']): r for r in rows}
    c4, full, r8 = (at[(s[0], 0.00079)] for s in SERIES)
    return {
        'p': 0.00079,
        'rate_per_s': {'C4 adaptive': c4['rate'], 'Carbon full': full['rate'], 'Carbon R8': r8['rate']},
        'rate_ratio_C4_over_R8': c4['rate'] / r8['rate'],
        'round_error_ratio_C4_over_R8': c4['eps_LQ'] / r8['eps_LQ'],
        'peak_qubits': r8['peak_qubits'], 'qubit_capacity': CAPACITY,
        'twoq_per_round': {'Carbon full': full['twoq_per_round'], 'Carbon R8': r8['twoq_per_round']},
        'twoq_saving': 1 - r8['twoq_per_round'] / full['twoq_per_round'],
        'round_ms': {'Carbon full': full['time_ms_per_round'], 'Carbon R8': r8['time_ms_per_round']},
        'round_time_saving': 1 - r8['time_ms_per_round'] / full['time_ms_per_round'],
        'R8_rate_below_full': 1 - r8['rate'] / full['rate'],
        'full_rate_above_R8': full['rate'] / r8['rate'] - 1,
        'measured_share': {k: at[(k, 0.00079)]['measured_share'] for k, _, _ in SERIES},
        'measured_share_p0.0003': {k: at[(k, 0.0003)]['measured_share'] for k, _, _ in SERIES},
    }


def sensitivity(summary):
    """Saved sensitivity estimates per second, with preparation timed under each variant."""
    out = {}
    for variant in ('helios_mem0.5', 'helios_mem2', 'helios_tm4'):
        rates = {}
        for code, rule in (('C4-Helix', 'mode=ase'), ('Carbon-Helix', 'full'),
                           ('Carbon-Helix', 'l2=on_event,l3=ambiguous,rot=4')):
            r = summary[f'{code}|{rule}|{variant}']
            t0 = prep_ms(code, rule, 0.00079, variant)
            rates[f'{code}|{rule}'] = dict(rate=rate(r['eps_LQ'], 4, t0, r['time_ms_per_round']),
                                          measured_share=r['measured_share'])
        c4 = rates['C4-Helix|mode=ase']['rate']
        out[variant] = dict(rates=rates,
                            ratio_C4_over_Carbon_full=c4 / rates['Carbon-Helix|full']['rate'],
                            ratio_C4_over_Carbon_R4=c4 / rates['Carbon-Helix|l2=on_event,l3=ambiguous,rot=4']['rate'])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--nboot', type=int, default=50)
    ap.add_argument('--saved', action='store_true')
    a = ap.parse_args()
    summary = json.loads(SUMMARY.read_text())
    saved = summary['value']
    saved = {p: saved[f'helios|{p}']['rows'] for p in POINTS}
    fresh = saved if a.saved else refit(a.nboot)
    if not a.saved:
        for p in POINTS:
            for key, rec in fresh[p].items():
                for field in FIELDS:
                    np.testing.assert_allclose(rec[field], saved[p][key][field], rtol=2e-7, atol=1e-14,
                                               err_msg=f'{key} p={p}: {field}')
    rows = table_rows(fresh)
    stored = list(csv.DictReader(io.StringIO(TABLE.read_text())))
    for new, old in zip(rows, stored, strict=True):
        for k, v in new.items():
            if isinstance(v, float):
                np.testing.assert_allclose(v, float(old[k]), rtol=2e-7, atol=1e-14, err_msg=f"{new['key']} {k}")
            else:
                assert str(v) == old[k], (new['key'], k, v, old[k])
    result = numbers(rows)
    result['device_assumptions'] = sensitivity(summary['sensitivity'])
    out = ROOT / 'build'
    out.mkdir(exist_ok=True)
    (out / 'headline.json').write_text(json.dumps({'numbers': result, 'rows': rows}, indent=2) + '\n')
    n = result
    print()
    print(f"p = {n['p']}, Helios-like noise and timing, X memory, 4 rounds after preparation")
    print(f"memory error per second: C4 adaptive {n['rate_per_s']['C4 adaptive']:.3g}, "
          f"Carbon full {n['rate_per_s']['Carbon full']:.3g}, Carbon R8 {n['rate_per_s']['Carbon R8']:.3g}")
    print(f"C4 adaptive / Carbon R8, per second: {n['rate_ratio_C4_over_R8']:.1f}x "
          f"(per round: {n['round_error_ratio_C4_over_R8']:.1f}x)")
    print(f"qubits: {n['peak_qubits']:.0f} of {CAPACITY}")
    print(f"two-qubit gates per round: {n['twoq_per_round']['Carbon R8']:.1f} vs "
          f"{n['twoq_per_round']['Carbon full']:.0f} ({100 * n['twoq_saving']:.1f}% fewer)")
    print(f"round time: {n['round_ms']['Carbon R8']:.1f} vs {n['round_ms']['Carbon full']:.1f} ms "
          f"({100 * n['round_time_saving']:.1f}% shorter)")
    print(f"R8 error per second vs full: {100 * n['R8_rate_below_full']:.1f}% lower "
          f"(full is {100 * n['full_rate_above_R8']:.1f}% higher)")
    print('share from sampled strata: ' + ', '.join(f'{k} {100 * v:.1f}%' for k, v in n['measured_share'].items())
          + '; at p = 0.0003: ' + ', '.join(f'{k} {100 * v:.1f}%' for k, v in n['measured_share_p0.0003'].items()))
    for variant, s in result['device_assumptions'].items():
        print(f"{variant}: C4 adaptive / Carbon full {s['ratio_C4_over_Carbon_full']:.1f}x, "
              f"C4 adaptive / Carbon R4 {s['ratio_C4_over_Carbon_R4']:.1f}x per second (saved estimates)")
    print(f'matches results/summary.json and {TABLE.relative_to(ROOT)}; written to build/headline.json')


if __name__ == '__main__':
    main()
