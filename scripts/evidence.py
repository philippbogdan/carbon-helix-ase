"""Check saved circuit bounds and export the exponent and interval tables.

--refit recomputes free-order fits and periodic ratios from raw counts.
--windows reruns the constructed six-fault failures without changing saved data.
"""
import argparse
import ast
from contextlib import redirect_stdout
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'analysis'), str(ROOT/'third_party/frontier')]
import numpy as np

TAGS = ('full', 'C', 'B4', 'R4', 'R8', 'R8F')
LABELS = ('Full', 'Any event', 'Ambiguous, T=4', 'R4', 'R8', 'R8F')
SOURCE = ROOT/'results/summary.json'


def check_bounds(s):
    """Require complete task coverage and every logical class above ten faults."""
    report = {}
    for name, expected in s['relaxed_distance'].items():
        files = sorted((ROOT/'results/relaxed_distance'/name).glob('*.json'))
        covered, subsets_seen, no_gadget = set(), set(), False
        for path in files:
            d = json.loads(path.read_text())
            assert d['W'] == 10 and d['n_subsets'] == expected['n_subsets'], path
            assert d['basis'] == expected['basis'] and d['kind'] == expected['kind'], path
            covered.update(d['subsets'])
            for subset, result in d['results'].items():
                subset = tuple(ast.literal_eval(subset))
                assert subset not in subsets_seen, (name, subset)
                subsets_seen.add(subset)
                assert set(result['min']) == {'(0,)', '(1,)', '(0, 1)'}, path
                assert all(v == '> 10' for v in result['min'].values()), path
            if 'no_gadget' in d:
                assert all(v == '> 10' for v in d['no_gadget'].values()), path
                no_gadget = True
        assert covered == set(range(expected['n_subsets'])), name
        assert len(subsets_seen) == len(covered) and no_gadget, name
        assert expected['complete'] and expected['lower_bound'] == 11, name
        report[name] = {'tasks': len(files), 'subsets': len(covered), 'lower_bound': 11}
    paths = 0
    for T in (2, 4, 8):
        for basis in 'XZ':
            d = json.loads((ROOT/f'results/inner_window/T{T}_{basis}.json').read_text())
            assert len(d['paths']) == 2**(T-1)
            for result in d['paths'].values():
                assert result['all_optimal'] and result['distance'] == 4
                assert all(c['optimal'] and c['weight'] >= 4 for c in result['classes'].values())
            paths += len(d['paths'])
    assert paths == 276
    # Check the actual 11-fault witness against this copy's detector model.
    from hase.protocols import make_protocol
    from hase.engine import PathDEM
    d = json.loads((ROOT/'results/distance/whole__k4cyc__ZX__Z__c1.json').read_text())
    proto = make_protocol('carbonhelix', 1e-3, d['basis'], d['rounds'], d['rule'], 'uniform')
    pd = PathDEM(proto, [tuple(p) for p in d['path']])
    assert len(set(d['solution'])) == 11
    assert not np.any(np.asarray(pd.H[:, d['solution']].sum(axis=1)) % 2)
    assert np.any(np.asarray(pd.L[:, d['solution']].sum(axis=1)) % 2)
    return {'relaxed_bounds': report, 'inner_paths': paths, 'eleven_fault_witness': 'valid',
            'limit': 'Saved proof outputs checked for coverage; searches not rerun.'}


def data_rows(s):
    orders, intervals = [], []
    for noise in ('uniform', 'helios'):
        for tag, label in zip(TAGS, LABELS):
            r = s['carbon'][noise]['X'][tag]
            v = r['fault_order_all']
            orders.append(dict(noise=noise, rule=tag, label=label, exponent=v['slope_at_p_star'],
                               low=v['slope_ci95'][0], high=v['slope_ci95'][1], p_star=v['p_star'],
                               shots=r['shots'], failures=r['fails']))
    r = s['interval']['P']
    for t in r['T']:
        key = str(t)
        for i, p in enumerate(r['ps']):
            share = min(r['measured_share']['full'][i], r['measured_share'][key][i])
            intervals.append(dict(T=t, p=p, ratio=r['ratio'][key][i],
                                  low=r['ratio_ci95'][key][0][i], high=r['ratio_ci95'][key][1][i],
                                  sampled_share_min=share, data_driven=share >= 0.5,
                                  shots=r['data'][key]['shots'], failures=r['data'][key]['fails'],
                                  full_shots=r['data']['full']['shots'], full_failures=r['data']['full']['fails'],
                                  sampled_share=r['measured_share'][key][i],
                                  full_sampled_share=r['measured_share']['full'][i]))
    return orders, intervals


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def draw(s):
    out = ROOT/'figures'
    orders, intervals = data_rows(s)
    write_csv(out/'exponents.csv', orders)
    write_csv(out/'intervals.csv', intervals)


def refit(s):
    from strat import load, StratModel, local_order
    from carbon import TAG_OF
    count = 0
    for noise, campaign in [('uniform', 'ch_main'), ('helios', 'ch_helios')]:
        for key, raw in load(str(ROOT/f'results/final/{campaign}/*/*/*/*.json')).items():
            tag = TAG_OF.get(key[1], key[1])
            if key[3] != 'X' or key[4] != 4 or tag not in TAGS:
                continue
            actual = local_order(StratModel(key, raw), use_all=True)
            expected = s['carbon'][noise]['X'][tag]['fault_order_all']
            for field in ('slope_at_p_star', 'slope_ci95', 'p_star'):
                np.testing.assert_allclose(actual[field], expected[field], rtol=2e-7, atol=1e-12)
            count += 1
            print(f'Refit free exponent: {noise} {tag}', flush=True)
    assert count == 12
    # Preserve the original all-family seed allocation for the periodic bootstrap.
    from interval import models, boot_all
    m = {k: v for k, v in models().items() if k[2] == 'X'}
    boot = boot_all(m, s['interval']['P']['ps'], nboot=100, seed=0)
    full = boot[('full', 1, 'X')]
    compared = 0
    for fam, expected in s['interval'].items():
        np.testing.assert_allclose(full['eps'], expected['eps_full'], rtol=2e-7, atol=1e-12)
        for t, saved_ratio in expected['ratio'].items():
            row = boot[(fam, np.inf if t == 'inf' else int(t), 'X')]
            ratio = row['eps']/full['eps']
            with np.errstate(divide='ignore', invalid='ignore'):
                replicas = row['reps']/full['reps']
            ci = np.nanpercentile(replicas, [2.5, 97.5], axis=0)
            for actual, saved in ((ratio, saved_ratio), (ci, expected['ratio_ci95'][t]),
                                  (row['eps'], expected['eps'][t]),
                                  (row['share'], expected['measured_share'][t])):
                np.testing.assert_allclose(actual, saved, rtol=2e-7, atol=1e-12)
            driven = [p for i, p in enumerate(expected['ps'])
                      if min(row['share'][i], full['share'][i]) >= 0.5]
            assert driven == expected['ratio_driven'][t]
            assert row['shots'] == expected['data'][t]['shots']
            assert row['fails'] == expected['data'][t]['fails']
            compared += 1
            print(f'Refit schedule ratios and intervals: {fam} T={t}', flush=True)
        if 'escalation_scan' in expected:
            from strat import path_resources
            scan = expected['escalation_scan']
            resources = path_resources(m[(fam, np.inf, 'X')], scan['p'], shots=scan['shots'])
            np.testing.assert_allclose([resources[p]['full_outer_fraction'] for p in scan['p']],
                                       scan['per_round'], rtol=2e-7, atol=1e-12)
            print(f'Replayed 8000-shot escalation scan: {fam}', flush=True)
    return {'exponents_refitted': count, 'schedule_configurations_refitted': compared,
            'bootstrap_replicas': 100, 'escalation_scans_replayed': 2}


def windows(s):
    import window_failures
    # The research function writes relative to ROOT. Redirect it to build, keeping the archive intact.
    window_failures.ROOT = str(ROOT/'build/window-replay')
    with (ROOT/'build/window-replay.log').open('w') as log, redirect_stdout(log):
        actual = window_failures.main()
    expected = json.loads((ROOT/'results/minimal/constructed.json').read_text())
    assert actual == expected, 'Constructed failures changed'
    print('Replayed all six-fault ties and window placements', flush=True)
    return actual


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--refit', action='store_true')
    ap.add_argument('--windows', action='store_true')
    args = ap.parse_args()
    s = json.loads(SOURCE.read_text())
    (ROOT/'build').mkdir(exist_ok=True)
    report = check_bounds(s)
    if args.refit:
        report['refits'] = refit(s)
    if args.windows:
        report['constructed_replay'] = windows(s)
    draw(s)
    (ROOT/'build/evidence-check.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Checked 12 saved bounds, 276 inner paths and the 11-fault witness; exported evidence tables', flush=True)

if __name__ == '__main__':
    main()
