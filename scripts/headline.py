"""Extract the six headline rows from the saved numerical summary."""
import json
import math
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from hase.protocols import make_protocol
from hase.timing import path_time_ms
SOURCE = ROOT/'results/summary.json'
SERIES = [
 ('C4 adaptive','C4-Helix|mode=ase|r4'),
 ('Carbon full','Carbon-Helix|full|r4'),
 ('Carbon R8','Carbon-Helix|l2=on_event,l3=ambiguous,rot=8|r4')]

def rate(e, rounds, prep_ms, round_ms):
    return -rounds*math.log1p(-2*e)/(2*(prep_ms+rounds*round_ms)/1000)

def rows():
    s=json.loads(SOURCE.read_text());out=[]
    for p in (0.00079,0.0003):
        for label,key in SERIES:
            r=s['value'][f'helios|{p}']['rows'][key]
            name='c4helix' if r['code']=='C4-Helix' else 'carbonhelix'
            pr=make_protocol(name,p,'X',4,r['rule'],'helios')
            prep=path_time_ms(pr,[(0,pt) for pt,_ in pr.plan(0)])
            record={k:r[k] for k in ('p','rounds','eps_LQ','measured_share','status','twoq_per_round','time_ms_per_round','peak_qubits','shots','fails')}
            record.update(label=label,key=key,prep_ms=prep,
                          ci_low=r['ci95_LQ'][0],ci_high=r['ci95_LQ'][1])
            for field,e in [('rate',r['eps_LQ']),('rate_low',r['ci95_LQ'][0]),('rate_high',r['ci95_LQ'][1])]:
                record[field]=rate(e,4,prep,r['time_ms_per_round'])
            out.append(record)
    return out
