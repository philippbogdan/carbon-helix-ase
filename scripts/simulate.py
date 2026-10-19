"""Run a local Carbon-Helix memory sample with live adaptive decisions."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rule',default='l2=on_event,l3=ambiguous,rot=8')
    p.add_argument('--noise',choices=['uniform','helios'],default='helios')
    p.add_argument('--p',type=float,default=0.00079)
    p.add_argument('--rounds',type=int,default=4)
    p.add_argument('--shots',type=int,default=128)
    p.add_argument('--basis',choices=['X','Z'],default='X')
    p.add_argument('--seed',type=int,default=19)
    p.add_argument('--out',type=Path,default=ROOT/'build/r8.json')
    a=p.parse_args()
    if a.shots<1 or a.rounds<1 or not 0<a.p<1: p.error('Use positive shots and rounds and 0 < p < 1.')
    if a.out.exists(): p.error(f'{a.out} already exists; choose a new output path.')
    env=os.environ.copy()
    env['PYTHONPATH']=os.pathsep.join([str(ROOT/'src'),str(ROOT/'third_party/frontier')])
    cmd=[sys.executable,str(ROOT/'scripts/job_protocol.py'),'--proto','carbonhelix']
    for key in ('rule','noise','p','rounds','shots','basis','seed','out'):
        cmd.extend(['--'+key,str(getattr(a,key))])
    subprocess.run(cmd,env=env,check=True)

if __name__=='__main__': main()
