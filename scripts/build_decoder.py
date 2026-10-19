"""Compile the bundled, unchanged Frontier C++ source."""
import os
from pathlib import Path
import shlex
import subprocess
import sys
import sysconfig

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'third_party/frontier/_frontier_native.cpp'
out = ROOT / 'build/native'
out.mkdir(parents=True, exist_ok=True)
target = out / ('_frontier_native' + sysconfig.get_config_var('EXT_SUFFIX'))
flags = ['-O3', '-std=c++17', '-shared', '-fPIC', '-DNDEBUG',
         '-I' + sysconfig.get_paths()['include']]
if sys.platform == 'darwin':
    flags += ['-undefined', 'dynamic_lookup']
subprocess.run(shlex.split(os.environ.get('CXX', 'c++')) + flags +
               [str(source), '-o', str(target)], check=True)
print('Built Frontier decoder')
