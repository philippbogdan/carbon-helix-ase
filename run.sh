#!/bin/sh
set -eu
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python3}
export PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export MPLCONFIGDIR="$PWD/build/matplotlib"
unset PYTHONPATH
if [ ! -x .venv/bin/python ]; then "$PYTHON" -m venv .venv; fi
.venv/bin/python -m pip install --quiet --only-binary=:all: -r requirements.txt
PY=.venv/bin/python
case "${1:-headline}" in
  headline) $PY scripts/headline.py ;;
  baseline) $PY scripts/c4_baseline.py ;;
  check) $PY scripts/check.py ;;
  evidence) $PY scripts/figure_data.py --check; $PY scripts/evidence.py ;;
  figures) $PY figures/make_figures.py ;;
  verify) $PY scripts/evidence.py --refit --windows; $PY analysis/aggregate.py --check ;;
  *) echo 'Usage: ./run.sh [headline|baseline|check|evidence|figures|verify]' >&2; exit 2 ;;
esac
