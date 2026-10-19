#!/bin/sh
set -eu
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python3}
export PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=1
export MPLCONFIGDIR="$PWD/build/matplotlib"
unset PYTHONPATH
if [ ! -x .venv/bin/python ]; then "$PYTHON" -m venv .venv; fi
.venv/bin/python -m pip install --quiet --only-binary=:all: -r requirements.txt
case "${1:-check}" in
  figures) .venv/bin/python scripts/figure_data.py --check; .venv/bin/python figures/make_figures.py --force; .venv/bin/python scripts/evidence.py ;;
  verify) .venv/bin/python scripts/evidence.py --refit --windows ;;
  reproduce) .venv/bin/python scripts/reproduce.py; .venv/bin/python scripts/figure_data.py --check ;;
  check) .venv/bin/python scripts/check.py ;;
  *) echo 'Usage: ./run.sh [check|figures|reproduce|verify]' >&2; exit 2 ;;
esac
