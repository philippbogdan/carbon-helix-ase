"""The 192-limb build of the patched Frontier engine (scripts/build_native_xwide.py), compiled
on first use into a machine-local directory, and `use_xwide()`, which makes this process
decode with it (as src/hase/native_wide.py does for 128 limbs). Only
scripts/job_strat_xwide.py calls it."""
import importlib
import os
import socket
import sys

LIMBS = 192
_mod = None


def get_native_xwide():
    global _mod
    if _mod is not None:
        return _mod
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    local = os.path.join("/tmp", f"hase_native_xwide{LIMBS}_{socket.gethostname()}_{os.getuid()}")
    sys.path.insert(0, local)
    modname = f"_hase_frontier_native{LIMBS}"
    try:
        _mod = importlib.import_module(modname)
        return _mod
    except ImportError:
        pass
    sys.path.insert(0, os.path.join(root, "scripts"))
    try:
        import build_native_xwide
    finally:
        sys.path.pop(0)
    build_native_xwide.build(local, LIMBS)
    sys.path_importer_cache.pop(local, None)
    importlib.invalidate_caches()
    _mod = importlib.import_module(modname)
    return _mod


def use_xwide():
    from . import native
    from .decoding import _frontier_modules
    native.get_native = get_native_xwide
    native._mod = None
    fp, fdec = _frontier_modules()
    fdec._NATIVE_MAX_DETECTOR_LIMBS = LIMBS
