"""The 128-limb build of the patched Frontier engine (scripts/build_native_wide.py), compiled
on first use into a machine-local directory, and `use_wide()`, which makes this process
decode with it: hase.native.get_native returns the wide module and the Frontier model
builder accepts up to 128 detector limbs. Only scripts/job_strat_wide.py calls it."""
import importlib
import os
import socket
import sys

_mod = None


def get_native_wide():
    global _mod
    if _mod is not None:
        return _mod
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    local = os.path.join("/tmp", f"hase_native_wide_{socket.gethostname()}_{os.getuid()}")
    sys.path.insert(0, local)
    try:
        _mod = importlib.import_module("_hase_frontier_native128")
        return _mod
    except ImportError:
        pass
    sys.path.insert(0, os.path.join(root, "scripts"))
    try:
        import build_native_wide
    finally:
        sys.path.pop(0)
    build_native_wide.build(local)
    # the directory did not exist at the first attempt: drop the cached (missing) finder
    sys.path_importer_cache.pop(local, None)
    importlib.invalidate_caches()
    _mod = importlib.import_module("_hase_frontier_native128")
    return _mod


def use_wide():
    from . import native
    from .decoding import _frontier_modules
    native.get_native = get_native_wide
    native._mod = None
    fp, fdec = _frontier_modules()
    fdec._NATIVE_MAX_DETECTOR_LIMBS = 128
