"""Load the patched Frontier native engine (see third_party/frontier/README.md).

Try the package extension, then build/adaptive_native/. If neither can be imported,
compile the extension there for the current Python interpreter and load it."""
import importlib
import os
import socket
import sys

_mod = None


def get_native():
    global _mod
    if _mod is not None:
        return _mod
    try:
        from hase import _hase_frontier_native as m
        _mod = m
        return m
    except ImportError:
        pass
    here = os.path.dirname(__file__)
    root = os.path.abspath(os.path.join(here, "..", ".."))
    local = os.path.join(root, "build", "adaptive_native")
    sys.path.insert(0, os.path.join(root, "scripts"))
    try:
        import build_native
    finally:
        sys.path.pop(0)
    ext_ok = False
    try:
        sys.path.insert(0, local)
        _mod = importlib.import_module("_hase_frontier_native")
        ext_ok = True
    except ImportError:
        sys.path.pop(0)
    if not ext_ok:
        build_native.build(local)
        importlib.invalidate_caches()
        sys.path.insert(0, local)
        _mod = importlib.import_module("_hase_frontier_native")
    return _mod
