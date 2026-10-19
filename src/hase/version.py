"""Code version stamp recorded in every result file."""
import hashlib
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TRACKED = ("src", "scripts", "analysis", "third_party")      # version-controlled code


def commit():
    here = os.path.dirname(__file__)
    stamp = os.path.join(here, "_commit.txt")
    if os.path.exists(stamp):
        return open(stamp).read().strip()
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=here,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _sha1(path):
    with open(path, "rb") as fh:
        return hashlib.sha1(fh.read()).hexdigest()


def provenance(extra=()):
    """Commit plus the SHA-1 of every source file of this repository that the running
    process has loaded (and of the vendored native sources and `extra` files), so that a
    result can be matched exactly against the code of any later commit."""
    files = set(extra)
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if not f or not f.endswith(".py"):
            continue
        f = os.path.abspath(f)
        rel = os.path.relpath(f, ROOT)
        if f.startswith(ROOT + os.sep) and rel.split(os.sep)[0] in TRACKED:
            files.add(rel)
    main = getattr(sys.modules.get("__main__"), "__file__", None)
    if main and os.path.abspath(main).startswith(ROOT + os.sep):
        files.add(os.path.relpath(os.path.abspath(main), ROOT))
    nat = os.path.join(ROOT, "third_party", "frontier")
    if os.path.isdir(nat):
        for dp, _, fns in os.walk(nat):
            for fn in fns:
                files.add(os.path.relpath(os.path.join(dp, fn), ROOT))
    files.add("scripts/build_native.py")
    hashes = {}
    for rel in sorted(files):
        p = os.path.join(ROOT, rel)
        if os.path.exists(p):
            hashes[rel] = _sha1(p)
    ext = {}
    try:
        from importlib import metadata
        for dist in ("frontier", "stim", "numpy", "scipy"):
            try:
                ext[dist] = metadata.version(dist)
            except metadata.PackageNotFoundError:
                pass
        # the Frontier sources: pinned git commit (requirements.txt), read from the install
        import frontier
        src = os.path.dirname(os.path.dirname(os.path.abspath(frontier.__file__)))
        try:
            ext["frontier_commit"] = subprocess.check_output(
                ["git", "-C", src, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            import json as _json
            du = metadata.distribution("frontier").read_text("direct_url.json") or "{}"
            ext["frontier_commit"] = _json.loads(du).get("vcs_info", {}).get("commit_id", "unknown")
    except Exception:
        pass
    return dict(commit=commit(), files=hashes, packages=ext)
