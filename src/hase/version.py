"""Package versions recorded in every result file."""


def versions():
    out = {}
    try:
        from importlib import metadata
        for dist in ("stim", "numpy", "scipy"):
            try:
                out[dist] = metadata.version(dist)
            except metadata.PackageNotFoundError:
                pass
    except Exception:
        pass
    return out
