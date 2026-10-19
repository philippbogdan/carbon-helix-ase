"""Render the documents from templates, filling every number from results/summary.json.

Placeholders: {{slash/separated/path|format}} where the path indexes summary.json (dict
keys, with '|' in keys written as '¦') and format is a Python format spec (default '.3g'). A missing
value renders as 'MISSING(path)' and is reported, so no stale number can survive.
"""
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PAT = re.compile(r"\{\{([^}|]+)(?:\|([^}]*))?\}\}")


def lookup(summary, path):
    cur = summary
    for part in path.strip().split("/"):
        part = part.replace("¦", "|")
        if cur is None:
            return None             # an undetermined interval or record: rendered as n/a
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            raise KeyError(path)
    return cur


def fmt(v, spec):
    if v is None:
        # the analysis leaves a value undetermined on purpose (too few failures, unbounded)
        return "n/a"
    if spec is None or spec == "":
        spec = ".3g"
    if spec == "pct":
        return f"{100 * v:.1f}%"
    if spec == "sci":
        return f"{v:.2e}"
    if spec == "int":
        return f"{int(round(v)):d}"
    if spec == "str":
        return str(v)
    return format(v, spec)


def render(template_text, summary, missing):
    def rep(m):
        path, spec = m.group(1), m.group(2)
        try:
            return fmt(lookup(summary, path), spec)
        except Exception:
            missing.append(path)
            return f"MISSING({path})"
    return PAT.sub(rep, template_text)


def main():
    summary = json.load(open(os.path.join(ROOT, "results/summary.json")))
    tdir = os.path.join(ROOT, "docs", "templates")
    missing = []
    for name in sorted(os.listdir(tdir)):
        if not name.endswith(".tmpl"):
            continue
        text = open(os.path.join(tdir, name)).read()
        out = render(text, summary, missing)
        target = os.path.join(ROOT, name[:-5])
        open(target, "w").write(out)
        print("rendered", target)
    if missing:
        print(f"{len(missing)} missing values:", sorted(set(missing))[:40])
        sys.exit(1 if "--strict" in sys.argv else 0)


if __name__ == "__main__":
    main()
