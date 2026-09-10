from __future__ import annotations

import hashlib
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def composite_source_fingerprint(import_manifest: Path, decisions_file: Path) -> str | None:
    imp = load_json(import_manifest)
    hatvp_fp = imp.get("source_fingerprint")
    decisions_fp = sha256_file(decisions_file)
    if not hatvp_fp or not decisions_fp:
        return None
    raw = f"hatvp={hatvp_fp}\ndecisions={decisions_fp}\n".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
