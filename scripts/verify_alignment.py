from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PATHS
from src.provenance import composite_source_fingerprint, load_json

imp_path = PATHS.data_imported / "import_manifest.json"
art_path = PATHS.artifacts / "manifest.json"
imp = load_json(imp_path)
art = load_json(art_path)

if not imp:
    raise SystemExit("❌ import_manifest.json absent")
if not PATHS.decisions_raw.exists():
    raise SystemExit("❌ fichier décisions publiques absent")
if not art:
    raise SystemExit("❌ artifacts/manifest.json absent")

expected = composite_source_fingerprint(imp_path, PATHS.decisions_raw)
if expected != art.get("source_fingerprint"):
    raise SystemExit(
        "❌ Désalignement HATVP/décisions/artefacts. Le rebuild doit être exécuté par la CI Forgejo."
    )
print("✅ Alignement HATVP + décisions + artefacts OK")
print("source_fingerprint:", expected)
print("imported_at:", imp.get("imported_at"))
print("built_at:", art.get("built_at"))
