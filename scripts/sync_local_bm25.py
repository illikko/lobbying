from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PATHS
from scripts.import_tricoteuses import run as import_tricoteuses


def run_step(label: str, command: list[str]) -> None:
    print(f"\n{label}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline local unique LobbySearch : Docker Tricoteuses HATVP + API décisions "
            "Canutes/Légifrance -> Parquet -> BM25. Aucun embedding ni FAISS."
        )
    )
    parser.add_argument(
        "--no-fetch-hatvp",
        action="store_true",
        help="Réutilise data/tricoteuses déjà présent (diagnostic seulement).",
    )
    parser.add_argument(
        "--no-fetch-decisions",
        action="store_true",
        help="Réutilise data/decisions/decisions.parquet déjà présent (diagnostic seulement).",
    )
    parser.add_argument("--force-import", action="store_true")
    parser.add_argument("--strict-hatvp", action="store_true")
    args = parser.parse_args()

    if not args.no_fetch_hatvp:
        fetch_cmd = [sys.executable, "scripts/fetch_tricoteuses.py"]
        if args.strict_hatvp:
            fetch_cmd.append("--strict")
        run_step(
            "[1/4] HATVP : Docker tricoteuses-hatvp -> JSON nettoyés",
            fetch_cmd,
        )
    else:
        print("\n[1/4] HATVP : réutilisation de data/tricoteuses", flush=True)

    if not args.no_fetch_decisions:
        run_step(
            "[2/4] Décisions publiques : API Canutes/Légifrance -> Parquet",
            [sys.executable, "scripts/import_decisions_tricoteuses.py"],
        )
    else:
        print("\n[2/4] Décisions : réutilisation de data/decisions/decisions.parquet", flush=True)

    print("\n[3/4] LobbySearch : JSON Tricoteuses -> tables Parquet", flush=True)
    result = import_tricoteuses(
        PATHS.tricoteuses_data,
        PATHS.data_imported,
        force=args.force_import,
    )
    print(
        f"      changed={result.get('changed')} "
        f"source_fingerprint={result.get('source_fingerprint')}",
        flush=True,
    )

    print("\n[4/4] LobbySearch : construction Parquet + BM25 (sans embeddings, sans FAISS)", flush=True)
    env = os.environ.copy()
    subprocess.run(
        [sys.executable, "build_artifacts.py", "--no-faiss"],
        cwd=ROOT,
        check=True,
        env=env,
    )

    print(
        "\n✅ Synchronisation locale BM25 terminée.\n"
        "Les données fraîches se trouvent dans data/ et les artefacts dans artifacts/.\n"
        "Lancez ensuite :\n"
        "  .\\.venv\\Scripts\\python.exe -m streamlit run app_serving_test.py",
        flush=True,
    )


if __name__ == "__main__":
    main()
