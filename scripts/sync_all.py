from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import os
import shutil
import subprocess
import tempfile

from src.config import PATHS
from src.provenance import composite_source_fingerprint, load_json
from scripts.import_tricoteuses import run as import_tricoteuses


def atomic_replace_dir(source: Path, destination: Path) -> None:
    backup = destination.with_name(destination.name + ".previous")
    if backup.exists():
        shutil.rmtree(backup)
    if destination.exists():
        destination.rename(backup)
    try:
        source.rename(destination)
    except Exception:
        if destination.exists():
            shutil.rmtree(destination)
        if backup.exists():
            backup.rename(destination)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)


def run_fetch_hatvp() -> None:
    subprocess.run([sys.executable, "scripts/fetch_tricoteuses.py"], cwd=ROOT, check=True)


def run_fetch_decisions() -> None:
    subprocess.run([sys.executable, "scripts/import_decisions_tricoteuses.py"], cwd=ROOT, check=True)


def run_build(temp_artifacts: Path) -> None:
    env = os.environ.copy()
    env["LOBBYSEARCH_ARTIFACTS_DIR"] = str(temp_artifacts.resolve())
    subprocess.run([sys.executable, "build_artifacts.py"], cwd=ROOT, check=True, env=env)


def _is_ci() -> bool:
    return os.getenv("CI", "").lower() in {"1", "true", "yes"} or bool(os.getenv("GITHUB_ACTIONS"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Synchronisation distante/CI : Tricoteuses HATVP + décisions Canutes -> import -> "
            "rebuild atomique Parquet/BM25/FAISS. Cette commande est volontairement bloquée en local."
        )
    )
    parser.add_argument("--no-fetch-hatvp", action="store_true")
    parser.add_argument("--no-fetch-decisions", action="store_true")
    parser.add_argument("--force-import", action="store_true")
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument(
        "--allow-local-build",
        action="store_true",
        help="Déverrouillage explicite pour maintenance. À éviter pour les tests locaux ordinaires.",
    )
    args = parser.parse_args()

    if not _is_ci() and not args.allow_local_build:
        raise SystemExit(
            "⛔ sync_all.py est réservé à la CI/Forgejo pour éviter un rebuild FAISS local.\n"
            "Pour tester en local sans FAISS : streamlit run app_serving_test.py\n"
            "Pour un besoin de maintenance exceptionnel : ajoutez --allow-local-build."
        )

    if not args.no_fetch_hatvp:
        print("[1/5] Récupération/nettoyage HATVP via tricoteuses-hatvp", flush=True)
        run_fetch_hatvp()
    else:
        print("[1/5] Données HATVP Tricoteuses déjà présentes", flush=True)

    if not args.no_fetch_decisions:
        print("[2/5] Import décisions publiques Canutes/Légifrance (amendements exclus)", flush=True)
        run_fetch_decisions()
    else:
        print("[2/5] Décisions publiques déjà présentes", flush=True)

    print("[3/5] Import relationnel léger HATVP -> Parquet", flush=True)
    import_result = import_tricoteuses(
        PATHS.tricoteuses_data,
        PATHS.data_imported,
        force=args.force_import,
    )

    import_manifest_path = PATHS.data_imported / "import_manifest.json"
    source_fp = composite_source_fingerprint(import_manifest_path, PATHS.decisions_raw)
    if not source_fp:
        raise RuntimeError("Impossible de calculer le fingerprint combiné HATVP + décisions publiques.")
    print(f"      source_fingerprint={source_fp}", flush=True)

    current_art_manifest = load_json(PATHS.artifacts / "manifest.json")
    artifacts_current = (
        current_art_manifest.get("source_fingerprint") == source_fp
        and (PATHS.artifacts / "df_activites_min.parquet").exists()
        and (PATHS.artifacts / "df_lois_min.parquet").exists()
        and (PATHS.artifacts / "bm25_activites.joblib").exists()
        and (PATHS.artifacts / "faiss_activites.index").exists()
        and (PATHS.artifacts / "faiss_lois.index").exists()
    )

    if artifacts_current and not args.force_build:
        print("[4/5] Artefacts déjà alignés; aucun rebuild.", flush=True)
        print("[5/5] Synchronisation terminée.", flush=True)
        return

    print("[4/5] Reconstruction complète Parquet/BM25/FAISS dans un dossier temporaire", flush=True)
    PATHS.artifacts.parent.mkdir(parents=True, exist_ok=True)
    temp_artifacts = Path(tempfile.mkdtemp(prefix="lobbysearch-artifacts-", dir=PATHS.artifacts.parent))
    try:
        run_build(temp_artifacts)
        built_manifest = load_json(temp_artifacts / "manifest.json")
        if built_manifest.get("source_fingerprint") != source_fp:
            raise RuntimeError(
                "Le manifest des artefacts ne correspond pas aux sources HATVP+décisions. "
                "Les anciens artefacts restent en place."
            )
        required = [
            "df_activites_min.parquet",
            "df_lois_min.parquet",
            "df_beneficiaires.parquet",
            "df_observations.parquet",
            "bm25_activites.joblib",
            "bm25_lois.joblib",
            "faiss_activites.index",
            "faiss_lois.index",
            "id_map_activites.parquet",
            "id_map_lois.parquet",
            "manifest.json",
        ]
        missing = [name for name in required if not (temp_artifacts / name).exists()]
        if missing:
            raise RuntimeError(f"Artefacts incomplets: {missing}")

        print("[5/5] Publication atomique du nouveau snapshot", flush=True)
        atomic_replace_dir(temp_artifacts, PATHS.artifacts)
        print("✅ Données HATVP, décisions et artefacts sont alignés.", flush=True)
    finally:
        if temp_artifacts.exists():
            shutil.rmtree(temp_artifacts, ignore_errors=True)


if __name__ == "__main__":
    main()
