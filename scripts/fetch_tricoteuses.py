from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PATHS


VERSION = "docker-node24-v2"

TRICOTEUSES_GIT_URL = (
    "https://git.tricoteuses.fr/logiciels/tricoteuses-hatvp.git"
)

NODE_IMAGE = "node:24-alpine"


def docker_available() -> bool:
    return shutil.which("docker") is not None


def docker_mount(host_dir: Path, container_dir: str) -> str:
    return (
        f"type=bind,"
        f"source={host_dir.resolve()},"
        f"target={container_dir}"
    )


def run_docker(data_dir: Path, strict: bool = False) -> None:
    """
    Exécute tricoteuses-hatvp entièrement dans Docker.

    Aucun Node.js/npm n'est nécessaire sur Windows.

    Le conteneur :
      1. part de node:24-alpine ;
      2. installe git ;
      3. clone tricoteuses-hatvp ;
      4. installe ses dépendances npm ;
      5. télécharge et nettoie le répertoire HATVP ;
      6. écrit le résultat dans data/tricoteuses.
    """

    if not docker_available():
        raise SystemExit(
            "Docker n'est pas disponible dans le PATH. "
            "Démarrez Docker Desktop puis relancez."
        )

    data_dir = data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    strict_arg = " --strict" if strict else ""

    shell_command = (
        "set -e; "
        "apk add --no-cache git; "
        "cd /tmp; "
        f"git clone --depth 1 {TRICOTEUSES_GIT_URL} tricoteuses-hatvp; "
        "cd /tmp/tricoteuses-hatvp; "
        "npm install --include=dev --ignore-scripts; "
        "npm run data:download -- /out --datasets repertoire"
        f"{strict_arg}"
    )

    cmd = [
        "docker",
        "run",
        "--rm",
        "--mount",
        docker_mount(data_dir, "/out"),
        NODE_IMAGE,
        "sh",
        "-lc",
        shell_command,
    ]

    print(
        f"fetch_tricoteuses version: {VERSION}",
        flush=True,
    )

    print(
        "Docker : téléchargement et nettoyage HATVP "
        "avec tricoteuses-hatvp",
        flush=True,
    )

    print(
        f"Image Docker : {NODE_IMAGE}",
        flush=True,
    )

    print(
        f"Destination : {data_dir}",
        flush=True,
    )

    subprocess.run(
        cmd,
        cwd=ROOT,
        check=True,
    )

    expected = (
        data_dir
        / "repertoire_representants_interets"
    )

    if not expected.exists():
        raise RuntimeError(
            "Le dossier attendu n'a pas été produit : "
            f"{expected}"
        )

    json_count = sum(
        1
        for _ in expected.glob("*.json")
    )

    if json_count == 0:
        raise RuntimeError(
            "Aucun fichier JSON représentant trouvé dans "
            f"{expected}"
        )

    print(
        f"Tricoteuses : {json_count:,} représentants "
        "JSON disponibles.",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Télécharge et nettoie le répertoire HATVP "
            "avec tricoteuses-hatvp exécuté dans Docker."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PATHS.tricoteuses_data,
    )

    parser.add_argument(
        "--strict",
        action="store_true",
    )

    args = parser.parse_args()

    run_docker(
        args.data_dir,
        strict=args.strict,
    )


if __name__ == "__main__":
    main()