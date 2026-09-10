"""Contrôle de présence au déploiement ; aucun chargement ou recalcul d'index."""
from src.config import PATHS
from src.serving_snapshot import check_snapshot_files


if __name__ == "__main__":
    check_snapshot_files(PATHS.artifacts)
    print("Snapshot BM25 présent : aucune reconstruction effectuée.")
