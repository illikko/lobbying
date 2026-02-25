from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    repo_root: Path = Path(__file__).resolve().parents[1]
    data_raw: Path = repo_root / "data" / "raw"
    artifacts: Path = repo_root / "artifacts"

PATHS = Paths()

@dataclass(frozen=True)
class ArtifactNames:
    manifest: str = "manifest.json"
    df_activites_min: str = "df_activites_min.parquet"
    df_lois_min: str = "df_lois_min.parquet"
    docs_activites: str = "docs_activites.parquet"
    bm25_activites: str = "bm25_activites.joblib"
    faiss_activites: str = "faiss_activites.index"
    id_map_activites: str = "id_map_activites.parquet"
    docs_lois: str = "docs_lois.parquet"
    bm25_lois: str = "bm25_lois.joblib"
    faiss_lois: str = "faiss_lois.index"
    id_map_lois: str = "id_map_lois.parquet"

ART = ArtifactNames()
