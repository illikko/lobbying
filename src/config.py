from dataclasses import dataclass, field
from pathlib import Path
import os

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")


@dataclass(frozen=True)
class Paths:
    repo_root: Path = REPO_ROOT
    data_raw: Path = REPO_ROOT / "data" / "raw"
    data_imported: Path = REPO_ROOT / "data" / "imported"
    tricoteuses_data: Path = field(
        default_factory=lambda: Path(
            os.getenv("TRICOTEUSES_DATA_DIR", str(REPO_ROOT / "data" / "tricoteuses"))
        ).resolve()
    )
    decisions_raw: Path = field(
        default_factory=lambda: Path(
            os.getenv("TRICOTEUSES_DECISIONS_FILE", str(REPO_ROOT / "data" / "decisions" / "decisions.parquet"))
        ).resolve()
    )
    artifacts: Path = field(
        default_factory=lambda: Path(
            os.getenv("LOBBYSEARCH_ARTIFACTS_DIR", str(REPO_ROOT / "artifacts"))
        ).resolve()
    )


PATHS = Paths()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TRICOTEUSES_IMAGE = os.getenv(
    "TRICOTEUSES_IMAGE",
    "git.tricoteuses.fr/logiciels/tricoteuses-hatvp:latest",
)

# API Canutes/Légifrance (PostgREST), sur le même modèle que LegiChat.
# Le service exact des textes peut être surchargé dans .env sans modifier le code.
TRICOTEUSES_DECISIONS_API_URL = (
    os.getenv("TRICOTEUSES_DECISIONS_API_URL")
    or "https://db.code4code.eu/canutes/texte_version"
)
TRICOTEUSES_DECISIONS_API_TOKEN = os.getenv("TRICOTEUSES_DECISIONS_API_TOKEN", "")
TRICOTEUSES_DECISIONS_FROM_YEAR = int(os.getenv("TRICOTEUSES_DECISIONS_FROM_YEAR", "2018"))
TRICOTEUSES_DECISIONS_PAGE_SIZE = int(os.getenv("TRICOTEUSES_DECISIONS_PAGE_SIZE", "1000"))



@dataclass(frozen=True)
class ArtifactNames:
    manifest: str = "manifest.json"
    df_activites_min: str = "df_activites_min.parquet"
    # Nom historique conservé pour compatibilité interne avec l'UI.
    # Le contenu est désormais l'ensemble des décisions publiques retenues.
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
