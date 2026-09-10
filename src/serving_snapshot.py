"""Chargement strict d'un snapshot BM25 existant, sans reconstruction."""
from pathlib import Path

import joblib
import pandas as pd

from .activity_analytics import ensure_activite_id


REQUIRED_FILES = (
    "df_activites_min.parquet",
    "df_lois_min.parquet",
    "bm25_activites.joblib",
    "bm25_lois.joblib",
    "df_observations.parquet",
    "df_beneficiaires.parquet",
    "df_affiliations.parquet",
    "df_informations_generales.parquet",
)


def check_snapshot_files(directory: Path) -> None:
    missing = [name for name in REQUIRED_FILES if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Snapshot BM25 incomplet : " + ", ".join(missing)
            + ". Publiez les artefacts préconstruits dans le dossier configuré. "
            "Aucune reconstruction n'est autorisée dans ce mode."
        )


def check_alignment(frame: pd.DataFrame, bundle, name: str) -> None:
    # Les recherches utilisent les identifiants enregistrés dans le bundle.
    normalize = lambda value: str(value).strip().removesuffix(".0")
    table_ids = {normalize(value) for value in frame.index}
    index_ids = {normalize(value) for value in bundle.doc_ids}
    if (
        len(frame) != len(bundle.doc_ids)
        or len(frame) != bundle.bm25.corpus_size
        or len(table_ids) != len(frame)
        or table_ids != index_ids
    ):
        raise ValueError(
            f"Snapshot BM25 désaligné ({name}) : {len(frame)} lignes, "
            f"{len(bundle.doc_ids)} documents indexés. Publiez un snapshot cohérent."
        )


def load_snapshot(directory: Path) -> dict:
    check_snapshot_files(directory)
    df_acts = ensure_activite_id(pd.read_parquet(directory / REQUIRED_FILES[0]))
    df_acts.index = df_acts["activite_id"].astype(str)
    df_lois = pd.read_parquet(directory / REQUIRED_FILES[1])
    activities_index = joblib.load(directory / "bm25_activites.joblib")
    laws_index = joblib.load(directory / "bm25_lois.joblib")
    check_alignment(df_acts, activities_index, "activités")
    check_alignment(df_lois, laws_index, "décisions publiques")
    return {
        "df_acts": df_acts,
        "df_lois": df_lois,
        "bm25_bundle": activities_index,
        "bm25_lois_bundle": laws_index,
        **{
            key: pd.read_parquet(directory / f"{key}.parquet")
            for key in (
                "df_observations", "df_beneficiaires", "df_affiliations",
                "df_informations_generales",
            )
        },
    }
