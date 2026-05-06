from __future__ import annotations
from dataclasses import asdict, dataclass
from datetime import datetime
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

from src.config import PATHS, ART
from src.prep import prepare_from_raw, minify_activites, minify_lois, build_docs_activites, build_docs_lois
from src.bm25_index import build_bm25
from src.embed_index import build_faiss_ivfpq
from src.io_artifacts import save_parquet, save_joblib, save_faiss, write_manifest


@dataclass
class BuildConfig:
    # sentence-transformers model (offline)
    st_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # 384 dims
    # FAISS
    nlist: int = 2048
    m: int = 32
    nbits: int = 8
    normalize: bool = True
    # misc
    created_by: str = "build_artifacts.py"


def _raw(p: str) -> str:
    return str((PATHS.data_raw / p).resolve())


def _read_raw_xlsx(filename: str) -> pd.DataFrame:
    """Lecture robuste des tables HATVP: les IDs doivent rester des chaînes."""
    path = PATHS.data_raw / filename
    if not path.exists():
        raise FileNotFoundError(f"Fichier source manquant: {path}")
    return pd.read_excel(path, dtype=str)


def _norm_id_series(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .mask(lambda x: x.str.lower().isin(["", "nan", "none", "<na>"]))
    )


def _join_unique(values: pd.Series, max_items: int = 1000) -> str:
    out: list[str] = []
    seen: set[str] = set()
    for value in values.dropna().astype(str):
        value = value.strip()
        if not value or value.lower() in {"nan", "none", "<na>"}:
            continue
        if value not in seen:
            seen.add(value)
            out.append(value)
        if len(out) >= max_items:
            break
    return "; ".join(out)


def _explode_ids(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    out[col] = _norm_id_series(out[col])
    out[col] = out[col].str.split(r"\s*[;,|]\s*", regex=True)
    out = out.explode(col)
    out[col] = _norm_id_series(out[col])
    return out.dropna(subset=[col])


def _add_categorie_to_activites(df_acts_min: pd.DataFrame, df_infos: pd.DataFrame) -> pd.DataFrame:
    """Ajoute label_categorie_organisation dans l'artefact activités.

    df_acts_min garde l'index activite_id. La clé fiable est representants_id quand elle existe.
    """
    out = df_acts_min.copy()
    if "label_categorie_organisation" in out.columns:
        return out
    if "representants_id" not in out.columns or "representants_id" not in df_infos.columns:
        return out
    if "label_categorie_organisation" not in df_infos.columns:
        return out

    left = out.reset_index().rename(columns={out.index.name or "index": "activite_id"})
    left = _explode_ids(left, "representants_id")

    right = df_infos[["representants_id", "label_categorie_organisation"]].drop_duplicates().copy()
    right = _explode_ids(right, "representants_id")
    right["label_categorie_organisation"] = right["label_categorie_organisation"].astype("string").str.strip()
    right = right.dropna(subset=["representants_id", "label_categorie_organisation"])

    mapped = left[["activite_id", "representants_id"]].merge(right, on="representants_id", how="left")
    cat_by_act = mapped.groupby("activite_id")["label_categorie_organisation"].agg(_join_unique)
    out["label_categorie_organisation"] = out.index.astype(str).map(cat_by_act).astype("string")
    return out


def _normalise_auxiliary_tables(
    df_affiliations: pd.DataFrame,
    df_beneficiaires: pd.DataFrame,
    df_observations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Garde uniquement les colonnes utiles aux jointures côté app."""
    aff = df_affiliations[["representants_id", "denomination_affiliation"]].copy()
    aff = _explode_ids(aff, "representants_id")
    aff["denomination_affiliation"] = aff["denomination_affiliation"].astype("string").str.strip()
    aff = aff.dropna(subset=["representants_id", "denomination_affiliation"]).drop_duplicates()

    benef = df_beneficiaires[["action_representation_interet_id", "beneficiaire_action_menee"]].copy()
    benef["action_representation_interet_id"] = _norm_id_series(benef["action_representation_interet_id"])
    benef["beneficiaire_action_menee"] = benef["beneficiaire_action_menee"].astype("string").str.strip()
    benef = benef.dropna(subset=["action_representation_interet_id", "beneficiaire_action_menee"]).drop_duplicates()

    obs = df_observations[["activite_id", "action_representation_interet_id"]].copy()
    obs["activite_id"] = _norm_id_series(obs["activite_id"])
    obs["action_representation_interet_id"] = _norm_id_series(obs["action_representation_interet_id"])
    obs = obs.dropna(subset=["activite_id", "action_representation_interet_id"]).drop_duplicates()

    return aff, benef, obs

def _build_beneficiaires_activites_globales(
    df_acts_min: pd.DataFrame,
    df_observations: pd.DataFrame,
    df_beneficiaires: pd.DataFrame,
    df_exercices: pd.DataFrame,
    df_objets: pd.DataFrame,
) -> pd.DataFrame:
    # --- NORMALISATION
    obs = df_observations.copy()
    benef = df_beneficiaires.copy()
    ex = df_exercices.copy()
    obj = df_objets.copy()

    obs["activite_id"] = _norm_id_series(obs["activite_id"])
    obs["action_representation_interet_id"] = _norm_id_series(obs["action_representation_interet_id"])

    benef["action_representation_interet_id"] = _norm_id_series(benef["action_representation_interet_id"])
    benef["beneficiaire_action_menee"] = benef["beneficiaire_action_menee"].astype("string").str.strip()

    ex["exercices_id"] = _norm_id_series(ex["exercices_id"])
    obj["exercices_id"] = _norm_id_series(obj["exercices_id"])
    obj["activite_id"] = _norm_id_series(obj["activite_id"])

    # --- BUDGET EXERCICE
    if {"montant_depense_inf", "montant_depense_sup"}.issubset(ex.columns):
        ex["_budget"] = (
            pd.to_numeric(ex["montant_depense_inf"], errors="coerce").fillna(0)
            + pd.to_numeric(ex["montant_depense_sup"], errors="coerce").fillna(0)
        ) / 2
    else:
        ex["_budget"] = pd.to_numeric(ex.get("montant_depense", 0), errors="coerce").fillna(0)

    # --- NB ACTIVITES REEL PAR EXERCICE
    nb_act = (
        obj.groupby("exercices_id")["activite_id"]
        .nunique()
        .rename("nb_activites_exercice")
        .reset_index()
    )

    ex = ex.merge(nb_act, on="exercices_id", how="left")

    ex["budget_activite"] = ex["_budget"] / ex["nb_activites_exercice"].replace(0, np.nan)
    ex["budget_activite"] = ex["budget_activite"].replace([np.inf, -np.inf], np.nan).fillna(0)

    # --- BUDGET PAR ACTIVITE
    budget_by_activity = (
        obj.merge(ex[["exercices_id", "budget_activite"]], on="exercices_id", how="left")
        .groupby("activite_id")["budget_activite"]
        .sum()
        .reset_index()
    )

    # --- MAPPING BENEFICIAIRE
    mapping = (
        obs.merge(benef, on="action_representation_interet_id", how="inner")
        .rename(columns={"beneficiaire_action_menee": "beneficiaire"})
    )

    mapping = mapping[["activite_id", "beneficiaire"]].drop_duplicates()

    # --- FINAL
    return mapping.merge(budget_by_activity, on="activite_id", how="left")

def build_all(cfg: BuildConfig) -> None:
    print("1/ Préparation coeur activités + lois", flush=True)
    # 1) Prepare coeur historique: activités, lois, BM25, FAISS.
    df_acts, df_lois = prepare_from_raw(
        xlsx_organisations=_raw("1_informations_generales.xlsx"),
        xlsx_activites=_raw("8_objets_activites.xlsx"),
        xlsx_exercices=_raw("15_exercices.xlsx"),
        xlsx_domaines=_raw("7_domaines_intervention.xlsx"),
        csv_ppl=_raw("ppl.csv"),
        csv_promulguees=_raw("promulguees.csv"),
    )

    print("2/ Lecture tables auxiliaires HATVP", flush=True)
    # 2) Nouvelles tables brutes HATVP, même snapshot que les autres fichiers.
    df_exercices_raw = _read_raw_xlsx("15_exercices.xlsx")
    df_objets_raw = _read_raw_xlsx("8_objets_activites.xlsx")
    df_infos = _read_raw_xlsx("1_informations_generales.xlsx")
    df_affiliations_raw = _read_raw_xlsx("5_affiliations.xlsx")
    df_beneficiaires_raw = _read_raw_xlsx("11_beneficiaires.xlsx")
    df_observations_raw = _read_raw_xlsx("14_observations.xlsx")

    df_affiliations, df_beneficiaires, df_observations = _normalise_auxiliary_tables(
        df_affiliations_raw,
        df_beneficiaires_raw,
        df_observations_raw,
    )

    print("3/ Normalisation tables auxiliaires terminée", flush=True)

    df_acts = df_acts.sort_index()
    df_acts_min = minify_activites(df_acts)

    # minify_activites ne gardait pas ces deux colonnes; elles sont nécessaires en production.
    if "representants_id" in df_acts.columns:
        df_acts_min["representants_id"] = _norm_id_series(df_acts["representants_id"])
    df_acts_min = _add_categorie_to_activites(df_acts_min, df_infos)

    df_lois_min = minify_lois(df_lois)

    print("5/ Construction docs + index lois", flush=True)
    # --- Docs lois
    df_lois_min = df_lois_min.reset_index(drop=True)  # index propre 0..N
    docs_lois = build_docs_lois(df_lois_min)
    loi_ids = docs_lois["loi_id"].to_numpy(dtype=object)
    loi_texts = docs_lois["doc_text"].fillna("").astype(str).tolist()

    save_parquet(docs_lois.set_index("loi_id"), ART.docs_lois)

    # --- BM25 lois
    bm25_lois_bundle = build_bm25(doc_ids=loi_ids, doc_texts=loi_texts)
    save_joblib(bm25_lois_bundle, ART.bm25_lois)

    # Ensure index is string activite_id
    df_acts_min.index = df_acts_min.index.astype(str)

    docs = build_docs_activites(df_acts_min)
    doc_ids = docs["activite_id"].to_numpy(dtype=object)
    doc_texts = docs["doc_text"].fillna("").astype(str).tolist()

    print("6/ Sauvegarde artefacts tables + auxiliaires", flush=True)
    # 3) Save minified + nouvelles tables propres.
    save_parquet(df_acts_min, ART.df_activites_min)
    save_parquet(df_lois_min, ART.df_lois_min)
    save_parquet(docs.set_index("activite_id"), ART.docs_activites)
    save_parquet(df_affiliations, "df_affiliations.parquet")
    save_parquet(df_beneficiaires, "df_beneficiaires.parquet")
    save_parquet(df_observations, "df_observations.parquet")

    print("7/ Construction df_beneficiaires_activites_globales", flush=True)
    df_benef_global = _build_beneficiaires_activites_globales(
        df_acts_min=df_acts_min,
        df_observations=df_observations,
        df_beneficiaires=df_beneficiaires,
        df_exercices=df_exercices_raw,
        df_objets=df_objets_raw,
    )
    save_parquet(df_benef_global, "df_beneficiaires_activites_globales.parquet")


    print("8/ Construction BM25 activités", flush=True)
    # 4) BM25 activités
    bm25_bundle = build_bm25(doc_ids=doc_ids, doc_texts=doc_texts)
    save_joblib(bm25_bundle, ART.bm25_activites)

    print("9/ Construction embeddings + FAISS", flush=True)
    # 5) Embeddings + FAISS (ACTIVITES + LOIS)
    model = SentenceTransformer(cfg.st_model)

    print("9a/ Encodage activités", flush=True)
    # --- Activités
    emb = model.encode(doc_texts, batch_size=128, show_progress_bar=True, convert_to_numpy=True)
    emb = emb.astype("float32", copy=False)

    faiss_bundle = build_faiss_ivfpq(
        embeddings=emb,
        doc_ids=doc_ids,
        nlist=cfg.nlist,
        m=cfg.m,
        nbits=cfg.nbits,
        normalize=cfg.normalize,
    )
    save_faiss(faiss_bundle.index, ART.faiss_activites)
    print("9b/ FAISS activités sauvegardé", flush=True)

    id_map = pd.DataFrame({"pos": np.arange(len(doc_ids), dtype=int), "activite_id": doc_ids.astype(str)})
    save_parquet(id_map.set_index("pos"), ART.id_map_activites)

    print("9c/ Encodage lois", flush=True)
    # --- Lois
    emb_lois = model.encode(loi_texts, batch_size=128, show_progress_bar=True, convert_to_numpy=True)
    emb_lois = emb_lois.astype("float32", copy=False)

    faiss_lois_bundle = build_faiss_ivfpq(
        embeddings=emb_lois,
        doc_ids=loi_ids,
        nlist=max(256, cfg.nlist // 4),
        m=cfg.m,
        nbits=cfg.nbits,
        normalize=cfg.normalize,
    )
    save_faiss(faiss_lois_bundle.index, ART.faiss_lois)

    id_map_lois = pd.DataFrame({"pos": np.arange(len(loi_ids), dtype=int), "loi_id": loi_ids.astype(str)})
    save_parquet(id_map_lois.set_index("pos"), ART.id_map_lois)
    print("9d/ FAISS lois sauvegardé", flush=True)

    print("10/ Écriture manifest", flush=True)
    # 6) manifest
    manifest = {
        "built_at": datetime.utcnow().isoformat() + "Z",
        "config": asdict(cfg),
        "counts": {
            "activites": int(df_acts_min.shape[0]),
            "lois": int(df_lois_min.shape[0]),
            "docs_activites": int(len(doc_texts)),
            "affiliations": int(df_affiliations.shape[0]),
            "beneficiaires": int(df_beneficiaires.shape[0]),
            "observations": int(df_observations.shape[0]),
        },
        "files": {
            "df_activites_min": ART.df_activites_min,
            "df_lois_min": ART.df_lois_min,
            "docs_activites": ART.docs_activites,
            "bm25_activites": ART.bm25_activites,
            "faiss_activites": ART.faiss_activites,
            "id_map_activites": ART.id_map_activites,
            "df_affiliations": "df_affiliations.parquet",
            "df_beneficiaires": "df_beneficiaires.parquet",
            "df_observations": "df_observations.parquet",
            "df_beneficiaires_activites_globales": "df_beneficiaires_activites_globales.parquet",
        },
    }
    write_manifest(manifest)


if __name__ == "__main__":
    print("Début build_artifacts", flush=True)
    cfg = BuildConfig()
    build_all(cfg)
    print("✅ Artifacts built in ./artifacts", flush=True)
