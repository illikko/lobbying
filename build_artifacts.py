from __future__ import annotations
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
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
    # BM25
    # (no special config)
    # misc
    created_by: str = "build_artifacts.py"

def _raw(p: str) -> str:
    return str((PATHS.data_raw / p).resolve())

def build_all(cfg: BuildConfig) -> None:
    
    # 1) Prepare
    df_acts, df_lois = prepare_from_raw(
        xlsx_organisations=_raw("1_informations_generales.xlsx"),
        xlsx_activites=_raw("8_objets_activites.xlsx"),
        xlsx_exercices=_raw("15_exercices.xlsx"),
        xlsx_domaines=_raw("7_domaines_intervention.xlsx"),
        csv_ppl=_raw("ppl.csv"),
        csv_promulguees=_raw("promulguees.csv"),
    )

    df_acts = df_acts.sort_index()
    df_acts_min = minify_activites(df_acts)
    df_lois_min = minify_lois(df_lois)
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

    # 2) Save minified
    save_parquet(df_acts_min, ART.df_activites_min)
    save_parquet(df_lois_min, ART.df_lois_min)
    save_parquet(docs.set_index("activite_id"), ART.docs_activites)

    # 3) BM25
    bm25_bundle = build_bm25(doc_ids=doc_ids, doc_texts=doc_texts)
    save_joblib(bm25_bundle, ART.bm25_activites)

    # 4) Embeddings + FAISS (ACTIVITES + LOIS)
    model = SentenceTransformer(cfg.st_model)

    # --- Activités
    emb = model.encode(doc_texts, batch_size=128, show_progress_bar=True, convert_to_numpy=True)
    emb = emb.astype("float32", copy=False)

    faiss_bundle = build_faiss_ivfpq(
        embeddings=emb,
        doc_ids=doc_ids,
        nlist=cfg.nlist,
        m=cfg.m,
        nbits=cfg.nbits,
        normalize=cfg.normalize
    )
    save_faiss(faiss_bundle.index, ART.faiss_activites)

    id_map = pd.DataFrame({"pos": np.arange(len(doc_ids), dtype=int), "activite_id": doc_ids.astype(str)})
    save_parquet(id_map.set_index("pos"), ART.id_map_activites)

    # --- Lois
    emb_lois = model.encode(loi_texts, batch_size=128, show_progress_bar=True, convert_to_numpy=True)
    emb_lois = emb_lois.astype("float32", copy=False)

    faiss_lois_bundle = build_faiss_ivfpq(
        embeddings=emb_lois,
        doc_ids=loi_ids,
        nlist=max(256, cfg.nlist // 4),
        m=cfg.m,
        nbits=cfg.nbits,
        normalize=cfg.normalize
    )
    save_faiss(faiss_lois_bundle.index, ART.faiss_lois)

    id_map_lois = pd.DataFrame({"pos": np.arange(len(loi_ids), dtype=int), "loi_id": loi_ids.astype(str)})
    save_parquet(id_map_lois.set_index("pos"), ART.id_map_lois)

    # 5) manifest
    manifest = {
        "built_at": datetime.utcnow().isoformat() + "Z",
        "config": asdict(cfg),
        "counts": {
            "activites": int(df_acts_min.shape[0]),
            "lois": int(df_lois_min.shape[0]),
            "docs_activites": int(len(doc_texts)),
        },
        "files": {
            "df_activites_min": ART.df_activites_min,
            "df_lois_min": ART.df_lois_min,
            "docs_activites": ART.docs_activites,
            "bm25_activites": ART.bm25_activites,
            "faiss_activites": ART.faiss_activites,
            "id_map_activites": ART.id_map_activites,
        }
    }
    write_manifest(manifest)

if __name__ == "__main__":
    cfg = BuildConfig()
    build_all(cfg)
    print("✅ Artifacts built in ./artifacts")