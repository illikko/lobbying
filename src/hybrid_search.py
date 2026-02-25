from __future__ import annotations
import numpy as np
import pandas as pd
from .bm25_index import BM25Bundle, bm25_search
from .embed_index import FaissBundle, faiss_search

def _minmax01(x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return x
    a = np.nanmin(x)
    b = np.nanmax(x)
    if not np.isfinite(a) or not np.isfinite(b) or b - a < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - a) / (b - a)

def _rrf_map(ids: np.ndarray, weight: float, k: int = 60) -> dict[str, float]:
    # ids are ordered best -> worst
    out = {}
    for rank, doc_id in enumerate(ids, start=1):
        out[str(doc_id)] = weight / (k + rank)
    return out

def hybrid_search_activites(
    query: str,
    df_activites_min: pd.DataFrame,
    bm25_bundle: BM25Bundle,
    faiss_bundle: FaissBundle,
    embed_query_fn,  # callable(query)->np.ndarray shape (384,)
    k_bm25: int = 400,
    k_vec: int = 400,
    nprobe: int = 16,
    alpha_vec: float = 0.55,
    topn: int = 50,
    year_range: tuple[int, int] | None = None,
    min_budget: float | None = None,
    fusion_mode: str = "rrf",   # "rrf" ou "mean"
    rrf_k: int = 60,
) -> pd.DataFrame:
    bm_ids, bm_scores = bm25_search(bm25_bundle, query, topk=k_bm25)

    qvec = embed_query_fn(query)
    vec_ids, vec_scores = faiss_search(faiss_bundle, qvec, topk=k_vec, nprobe=nprobe)

    if len(bm_ids) == 0 and len(vec_ids) == 0:
        return df_activites_min.iloc[0:0].copy()

    all_ids = np.unique(np.concatenate([bm_ids.astype(object), vec_ids.astype(object)]))
    if len(all_ids) == 0:
        return df_activites_min.iloc[0:0].copy()

    df = df_activites_min.copy()
    df.index = df.index.astype(str)
    res = df.loc[df.index.intersection(pd.Index(all_ids.astype(str)))].copy()
    if res.empty:
        return res
    
    # attach raw scores
    bm_map = {str(i): float(s) for i, s in zip(bm_ids, bm_scores)}
    vec_map = {str(i): float(s) for i, s in zip(vec_ids, vec_scores)}

    res["bm25_score"] = res.index.map(lambda x: bm_map.get(str(x), 0.0)).astype(float)
    res["vec_score"] = res.index.map(lambda x: vec_map.get(str(x), 0.0)).astype(float)

    if fusion_mode == "mean":
        res["bm25_norm"] = _minmax01(res["bm25_score"].to_numpy())
        res["vec_norm"] = _minmax01(res["vec_score"].to_numpy())
        res["hybrid_score"] = alpha_vec * res["vec_norm"] + (1.0 - alpha_vec) * res["bm25_norm"]
    else:
        bm_rrf = _rrf_map(bm_ids, weight=(1.0 - alpha_vec), k=rrf_k)
        vec_rrf = _rrf_map(vec_ids, weight=alpha_vec, k=rrf_k)

        def fused(doc_id: str) -> float:
            return bm_rrf.get(doc_id, 0.0) + vec_rrf.get(doc_id, 0.0)

        res["hybrid_score"] = res.index.map(fused).astype(float)

    # filters
    if year_range is not None and "date_publication_activite" in res.columns:
        y0, y1 = year_range
        dt = pd.to_datetime(res["date_publication_activite"], errors="coerce")
        res = res[dt.dt.year.between(int(y0), int(y1))]

    if min_budget is not None and "budget_moyen_activite" in res.columns:
        b = pd.to_numeric(res["budget_moyen_activite"], errors="coerce").fillna(0.0)
        res = res[b >= float(min_budget)]

    res = res.sort_values("hybrid_score", ascending=False).head(int(topn))
    return res
    

def hybrid_search_lois(query: str, topn: int = 50):
    # BM25
    bm_ids, bm_scores = bm25_lois_bundle.bm25.get_scores(tokenize(query)), None

    # attach raw scores (useful for debugging)
    bm_map = {str(i): float(s) for i, s in zip(bm_ids, bm_scores)}
    vec_map = {str(i): float(s) for i, s in zip(vec_ids, vec_scores)}  # cosine-ish if normalized
    res["bm25_score"] = res.index.map(lambda x: bm_map.get(str(x), 0.0)).astype(float)
    res["vec_score"] = res.index.map(lambda x: vec_map.get(str(x), 0.0)).astype(float)

    if fusion_mode == "mean":
        res["bm25_norm"] = _minmax01(res["bm25_score"].to_numpy())
        res["vec_norm"] = _minmax01(res["vec_score"].to_numpy())
        res["hybrid_score"] = alpha_vec * res["vec_norm"] + (1.0 - alpha_vec) * res["bm25_norm"]
    else:
        # default: RRF
        bm_rrf = _rrf_map(bm_ids, weight=(1.0 - alpha_vec), k=rrf_k)
        vec_rrf = _rrf_map(vec_ids, weight=alpha_vec, k=rrf_k)

        def fused(doc_id: str) -> float:
            return bm_rrf.get(doc_id, 0.0) + vec_rrf.get(doc_id, 0.0)

        res["hybrid_score"] = res.index.map(fused).astype(float)

    # filters
    if year_range is not None and "date_publication_activite" in res.columns:
        y0, y1 = year_range
        dt = pd.to_datetime(res["date_publication_activite"], errors="coerce")
        res = res[dt.dt.year.between(int(y0), int(y1))]

    if min_budget is not None and "budget_moyen_activite" in res.columns:
        b = pd.to_numeric(res["budget_moyen_activite"], errors="coerce").fillna(0.0)
        res = res[b >= float(min_budget)]

    res = res.sort_values("hybrid_score", ascending=False).head(int(topn))
    return res