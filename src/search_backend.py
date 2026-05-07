from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .bm25_index import BM25Bundle, bm25_search
from .embed_index import FaissBundle, faiss_search


def configured_search_backend(default: str = "hybrid") -> str:
    return str(os.getenv("SEARCH_BACKEND", default)).strip().lower()


def _minmax01(x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return x
    a = np.nanmin(x)
    b = np.nanmax(x)
    if not np.isfinite(a) or not np.isfinite(b) or b - a < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - a) / (b - a)


def _rrf_map(ids: np.ndarray, weight: float, k: int = 60) -> dict[str, float]:
    out = {}
    for rank, doc_id in enumerate(ids, start=1):
        out[str(doc_id)] = weight / (k + rank)
    return out


@dataclass
class SearchArtifacts:
    df_activites_min: pd.DataFrame
    bm25_bundle: BM25Bundle
    faiss_bundle: FaissBundle | None = None
    embed_query_fn: callable | None = None


def bm25_search_activites(
    query: str,
    df_activites_min: pd.DataFrame,
    bm25_bundle: BM25Bundle,
    topn: int = 50,
    k_bm25: int = 400,
    year_range: tuple[int, int] | None = None,
    min_budget: float | None = None,
) -> pd.DataFrame:
    bm_ids, bm_scores = bm25_search(bm25_bundle, query, topk=k_bm25)
    if len(bm_ids) == 0:
        return df_activites_min.iloc[0:0].copy()

    df = df_activites_min.copy()
    df.index = df.index.astype(str)
    res = df.loc[df.index.intersection(pd.Index(bm_ids.astype(str)))].copy()
    if res.empty:
        return res

    bm_map = {str(i): float(s) for i, s in zip(bm_ids, bm_scores)}
    res["bm25_score"] = res.index.map(lambda x: bm_map.get(str(x), 0.0)).astype(float)
    res["vec_score"] = 0.0
    res["hybrid_score"] = _minmax01(res["bm25_score"].to_numpy())

    if year_range is not None and "date_publication_activite" in res.columns:
        y0, y1 = year_range
        dt = pd.to_datetime(res["date_publication_activite"], errors="coerce")
        res = res[dt.dt.year.between(int(y0), int(y1))]

    if min_budget is not None:
        budget_col = "budget_activite" if "budget_activite" in res.columns else "budget_moyen_activite"
        if budget_col in res.columns:
            b = pd.to_numeric(res[budget_col], errors="coerce").fillna(0.0)
            res = res[b >= float(min_budget)]

    return res.sort_values(["hybrid_score", "bm25_score"], ascending=False).head(int(topn))


def hybrid_or_bm25_search_activites(
    query: str,
    df_activites_min: pd.DataFrame,
    bm25_bundle: BM25Bundle,
    faiss_bundle: FaissBundle | None,
    embed_query_fn,
    search_backend: str | None = None,
    k_bm25: int = 400,
    k_vec: int = 400,
    nprobe: int = 16,
    alpha_vec: float = 0.55,
    topn: int = 50,
    year_range: tuple[int, int] | None = None,
    min_budget: float | None = None,
    fusion_mode: str = "rrf",
    rrf_k: int = 60,
) -> pd.DataFrame:
    backend = (search_backend or configured_search_backend()).lower()
    if backend == "bm25" or faiss_bundle is None or embed_query_fn is None:
        return bm25_search_activites(
            query=query,
            df_activites_min=df_activites_min,
            bm25_bundle=bm25_bundle,
            topn=topn,
            k_bm25=k_bm25,
            year_range=year_range,
            min_budget=min_budget,
        )

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
        res["hybrid_score"] = res.index.map(lambda x: bm_rrf.get(str(x), 0.0) + vec_rrf.get(str(x), 0.0)).astype(float)

    if year_range is not None and "date_publication_activite" in res.columns:
        y0, y1 = year_range
        dt = pd.to_datetime(res["date_publication_activite"], errors="coerce")
        res = res[dt.dt.year.between(int(y0), int(y1))]

    if min_budget is not None:
        budget_col = "budget_activite" if "budget_activite" in res.columns else "budget_moyen_activite"
        if budget_col in res.columns:
            b = pd.to_numeric(res[budget_col], errors="coerce").fillna(0.0)
            res = res[b >= float(min_budget)]

    return res.sort_values("hybrid_score", ascending=False).head(int(topn))
