from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from rank_bm25 import BM25Okapi
from .textnorm import tokenize

@dataclass
class BM25Bundle:
    bm25: BM25Okapi
    doc_ids: np.ndarray  # same order as bm25 corpus

def build_bm25(doc_ids: np.ndarray, doc_texts: list[str]) -> BM25Bundle:
    tokenized_docs = [tokenize(t) for t in doc_texts]
    bm25 = BM25Okapi(tokenized_docs)
    return BM25Bundle(bm25=bm25, doc_ids=doc_ids)

def bm25_search(bundle: BM25Bundle, query: str, topk: int = 400) -> tuple[np.ndarray, np.ndarray]:
    q = tokenize(query)
    if not q:
        return np.array([], dtype=object), np.array([], dtype=float)
    scores = np.asarray(bundle.bm25.get_scores(q), dtype=float)
    idx = np.argsort(scores)[::-1][:topk]
    return bundle.doc_ids[idx], scores[idx]