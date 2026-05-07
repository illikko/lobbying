from __future__ import annotations
from dataclasses import dataclass
import numpy as np
try:
    import faiss
except ImportError:
    faiss = None

@dataclass
class FaissBundle:
    index: object
    doc_ids: np.ndarray  # position -> doc id (activite_id)
    normalize: bool = True  # cosine via inner product

def _l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

def build_faiss_ivfpq(
    embeddings: np.ndarray,
    doc_ids: np.ndarray,
    nlist: int = 2048,
    m: int = 32,
    nbits: int = 8,
    normalize: bool = True,
) -> FaissBundle:
    if faiss is None:
        raise ImportError("faiss n'est pas disponible dans cet environnement.")
    emb = embeddings.astype("float32", copy=False)
    if normalize:
        emb = _l2_normalize(emb)

    d = emb.shape[1]
    quantizer = faiss.IndexFlatIP(d) if normalize else faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)

    # train then add
    index.train(emb)
    index.add(emb)

    return FaissBundle(index=index, doc_ids=doc_ids.astype(object), normalize=normalize)

def faiss_search(bundle: FaissBundle, query_vec: np.ndarray, topk: int = 400, nprobe: int = 16):
    if faiss is None or bundle is None or bundle.index is None:
        raise RuntimeError("Recherche FAISS indisponible.")
    v = query_vec.astype("float32").reshape(1, -1)
    if bundle.normalize:
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    bundle.index.nprobe = int(nprobe)
    D, I = bundle.index.search(v, topk)
    I = I.reshape(-1)
    D = D.reshape(-1)
    ok = I >= 0
    I = I[ok]
    D = D[ok]
    # Convert FAISS positions to doc_ids
    doc_ids = bundle.doc_ids[I]
    return doc_ids, D
