from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.serving_snapshot import check_alignment, load_snapshot


def test_missing_snapshot_never_loads_or_rebuilds(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Un snapshot incomplet ne doit lancer aucun chargement ni build")

    monkeypatch.setattr("src.serving_snapshot.pd.read_parquet", forbidden)
    monkeypatch.setattr("src.serving_snapshot.joblib.load", forbidden)
    monkeypatch.setattr("src.bm25_index.build_bm25", forbidden)
    with pytest.raises(FileNotFoundError, match="Aucune reconstruction"):
        load_snapshot(tmp_path)


def test_alignment_rejects_index_from_another_snapshot():
    bundle = SimpleNamespace(doc_ids=np.array(["a", "c"]), bm25=SimpleNamespace(corpus_size=2))
    with pytest.raises(ValueError, match="désaligné"):
        check_alignment(pd.DataFrame(index=["a", "b"]), bundle, "activités")


def test_alignment_allows_different_row_order():
    bundle = SimpleNamespace(doc_ids=np.array(["b", "a"]), bm25=SimpleNamespace(corpus_size=2))
    check_alignment(pd.DataFrame(index=["a", "b"]), bundle, "activités")
