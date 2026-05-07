from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import joblib
try:
    import faiss
except ImportError:
    faiss = None
from .config import PATHS, ART

def artifacts_dir() -> Path:
    PATHS.artifacts.mkdir(parents=True, exist_ok=True)
    return PATHS.artifacts

def save_parquet(df: pd.DataFrame, name: str) -> None:
    p = artifacts_dir() / name
    df.to_parquet(p, index=True)

def load_parquet(name: str) -> pd.DataFrame:
    p = artifacts_dir() / name
    return pd.read_parquet(p)

def save_joblib(obj, name: str) -> None:
    p = artifacts_dir() / name
    joblib.dump(obj, p)

def load_joblib(name: str):
    p = artifacts_dir() / name
    return joblib.load(p)

def save_faiss(index, name: str) -> None:
    if faiss is None:
        raise ImportError("faiss n'est pas disponible dans cet environnement.")
    p = artifacts_dir() / name
    faiss.write_index(index, str(p))

def load_faiss(name: str):
    if faiss is None:
        raise ImportError("faiss n'est pas disponible dans cet environnement.")
    p = artifacts_dir() / name
    return faiss.read_index(str(p))

def write_manifest(d: dict) -> None:
    p = artifacts_dir() / ART.manifest
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

def read_manifest() -> dict:
    p = artifacts_dir() / ART.manifest
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # manifest corrompu / partiellement écrit
        return {"_error": "manifest.json invalide (corrompu ou incomplet). Supprimez-le et relancez un build."}
