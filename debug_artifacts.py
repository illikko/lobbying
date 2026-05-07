import pandas as pd
from pathlib import Path

ARTIFACTS = Path("artifacts")

def read(name):
    p = ARTIFACTS / name
    print(f"\n{name}: exists={p.exists()} size={p.stat().st_size if p.exists() else 0}")
    if not p.exists():
        return None
    return pd.read_parquet(p)

df_acts = read("df_activites_min.parquet")
docs = read("docs_activites.parquet")
id_map = read("id_map_activites.parquet")
obs = read("df_observations.parquet")
benef = read("df_beneficiaires.parquet")

def ids_from_index(df):
    return set(df.index.astype(str)) if df is not None else set()

def ids_from_col(df, col):
    return set(df[col].astype(str)) if df is not None and col in df.columns else set()

acts_ids = ids_from_index(df_acts)
docs_ids = ids_from_index(docs)
idmap_ids = ids_from_col(id_map, "activite_id")
obs_ids = ids_from_col(obs, "activite_id")

print("\n--- Cardinalités ---")
print("df_activites_min:", len(acts_ids))
print("docs_activites:", len(docs_ids))
print("id_map activites:", len(idmap_ids))
print("observations activites:", len(obs_ids))

print("\n--- Incohérences ---")
print("docs absents de df_activites_min:", len(docs_ids - acts_ids))
print("id_map absents de df_activites_min:", len(idmap_ids - acts_ids))
print("df_activites_min sans observations:", len(acts_ids - obs_ids))
print("observations hors df_activites_min:", len(obs_ids - acts_ids))