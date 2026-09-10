# Artefacts générés

Ce dossier est rempli par `build_artifacts.py`.

En local, `scripts/sync_local_bm25.py` y produit les Parquet applicatifs et les index BM25, sans FAISS.
En CI/production, `scripts/sync_all.py` peut également y produire les embeddings et index FAISS.

Tout le contenu généré de ce dossier peut être reconstruit à partir des données sources.
