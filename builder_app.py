from __future__ import annotations

import subprocess
import sys

import streamlit as st

from src.io_artifacts import read_manifest

st.set_page_config(page_title="Builder LobbySearch", layout="wide")
st.title("Synchronisation et construction des artefacts")
st.info(
    "Utilisez ce builder pour garder les données Tricoteuses et tous les artefacts "
    "Parquet/BM25/FAISS sur le même snapshot."
)

manifest = read_manifest()
if manifest:
    st.write("Snapshot actuellement servi")
    st.json(manifest)

use_existing = st.checkbox(
    "Utiliser les données Tricoteuses déjà présentes (ne pas lancer Docker)",
    value=False,
)
force = st.checkbox("Forcer la reconstruction des artefacts", value=False)

if st.button("🔄 Synchroniser et reconstruire", type="primary"):
    cmd = [sys.executable, "scripts/sync_all.py"]
    if use_existing:
        cmd.append("--no-fetch")
    if force:
        cmd.append("--force-build")
    with st.spinner("Synchronisation en cours…"):
        proc = subprocess.run(cmd, capture_output=True, text=True)
    st.code((proc.stdout or "") + (proc.stderr or ""))
    if proc.returncode == 0:
        st.success("Données et artefacts alignés.")
        st.rerun()
    else:
        st.error("Échec de la synchronisation. Les anciens artefacts cohérents ont été conservés.")
