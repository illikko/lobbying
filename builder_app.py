import streamlit as st
from dataclasses import asdict
from src.io_artifacts import read_manifest
from build_artifacts import BuildConfig, build_all

st.set_page_config(page_title="Builder (local)", layout="wide")
st.title("Builder artefacts (local)")

st.info("Cette app est à lancer en local. Elle construit les artefacts dans ./artifacts/ (à committer dans Git).")

manifest = read_manifest()
if manifest:
    st.success(f"Manifest actuel: built_at={manifest.get('built_at')}")
    st.json(manifest)

st.markdown("## Paramètres")

cfg = BuildConfig(
    st_model=st.text_input("Sentence-Transformers model", value=BuildConfig.st_model),
    nlist=st.number_input("FAISS nlist", min_value=256, max_value=16384, value=BuildConfig.nlist, step=256),
    m=st.number_input("FAISS PQ m", min_value=8, max_value=64, value=BuildConfig.m, step=4),
    nbits=st.selectbox("FAISS PQ nbits", options=[4, 6, 8], index=[4,6,8].index(BuildConfig.nbits)),
    normalize=st.checkbox("Normalize (cosine via inner product)", value=BuildConfig.normalize),
)

st.code(asdict(cfg), language="python")

if st.button("🚀 Build artifacts", type="primary"):
    with st.spinner("Build en cours (offline)…"):
        build_all(cfg)
    st.success("Artefacts générés dans ./artifacts ✅")
    st.rerun()