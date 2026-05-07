from __future__ import annotations

from pathlib import Path
import re
import zlib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from rank_bm25 import BM25Okapi
import streamlit as st

from src.activity_analytics import (
    build_llm_payload,
    build_search_analytics,
    ensure_activite_id,
    explode_beneficiary_domains,
    join_unique,
    mean_distinct_numeric,
    norm_id_series,
    to_int64_safe,
)
from src.config import ART
from src.embed_index import FaissBundle, faiss_search
from src.io_artifacts import load_joblib, load_parquet, load_faiss, read_manifest
from src.llm_summarize import summarize_activites
from src.search_backend import configured_search_backend, hybrid_or_bm25_search_activites
from src.textnorm import tokenize

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


st.set_page_config(page_title="Cartographie des influences", layout="wide")
st.title("Cartographie des influences")


def load_parquet_or_raw(parquet_name: str, raw_name: str) -> pd.DataFrame:
    try:
        df = load_parquet(parquet_name)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    except FileNotFoundError:
        pass
    raw_path = Path("data/raw") / raw_name
    if raw_path.exists():
        return pd.read_excel(raw_path, dtype=str)
    return pd.DataFrame()


@st.cache_resource(show_spinner="Chargement des artefacts…")
def load_all():
    manifest = read_manifest()
    df_acts = load_parquet(ART.df_activites_min)
    df_lois = load_parquet(ART.df_lois_min)
    bm25_bundle = load_joblib(ART.bm25_activites)
    bm25_lois_bundle = load_joblib(ART.bm25_lois)

    faiss_bundle = None
    faiss_lois_bundle = None
    try:
        faiss_index = load_faiss(ART.faiss_activites)
        id_map = load_parquet(ART.id_map_activites)
        faiss_bundle = FaissBundle(
            index=faiss_index,
            doc_ids=id_map["activite_id"].astype(object).to_numpy(),
            normalize=True,
        )
    except (FileNotFoundError, ImportError):
        faiss_bundle = None

    try:
        faiss_lois_index = load_faiss(ART.faiss_lois)
        id_map_lois = load_parquet(ART.id_map_lois)
        faiss_lois_bundle = FaissBundle(
            index=faiss_lois_index,
            doc_ids=id_map_lois["loi_id"].astype(object).to_numpy(),
            normalize=True,
        )
    except (FileNotFoundError, ImportError):
        faiss_lois_bundle = None

    search_backend = configured_search_backend()
    embedder = None
    if search_backend != "bm25" and faiss_bundle is not None and SentenceTransformer is not None:
        st_model = (manifest.get("config", {}) or {}).get(
            "st_model",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        embedder = SentenceTransformer(st_model)

    return {
        "manifest": manifest,
        "df_acts": df_acts,
        "df_lois": df_lois,
        "bm25_bundle": bm25_bundle,
        "bm25_lois_bundle": bm25_lois_bundle,
        "faiss_bundle": faiss_bundle,
        "faiss_lois_bundle": faiss_lois_bundle,
        "embedder": embedder,
        "df_affiliations": load_parquet_or_raw("df_affiliations.parquet", "5_affiliations.xlsx"),
        "df_beneficiaires": load_parquet_or_raw("df_beneficiaires.parquet", "11_beneficiaires.xlsx"),
        "df_observations": load_parquet_or_raw("df_observations.parquet", "14_observations.xlsx"),
    }


STATE = load_all()
df_acts = STATE["df_acts"]
df_lois = STATE["df_lois"]
bm25_bundle = STATE["bm25_bundle"]
bm25_lois_bundle = STATE["bm25_lois_bundle"]
faiss_bundle = STATE["faiss_bundle"]
faiss_lois_bundle = STATE["faiss_lois_bundle"]
embedder = STATE["embedder"]
df_affiliations = STATE["df_affiliations"]
df_beneficiaires = STATE["df_beneficiaires"]
df_observations = STATE["df_observations"]

if "results" not in st.session_state:
    st.session_state.results = None


def embed_query_fn(q: str) -> np.ndarray:
    if embedder is None:
        raise RuntimeError("Embedder indisponible.")
    vector = embedder.encode([str(q or "")], convert_to_numpy=True)[0]
    return vector.astype("float32", copy=False)


def build_affiliations_organisations(org_search: pd.DataFrame) -> pd.DataFrame:
    if org_search is None or org_search.empty:
        return pd.DataFrame()
    if "representants_id" not in org_search.columns:
        return pd.DataFrame()
    required = {"representants_id", "denomination_affiliation"}
    if df_affiliations.empty or not required.issubset(df_affiliations.columns):
        return pd.DataFrame()

    left = org_search[["denomination", "representants_id"]].drop_duplicates().copy()
    left["representants_id"] = norm_id_series(left["representants_id"])
    left["representants_id"] = left["representants_id"].str.split(r"\s*[;,|]\s*", regex=True)
    left = left.explode("representants_id").dropna(subset=["representants_id"])

    right = df_affiliations[["representants_id", "denomination_affiliation"]].copy()
    right["representants_id"] = norm_id_series(right["representants_id"])
    right["denomination_affiliation"] = right["denomination_affiliation"].astype("string").str.strip()
    right = right.dropna(subset=["representants_id", "denomination_affiliation"]).drop_duplicates()

    detail = left.merge(right, on="representants_id", how="inner")
    if detail.empty:
        return pd.DataFrame()

    return (
        detail.groupby("denomination", dropna=False)
        .agg(
            nb_affiliations=("denomination_affiliation", "nunique"),
            affiliations=("denomination_affiliation", lambda s: join_unique(s, max_items=120)),
        )
        .reset_index()
    )


def format_date_yyyy_mm_dd(series: pd.Series) -> pd.Series:
    dates = pd.to_datetime(series, errors="coerce", dayfirst=True)
    return dates.dt.strftime("%Y-%m-%d").fillna("")


def search_lois(query_text: str, topn_laws: int, search_backend: str) -> pd.DataFrame:
    qtok = tokenize(query_text)
    if not qtok:
        return df_lois.iloc[0:0].copy()

    scores_bm = np.asarray(bm25_lois_bundle.bm25.get_scores(qtok), dtype=float)
    idx_bm = np.argsort(scores_bm)[::-1][: max(topn_laws, 50)]
    bm_ids = bm25_lois_bundle.doc_ids[idx_bm].astype(object)
    bm_scores = scores_bm[idx_bm]

    laws_df = df_lois.copy().reset_index(drop=True)
    laws_df.index = laws_df.index.astype(str)
    laws = laws_df.loc[laws_df.index.intersection(pd.Index(bm_ids.astype(str)))].copy()
    if laws.empty:
        return laws

    bm_map = {str(i): float(s) for i, s in zip(bm_ids, bm_scores)}
    laws["bm25_score"] = laws.index.map(lambda x: bm_map.get(str(x), 0.0)).astype(float)
    laws["vec_score"] = 0.0
    laws["hybrid_score"] = laws["bm25_score"]

    if search_backend != "bm25" and faiss_lois_bundle is not None and embedder is not None:
        qvec = embed_query_fn(query_text)
        vec_ids, vec_scores = faiss_search(faiss_lois_bundle, qvec, topk=max(topn_laws, 50), nprobe=16)
        vec_map = {str(i): float(s) for i, s in zip(vec_ids, vec_scores)}
        laws["vec_score"] = laws.index.map(lambda x: vec_map.get(str(x), 0.0)).astype(float)
        laws["hybrid_score"] = laws["bm25_score"] + laws["vec_score"]

    return laws.sort_values("hybrid_score", ascending=False).head(int(topn_laws))


with st.expander("À propos", expanded=False):
    st.markdown(
        """
        Recherche d'activités de lobbying avec tables analytiques cohérentes.
        Les bénéficiaires sont rattachés uniquement via `activite_id -> observations -> bénéficiaires`.
        """
    )

years = pd.to_datetime(df_acts["date_publication_activite"], errors="coerce").dropna().dt.year
y_min = int(years.min()) if not years.empty else 2018
y_max = int(years.max()) if not years.empty else 2026

query = st.text_input("Mots-clés / requête", value="")
c1, c2, c3 = st.columns(3)
with c1:
    topn = st.slider("Nombre d'activités", 10, 60, 20, 10)
with c2:
    min_budget = st.slider("Budget moyen/activité minimum (€)", 0, 5000, 0, 100)
with c3:
    year_range = st.slider("Période (année)", y_min, y_max, (y_min, y_max))

search_backend = configured_search_backend()
effective_search_backend = "hybrid" if search_backend != "bm25" and faiss_bundle is not None and embedder is not None else "bm25"
st.caption(f"Backend activites: `{effective_search_backend}`")

if st.button("Lancer", type="primary"):
    with st.spinner("Recherche…"):
        res = hybrid_or_bm25_search_activites(
            query=query,
            df_activites_min=df_acts,
            bm25_bundle=bm25_bundle,
            faiss_bundle=faiss_bundle,
            embed_query_fn=embed_query_fn if embedder is not None else None,
            search_backend=search_backend,
            k_bm25=400,
            k_vec=400,
            nprobe=16,
            alpha_vec=0.55,
            topn=int(topn),
            year_range=year_range,
            min_budget=float(min_budget),
            fusion_mode="rrf",
            rrf_k=60,
        )
    st.session_state.results = ensure_activite_id(res)
    st.session_state.selected_results = None
    st.session_state.last_query = query

res = st.session_state.get("results")
if not isinstance(res, pd.DataFrame) or res.empty:
    st.info("Lancez une recherche pour afficher des résultats.")
    st.stop()

analytics_for_results = build_search_analytics(
    selected_activities=res,
    df_observations=df_observations,
    df_beneficiaires=df_beneficiaires,
)
activities_display = analytics_for_results.table_activites.copy()
activities_display.insert(0, "selected", True)
activities_display = activities_display.rename(
    columns={
        "beneficiaires": "beneficiaire(s)",
        "hybrid_score": "score de pertinence",
        "budget_utilise": "budget utilisé",
    }
)
if "date_publication_activite" in activities_display.columns:
    activities_display["date_publication_activite"] = format_date_yyyy_mm_dd(activities_display["date_publication_activite"])
for col in ["budget utilisé", "score de pertinence", "nb_beneficiaires"]:
    if col in activities_display.columns:
        activities_display[col] = activities_display[col]

st.markdown("### Table Activités")
edited = st.data_editor(
    activities_display,
    use_container_width=True,
    disabled=[c for c in activities_display.columns if c != "selected"],
    column_config={"selected": st.column_config.CheckboxColumn("Sélection", default=True)},
    key="results_editor",
)

if st.button("Valider la sélection", type="primary"):
    selected_ids = set(edited.loc[edited["selected"] == True, "activite_id"].astype(str))
    st.session_state.selected_results = res[res["activite_id"].astype(str).isin(selected_ids)].copy()
    st.success(f"{len(selected_ids)} activités retenues.")

selected_res = st.session_state.get("selected_results")
if not isinstance(selected_res, pd.DataFrame) or selected_res.empty:
    selected_res = res.copy()

affiliations_by_org = build_affiliations_organisations(
    selected_res[["denomination", "representants_id"]].drop_duplicates()
    if {"denomination", "representants_id"}.issubset(selected_res.columns)
    else pd.DataFrame()
)
analytics = build_search_analytics(
    selected_activities=selected_res,
    df_observations=df_observations,
    df_beneficiaires=df_beneficiaires,
    affiliations_by_org=affiliations_by_org,
)

table_activites = analytics.table_activites.copy().rename(
    columns={
        "beneficiaires": "beneficiaire(s)",
        "budget_utilise": "budget utilisé",
        "hybrid_score": "score de pertinence",
    }
)
if "date_publication_activite" in table_activites.columns:
    table_activites["date_publication_activite"] = format_date_yyyy_mm_dd(table_activites["date_publication_activite"])
if "budget utilisé" in table_activites.columns:
    table_activites["budget utilisé"] = to_int64_safe(table_activites["budget utilisé"])

st.dataframe(table_activites, use_container_width=True)

st.markdown("### Table Organisations déclarantes")
st.dataframe(analytics.table_organisations, use_container_width=True)

st.markdown("### Table Bénéficiaires")
st.dataframe(analytics.table_beneficiaires, use_container_width=True)

st.markdown("### Matrice bénéficiaires × domaines")
matrix_detail = explode_beneficiary_domains(analytics.enriched)
if matrix_detail.empty:
    st.info("Aucun bénéficiaire déclaré dans la sélection.")
else:
    cell = (
        matrix_detail.groupby(["beneficiaire", "domaine"], as_index=False)
        .agg(
            budget_estime_recherche=("budget_beneficiaire_domaine", "sum"),
            nb_activites_matching=("activite_id", "nunique"),
            score_moyen=("hybrid_score", "mean"),
            objets=("objet_activite", lambda s: join_unique(s, max_items=20)) if "objet_activite" in matrix_detail.columns else ("activite_id", "count"),
        )
    )
    cell = cell[cell["budget_estime_recherche"] > 0].copy()
    if cell.empty:
        st.info("Aucune cellule de matrice avec budget réparti.")
    else:
        top_domains = (
            cell.groupby("domaine")["budget_estime_recherche"]
            .sum()
            .sort_values(ascending=False)
            .head(12)
            .index
        )
        cell = cell[cell["domaine"].isin(top_domains)].copy()
        cell["hover"] = (
            "<b>Bénéficiaire :</b> " + cell["beneficiaire"].astype(str)
            + "<br><b>Domaine :</b> " + cell["domaine"].astype(str)
            + "<br><b>Budget réparti :</b> " + cell["budget_estime_recherche"].round(0).astype(int).astype(str)
            + "<br><b>Nb activités :</b> " + cell["nb_activites_matching"].astype(str)
            + "<br><b>Score moyen :</b> " + cell["score_moyen"].fillna(0).round(3).astype(str)
            + "<br><b>Objets :</b> " + cell["objets"].astype(str)
        )
        max_val = float(cell["budget_estime_recherche"].max()) if len(cell) else 1.0
        sizeref = 2.0 * max_val / (40 ** 2) if max_val > 0 else 1.0
        figm = go.Figure(
            data=go.Scatter(
                x=cell["domaine"].astype(str),
                y=cell["beneficiaire"].astype(str),
                mode="markers",
                marker=dict(
                    size=cell["budget_estime_recherche"],
                    sizemode="area",
                    sizeref=sizeref,
                    sizemin=3,
                    opacity=0.75,
                ),
                text=cell["hover"],
                hovertemplate="%{text}<extra></extra>",
            )
        )
        figm.update_layout(height=max(650, 28 * cell["beneficiaire"].nunique() + 200))
        st.plotly_chart(figm, use_container_width=True)

st.markdown("### Lois correspondant à la recherche")
topn_laws = st.slider("Nombre de lois", 5, 200, 20, 5)
lois_res = search_lois(st.session_state.get("last_query", ""), int(topn_laws), effective_search_backend)
show_laws = lois_res.copy()
if not show_laws.empty:
    show_laws.insert(0, "selected", True)
    for date_col in ["Date initiale", "Date de promulgation"]:
        if date_col in show_laws.columns:
            show_laws[date_col] = format_date_yyyy_mm_dd(show_laws[date_col])
    edited_laws = st.data_editor(
        show_laws,
        use_container_width=True,
        disabled=[c for c in show_laws.columns if c != "selected"],
        column_config={"selected": st.column_config.CheckboxColumn("Sélection", default=True)},
        key="laws_editor",
    )
    selected_lois = edited_laws.loc[edited_laws["selected"] == True].drop(columns=["selected"], errors="ignore").copy()
else:
    selected_lois = pd.DataFrame()
    st.info("Aucune loi trouvée.")


def first_theme(value) -> str:
    parts = [p.strip() for p in re.split(r"[;,/|]", str(value or "")) if p.strip()]
    return parts[0] if parts else "Thème inconnu"


def stable_jitter(value: str, scale: float = 0.25) -> float:
    hashed = zlib.crc32(value.encode("utf-8")) % 10000
    return (hashed / 10000 - 0.5) * 2 * scale


st.markdown("### Chronologie activités & lois")
acts = analytics.table_activites.copy()
acts["date_evt"] = pd.to_datetime(acts.get("date_publication_activite"), errors="coerce")
acts = acts.dropna(subset=["date_evt"])
acts["theme"] = acts.get("domaines", "").fillna("").apply(first_theme)
acts["label_full"] = acts.get("beneficiaires", acts.get("denomination", "")).fillna("").astype(str)
budget_series = pd.to_numeric(acts.get("budget_utilise", 0), errors="coerce").fillna(0)
budget_max = float(budget_series.max()) if len(acts) else 1.0
acts["x"] = (np.sqrt(budget_series + 1) / np.sqrt(budget_max + 1) * 6.0) + 1.0
acts["x"] = acts["label_full"].apply(lambda s: stable_jitter(str(s), 0.25)) + acts["x"]

laws = selected_lois.copy()
if not laws.empty:
    laws["date_evt"] = pd.to_datetime(laws.get("Date initiale"), errors="coerce", dayfirst=True)
    laws = laws.dropna(subset=["date_evt"])
    laws["theme"] = laws.get("Thèmes", "").fillna("").apply(first_theme)
    laws["label_full"] = laws.get("Titre", "").astype(str)
    laws["x"] = -1.0 + laws["label_full"].apply(lambda s: stable_jitter(str(s), 0.15))

themes = sorted(set(acts.get("theme", pd.Series(dtype=str))).union(set(laws.get("theme", pd.Series(dtype=str)))))
palette = px.colors.qualitative.Safe
color_map = {theme: palette[i % len(palette)] for i, theme in enumerate(themes)}
figt = go.Figure()
if not laws.empty:
    figt.add_trace(
        go.Scatter(
            x=laws["x"],
            y=laws["date_evt"],
            mode="markers",
            name="Lois",
            marker=dict(size=10, symbol="square", color=[color_map[t] for t in laws["theme"]]),
            customdata=np.stack([laws["label_full"], laws["theme"]], axis=1),
            hovertemplate="<b>Loi</b><br>%{customdata[0]}<br>Thème: %{customdata[1]}<extra></extra>",
        )
    )
if not acts.empty:
    figt.add_trace(
        go.Scatter(
            x=acts["x"],
            y=acts["date_evt"],
            mode="markers",
            name="Activités de lobbying",
            marker=dict(size=9, symbol="circle", color=[color_map[t] for t in acts["theme"]]),
            customdata=np.stack([acts["label_full"], acts.get("objet_activite", ""), acts.get("domaines", "")], axis=1),
            hovertemplate="<b>Activité</b><br>%{customdata[0]}<br>%{customdata[1]}<br>Domaines: %{customdata[2]}<extra></extra>",
        )
    )
figt.update_layout(height=900, xaxis_title="Lois ← | → Activités", yaxis_title="Date")
st.plotly_chart(figt, use_container_width=True)

st.markdown("### Synthèse des résultats")
payload_llm = build_llm_payload(st.session_state.get("last_query", ""), analytics)
st.json(payload_llm, expanded=False)
if st.button("Générer la synthèse (par LLM)", type="primary"):
    with st.spinner("Synthèse…"):
        st.session_state.synth = summarize_activites(payload_llm)
if st.session_state.get("synth"):
    st.text_area("Synthèse", st.session_state.synth, height=450)
