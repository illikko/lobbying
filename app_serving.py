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
    norm_id_series,
    to_int64_safe,
)
from src.config import ART
from src.embed_index import FaissBundle
from src.io_artifacts import load_joblib, load_parquet, load_faiss, read_manifest
from src.llm_summarize import summarize_activites
from src.search_backend import configured_search_backend, hybrid_or_bm25_search_activites
from src.textnorm import tokenize

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


st.set_page_config(page_title="Cartographie des influences", layout="wide")


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


def explode_ids(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    out[col] = norm_id_series(out[col])
    out[col] = out[col].str.split(r"\s*[;,|]\s*", regex=True)
    out = out.explode(col)
    out[col] = norm_id_series(out[col])
    return out.dropna(subset=[col])


def fill_missing_category_from_infos(df_acts: pd.DataFrame, df_infos: pd.DataFrame) -> pd.DataFrame:
    if df_acts.empty or df_infos.empty:
        return df_acts
    if "representants_id" not in df_acts.columns:
        return df_acts
    if not {"representants_id", "label_categorie_organisation"}.issubset(df_infos.columns):
        return df_acts

    out = df_acts.copy()
    if "label_categorie_organisation" in out.columns:
        missing_mask = out["label_categorie_organisation"].isna() | (out["label_categorie_organisation"].astype("string").str.strip() == "")
    else:
        out["label_categorie_organisation"] = pd.NA
        missing_mask = pd.Series(True, index=out.index)

    left = explode_ids(out.loc[missing_mask, ["activite_id", "representants_id"]].drop_duplicates(), "representants_id")
    right = explode_ids(df_infos[["representants_id", "label_categorie_organisation"]].copy(), "representants_id")
    right["label_categorie_organisation"] = right["label_categorie_organisation"].astype("string").str.strip()
    right = right.dropna(subset=["representants_id", "label_categorie_organisation"]).drop_duplicates()
    if left.empty or right.empty:
        return out

    mapped = left.merge(right, on="representants_id", how="left")
    categories = (
        mapped.groupby("activite_id", dropna=False)["label_categorie_organisation"]
        .agg(lambda s: join_unique(s, max_items=10))
        .replace("", pd.NA)
    )
    out.loc[missing_mask, "label_categorie_organisation"] = out.loc[missing_mask, "activite_id"].astype(str).map(categories)
    return out


def build_affiliations_by_org(selected_activities: pd.DataFrame, df_affiliations: pd.DataFrame) -> pd.DataFrame:
    if selected_activities.empty or df_affiliations.empty:
        return pd.DataFrame()
    if "representants_id" not in selected_activities.columns:
        return pd.DataFrame()
    if not {"representants_id", "denomination_affiliation"}.issubset(df_affiliations.columns):
        return pd.DataFrame()

    left = explode_ids(selected_activities[["denomination", "representants_id"]].dropna(subset=["denomination"]).drop_duplicates(), "representants_id")
    right = explode_ids(df_affiliations[["representants_id", "denomination_affiliation"]].copy(), "representants_id")
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


@st.cache_resource(show_spinner="Chargement des artefacts…")
def load_all():
    manifest = read_manifest()
    df_acts = ensure_activite_id(load_parquet(ART.df_activites_min))
    df_lois = load_parquet(ART.df_lois_min)
    df_infos = load_parquet_or_raw("df_informations_generales.parquet", "1_informations_generales.xlsx")
    df_acts = fill_missing_category_from_infos(df_acts, df_infos)
    df_acts.index = df_acts["activite_id"].astype(str)

    bm25_bundle = load_joblib(ART.bm25_activites)
    bm25_lois_bundle = load_joblib(ART.bm25_lois)

    faiss_bundle = None
    try:
        faiss_index = load_faiss(ART.faiss_activites)
        id_map = load_parquet(ART.id_map_activites)
        faiss_bundle = FaissBundle(index=faiss_index, doc_ids=id_map["activite_id"].astype(object).to_numpy(), normalize=True)
    except (FileNotFoundError, ImportError):
        faiss_bundle = None

    search_backend = configured_search_backend()
    embedder = None
    if search_backend != "bm25" and faiss_bundle is not None and SentenceTransformer is not None:
        st_model = (manifest.get("config", {}) or {}).get(
            "st_model",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        embedder = SentenceTransformer(st_model)

    return {
        "df_acts": df_acts,
        "df_lois": df_lois,
        "bm25_bundle": bm25_bundle,
        "bm25_lois_bundle": bm25_lois_bundle,
        "faiss_bundle": faiss_bundle,
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
embedder = STATE["embedder"]
df_affiliations = STATE["df_affiliations"]
df_beneficiaires = STATE["df_beneficiaires"]
df_observations = STATE["df_observations"]


def embed_query_fn(q: str) -> np.ndarray:
    if embedder is None:
        raise RuntimeError("Embedder indisponible.")
    vector = embedder.encode([str(q or "")], convert_to_numpy=True)[0]
    return vector.astype("float32", copy=False)


def search_lois(query_text: str, topn_laws: int) -> pd.DataFrame:
    qtok = tokenize(query_text)
    if not qtok:
        return df_lois.iloc[0:0].copy()
    scores = np.asarray(bm25_lois_bundle.bm25.get_scores(qtok), dtype=float)
    idx = np.argsort(scores)[::-1][: int(topn_laws)]
    out = df_lois.iloc[idx].copy()
    out["bm25_score"] = scores[idx]
    return out.sort_values("bm25_score", ascending=False)


def first_theme(value) -> str:
    parts = [p.strip() for p in re.split(r"[;,/|]", str(value or "")) if p.strip()]
    return parts[0] if parts else "Thème inconnu"


def stable_jitter(value: str, scale: float = 0.25) -> float:
    hashed = zlib.crc32(value.encode("utf-8")) % 10000
    return (hashed / 10000 - 0.5) * 2 * scale


# application Streamlit
st.title("Cartographie des influences")


with st.expander("ℹ️ À propos de l'application", expanded=False):
        st.markdown("""
        Cette application permet de cartographier les activités de lobbying en France.  
        Elle utilise une recherche hybride lexicale et sémantique pour trouver les activités les plus pertinentes par rapport à une requête comprenant plusieurs mots clé ou une/plusieurs phrases.  
        &nbsp;&nbsp;&nbsp;&nbsp; par exemple: transport, fret, fiscalité  

        Les résultats de la recherche sont présentés sous forme de:  
        • tableau avec des filtres interactifs  
        • matrice à bulles représentant les organisations et les domaines d'activité  
        • frise chronoligique des activités de lobbying et des lois  
        • synthèse par IA des activités de lobbying  
        
        Elle utilise les données ouvertes de la HATPV sur les activités de lobbying (environ 100K activités) et les lois du Sénat (envrion 6000 lois).

        L'application est à un stade d'expérimentation, et évolue en fonction des retours de ses utilisateurs.
        Elle est développée par Vincent Castaignet, un data analyst/scientist freelance.
        """)

with st.expander("ℹ️ Comment ça marche ?", expanded=False):
        st.markdown("""
        • écrire les mots clés recherchés ou les phrases dans le formulaire "mots-clés / requête", puis cliquer sur "Lancer" pour obtenir les résultats de recherche  
        • lire les résultats (objet_activite), et dé-sélectionner les activités qui ne sont pas pertinentes (colonne "Sélection"), puis cliquer sur "Valider la sélection" pour confirmer les activités retenues  
        • faire de même pour les lois correspondant à la recherche (dé-sélectionner les lois non pertinentes, puis valider)  
        • explorer les différentes visualisations (matrice à bulles, frise chronologique)  
        • dans l'onglet "Synthèse", cliquer sur "Générer la synthèse" pour obtenir une synthèse textuelle des activités de lobbying retenues, par un modèle de langage (LLM)  
        • utiliser les filtres pour ajuster la recherche: nombre de résultats demandés, budget, période  
        """)

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
st.caption(f"Backend activités: `{effective_search_backend}`")

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
activities_display = analytics_for_results.table_activites.copy().rename(
    columns={
        "beneficiaires": "beneficiaire(s)",
        "hybrid_score": "score de pertinence",
        "budget_utilise": "budget utilisé",
    }
)
activities_display.insert(0, "selected", True)
if "date_publication_activite" in activities_display.columns:
    activities_display["date_publication_activite"] = format_date_yyyy_mm_dd(activities_display["date_publication_activite"])
if "budget utilisé" in activities_display.columns:
    activities_display["budget utilisé"] = to_int64_safe(activities_display["budget utilisé"])
drop_cols = [c for c in ["bm25_score", "vec_score", "hybrid_score"] if c in activities_display.columns]
if drop_cols:
    activities_display = activities_display.drop(columns=drop_cols)
activity_column_order = [
    "selected",
    "activite_id",
    "objet_activite",
    "denomination",
    "beneficiaire(s)",
    "nb_beneficiaires",
    "date_publication_activite",
    "budget utilisé",
    "domaines",
    "score de pertinence",
]
activities_display = activities_display[[c for c in activity_column_order if c in activities_display.columns]]

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

affiliations_by_org = build_affiliations_by_org(selected_res, df_affiliations)
analytics = build_search_analytics(
    selected_activities=selected_res,
    df_observations=df_observations,
    df_beneficiaires=df_beneficiaires,
    affiliations_by_org=affiliations_by_org,
)

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
            cell.groupby("domaine")["budget_estime_recherche"].sum().sort_values(ascending=False).head(12).index
        )
        cell = cell[cell["domaine"].isin(top_domains)].copy()
        top_beneficiaries = (
            cell.groupby("beneficiaire")["budget_estime_recherche"].sum().sort_values(ascending=False).index.tolist()
        )
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
                marker=dict(size=cell["budget_estime_recherche"], sizemode="area", sizeref=sizeref, sizemin=3, opacity=0.75),
                text=cell["hover"],
                hovertemplate="%{text}<extra></extra>",
            )
        )
        figm.update_xaxes(categoryorder="array", categoryarray=list(top_domains))
        figm.update_yaxes(categoryorder="array", categoryarray=top_beneficiaries, autorange="reversed")
        figm.update_layout(height=max(650, 28 * cell["beneficiaire"].nunique() + 200))
        st.plotly_chart(figm, use_container_width=True)

st.markdown("### Lois correspondant à la recherche")
topn_laws = st.slider("Nombre de lois", 5, 200, 20, 5)
lois_res = search_lois(st.session_state.get("last_query", ""), int(topn_laws))
show_laws = lois_res.copy()
if not show_laws.empty:
    show_laws.insert(0, "selected", True)
    for date_col in ["Date initiale", "Date de promulgation"]:
        if date_col in show_laws.columns:
            show_laws[date_col] = format_date_yyyy_mm_dd(show_laws[date_col])
    laws_display = show_laws.rename(columns={"bm25_score": "score de pertinence"})
    law_column_order = [
        "selected",
        "Titre",
        "Numéro de la loi",
        "Thèmes",
        "Date initiale",
        "Date de promulgation",
        "État du dossier",
        "URL du dossier",
        "score de pertinence",
    ]
    laws_display = laws_display[[c for c in law_column_order if c in laws_display.columns]]
    edited_laws = st.data_editor(
        laws_display,
        use_container_width=True,
        disabled=[c for c in laws_display.columns if c != "selected"],
        column_config={"selected": st.column_config.CheckboxColumn("Sélection", default=True)},
        key="laws_editor",
    )
    selected_lois = edited_laws.loc[edited_laws["selected"] == True].drop(columns=["selected"], errors="ignore").copy()
else:
    selected_lois = pd.DataFrame()
    st.info("Aucune loi trouvée.")

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
with st.expander("Payload LLM", expanded=False):
    st.json(payload_llm, expanded=False)
if st.button("Générer la synthèse (par LLM)", type="primary"):
    with st.spinner("Synthèse…"):
        st.session_state.synth = summarize_activites(payload_llm)
if st.session_state.get("synth"):
    st.text_area("Synthèse", st.session_state.synth, height=450)
