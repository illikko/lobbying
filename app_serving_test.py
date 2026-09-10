from __future__ import annotations

from pathlib import Path
import os
import re
import zlib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.activity_analytics import (
    build_llm_payload,
    build_search_analytics,
    ensure_activite_id,
    explode_beneficiary_domains,
    join_unique,
    to_int64_safe,
    validate_beneficiary_budget_not_overcounted,
    validate_no_beneficiary_join_on_text_keys,
    validate_selected_ids_exist_in_observations_when_expected,
    validate_three_tables_share_same_enriched_source,
)
from src.bm25_index import build_bm25
from src.config import ART, PATHS
from src.io_artifacts import load_joblib, load_parquet
from src.prep import (
    clean_columns,
    concat_sur_liste_colonnes,
    minify_activites,
    minify_lois,
    prepare_from_imported,
)
from src.search_backend import bm25_search_activites, hybrid_or_bm25_search_lois
from src.textnorm import tokenize


PREBUILT_ONLY = os.getenv("LOBBYSEARCH_PREBUILT_ONLY", "0") == "1"
ENABLE_LLM = os.getenv("LOBBYSEARCH_ENABLE_LLM", "1") == "1"

st.set_page_config(page_title="Test BM25 LobbySearch", layout="wide")
st.title("Test BM25 uniquement")

LOCAL_TEST_ARTIFACTS = Path(__file__).resolve().parent / "local_test_artifacts"


def norm_id_series(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .mask(lambda x: x.str.lower().isin(["", "nan", "none", "<na>"]))
    )


def explode_ids(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    out[col] = norm_id_series(out[col])
    out[col] = out[col].str.split(r"\s*[;,|]\s*", regex=True)
    out = out.explode(col)
    out[col] = norm_id_series(out[col])
    return out.dropna(subset=[col])


def load_repo_table(parquet_name: str, imported_name: str | None = None) -> pd.DataFrame:
    try:
        return load_parquet(parquet_name)
    except FileNotFoundError:
        pass
    local_snapshot = LOCAL_TEST_ARTIFACTS / parquet_name
    if local_snapshot.exists():
        return pd.read_parquet(local_snapshot)
    if imported_name is not None:
        path = PATHS.data_imported / imported_name
        if path.exists():
            return pd.read_parquet(path)
    return pd.DataFrame()


def load_local_activities() -> pd.DataFrame:
    try:
        df = ensure_activite_id(load_parquet(ART.df_activites_min))
        df.index = df["activite_id"].astype(str)
        return df
    except FileNotFoundError:
        pass
    local_snapshot = LOCAL_TEST_ARTIFACTS / ART.df_activites_min
    if local_snapshot.exists():
        df = ensure_activite_id(pd.read_parquet(local_snapshot))
        df.index = df["activite_id"].astype(str)
        return df
    for path in [Path("activites.csv"), Path("df.csv")]:
        if path.exists():
            df = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig")
            df = ensure_activite_id(df)
            for col in ["budget_total", "nb_activites_total", "budget_moyen_activite", "bm25_score", "vec_score", "hybrid_score"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df.index = df["activite_id"].astype(str)
            return df
    raise FileNotFoundError("Aucune base activités locale trouvée.")


def load_local_activities_from_imported() -> pd.DataFrame:
    required = [
        PATHS.data_imported / "informations_generales.parquet",
        PATHS.data_imported / "objets_activites.parquet",
        PATHS.data_imported / "exercices.parquet",
        PATHS.data_imported / "domaines_intervention.parquet",
        PATHS.decisions_raw,
    ]
    if not all(path.exists() for path in required):
        raise FileNotFoundError(
            "Aucun artefact local et tables importées incomplètes. "
            "Exécutez d’abord : python scripts/sync_local_bm25.py. "
            "Cette synchronisation locale utilise Docker et reconstruit uniquement Parquet + BM25, sans FAISS."
        )

    df_acts_full, _ = prepare_from_imported(
        parquet_organisations=str(PATHS.data_imported / "informations_generales.parquet"),
        parquet_activites=str(PATHS.data_imported / "objets_activites.parquet"),
        parquet_exercices=str(PATHS.data_imported / "exercices.parquet"),
        parquet_domaines=str(PATHS.data_imported / "domaines_intervention.parquet"),
        parquet_decisions=str(PATHS.decisions_raw),
    )
    df = minify_activites(df_acts_full).reset_index()
    if "representants_id" in df_acts_full.columns:
        representants = (
            df_acts_full[["representants_id"]]
            .reset_index()
            .drop_duplicates(subset=["activite_id"])
            .copy()
        )
        representants["activite_id"] = representants["activite_id"].astype(str)
        df["activite_id"] = df["activite_id"].astype(str)
        df = df.merge(representants, on="activite_id", how="left")
    df = ensure_activite_id(df)
    df.index = df["activite_id"].astype(str)
    return df


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
    categories = mapped.groupby("activite_id", dropna=False)["label_categorie_organisation"].agg(lambda s: join_unique(s, max_items=10)).replace("", pd.NA)
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


def build_local_bm25_bundle(df_acts: pd.DataFrame):
    try:
        return load_joblib(ART.bm25_activites)
    except FileNotFoundError:
        local_snapshot = LOCAL_TEST_ARTIFACTS / ART.bm25_activites
        if local_snapshot.exists():
            return __import__("joblib").load(local_snapshot)
        docs = (df_acts.get("objet_activite", "").fillna("").astype(str) + " " + df_acts.get("denomination", "").fillna("").astype(str) + " " + df_acts.get("domaines", "").fillna("").astype(str)).tolist()
        doc_ids = df_acts["activite_id"].astype(object).to_numpy()
        return build_bm25(doc_ids=doc_ids, doc_texts=docs)


def build_lois_bm25_bundle(df_lois: pd.DataFrame):
    # 1. Utiliser en priorité le BM25 construit à partir
    #    des décisions actuellement présentes dans artifacts/
    try:
        return load_joblib(ART.bm25_lois)
    except FileNotFoundError:
        pass

    # 2. Snapshot historique uniquement comme dernier recours
    local_snapshot = LOCAL_TEST_ARTIFACTS / ART.bm25_lois

    if local_snapshot.exists():
        return __import__("joblib").load(local_snapshot)

    # 3. À défaut, reconstruire un BM25 en mémoire
    docs = (
        df_lois.get("Titre", "").fillna("").astype(str)
        + " "
        + df_lois.get("Description", "").fillna("").astype(str)
        + " "
        + df_lois.get("Thèmes", "").fillna("").astype(str)
    ).tolist()

    doc_ids = np.arange(
        len(df_lois),
        dtype=object,
    )

    return build_bm25(
        doc_ids=doc_ids,
        doc_texts=docs,
    )


def load_local_laws() -> pd.DataFrame:
    # Nom historique dans l'UI : ce DataFrame contient désormais les décisions publiques
    # (lois, décrets, ordonnances, arrêtés), jamais les amendements.
    try:
        return load_parquet(ART.df_lois_min)
    except FileNotFoundError:
        pass
    local_snapshot = LOCAL_TEST_ARTIFACTS / ART.df_lois_min
    if local_snapshot.exists():
        df = pd.read_parquet(local_snapshot)
        # Le snapshot local historique provient des anciens corpus législatifs
        # (PPL/promulguées) : ce sont des lois. Les snapshots Canutes récents
        # fournissent directement type_decision et ne passent pas par ce fallback.
        if "type_decision" not in df.columns:
            df.insert(0, "type_decision", "loi")
        return df
    if not PATHS.decisions_raw.exists():
        return pd.DataFrame()
    df = pd.read_parquet(PATHS.decisions_raw)
    if "type_decision" in df.columns:
        df = df[df["type_decision"].isin(["loi", "decret", "ordonnance", "arrete"])].copy()
    return minify_lois(df)


def format_date_yyyy_mm_dd(series: pd.Series) -> pd.Series:
    dates = pd.to_datetime(series, errors="coerce", dayfirst=True)
    return dates.dt.strftime("%Y-%m-%d").fillna("")


def first_theme(value) -> str:
    parts = [p.strip() for p in re.split(r"[;,/|]", str(value or "")) if p.strip()]
    return parts[0] if parts else "Thème inconnu"


def stable_jitter(value: str, scale: float = 0.25) -> float:
    hashed = zlib.crc32(value.encode("utf-8")) % 10000
    return (hashed / 10000 - 0.5) * 2 * scale


@st.cache_resource(show_spinner="Chargement des fichiers du repo…")
def load_local_state():
    if PREBUILT_ONLY:
        from src.serving_snapshot import load_snapshot

        state = load_snapshot(PATHS.artifacts)
        state["df_acts"] = fill_missing_category_from_infos(
            state["df_acts"], state.pop("df_informations_generales")
        )
        return state
    try:
        df_acts = ensure_activite_id(load_parquet(ART.df_activites_min))
        df_acts.index = df_acts["activite_id"].astype(str)
    except FileNotFoundError:
        df_acts = load_local_activities()
    df_infos = load_repo_table("df_informations_generales.parquet", "informations_generales.parquet")
    df_acts = fill_missing_category_from_infos(df_acts, df_infos)
    df_lois = load_local_laws()
    return {
        "df_acts": df_acts,
        "df_lois": df_lois,
        "bm25_bundle": build_local_bm25_bundle(df_acts),
        "bm25_lois_bundle": build_lois_bm25_bundle(df_lois) if not df_lois.empty else None,
        "df_observations": load_repo_table("df_observations.parquet", "observations.parquet"),
        "df_beneficiaires": load_repo_table("df_beneficiaires.parquet", "beneficiaires.parquet"),
        "df_affiliations": load_repo_table("df_affiliations.parquet", "affiliations.parquet"),
    }


try:
    STATE = load_local_state()
except (FileNotFoundError, ValueError) as exc:
    st.error(str(exc))
    st.stop()
df_acts = STATE["df_acts"]
df_lois = STATE["df_lois"]
bm25_bundle = STATE["bm25_bundle"]
bm25_lois_bundle = STATE["bm25_lois_bundle"]
df_observations = STATE["df_observations"]
df_beneficiaires = STATE["df_beneficiaires"]
df_affiliations = STATE["df_affiliations"]

query = st.text_input("Requête BM25", value="", key="test_query_input")
topn = st.slider("Nombre d'activités", 10, 100, 30, 10)

current_query = st.session_state.get("test_query_input", "")

if "last_submitted_test_query" not in st.session_state:
    st.session_state.last_submitted_test_query = None

if st.session_state.last_submitted_test_query is not None and current_query != st.session_state.last_submitted_test_query:
    st.session_state.pop("test_results", None)
    st.session_state.pop("test_synth", None)
    st.info("La requête a changé. Cliquez sur `Tester BM25` pour recalculer les résultats.")

if st.button("Tester BM25", type="primary"):
    results = bm25_search_activites(query=current_query, df_activites_min=df_acts, bm25_bundle=bm25_bundle, topn=int(topn), k_bm25=max(200, int(topn)))
    st.session_state.test_results = ensure_activite_id(results)
    st.session_state.last_submitted_test_query = current_query

results = st.session_state.get("test_results")
if not isinstance(results, pd.DataFrame) or results.empty:
    st.info("Lancez une requête BM25.")
    st.stop()

analytics = build_search_analytics(
    selected_activities=results,
    df_observations=df_observations,
    df_beneficiaires=df_beneficiaires,
    affiliations_by_org=build_affiliations_by_org(results, df_affiliations),
)

validate_no_beneficiary_join_on_text_keys()
validate_selected_ids_exist_in_observations_when_expected(analytics.enriched, df_observations)
validate_beneficiary_budget_not_overcounted(analytics.enriched)
validate_three_tables_share_same_enriched_source(analytics)

st.markdown("### Table Activités")
table_activites = analytics.table_activites.copy().rename(columns={"beneficiaires": "beneficiaire(s)", "budget_utilise": "budget utilisé", "hybrid_score": "score de pertinence"})
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
matrix = explode_beneficiary_domains(analytics.enriched)
if matrix.empty:
    st.info("Aucun bénéficiaire déclaré.")
else:
    matrix_table = (
        matrix.groupby(["beneficiaire", "domaine"], dropna=False)
        .agg(
            budget_estime_recherche=("budget_beneficiaire_domaine", "sum"),
            nb_activites_matching=("activite_id", "nunique"),
        )
        .reset_index()
        .sort_values("budget_estime_recherche", ascending=False)
    )
    matrix_table["budget_estime_recherche"] = to_int64_safe(matrix_table["budget_estime_recherche"])
    top_domains = (
        matrix_table.groupby("domaine")["budget_estime_recherche"]
        .sum()
        .sort_values(ascending=False)
        .head(12)
        .index
    )
    matrix_table = matrix_table[matrix_table["domaine"].isin(top_domains)].copy()
    top_beneficiaries = (
        matrix_table.groupby("beneficiaire")["budget_estime_recherche"]
        .sum()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    matrix_table["hover"] = (
        "<b>Bénéficiaire :</b> " + matrix_table["beneficiaire"].astype(str)
        + "<br><b>Domaine :</b> " + matrix_table["domaine"].astype(str)
        + "<br><b>Budget réparti :</b> " + matrix_table["budget_estime_recherche"].round(0).astype(int).astype(str)
        + "<br><b>Nb activités :</b> " + matrix_table["nb_activites_matching"].astype(str)
    )
    max_val = float(matrix_table["budget_estime_recherche"].max()) if len(matrix_table) else 1.0
    sizeref = 2.0 * max_val / (40 ** 2) if max_val > 0 else 1.0
    figm = go.Figure(
        data=go.Scatter(
            x=matrix_table["domaine"].astype(str),
            y=matrix_table["beneficiaire"].astype(str),
            mode="markers",
            marker=dict(
                size=matrix_table["budget_estime_recherche"],
                sizemode="area",
                sizeref=sizeref,
                sizemin=3,
                opacity=0.75,
            ),
            text=matrix_table["hover"],
            hovertemplate="%{text}<extra></extra>",
        )
    )
    figm.update_xaxes(categoryorder="array", categoryarray=list(top_domains))
    figm.update_yaxes(categoryorder="array", categoryarray=top_beneficiaries, autorange="reversed")
    figm.update_layout(height=max(650, 28 * matrix_table["beneficiaire"].nunique() + 200))
    st.plotly_chart(figm, use_container_width=True)
    st.dataframe(matrix_table.drop(columns=["hover"]), use_container_width=True)

st.markdown("### Décisions publiques correspondant à la recherche")
if bm25_lois_bundle is None or df_lois.empty:
    selected_lois = pd.DataFrame()
    st.info("Aucune base de décisions publiques locale disponible.")
else:
    topn_laws = st.slider("Nombre de décisions", 5, 100, 20, 5)
    lois_res = hybrid_or_bm25_search_lois(
        query=st.session_state.get("last_submitted_test_query", ""),
        df_lois_min=df_lois,
        bm25_bundle=bm25_lois_bundle,
        faiss_bundle=None,
        embed_query_fn=None,
        search_backend="bm25",
        k_bm25=400,
        topn=int(topn_laws),
        fusion_mode="rrf",
        rrf_k=60,
    )
    qtok = tokenize(current_query)
    all_law_scores = np.asarray(bm25_lois_bundle.bm25.get_scores(qtok), dtype=float) if qtok else np.asarray([], dtype=float)
    st.caption(
        f"Décisions publiques avec score BM25 strictement positif sur tout le corpus : {int((all_law_scores > 0).sum()) if len(all_law_scores) else 0}"
    )
    hide_zero_scores = st.checkbox("Masquer les décisions à score nul", value=True)
    show_laws = lois_res.copy()
    if hide_zero_scores and "bm25_score" in show_laws.columns:
        show_laws = show_laws[show_laws["bm25_score"] > 0].copy()
    if not show_laws.empty:
        for date_col in ["Date initiale", "Date de promulgation"]:
            if date_col in show_laws.columns:
                show_laws[date_col] = format_date_yyyy_mm_dd(show_laws[date_col])
        laws_display = show_laws.rename(columns={"hybrid_score": "score de pertinence"}).copy()
        law_column_order = [
            "type_decision",
            "Titre",
            "Description",
            "Numéro de la loi",
            "Thèmes",
            "Date initiale",
            "Date de promulgation",
            "État du dossier",
            "URL du dossier",
            "score de pertinence",
        ]
        laws_display = laws_display[[c for c in law_column_order if c in laws_display.columns]]
        if "type_decision" in laws_display.columns:
            laws_display["type_decision"] = laws_display["type_decision"].astype(str).str.upper()
        st.dataframe(laws_display, use_container_width=True)
        selected_lois = show_laws.copy()
    else:
        selected_lois = pd.DataFrame()
        st.info("Aucune décision publique trouvée.")

st.markdown("### Chronologie activités & décisions publiques")
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
            name="Décisions publiques",
            marker=dict(size=10, symbol="square", color=[color_map[t] for t in laws["theme"]]),
            customdata=np.stack(
                [
                    laws.get("Titre", pd.Series([""] * len(laws))).astype(str).to_numpy(),
                    laws.get("Description", pd.Series([""] * len(laws))).astype(str).to_numpy(),
                    laws.get("Thèmes", pd.Series([""] * len(laws))).astype(str).to_numpy(),
                    laws.get("Date de promulgation", pd.Series([""] * len(laws))).astype(str).to_numpy(),
                    laws.get("bm25_score", pd.Series([0.0] * len(laws))).round(3).astype(str).to_numpy(),
                ],
                axis=1,
            ),
            hovertemplate="<b>Décision publique</b><br>Titre: %{customdata[0]}<br>Description: %{customdata[1]}<br>Thèmes: %{customdata[2]}<br>Promulgation: %{customdata[3]}<br>Score BM25: %{customdata[4]}<extra></extra>",
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
            customdata=np.stack(
                [
                    acts.get("beneficiaires", acts.get("denomination", pd.Series([""] * len(acts)))).fillna("").astype(str).to_numpy(),
                    acts.get("objet_activite", pd.Series([""] * len(acts))).astype(str).to_numpy(),
                    acts.get("domaines", pd.Series([""] * len(acts))).astype(str).to_numpy(),
                    pd.to_numeric(acts.get("budget_utilise", 0), errors="coerce").fillna(0).round(0).astype(int).astype(str).to_numpy(),
                ],
                axis=1,
            ),
            hovertemplate="<b>Activité</b><br>Bénéficiaire(s): %{customdata[0]}<br>Objet: %{customdata[1]}<br>Domaines: %{customdata[2]}<br>Budget utilisé: %{customdata[3]}<extra></extra>",
        )
    )
figt.update_layout(height=900, xaxis_title="Décisions publiques ← | → Activités", yaxis_title="Date")
st.plotly_chart(figt, use_container_width=True)

if ENABLE_LLM:
    st.markdown("### Payload LLM")
    payload = build_llm_payload(current_query, analytics)
    st.json(payload, expanded=False)
    if st.button("Générer la synthèse test"):
        from src.llm_summarize import summarize_activites

        st.session_state.test_synth = summarize_activites(payload)
    if st.session_state.get("test_synth"):
        st.text_area("Synthèse", st.session_state.test_synth, height=400)
