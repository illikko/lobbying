from __future__ import annotations

from pathlib import Path
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
from src.config import ART
from src.io_artifacts import load_joblib, load_parquet
from src.llm_summarize import summarize_activites
from src.prep import (
    clean_columns,
    concat_sur_liste_colonnes,
    minify_activites,
    minify_lois,
    prepare_from_raw,
)
from src.search_backend import bm25_search_activites
from src.textnorm import tokenize


st.set_page_config(page_title="Test local BM25 lobbying", layout="wide")
st.title("Test local BM25 uniquement")


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


def load_repo_table(parquet_name: str, raw_name: str | None = None) -> pd.DataFrame:
    try:
        return load_parquet(parquet_name)
    except FileNotFoundError:
        if raw_name is None:
            return pd.DataFrame()
        raw_path = Path("data/raw") / raw_name
        if raw_path.suffix.lower() in {".xlsx", ".xls"} and raw_path.exists():
            return pd.read_excel(raw_path, dtype=str)
        return pd.DataFrame()


def load_local_activities() -> pd.DataFrame:
    try:
        df = ensure_activite_id(load_parquet(ART.df_activites_min))
        df.index = df["activite_id"].astype(str)
        return df
    except FileNotFoundError:
        pass
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


def load_local_activities_from_raw() -> pd.DataFrame:
    raw_paths = {
        "xlsx_organisations": Path("data/raw/1_informations_generales.xlsx"),
        "xlsx_activites": Path("data/raw/8_objets_activites.xlsx"),
        "xlsx_exercices": Path("data/raw/15_exercices.xlsx"),
        "xlsx_domaines": Path("data/raw/7_domaines_intervention.xlsx"),
        "csv_ppl": Path("data/raw/ppl.csv"),
        "csv_promulguees": Path("data/raw/promulguees.csv"),
    }
    if not all(path.exists() for path in raw_paths.values()):
        raise FileNotFoundError(
            "Aucune base activités locale trouvée. Attendu: "
            "artifacts/df_activites_min.parquet ou reconstruction depuis data/raw/."
        )

    df_acts_full, _ = prepare_from_raw(
        xlsx_organisations=str(raw_paths["xlsx_organisations"]),
        xlsx_activites=str(raw_paths["xlsx_activites"]),
        xlsx_exercices=str(raw_paths["xlsx_exercices"]),
        xlsx_domaines=str(raw_paths["xlsx_domaines"]),
        csv_ppl=str(raw_paths["csv_ppl"]),
        csv_promulguees=str(raw_paths["csv_promulguees"]),
    )
    df = minify_activites(df_acts_full)
    df = df.reset_index()
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
        docs = (df_acts.get("objet_activite", "").fillna("").astype(str) + " " + df_acts.get("denomination", "").fillna("").astype(str) + " " + df_acts.get("domaines", "").fillna("").astype(str)).tolist()
        doc_ids = df_acts["activite_id"].astype(object).to_numpy()
        return build_bm25(doc_ids=doc_ids, doc_texts=docs)


def build_lois_bm25_bundle(df_lois: pd.DataFrame):
    docs = (df_lois.get("Titre", "").fillna("").astype(str) + " " + df_lois.get("Thèmes", "").fillna("").astype(str)).tolist()
    doc_ids = np.arange(len(df_lois), dtype=object)
    return build_bm25(doc_ids=doc_ids, doc_texts=docs)


def load_local_laws() -> pd.DataFrame:
    try:
        df_lois = load_parquet(ART.df_lois_min)
        return df_lois.reset_index(drop=True)
    except FileNotFoundError:
        pass

    ppl_path = Path("data/raw/ppl.csv")
    prom_path = Path("data/raw/promulguees.csv")
    if not ppl_path.exists() or not prom_path.exists():
        return pd.DataFrame()

    df_ppl = pd.read_csv(ppl_path, sep=";", encoding="latin-1")
    df_prom = pd.read_csv(prom_path, sep=";", encoding="latin-1")
    df_ppl = clean_columns(df_ppl)
    df_prom = clean_columns(df_prom)
    if "Date de dépôt" in df_ppl.columns:
        df_ppl = df_ppl.rename(columns={"Date de dépôt": "Date initiale"})

    df_lois = concat_sur_liste_colonnes(
        df_ppl,
        df_prom,
        cols=["Date initiale", "Date de promulgation", "Titre", "Numéro de la loi", "Thèmes", "État du dossier", "URL du dossier"],
    )
    df_lois["Date initiale"] = pd.to_datetime(df_lois["Date initiale"], errors="coerce", dayfirst=True)
    df_lois["Date de promulgation"] = pd.to_datetime(df_lois["Date de promulgation"], errors="coerce", dayfirst=True)
    df_lois = df_lois[df_lois["Date initiale"] >= pd.Timestamp("2018-01-01")]
    df_lois = minify_lois(df_lois).reset_index(drop=True)
    return df_lois


def search_lois(df_lois: pd.DataFrame, bm25_lois_bundle, query_text: str, topn_laws: int) -> pd.DataFrame:
    qtok = tokenize(query_text)
    if not qtok:
        return df_lois.iloc[0:0].copy()
    scores = np.asarray(bm25_lois_bundle.bm25.get_scores(qtok), dtype=float)
    idx = np.argsort(scores)[::-1][: int(topn_laws)]
    out = df_lois.iloc[idx].copy()
    out["bm25_score"] = scores[idx]
    return out.sort_values("bm25_score", ascending=False)


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
    try:
        df_acts = ensure_activite_id(load_parquet(ART.df_activites_min))
        df_acts.index = df_acts["activite_id"].astype(str)
    except FileNotFoundError:
        df_acts = load_local_activities_from_raw()
    df_infos = load_repo_table("df_informations_generales.parquet", "1_informations_generales.xlsx")
    df_acts = fill_missing_category_from_infos(df_acts, df_infos)
    df_lois = load_local_laws()
    return {
        "df_acts": df_acts,
        "df_lois": df_lois,
        "bm25_bundle": build_local_bm25_bundle(df_acts),
        "bm25_lois_bundle": build_lois_bm25_bundle(df_lois) if not df_lois.empty else None,
        "df_observations": load_repo_table("df_observations.parquet", "14_observations.xlsx"),
        "df_beneficiaires": load_repo_table("df_beneficiaires.parquet", "11_beneficiaires.xlsx"),
        "df_affiliations": load_repo_table("df_affiliations.parquet", "5_affiliations.xlsx"),
    }


STATE = load_local_state()
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

st.markdown("### Debug")
debug_detail = analytics.enriched.copy()
debug_display = debug_detail[[c for c in ["activite_id", "denomination", "objet_activite", "beneficiaire", "budget_utilise"] if c in debug_detail.columns]].head(20)
st.json(
    {
        "nb_activites_resultats": int(results["activite_id"].nunique()),
        "nb_activites_selectionnees": int(analytics.table_activites["activite_id"].nunique()),
        "nb_activites_avec_beneficiaire": int(debug_detail.loc[debug_detail["beneficiaire"].notna(), "activite_id"].nunique()),
        "nb_beneficiaires_distincts": int(debug_detail["beneficiaire"].dropna().nunique()),
    }
)
st.dataframe(debug_display, use_container_width=True)

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
        "<b>BÃ©nÃ©ficiaire :</b> " + matrix_table["beneficiaire"].astype(str)
        + "<br><b>Domaine :</b> " + matrix_table["domaine"].astype(str)
        + "<br><b>Budget rÃ©parti :</b> " + matrix_table["budget_estime_recherche"].round(0).astype(int).astype(str)
        + "<br><b>Nb activitÃ©s :</b> " + matrix_table["nb_activites_matching"].astype(str)
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

st.markdown("### Lois correspondant à la recherche")
if bm25_lois_bundle is None or df_lois.empty:
    selected_lois = pd.DataFrame()
    st.info("Aucune base lois locale disponible.")
else:
    topn_laws = st.slider("Nombre de lois", 5, 100, 20, 5)
    lois_res = search_lois(df_lois, bm25_lois_bundle, current_query, int(topn_laws))
    qtok = tokenize(current_query)
    all_law_scores = np.asarray(bm25_lois_bundle.bm25.get_scores(qtok), dtype=float) if qtok else np.asarray([], dtype=float)
    st.caption(
        f"Lois avec score BM25 strictement positif sur tout le corpus : {int((all_law_scores > 0).sum()) if len(all_law_scores) else 0}"
    )
    hide_zero_scores = st.checkbox("Masquer les lois à score nul", value=True)
    show_laws = lois_res.copy()
    if hide_zero_scores and "bm25_score" in show_laws.columns:
        show_laws = show_laws[show_laws["bm25_score"] > 0].copy()
    if not show_laws.empty:
        for date_col in ["Date initiale", "Date de promulgation"]:
            if date_col in show_laws.columns:
                show_laws[date_col] = format_date_yyyy_mm_dd(show_laws[date_col])
        st.dataframe(show_laws, use_container_width=True)
        selected_lois = show_laws.copy()
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
            customdata=np.stack(
                [
                    laws.get("Titre", pd.Series([""] * len(laws))).astype(str).to_numpy(),
                    laws.get("Thèmes", pd.Series([""] * len(laws))).astype(str).to_numpy(),
                    laws.get("Date de promulgation", pd.Series([""] * len(laws))).astype(str).to_numpy(),
                    laws.get("bm25_score", pd.Series([0.0] * len(laws))).round(3).astype(str).to_numpy(),
                ],
                axis=1,
            ),
            hovertemplate="<b>Loi</b><br>Titre: %{customdata[0]}<br>Thèmes: %{customdata[1]}<br>Promulgation: %{customdata[2]}<br>Score BM25: %{customdata[3]}<extra></extra>",
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
figt.update_layout(height=900, xaxis_title="Lois ← | → Activités", yaxis_title="Date")
st.plotly_chart(figt, use_container_width=True)

st.markdown("### Payload LLM")
payload = build_llm_payload(current_query, analytics)
st.json(payload, expanded=False)
if st.button("Générer la synthèse test"):
    st.session_state.test_synth = summarize_activites(payload)
if st.session_state.get("test_synth"):
    st.text_area("Synthèse", st.session_state.test_synth, height=400)
