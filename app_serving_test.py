from __future__ import annotations

import io
import json
import re
import zlib
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from src.llm_summarize import summarize_activites

st.set_page_config(page_title="Test budgets bénéficiaires", layout="wide")
st.title("Test artefacts bénéficiaires + budgets, sans FAISS")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _norm_id_series(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .mask(lambda x: x.str.lower().isin(["", "nan", "none", "<na>"]))
    )


def _join_unique(values, max_items: int = 80) -> str:
    out: list[str] = []
    seen: set[str] = set()
    for value in pd.Series(values).dropna().astype(str):
        value = value.strip()
        if not value or value.lower() in {"nan", "none", "<na>"}:
            continue
        if value not in seen:
            seen.add(value)
            out.append(value)
        if len(out) >= max_items:
            break
    if len(seen) > max_items:
        out.append(f"… (+{len(seen) - max_items} autres)")
    return "; ".join(out)


def to_int64_safe(series: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .round(0)
        .astype("Int64")
    )


def _clean_money_series(s: pd.Series) -> pd.Series:
    """Convertit des montants FR/EN en numérique."""
    x = s.astype("string").str.strip()
    x = x.str.replace("\u00a0", "", regex=False)
    x = x.str.replace(" ", "", regex=False)
    x = x.str.replace("€", "", regex=False)
    # Si virgule décimale, conversion simple. Les montants sont normalement entiers.
    x = x.str.replace(",", ".", regex=False)
    return pd.to_numeric(x, errors="coerce")


def read_table(path_or_file, *, default_sep: str = ";") -> pd.DataFrame:
    name = getattr(path_or_file, "name", str(path_or_file)).lower()
    if name.endswith(".parquet"):
        return pd.read_parquet(path_or_file)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(path_or_file, dtype=str)
    if name.endswith(".csv"):
        return pd.read_csv(path_or_file, sep=default_sep, dtype=str, encoding="utf-8-sig")
    raise ValueError(f"Format non supporté: {name}")


def load_from_candidates(label: str, candidates: list[str], key: str, sep: str = ";") -> Optional[pd.DataFrame]:
    uploaded = st.sidebar.file_uploader(label, type=["csv", "xlsx", "xls", "parquet"], key=key)
    if uploaded is not None:
        return read_table(uploaded, default_sep=sep)
    for c in candidates:
        p = Path(c)
        if p.exists():
            return read_table(p, default_sep=sep)
    return None


def ensure_activite_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "activite_id" not in out.columns:
        unnamed = [c for c in out.columns if str(c).startswith("Unnamed")]
        if unnamed:
            out = out.rename(columns={unnamed[0]: "activite_id"})
    if "activite_id" not in out.columns:
        out = out.reset_index().rename(columns={"index": "activite_id"})
    out["activite_id"] = _norm_id_series(out["activite_id"])
    return out


def normalise_observations(df_observations: pd.DataFrame) -> pd.DataFrame:
    required = {"activite_id", "action_representation_interet_id"}
    missing = required - set(df_observations.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans observations: {sorted(missing)}")
    obs = df_observations[["activite_id", "action_representation_interet_id"]].copy()
    obs["activite_id"] = _norm_id_series(obs["activite_id"])
    obs["action_representation_interet_id"] = _norm_id_series(obs["action_representation_interet_id"])
    return obs.dropna(subset=["activite_id", "action_representation_interet_id"]).drop_duplicates()


def normalise_beneficiaires(df_beneficiaires: pd.DataFrame) -> pd.DataFrame:
    required = {"action_representation_interet_id", "beneficiaire_action_menee"}
    missing = required - set(df_beneficiaires.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans bénéficiaires: {sorted(missing)}")
    benef = df_beneficiaires[["action_representation_interet_id", "beneficiaire_action_menee"]].copy()
    benef["action_representation_interet_id"] = _norm_id_series(benef["action_representation_interet_id"])
    benef["beneficiaire"] = benef["beneficiaire_action_menee"].astype("string").str.strip()
    return benef.dropna(subset=["action_representation_interet_id", "beneficiaire"]).drop_duplicates(
        subset=["action_representation_interet_id", "beneficiaire"]
    )[["action_representation_interet_id", "beneficiaire"]]


def build_action_beneficiaire_mapping(df_observations: pd.DataFrame, df_beneficiaires: pd.DataFrame) -> pd.DataFrame:
    obs = normalise_observations(df_observations)
    benef = normalise_beneficiaires(df_beneficiaires)
    mapping = obs.merge(benef, on="action_representation_interet_id", how="inner")
    return mapping[["activite_id", "action_representation_interet_id", "beneficiaire"]].drop_duplicates()


# -----------------------------------------------------------------------------
# Budget activité depuis 15_exercices
# -----------------------------------------------------------------------------

def detect_budget_columns(df_exercices: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
    """Retourne (id_col, budget_col). Gère plusieurs exports possibles."""
    cols = list(df_exercices.columns)
    lower = {str(c).strip().lower(): c for c in cols}

    id_candidates = [
        "activite_id",
        "action_representation_interet_id",
        "representants_id",
    ]
    id_col = next((lower[c] for c in id_candidates if c in lower), None)

    budget_candidates_exact = [
        "budget_activite",
        "budget_moyen_activite",
        "budget_total",
        "montant_budget",
        "budget",
        "montant",
    ]
    budget_col = next((lower[c] for c in budget_candidates_exact if c in lower), None)

    if budget_col is None:
        # Recherche souple sur colonnes contenant budget ou montant.
        fuzzy = [c for c in cols if re.search(r"budget|montant", str(c), flags=re.I)]
        budget_col = fuzzy[0] if fuzzy else None

    return id_col, budget_col


def build_budget_by_activity_from_exercices_and_objets(
    df_exercices: pd.DataFrame,
    df_objets_activites: pd.DataFrame,
) -> pd.DataFrame:
    ex = df_exercices.copy()
    obj = df_objets_activites.copy()

    if "exercices_id" not in ex.columns:
        raise ValueError(f"15_exercices doit contenir exercices_id. Colonnes: {ex.columns.tolist()}")

    if not {"exercices_id", "activite_id"}.issubset(obj.columns):
        raise ValueError(
            f"8_objets_activites doit contenir exercices_id et activite_id. Colonnes: {obj.columns.tolist()}"
        )

    ex["exercices_id"] = _norm_id_series(ex["exercices_id"])
    obj["exercices_id"] = _norm_id_series(obj["exercices_id"])
    obj["activite_id"] = _norm_id_series(obj["activite_id"])

    # Budget exercice : on privilégie les bornes HATVP lorsqu’elles existent.
    # montant_depense est parfois un libellé; montant_depense_inf/sup donnent une valeur exploitable.
    if {"montant_depense_inf", "montant_depense_sup"}.issubset(ex.columns):
        ex["montant_depense_inf"] = _clean_money_series(ex["montant_depense_inf"])
        ex["montant_depense_sup"] = _clean_money_series(ex["montant_depense_sup"])
        ex["_budget_exercice"] = (
            ex["montant_depense_inf"].fillna(0)
            + ex["montant_depense_sup"].fillna(0)
        ) / 2
        budget_col = "_budget_exercice"
    elif "montant_depense" in ex.columns:
        ex["_budget_exercice"] = _clean_money_series(ex["montant_depense"])
        budget_col = "_budget_exercice"
    else:
        budget_col = next(
            (
                c for c in [
                    "budget_total",
                    "budget",
                    "montant",
                    "montant_budget",
                    "budget_lobbying",
                    "budget_depenses",
                    "montant_depenses",
                ]
                if c in ex.columns
            ),
            None,
        )
        if budget_col is None:
            raise ValueError(f"Aucune colonne budget trouvée dans 15_exercices. Colonnes: {ex.columns.tolist()}")
        ex[budget_col] = _clean_money_series(ex[budget_col])

    nb_activites = (
        obj.dropna(subset=["exercices_id", "activite_id"])
        .groupby("exercices_id", dropna=False)
        .agg(nb_activites_exercice_recalcule=("activite_id", "nunique"))
        .reset_index()
    )

    ex = ex.merge(nb_activites, on="exercices_id", how="left")

    ex["budget_activite"] = (
        ex[budget_col].fillna(0)
        / ex["nb_activites_exercice_recalcule"]
    )

    # 🔥 nettoyage
    ex["budget_activite"] = (
        ex["budget_activite"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    budget_by_activity = obj.merge(
        ex[["exercices_id", "budget_activite", "nb_activites_exercice_recalcule"]],
        on="exercices_id",
        how="left",
    )

    budget_by_activity = (
        budget_by_activity
        .groupby("activite_id", dropna=False)
        .agg(
            budget_activite=("budget_activite", "sum"),
            nb_exercices_activite=("exercices_id", "nunique"),
            nb_activites_exercice_recalcule=("nb_activites_exercice_recalcule", "max"),
        )
        .reset_index()
    )

    return budget_by_activity


def fallback_budget_by_activity_from_acts(df_acts_full: pd.DataFrame) -> pd.DataFrame:
    acts = ensure_activite_id(df_acts_full)
    if "budget_moyen_activite" in acts.columns:
        budget_col = "budget_moyen_activite"
    elif "budget_total" in acts.columns:
        budget_col = "budget_total"
    else:
        raise ValueError("Aucune colonne budget_moyen_activite ou budget_total dans la base activités complète.")
    acts[budget_col] = _clean_money_series(acts[budget_col])
    return acts[["activite_id", budget_col]].rename(columns={budget_col: "budget_activite"}).drop_duplicates("activite_id")


# -----------------------------------------------------------------------------
# Artefacts cibles
# -----------------------------------------------------------------------------

def build_beneficiaires_activites_globales(
    df_observations: pd.DataFrame,
    df_beneficiaires: pd.DataFrame,
    df_budget_by_activity: pd.DataFrame,
    df_acts_full: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    mapping = build_action_beneficiaire_mapping(df_observations, df_beneficiaires)
    budget = df_budget_by_activity.copy()
    budget["activite_id"] = _norm_id_series(budget["activite_id"])
    budget["budget_activite"] = pd.to_numeric(budget["budget_activite"], errors="coerce")

    detail = mapping.merge(budget, on="activite_id", how="left")

    if df_acts_full is not None and not df_acts_full.empty:
        acts = ensure_activite_id(df_acts_full)
        keep = [
            c for c in [
                "activite_id",
                "denomination",
                "objet_activite",
                "domaines",
                "label_categorie_organisation",
                "date_publication_activite",
            ] if c in acts.columns
        ]
        detail = detail.merge(acts[keep].drop_duplicates("activite_id"), on="activite_id", how="left")

    return detail.drop_duplicates()


def build_beneficiaire_global_stats(df_detail_global: pd.DataFrame) -> pd.DataFrame:
    out = (
        df_detail_global.groupby("beneficiaire", dropna=False)
        .agg(
            nb_activites_total_beneficiaire=("activite_id", "nunique"),
            nb_actions_total_beneficiaire=("action_representation_interet_id", "nunique"),
            budget_total_beneficiaire=("budget_activite", "sum"),
        )
        .reset_index()
    )
    for col in ["nb_activites_total_beneficiaire", "nb_actions_total_beneficiaire", "budget_total_beneficiaire"]:
        out[col] = to_int64_safe(out[col])
    return out.sort_values("budget_total_beneficiaire", ascending=False)


def add_beneficiaire_column_to_selection(
    selected_df: pd.DataFrame,
    df_observations: pd.DataFrame,
    df_beneficiaires: pd.DataFrame,
) -> pd.DataFrame:
    selected = ensure_activite_id(selected_df)
    mapping = build_action_beneficiaire_mapping(df_observations, df_beneficiaires)
    mapping = mapping[mapping["activite_id"].isin(set(selected["activite_id"].dropna()))]
    by_activity = (
        mapping.groupby("activite_id", dropna=False)
        .agg(
            beneficiaire=("beneficiaire", lambda s: _join_unique(s, max_items=80)),
            nb_beneficiaires=("beneficiaire", "nunique"),
        )
        .reset_index()
    )
    out = selected.merge(by_activity, on="activite_id", how="left")
    if "denomination" in out.columns:
        out["beneficiaire"] = out["beneficiaire"].fillna(out["denomination"])
    return out


def build_selected_beneficiaire_budget(selected_enriched: pd.DataFrame, global_stats: pd.DataFrame) -> pd.DataFrame:
    df = selected_enriched.copy()
    if "budget_moyen_activite" in df.columns:
        df["budget_moyen_activite"] = _clean_money_series(df["budget_moyen_activite"])
    else:
        df["budget_moyen_activite"] = 0.0

    agg = {
        "denomination": ("denomination", lambda s: _join_unique(s, max_items=50)) if "denomination" in df.columns else ("beneficiaire", "count"),
        "nb_activites_matching": ("activite_id", "nunique"),
        "budget_estime_recherche": ("budget_moyen_activite", "sum"),
    }
    if "label_categorie_organisation" in df.columns:
        agg["label_categorie_organisation"] = ("label_categorie_organisation", lambda s: _join_unique(s, max_items=20))

    out = df.groupby("beneficiaire", dropna=False).agg(**agg).reset_index()
    if "denomination" in out.columns and out["denomination"].dtype != object:
        out = out.drop(columns=["denomination"])
    out = out.merge(global_stats, on="beneficiaire", how="left") if not global_stats.empty else out

    for col in [
        "nb_activites_matching",
        "budget_estime_recherche",
        "nb_activites_total_beneficiaire",
        "nb_actions_total_beneficiaire",
        "budget_total_beneficiaire",
    ]:
        if col in out.columns:
            out[col] = to_int64_safe(out[col])
    return out.sort_values("budget_estime_recherche", ascending=False)


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, sep=";", encoding="utf-8-sig").encode("utf-8-sig")


def df_to_parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


# -----------------------------------------------------------------------------
# UI loading
# -----------------------------------------------------------------------------
st.sidebar.header("Fichiers")
st.sidebar.caption("Tous les calculs se font ici, sans FAISS et sans build_artifacts.")

selected_raw = load_from_candidates(
    "Activités sélectionnées CSV ;",
    ["/mnt/data/activites.csv", "activites.csv", "data/activites.csv"],
    "selected",
    sep=";",
)
objets_raw = load_from_candidates(
    "8_objets_activites",
    ["data/8_objets_activites.xlsx", "/mnt/data/8_objets_activites.xlsx"],
    "objets",
)
obs_raw = load_from_candidates(
    "14_observations / df_observations",
    ["/mnt/data/14_observations.xlsx", "data/14_observations.xlsx", "df_observations.parquet", "artifacts/df_observations.parquet"],
    "obs",
)
benef_raw = load_from_candidates(
    "11_beneficiaires / df_beneficiaires",
    ["/mnt/data/11_beneficiaires.xlsx", "data/11_beneficiaires.xlsx", "df_beneficiaires.parquet", "artifacts/df_beneficiaires.parquet"],
    "benef",
)
exercices_raw = load_from_candidates(
    "15_exercices, recommandé pour budget activité",
    ["/mnt/data/15_exercices.xlsx", "data/15_exercices.xlsx"],
    "exercices",
)
acts_full_raw = load_from_candidates(
    "Base activités complète/minifiée, optionnel",
    ["/mnt/data/df_activites_min.parquet", "artifacts/df_activites_min.parquet", "df_activites_min.parquet"],
    "acts_full",
)

if selected_raw is None or obs_raw is None or benef_raw is None:
    st.warning("Chargez au minimum activités sélectionnées, observations et bénéficiaires.")
    st.stop()

# -----------------------------------------------------------------------------
# Préparation budget activité
# -----------------------------------------------------------------------------
st.markdown("## 1. Construction du budget par activité")

budget_by_activity = None
budget_source = None

try:
    if exercices_raw is not None and not exercices_raw.empty:
        with st.expander("Colonnes détectées dans 15_exercices", expanded=False):
            st.write(exercices_raw.columns.tolist())
        detected_id, detected_budget = detect_budget_columns(exercices_raw)
        col1, col2 = st.columns(2)
        id_options = [""] + list(exercices_raw.columns)
        budget_options = [""] + list(exercices_raw.columns)
        id_choice = col1.selectbox(
            "Colonne identifiant 15_exercices",
            id_options,
            index=(id_options.index(detected_id) if detected_id in id_options else 0),
        )
        budget_choice = col2.selectbox(
            "Colonne budget/montant 15_exercices",
            budget_options,
            index=(budget_options.index(detected_budget) if detected_budget in budget_options else 0),
        )
        budget_by_activity = build_budget_by_activity_from_exercices_and_objets(
            exercices_raw,
            objets_raw,
        )
        budget_source = "15_exercices"
    elif acts_full_raw is not None and not acts_full_raw.empty:
        budget_by_activity = fallback_budget_by_activity_from_acts(acts_full_raw)
        budget_source = "base activités complète/minifiée"
    else:
        st.error("Il faut charger 15_exercices ou une base activités complète avec budget_moyen_activite.")
        st.stop()
except Exception as e:
    st.error(f"Erreur budget par activité: {e}")
    st.stop()

st.success(f"Budget par activité construit depuis : {budget_source}")
c1, c2, c3 = st.columns(3)
c1.metric("Activités avec budget", int(budget_by_activity["activite_id"].nunique()))
c2.metric("Budget total activité", int(pd.to_numeric(budget_by_activity["budget_activite"], errors="coerce").replace([np.inf, -np.inf], 0).fillna(0).sum()))
c3.metric("Budget moyen activité", int(pd.to_numeric(budget_by_activity["budget_activite"], errors="coerce").fillna(0).mean()))
st.dataframe(budget_by_activity.head(30), use_container_width=True)

# -----------------------------------------------------------------------------
# Artefact global bénéficiaires x activités
# -----------------------------------------------------------------------------
st.markdown("## 2. Artefact global bénéficiaires × activités")
try:
    detail_global = build_beneficiaires_activites_globales(obs_raw, benef_raw, budget_by_activity, acts_full_raw)
    global_stats = build_beneficiaire_global_stats(detail_global)
except Exception as e:
    st.error(f"Erreur artefact global bénéficiaires: {e}")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Lignes détail", len(detail_global))
c2.metric("Bénéficiaires", int(detail_global["beneficiaire"].nunique()))
c3.metric("Activités mappées", int(detail_global["activite_id"].nunique()))
c4.metric("Actions mappées", int(detail_global["action_representation_interet_id"].nunique()))

with st.expander("Debug détail global", expanded=False):
    st.dataframe(detail_global.head(50), use_container_width=True)
    st.write("Colonnes détail global", detail_global.columns.tolist())

st.markdown("### Stats globales par bénéficiaire")
st.dataframe(global_stats.head(100), use_container_width=True)

# Downloads
col_dl1, col_dl2, col_dl3, col_dl4 = st.columns(4)
col_dl1.download_button(
    "Télécharger détail global CSV",
    data=df_to_csv_bytes(detail_global),
    file_name="df_beneficiaires_activites_globales.csv",
    mime="text/csv",
)
col_dl2.download_button(
    "Télécharger stats globales CSV",
    data=df_to_csv_bytes(global_stats),
    file_name="df_beneficiaires_global_stats.csv",
    mime="text/csv",
)
col_dl3.download_button(
    "Télécharger détail global parquet",
    data=df_to_parquet_bytes(detail_global),
    file_name="df_beneficiaires_activites_globales.parquet",
    mime="application/octet-stream",
)
col_dl4.download_button(
    "Télécharger stats globales parquet",
    data=df_to_parquet_bytes(global_stats),
    file_name="df_beneficiaires_global_stats.parquet",
    mime="application/octet-stream",
)

# -----------------------------------------------------------------------------
# Sélection enrichie et analyse bénéficiaire sur recherche
# -----------------------------------------------------------------------------
st.markdown("## 3. Test sur les activités sélectionnées")
try:
    selected_enriched = add_beneficiaire_column_to_selection(selected_raw, obs_raw, benef_raw)
    selected_budget = build_selected_beneficiaire_budget(selected_enriched, global_stats)
except Exception as e:
    st.error(f"Erreur sélection enrichie: {e}")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Activités sélectionnées", len(selected_enriched))
c2.metric("Bénéficiaires sélection", int(selected_enriched["beneficiaire"].nunique()))
c3.metric("Sans bénéficiaire déclaré", int(selected_enriched.get("nb_beneficiaires", pd.Series([pd.NA]*len(selected_enriched))).isna().sum()))

with st.expander("Activités sélectionnées enrichies", expanded=True):
    priority = [
        "activite_id", "objet_activite", "denomination", "beneficiaire", "nb_beneficiaires",
        "date_publication_activite", "budget_moyen_activite", "budget_total", "domaines", "label_categorie_organisation"
    ]
    st.dataframe(selected_enriched[[c for c in priority if c in selected_enriched.columns] + [c for c in selected_enriched.columns if c not in priority]].head(200), use_container_width=True)

st.markdown("### Analyse budget par bénéficiaire sur la sélection + stats globales")
st.dataframe(selected_budget, use_container_width=True)

st.download_button(
    "Télécharger activités sélectionnées enrichies CSV",
    data=df_to_csv_bytes(selected_enriched),
    file_name="activites_selectionnees_enrichies_beneficiaires.csv",
    mime="text/csv",
)
st.download_button(
    "Télécharger analyse sélection bénéficiaires CSV",
    data=df_to_csv_bytes(selected_budget),
    file_name="analyse_selection_beneficiaires.csv",
    mime="text/csv",
)



# -----------------------------------------------------------------------------
# 4. Tests applicatifs complets : matrice, frise, payload LLM
# -----------------------------------------------------------------------------

def first_theme(s):
    s = "" if pd.isna(s) else str(s)
    parts = [p.strip() for p in re.split(r"[;,/|]", s) if p.strip()]
    return parts[0] if parts else "Thème inconnu"


def stable_jitter(s: str, scale=0.25) -> float:
    h = zlib.crc32(s.encode("utf-8")) % 10_000
    return (h / 10_000 - 0.5) * 2 * scale


def build_llm_payload(selected_enriched: pd.DataFrame, global_stats: pd.DataFrame, requete: str) -> dict:
    df_llm = selected_enriched.copy()
    for col in ["budget_moyen_activite", "budget_total", "nb_activites_total", "hybrid_score"]:
        if col in df_llm.columns:
            df_llm[col] = _clean_money_series(df_llm[col]) if df_llm[col].dtype == object else pd.to_numeric(df_llm[col], errors="coerce")

    df_llm["domaines"] = df_llm.get("domaines", "").fillna("").astype(str)
    df_llm["domaines_list"] = df_llm["domaines"].apply(
        lambda s: [d.strip() for d in s.split(";") if d.strip()] if s.strip() else ["Non renseigné"]
    )
    df_llm["nb_domaines"] = df_llm["domaines_list"].apply(lambda x: max(len(x), 1))
    df_llm["budget_reparti"] = df_llm.get("budget_moyen_activite", pd.Series([0] * len(df_llm))).fillna(0.0) / df_llm["nb_domaines"]

    exploded = df_llm.explode("domaines_list").rename(columns={"domaines_list": "domaine"})
    exploded["domaine"] = exploded["domaine"].fillna("Non renseigné").astype(str)

    benef_llm = (
        exploded.groupby("beneficiaire", dropna=False)
        .agg(
            organisations_declarantes=("denomination", lambda s: list(pd.Series(s).dropna().astype(str).unique())[:20]),
            budget_cumule_recherche=("budget_reparti", "sum"),
            nb_activites_matching=("activite_id", "nunique"),
        )
        .reset_index()
    )
    if global_stats is not None and not global_stats.empty:
        benef_llm = benef_llm.merge(global_stats, on="beneficiaire", how="left")

    budget_total_recherche = float(pd.to_numeric(benef_llm["budget_cumule_recherche"], errors="coerce").fillna(0).sum()) if not benef_llm.empty else 0.0
    benef_llm["part_budget_recherche_pct"] = np.where(
        budget_total_recherche > 0,
        100 * pd.to_numeric(benef_llm["budget_cumule_recherche"], errors="coerce").fillna(0) / budget_total_recherche,
        0.0,
    )
    for col in ["budget_cumule_recherche", "nb_activites_matching", "nb_activites_total_beneficiaire", "nb_actions_total_beneficiaire", "budget_total_beneficiaire"]:
        if col in benef_llm.columns:
            benef_llm[col] = to_int64_safe(benef_llm[col])
    benef_llm["part_budget_recherche_pct"] = benef_llm["part_budget_recherche_pct"].round(1)
    benef_llm = benef_llm.sort_values("budget_cumule_recherche", ascending=False)

    benef_theme_llm = (
        exploded.groupby(["beneficiaire", "domaine"], dropna=False)
        .agg(
            budget_cumule_recherche_beneficiaire_domaine=("budget_reparti", "sum"),
            nb_activites_matching_beneficiaire_domaine=("activite_id", "nunique"),
            organisations_declarantes=("denomination", lambda s: list(pd.Series(s).dropna().astype(str).unique())[:12]),
            objets_activite=("objet_activite", lambda s: list(pd.Series(s).dropna().astype(str).unique())[:12]),
        )
        .reset_index()
    )
    benef_theme_llm = benef_theme_llm.merge(
        benef_llm[["beneficiaire", "budget_cumule_recherche"]],
        on="beneficiaire",
        how="left",
    )
    denom = pd.to_numeric(benef_theme_llm["budget_cumule_recherche"], errors="coerce").fillna(0)
    benef_theme_llm["part_du_budget_recherche_beneficiaire_pct"] = np.where(
        denom > 0,
        100 * pd.to_numeric(benef_theme_llm["budget_cumule_recherche_beneficiaire_domaine"], errors="coerce").fillna(0) / denom,
        0.0,
    )
    benef_theme_llm["part_du_budget_total_recherche_pct"] = np.where(
        budget_total_recherche > 0,
        100 * pd.to_numeric(benef_theme_llm["budget_cumule_recherche_beneficiaire_domaine"], errors="coerce").fillna(0) / budget_total_recherche,
        0.0,
    )
    for col in ["budget_cumule_recherche_beneficiaire_domaine", "nb_activites_matching_beneficiaire_domaine"]:
        benef_theme_llm[col] = to_int64_safe(benef_theme_llm[col])
    benef_theme_llm["part_du_budget_recherche_beneficiaire_pct"] = benef_theme_llm["part_du_budget_recherche_beneficiaire_pct"].round(1)
    benef_theme_llm["part_du_budget_total_recherche_pct"] = benef_theme_llm["part_du_budget_total_recherche_pct"].round(1)
    benef_theme_llm = benef_theme_llm.sort_values(["budget_cumule_recherche_beneficiaire_domaine", "nb_activites_matching_beneficiaire_domaine"], ascending=False)

    theme_llm = (
        benef_theme_llm.groupby("domaine", dropna=False)
        .agg(
            budget_cumule_recherche_domaine=("budget_cumule_recherche_beneficiaire_domaine", "sum"),
            nb_beneficiaires=("beneficiaire", "nunique"),
            nb_activites_matching=("nb_activites_matching_beneficiaire_domaine", "sum"),
        )
        .reset_index()
    )
    theme_llm["part_budget_recherche_pct"] = np.where(
        budget_total_recherche > 0,
        100 * pd.to_numeric(theme_llm["budget_cumule_recherche_domaine"], errors="coerce") / budget_total_recherche,
        0.0,
    )
    for col in ["budget_cumule_recherche_domaine", "nb_beneficiaires", "nb_activites_matching"]:
        theme_llm[col] = to_int64_safe(theme_llm[col])
    theme_llm["part_budget_recherche_pct"] = theme_llm["part_budget_recherche_pct"].round(1)
    theme_llm = theme_llm.sort_values("budget_cumule_recherche_domaine", ascending=False)

    activites_cols = [c for c in ["beneficiaire", "denomination", "domaines", "objet_activite"] if c in df_llm.columns]
    return {
        "contexte": {
            "requete": requete,
            "unite_analyse": "beneficiaire",
            "nb_activites_retenues": int(df_llm["activite_id"].nunique()) if "activite_id" in df_llm.columns else int(len(df_llm)),
            "budget_total_estime_recherche": round(budget_total_recherche),
        },
        "tableau_beneficiaires": benef_llm.to_dict(orient="records"),
        "tableau_beneficiaires_domaines": benef_theme_llm.to_dict(orient="records"),
        "tableau_domaines": theme_llm.to_dict(orient="records"),
        "activites_selectionnees": df_llm[activites_cols].to_dict(orient="records"),
    }


st.markdown("## 4. Matrice bénéficiaires × domaines")
min_budget_matrix = st.slider("Budget minimum par cellule de matrice", 0, 5000, 0, 100, key="matrix_min")
top_domains_matrix = st.slider("Nombre de domaines affichés dans la matrice", 3, 20, 12, 1, key="matrix_top")

df2 = selected_enriched.copy()
df2["domaines"] = df2.get("domaines", "").fillna("").astype(str)
df2["domaines_list"] = df2["domaines"].str.split(";")
df2["nb_domaines"] = df2["domaines_list"].apply(lambda xs: len([x.strip() for x in xs if str(x).strip()]) if isinstance(xs, list) else 0)
df2 = df2.explode("domaines_list")
df2["domaines_list"] = df2["domaines_list"].astype(str).str.strip()
df2 = df2[df2["domaines_list"] != ""]
df2["budget_moyen_activite"] = _clean_money_series(df2.get("budget_moyen_activite", pd.Series([0] * len(df2))))
df2["budget_reparti"] = np.where(df2["nb_domaines"] > 0, df2["budget_moyen_activite"].fillna(0) / df2["nb_domaines"], 0.0)

def _join_objets_html(s, max_items=30):
    seen = set(); out = []
    for x in s.dropna().astype(str):
        if x and x not in seen:
            seen.add(x); out.append(x)
        if len(out) >= max_items:
            break
    if len(seen) > max_items:
        out.append(f"... (+{len(seen)-max_items} autres)")
    return "<br>• " + "<br>• ".join(out) if out else ""

cell = (
    df2.groupby(["beneficiaire", "domaines_list"], as_index=False)
    .agg(
        budget_total=("budget_reparti", "sum"),
        nb_activites=("activite_id", "nunique"),
        objets=("objet_activite", lambda s: _join_objets_html(s, max_items=30)),
    )
)
cell = cell[cell["budget_total"] >= float(min_budget_matrix)].copy()
if not cell.empty:
    top_domains = cell.groupby("domaines_list")["budget_total"].sum().sort_values(ascending=False).head(top_domains_matrix).index
    cell = cell[cell["domaines_list"].isin(top_domains)].copy()

if cell.empty:
    st.warning("Aucune donnée après filtrage de la matrice.")
else:
    benef_order = cell.groupby("beneficiaire")["budget_total"].sum().sort_values(ascending=False).index.tolist()
    dom_order = cell.groupby("domaines_list")["budget_total"].sum().sort_values(ascending=False).index.tolist()
    cell["beneficiaire"] = pd.Categorical(cell["beneficiaire"], categories=benef_order, ordered=True)
    cell["domaines_list"] = pd.Categorical(cell["domaines_list"], categories=dom_order, ordered=True)
    cell = cell.sort_values(["beneficiaire", "domaines_list"])
    cell["hover"] = (
        "<b>Bénéficiaire :</b> " + cell["beneficiaire"].astype(str) +
        "<br><b>Domaine :</b> " + cell["domaines_list"].astype(str) +
        "<br><b>Budget recherche réparti :</b> " + cell["budget_total"].fillna(0).round(0).astype(int).astype(str) +
        "<br><b>Nb activités matching :</b> " + cell["nb_activites"].fillna(0).astype(int).astype(str) +
        "<br><b>Objets :</b> " + cell["objets"].astype(str)
    )
    max_val = float(cell["budget_total"].max()) if len(cell) else 1.0
    sizeref = 2.0 * max_val / (40 ** 2) if max_val > 0 else 1.0
    figm = go.Figure(data=go.Scatter(
        x=cell["domaines_list"].astype(str),
        y=cell["beneficiaire"].astype(str),
        mode="markers",
        marker=dict(size=cell["budget_total"], sizemode="area", sizeref=sizeref, sizemin=3, opacity=0.75),
        text=cell["hover"],
        hovertemplate="%{text}<extra></extra>",
    ))
    left_margin = min(620, max(180, 7 * max((len(str(x)) for x in benef_order), default=10)))
    figm.update_layout(
        title="Répartition du budget de lobbying par bénéficiaire et domaine",
        xaxis=dict(title="Domaines", categoryorder="array", categoryarray=dom_order, tickangle=45),
        yaxis=dict(title="Bénéficiaires", categoryorder="array", categoryarray=benef_order, autorange="reversed"),
        height=max(650, 28 * len(benef_order) + 240),
        margin=dict(l=left_margin, r=10, t=80, b=120),
    )
    st.plotly_chart(figm, use_container_width=True)

st.markdown("## 5. Chronologie activités & lois")
st.caption("Dans ce test, les lois ne sont affichées que si vous chargez un fichier déjà sélectionné. En prod, la sélection lois restera pilotée par la recherche.")
lois_test = load_from_candidates(
    "Lois sélectionnées pour frise, optionnel",
    ["/mnt/data/lois_selectionnees.csv", "lois_selectionnees.csv", "data/lois_selectionnees.csv"],
    "lois_selected_test",
    sep=";",
)
START_DATE = pd.Timestamp("2018-01-01")
acts = selected_enriched.copy()
acts["date_evt"] = pd.to_datetime(acts.get("date_publication_activite"), errors="coerce")
acts = acts.dropna(subset=["date_evt"])
acts = acts[acts["date_evt"] >= START_DATE]
acts["theme"] = acts.get("domaines", "").fillna("").apply(first_theme)
acts["label_full"] = acts.get("beneficiaire", acts.get("denomination", "")).astype(str)
m = _clean_money_series(acts.get("budget_moyen_activite", pd.Series([0] * len(acts))))
mmax = float(m.max()) if len(m) and pd.notna(m.max()) else 1.0
acts["x"] = (np.sqrt(m.fillna(0) + 1) / np.sqrt(mmax + 1) * 6.0) + 1.0
acts["x"] = acts["label_full"].apply(lambda s: stable_jitter(str(s), 0.25)) + acts["x"]

if lois_test is not None and not lois_test.empty:
    laws = lois_test.copy()
    date_col = "Date initiale" if "Date initiale" in laws.columns else ("date_evt" if "date_evt" in laws.columns else None)
    if date_col:
        laws["date_evt"] = pd.to_datetime(laws.get(date_col), errors="coerce", dayfirst=True)
        laws = laws.dropna(subset=["date_evt"])
        laws = laws[laws["date_evt"] >= START_DATE]
        laws["theme"] = laws.get("Thèmes", "").fillna("").apply(first_theme)
        laws["label_full"] = laws.get("Titre", "").astype(str)
        laws["x"] = -1.0 + laws["label_full"].apply(lambda s: stable_jitter(str(s), 0.15))
    else:
        laws = pd.DataFrame(columns=["x", "date_evt", "theme", "label_full"])
else:
    laws = pd.DataFrame(columns=["x", "date_evt", "theme", "label_full"])

themes = sorted(set(acts["theme"]).union(set(laws["theme"]))) if len(acts) or len(laws) else []
palette = px.colors.qualitative.Safe
color_map = {t: palette[i % len(palette)] for i, t in enumerate(themes)}
figt = go.Figure()
if len(laws):
    figt.add_trace(go.Scatter(
        x=laws["x"], y=laws["date_evt"], mode="markers", name="Lois",
        marker=dict(size=10, symbol="square", color=[color_map[t] for t in laws["theme"]]),
        customdata=np.stack([laws["label_full"].to_numpy(), laws["theme"].to_numpy()], axis=1),
        hovertemplate="<b>Loi</b><br>Titre: %{customdata[0]}<br>Thème: %{customdata[1]}<extra></extra>",
    ))
if len(acts):
    figt.add_trace(go.Scatter(
        x=acts["x"], y=acts["date_evt"], mode="markers", name="Activités de lobbying",
        marker=dict(size=9, symbol="circle", color=[color_map[t] for t in acts["theme"]]),
        customdata=np.stack([
            acts["label_full"].to_numpy(),
            acts["theme"].to_numpy(),
            acts.get("objet_activite", pd.Series([""] * len(acts))).astype(str).to_numpy(),
            acts.get("domaines", pd.Series([""] * len(acts))).astype(str).to_numpy(),
            acts.get("denomination", pd.Series([""] * len(acts))).astype(str).to_numpy(),
            _clean_money_series(acts.get("budget_moyen_activite", pd.Series([0] * len(acts)))).fillna(0).to_numpy(),
        ], axis=1),
        hovertemplate=(
            "<b>Activité de lobbying</b><br>"
            "Bénéficiaire: %{customdata[0]}<br>"
            "Organisation déclarante: %{customdata[4]}<br>"
            "Objet: %{customdata[2]}<br>"
            "Domaines: %{customdata[3]}<br>"
            "Budget moyen/activité: %{customdata[5]:.0f} €<extra></extra>"
        ),
    ))
figt.add_vline(x=0, line_width=1, opacity=0.3)
figt.update_layout(height=850, xaxis_title="Lois  ←   |   →  Bénéficiaires (budget_moyen_activite)", yaxis_title="Date", margin=dict(l=30, r=30, t=60, b=30))
st.plotly_chart(figt, use_container_width=True)

st.markdown("## 6. Payload et appel LLM")
requete_test = st.text_input("Requête test pour le payload LLM", value="")
info_llm = build_llm_payload(selected_enriched, global_stats, requete_test)
st.json(info_llm, expanded=False)

st.download_button(
    "Télécharger payload LLM JSON",
    data=json.dumps(info_llm, ensure_ascii=False, indent=2).encode("utf-8"),
    file_name="payload_llm_beneficiaires.json",
    mime="application/json",
)

if st.button("Tester l'appel LLM avec ce payload", type="primary"):
    if summarize_activites is None:
        st.error("Module summarize_activites introuvable.")
    else:
        with st.spinner("Synthèse…"):
            txt = summarize_activites(info_llm)
        st.text_area("Synthèse", txt, height=500)

st.warning("Le module llm_summarize.py actuel doit encore être renommé côté prompt : organisations → bénéficiaires. Ce test valide déjà la structure du payload.")


st.caption(
    "Une fois les résultats validés, les deux artefacts utiles pour l'app finale sont "
    "df_beneficiaires_activites_globales et df_beneficiaires_global_stats. "
    "Ils pourront être produits ensuite dans build_artifacts.py sans changer la logique applicative."
)
