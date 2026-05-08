from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import pandas as pd


def norm_id_series(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .mask(lambda x: x.str.lower().isin(["", "nan", "none", "<na>"]))
    )


def join_unique(values, max_items: int = 80) -> str:
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
        out.append(f"... (+{len(seen) - max_items} autres)")
    return "; ".join(out)


def to_int64_safe(series: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .round(0)
        .astype("Int64")
    )


def parse_numeric_values(value) -> list[float]:
    if pd.isna(value):
        return []
    values: list[float] = []
    for part in re.split(r"[;,|]", str(value)):
        part = part.strip().replace("\u00a0", " ").replace(" ", "").replace(",", ".")
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError:
            continue
    return values


def mean_distinct_numeric(values) -> float:
    nums: list[float] = []
    seen: set[float] = set()
    for value in values:
        for num in parse_numeric_values(value):
            key = round(float(num), 6)
            if key not in seen:
                seen.add(key)
                nums.append(float(num))
    return float(np.mean(nums)) if nums else np.nan


def clean_money_series(series: pd.Series) -> pd.Series:
    x = series.astype("string").str.strip()
    x = x.str.replace("\u00a0", "", regex=False)
    x = x.str.replace(" ", "", regex=False)
    x = x.str.replace("€", "", regex=False)
    x = x.str.replace(",", ".", regex=False)
    return pd.to_numeric(x, errors="coerce")


def ensure_activite_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "activite_id" not in out.columns:
        unnamed = [c for c in out.columns if str(c).startswith("Unnamed")]
        if unnamed:
            out = out.rename(columns={unnamed[0]: "activite_id"})
    if "activite_id" not in out.columns:
        out = out.reset_index().rename(columns={"index": "activite_id"})
    out["activite_id"] = norm_id_series(out["activite_id"])
    return out


def normalise_observations(df_observations: pd.DataFrame) -> pd.DataFrame:
    required = {"activite_id", "action_representation_interet_id"}
    missing = required - set(df_observations.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans observations: {sorted(missing)}")
    obs = df_observations[["activite_id", "action_representation_interet_id"]].copy()
    obs["activite_id"] = norm_id_series(obs["activite_id"])
    obs["action_representation_interet_id"] = norm_id_series(obs["action_representation_interet_id"])
    return obs.dropna(subset=["activite_id", "action_representation_interet_id"]).drop_duplicates()


def normalise_beneficiaires(df_beneficiaires: pd.DataFrame) -> pd.DataFrame:
    required = {"action_representation_interet_id", "beneficiaire_action_menee"}
    missing = required - set(df_beneficiaires.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans bénéficiaires: {sorted(missing)}")
    benef = df_beneficiaires[["action_representation_interet_id", "beneficiaire_action_menee"]].copy()
    benef["action_representation_interet_id"] = norm_id_series(benef["action_representation_interet_id"])
    benef["beneficiaire"] = benef["beneficiaire_action_menee"].astype("string").str.strip()
    benef["beneficiaire"] = benef["beneficiaire"].mask(
        benef["beneficiaire"].str.lower().isin(["", "nan", "none", "<na>"])
    )
    return benef.dropna(subset=["action_representation_interet_id", "beneficiaire"]).drop_duplicates(
        subset=["action_representation_interet_id", "beneficiaire"]
    )[["action_representation_interet_id", "beneficiaire"]]


def get_activity_budget_column(df: pd.DataFrame) -> str | None:
    for col in ("budget_activite", "budget_moyen_activite"):
        if col in df.columns:
            return col
    return None


def add_budget_used_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    budget_col = get_activity_budget_column(out)
    if budget_col is None:
        out["budget_utilise"] = 0.0
    else:
        out["budget_utilise"] = pd.to_numeric(out[budget_col], errors="coerce").fillna(0.0)
    return out


def enrich_activities_with_beneficiaries(
    selected_activities: pd.DataFrame,
    df_observations: pd.DataFrame,
    df_beneficiaires: pd.DataFrame,
) -> pd.DataFrame:
    selected = add_budget_used_column(ensure_activite_id(selected_activities))
    if selected.empty:
        return selected.assign(
            action_representation_interet_id=pd.Series(dtype="string"),
            beneficiaire=pd.Series(dtype="string"),
            nb_beneficiaires=pd.Series(dtype="Int64"),
            budget_beneficiaire=pd.Series(dtype="float64"),
        )

    obs = normalise_observations(df_observations)
    benef = normalise_beneficiaires(df_beneficiaires)

    mapping = obs.merge(benef, on="action_representation_interet_id", how="left")
    mapping = mapping.drop_duplicates(subset=["activite_id", "action_representation_interet_id", "beneficiaire"])
    nb_benef = (
        mapping.dropna(subset=["beneficiaire"])
        .groupby("activite_id", dropna=False)["beneficiaire"]
        .nunique()
        .rename("nb_beneficiaires")
        .reset_index()
    )

    selected_cols = list(selected.columns)
    enriched = selected.merge(mapping, on="activite_id", how="left")
    enriched = enriched.merge(nb_benef, on="activite_id", how="left")
    enriched["nb_beneficiaires"] = (
        pd.to_numeric(enriched["nb_beneficiaires"], errors="coerce")
        .fillna(0)
        .astype("Int64")
    )
    enriched["budget_beneficiaire"] = np.where(
        enriched["nb_beneficiaires"].fillna(0) > 0,
        enriched["budget_utilise"] / enriched["nb_beneficiaires"].astype(float),
        0.0,
    )

    # Preserve one row per selected activity when there is no beneficiary mapping.
    if "beneficiaire" not in enriched.columns:
        enriched["beneficiaire"] = pd.NA

    return enriched[selected_cols + [
        "action_representation_interet_id",
        "beneficiaire",
        "nb_beneficiaires",
        "budget_beneficiaire",
    ]]


def build_activity_table(enriched: pd.DataFrame) -> pd.DataFrame:
    if enriched.empty:
        return enriched.iloc[0:0].copy()

    acts = (
        enriched.groupby("activite_id", dropna=False)
        .agg(
            objet_activite=("objet_activite", "first") if "objet_activite" in enriched.columns else ("activite_id", "first"),
            denomination=("denomination", "first") if "denomination" in enriched.columns else ("activite_id", "first"),
            beneficiaires=("beneficiaire", lambda s: join_unique(s, max_items=80)),
            nb_beneficiaires=("nb_beneficiaires", "max"),
            date_publication_activite=("date_publication_activite", "first") if "date_publication_activite" in enriched.columns else ("activite_id", "first"),
            budget_utilise=("budget_utilise", "max"),
            domaines=("domaines", "first") if "domaines" in enriched.columns else ("activite_id", "first"),
            hybrid_score=("hybrid_score", "max") if "hybrid_score" in enriched.columns else ("activite_id", "count"),
        )
        .reset_index()
    )
    acts["beneficiaires"] = acts["beneficiaires"].replace("", pd.NA)
    if "hybrid_score" in acts.columns:
        acts = acts.sort_values("hybrid_score", ascending=False, kind="stable")
    return acts


def build_organisation_table(enriched: pd.DataFrame, affiliations_by_org: pd.DataFrame | None = None) -> pd.DataFrame:
    if enriched.empty or "denomination" not in enriched.columns:
        return pd.DataFrame()

    per_activity = enriched.drop_duplicates(subset=["activite_id"]).copy()
    agg = {
        "nb_activites_matching": ("activite_id", "nunique"),
        "budget_estime_recherche": ("budget_utilise", "sum"),
    }
    if "budget_total" in per_activity.columns:
        agg["budget_total"] = ("budget_total", "max")
    if "nb_activites_total" in per_activity.columns:
        agg["nb_activites_total"] = ("nb_activites_total", "max")
    if "nombre_salaries" in per_activity.columns:
        agg["nombre_salaries"] = ("nombre_salaries", mean_distinct_numeric)
    if "label_categorie_organisation" in per_activity.columns:
        agg["categorie"] = ("label_categorie_organisation", lambda s: join_unique(s, max_items=20))

    out = per_activity.groupby("denomination", dropna=False).agg(**agg).reset_index()
    if affiliations_by_org is not None and not affiliations_by_org.empty:
        out = out.merge(affiliations_by_org, on="denomination", how="left")
    for col in ["nb_activites_matching", "budget_estime_recherche", "budget_total", "nb_activites_total", "nombre_salaries", "nb_affiliations"]:
        if col in out.columns:
            out[col] = to_int64_safe(out[col])
    return out.sort_values("budget_estime_recherche", ascending=False)


def build_beneficiaire_table(enriched: pd.DataFrame) -> pd.DataFrame:
    if enriched.empty:
        return pd.DataFrame()
    detail = enriched.dropna(subset=["beneficiaire"]).copy()
    if detail.empty:
        return pd.DataFrame(columns=["beneficiaire", "nb_activites_matching", "budget_estime_recherche"])

    agg = {
        "nb_activites_matching": ("activite_id", "nunique"),
        "budget_estime_recherche": ("budget_beneficiaire", "sum"),
    }
    if "denomination" in detail.columns:
        agg["organisations_declarantes"] = ("denomination", lambda s: join_unique(s, max_items=30))
    if "domaines" in detail.columns:
        agg["domaines"] = ("domaines", lambda s: join_unique(s, max_items=30))

    out = detail.groupby("beneficiaire", dropna=False).agg(**agg).reset_index()
    out["budget_estime_recherche"] = to_int64_safe(out["budget_estime_recherche"])
    out["nb_activites_matching"] = to_int64_safe(out["nb_activites_matching"])
    return out.sort_values("budget_estime_recherche", ascending=False)


def explode_beneficiary_domains(enriched: pd.DataFrame) -> pd.DataFrame:
    detail = enriched.dropna(subset=["beneficiaire"]).copy()
    if detail.empty:
        return detail.assign(domaine=pd.Series(dtype="string"), budget_beneficiaire_domaine=pd.Series(dtype="float64"))

    detail["domaines"] = detail.get("domaines", "").fillna("").astype(str)
    detail["domaine"] = detail["domaines"].apply(
        lambda s: [d.strip() for d in s.split(";") if d.strip()] if s.strip() else ["Non renseigné"]
    )
    detail["nb_domaines"] = detail["domaine"].apply(lambda x: max(len(x), 1))
    detail = detail.explode("domaine")
    detail["budget_beneficiaire_domaine"] = detail["budget_beneficiaire"] / detail["nb_domaines"]
    return detail


def build_beneficiary_domain_matrix(enriched: pd.DataFrame) -> pd.DataFrame:
    detail = explode_beneficiary_domains(enriched)
    if detail.empty:
        return pd.DataFrame()
    out = (
        detail.groupby(["beneficiaire", "domaine"], dropna=False)
        .agg(
            budget_estime_recherche=("budget_beneficiaire_domaine", "sum"),
            nb_activites_matching=("activite_id", "nunique"),
            organisations_declarantes=("denomination", lambda s: join_unique(s, max_items=20)) if "denomination" in detail.columns else ("beneficiaire", "count"),
        )
        .reset_index()
    )
    out["budget_estime_recherche"] = to_int64_safe(out["budget_estime_recherche"])
    out["nb_activites_matching"] = to_int64_safe(out["nb_activites_matching"])
    return out.sort_values("budget_estime_recherche", ascending=False)


@dataclass
class AnalyticsBundle:
    enriched: pd.DataFrame
    table_activites: pd.DataFrame
    table_organisations: pd.DataFrame
    table_beneficiaires: pd.DataFrame
    matrix_beneficiaires_domaines: pd.DataFrame
    table_domaines: pd.DataFrame


def build_domain_table(enriched: pd.DataFrame) -> pd.DataFrame:
    detail = explode_beneficiary_domains(enriched)
    if detail.empty:
        return pd.DataFrame(columns=["domaine", "budget_estime_recherche", "nb_beneficiaires", "nb_activites_matching"])
    out = (
        detail.groupby("domaine", dropna=False)
        .agg(
            budget_estime_recherche=("budget_beneficiaire_domaine", "sum"),
            nb_beneficiaires=("beneficiaire", "nunique"),
            nb_activites_matching=("activite_id", "nunique"),
        )
        .reset_index()
    )
    out["budget_estime_recherche"] = to_int64_safe(out["budget_estime_recherche"])
    out["nb_beneficiaires"] = to_int64_safe(out["nb_beneficiaires"])
    out["nb_activites_matching"] = to_int64_safe(out["nb_activites_matching"])
    return out.sort_values("budget_estime_recherche", ascending=False)


def build_search_analytics(
    selected_activities: pd.DataFrame,
    df_observations: pd.DataFrame,
    df_beneficiaires: pd.DataFrame,
    affiliations_by_org: pd.DataFrame | None = None,
) -> AnalyticsBundle:
    enriched = enrich_activities_with_beneficiaries(
        selected_activities=selected_activities,
        df_observations=df_observations,
        df_beneficiaires=df_beneficiaires,
    )
    return AnalyticsBundle(
        enriched=enriched,
        table_activites=build_activity_table(enriched),
        table_organisations=build_organisation_table(enriched, affiliations_by_org=affiliations_by_org),
        table_beneficiaires=build_beneficiaire_table(enriched),
        matrix_beneficiaires_domaines=build_beneficiary_domain_matrix(enriched),
        table_domaines=build_domain_table(enriched),
    )


def build_llm_payload(
    requete: str,
    analytics: AnalyticsBundle,
) -> dict:
    budget_total = 0
    if not analytics.table_beneficiaires.empty and "budget_estime_recherche" in analytics.table_beneficiaires.columns:
        budget_total = int(
            pd.to_numeric(analytics.table_beneficiaires["budget_estime_recherche"], errors="coerce").fillna(0).sum()
        )
    activity_cols = [c for c in [
        "activite_id",
        "denomination",
        "objet_activite",
        "beneficiaires",
        "nb_beneficiaires",
        "domaines",
        "budget_utilise",
        "hybrid_score",
    ] if c in analytics.table_activites.columns]
    activites_selectionnees = analytics.table_activites[activity_cols].to_dict(orient="records")
    return {
        "contexte": {
            "requete": requete,
            "nb_activites_retenues": int(analytics.table_activites["activite_id"].nunique()) if not analytics.table_activites.empty else 0,
            "budget_total_estime_recherche": budget_total,
        },
        "tableau_activites": activites_selectionnees,
        "tableau_organisations": analytics.table_organisations.to_dict(orient="records"),
        "tableau_beneficiaires": analytics.table_beneficiaires.to_dict(orient="records"),
        "tableau_beneficiaires_domaines": analytics.matrix_beneficiaires_domaines.to_dict(orient="records"),
        "tableau_domaines": analytics.table_domaines.to_dict(orient="records"),
        "activites_selectionnees": activites_selectionnees,
    }


def validate_no_beneficiary_join_on_text_keys() -> None:
    sample_selected = pd.DataFrame(
        {
            "activite_id": ["1"],
            "denomination": ["Org A"],
            "objet_activite": ["Objet A"],
            "budget_moyen_activite": [100.0],
        }
    )
    sample_obs = pd.DataFrame({"activite_id": ["1"], "action_representation_interet_id": ["10"]})
    sample_benef = pd.DataFrame({"action_representation_interet_id": ["10"], "beneficiaire_action_menee": ["Benef A"]})

    left = enrich_activities_with_beneficiaries(sample_selected, sample_obs, sample_benef)
    sample_selected["denomination"] = "Org B"
    sample_selected["objet_activite"] = "Objet B"
    right = enrich_activities_with_beneficiaries(sample_selected, sample_obs, sample_benef)

    assert left["beneficiaire"].fillna("").tolist() == right["beneficiaire"].fillna("").tolist()


def validate_selected_ids_exist_in_observations_when_expected(enriched: pd.DataFrame, df_observations: pd.DataFrame) -> None:
    if enriched.empty:
        return
    selected_ids = set(enriched["activite_id"].dropna().astype(str))
    observed_ids = set(normalise_observations(df_observations)["activite_id"].dropna().astype(str))
    missing = sorted(selected_ids - observed_ids)
    if missing and enriched["beneficiaire"].notna().any():
        raise AssertionError(f"activite_id avec bénéficiaire mais absent des observations: {missing[:10]}")


def validate_beneficiary_budget_not_overcounted(enriched: pd.DataFrame) -> None:
    if enriched.empty:
        return
    per_activity = enriched.groupby("activite_id", dropna=False).agg(
        budget_utilise=("budget_utilise", "max"),
        budget_beneficiaire_total=("budget_beneficiaire", "sum"),
        nb_beneficiaires=("nb_beneficiaires", "max"),
    )
    overcounted = (
        per_activity["budget_beneficiaire_total"].fillna(0)
        > per_activity["budget_utilise"].fillna(0) + 1e-6
    )
    if overcounted.any():
        raise AssertionError("Le budget bénéficiaire dépasse artificiellement le budget activité.")
    with_beneficiaries = per_activity["nb_beneficiaires"].fillna(0) > 0
    if not np.allclose(
        per_activity.loc[with_beneficiaries, "budget_beneficiaire_total"].fillna(0).to_numpy(),
        per_activity.loc[with_beneficiaries, "budget_utilise"].fillna(0).to_numpy(),
        rtol=0,
        atol=1e-6,
    ):
        raise AssertionError("Le budget réparti par bénéficiaire ne recompose pas le budget activité.")


def validate_three_tables_share_same_enriched_source(analytics: AnalyticsBundle) -> None:
    enriched_ids = set(analytics.enriched["activite_id"].dropna().astype(str))
    activity_ids = set(analytics.table_activites["activite_id"].dropna().astype(str)) if not analytics.table_activites.empty else set()
    org_ids = set(analytics.table_organisations["denomination"].dropna().astype(str)) if not analytics.table_organisations.empty else set()
    benef_ids = set(analytics.table_beneficiaires["beneficiaire"].dropna().astype(str)) if not analytics.table_beneficiaires.empty else set()

    assert activity_ids.issubset(enriched_ids)
    if org_ids and "denomination" in analytics.enriched.columns:
        assert org_ids.issubset(set(analytics.enriched["denomination"].dropna().astype(str)))
    if benef_ids:
        assert benef_ids.issubset(set(analytics.enriched["beneficiaire"].dropna().astype(str)))
