from __future__ import annotations
import pandas as pd
import numpy as np

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.replace("\xa0", " ", regex=False)
        .str.strip()
    )
    return df

def concat_sur_liste_colonnes(df1: pd.DataFrame, df2: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    common_in_order = [c for c in cols if c in df1.columns and c in df2.columns]
    if not common_in_order:
        raise ValueError("Aucune des colonnes demandées n'est présente dans les deux DataFrames.")
    out = pd.concat(
        [df1.loc[:, common_in_order], df2.loc[:, common_in_order]],
        axis=0,
        ignore_index=True
    )
    return out

def prepare_from_raw(
    xlsx_organisations: str,
    xlsx_activites: str,
    xlsx_exercices: str,
    xlsx_domaines: str,
    csv_ppl: str,
    csv_promulguees: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # --- Load
    df_organisations = pd.read_excel(xlsx_organisations).set_index("representants_id")
    df_activites = pd.read_excel(xlsx_activites).set_index("activite_id")
    df_exercices = pd.read_excel(xlsx_exercices).set_index("exercices_id")
    df_domaines_intervention = pd.read_excel(xlsx_domaines).set_index("activite_id")
    df_ppl = pd.read_csv(csv_ppl, sep=";", encoding="latin-1")
    df_promulguees = pd.read_csv(csv_promulguees, sep=";", encoding="latin-1")

    df_ppl = clean_columns(df_ppl)
    df_promulguees = clean_columns(df_promulguees)

    if "Date de dépôt" in df_ppl.columns:
        df_ppl = df_ppl.rename(columns={"Date de dépôt": "Date initiale"})

    # =========================
    # 1) STATS BUDGETS AU BON NIVEAU : EXERCICE -> ORGANISATION
    # =========================
    ex = df_exercices.copy()

    ex["budget_exercice"] = (ex["montant_depense_inf"] + ex["montant_depense_sup"]) / 2
    ex["budget_exercice"] = pd.to_numeric(ex["budget_exercice"], errors="coerce")
    ex["nombre_activites"] = pd.to_numeric(ex.get("nombre_activites"), errors="coerce")

    ex["denomination"] = ex["representants_id"].map(df_organisations["denomination"])

    org_stats = ex.groupby("denomination", dropna=False).agg(
        budget_total=("budget_exercice", "sum"),
        nb_activites_total=("nombre_activites", "sum"),
    )
    org_stats["budget_moyen_activite"] = (
        org_stats["budget_total"] / org_stats["nb_activites_total"].replace(0, np.nan)
    )

    # =========================
    # 2) ACTIVITES : joindre uniquement infos utiles
    # =========================
    df_activites = df_activites.merge(
        df_exercices[["nombre_salaries", "representants_id"]],
        left_on="exercices_id",
        right_index=True,
        how="left"
    )
    df_activites["denomination"] = df_activites["representants_id"].map(df_organisations["denomination"])
    df_activites = df_activites.join(
        org_stats[["budget_total", "nb_activites_total", "budget_moyen_activite"]],
        on="denomination"
    )

    df_activites["date_publication_activite"] = pd.to_datetime(
        df_activites["date_publication_activite"], errors="coerce"
    )

    # Domaines d’intervention agrégés
    activites_domaines = df_domaines_intervention.copy()
    activites_domaines["objet_activite"] = activites_domaines.index.map(df_activites["objet_activite"])

    domains_str = (
        activites_domaines.reset_index()
        .groupby("activite_id")["domaines_intervention_actions_menees"]
        .apply(lambda s: "; ".join(sorted(set(s.dropna().astype(str).str.strip()))))
        .rename("domaines")
    )
    df_activites = df_activites.join(domains_str)

    # =========================
    # 3) Lois
    # =========================
    df_lois = concat_sur_liste_colonnes(
        df_ppl, df_promulguees,
        cols=["Date initiale", "Date de promulgation", "Titre", "Numéro de la loi", "Thèmes", "État du dossier", "URL du dossier"]
    )

    df_lois["Date initiale"] = pd.to_datetime(df_lois["Date initiale"], errors="coerce", dayfirst=True)
    df_lois["Date de promulgation"] = pd.to_datetime(df_lois["Date de promulgation"], errors="coerce", dayfirst=True)
    df_lois = df_lois[df_lois["Date initiale"] >= pd.Timestamp("2018-01-01")]

    return df_activites, df_lois

def minify_activites(df_activites: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "date_publication_activite",
        "denomination",
        "budget_total",
        "nb_activites_total",
        "budget_moyen_activite",
        "domaines",
        "objet_activite",
        "nombre_salaries",
    ]
    cols = [c for c in keep if c in df_activites.columns]
    out = df_activites.loc[:, cols].copy()

    # types compacts
    for c in ["budget_total", "nb_activites_total", "budget_moyen_activite", "nombre_salaries"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if "denomination" in out.columns:
        out["denomination"] = out["denomination"].astype("string")
    if "domaines" in out.columns:
        out["domaines"] = out["domaines"].astype("string")
    if "objet_activite" in out.columns:
        out["objet_activite"] = out["objet_activite"].astype("string")

    return out

def minify_lois(df_lois: pd.DataFrame) -> pd.DataFrame:
    keep = ["Date initiale", "Date de promulgation", "Titre", "Numéro de la loi", "Thèmes", "État du dossier", "URL du dossier"]
    cols = [c for c in keep if c in df_lois.columns]
    out = df_lois.loc[:, cols].copy()
    for c in ["Titre", "Thèmes", "État du dossier", "URL du dossier", "Numéro de la loi"]:
        if c in out.columns:
            out[c] = out[c].astype("string")
    return out

def build_docs_activites(df_activites_min: pd.DataFrame) -> pd.DataFrame:
    doc_text = (
        df_activites_min.get("objet_activite", "").fillna("").astype(str)
        + " "
        + df_activites_min.get("domaines", "").fillna("").astype(str)
    ).str.strip()

    docs = pd.DataFrame({"activite_id": df_activites_min.index.astype(str), "doc_text": doc_text.astype("string")})
    return docs

def build_docs_lois(df_lois_min: pd.DataFrame) -> pd.DataFrame:
    doc_text = (
        df_lois_min.get("Titre", "").fillna("").astype(str)
        + " "
        + df_lois_min.get("Thèmes", "").fillna("").astype(str)
    ).str.strip()

    docs = pd.DataFrame({"loi_id": df_lois_min.index.astype(str), "doc_text": doc_text.astype("string")})
    return docs