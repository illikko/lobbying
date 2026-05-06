import streamlit as st
import pandas as pd
import numpy as np
from src.config import ART
from src.io_artifacts import load_parquet
from src.llm_summarize import summarize_activites
import re
import zlib
import plotly.graph_objects as go
import plotly.express as px
from rank_bm25 import BM25Okapi
from src.textnorm import tokenize
from pathlib import Path


# application Streamlit
st.set_page_config(page_title="Cartographie des influences", layout="wide")
st.title("Cartographie des influences")

if "results" not in st.session_state:
    st.session_state.results = None

with st.expander("ℹ️ À propos de l'application", expanded=False):
        st.write("""
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
        st.write("""
        • écrire les mots clés recherchés ou les phrases dans le formulaire "mots-clés / requête", puis cliquer sur "Lancer" pour obtenir les résultats de recherche  
        • lire les résultats (objet_activite), et dé-sélectionner les activités qui ne sont pas pertinentes (colonne "Sélection"), puis cliquer sur "Valider la sélection" pour confirmer les activités retenues  
        • faire de même pour les lois correspondant à la recherche (dé-sélectionner les lois non pertinentes, puis valider)  
        • explorer les différentes visualisations (matrice à bulles, frise chronologique)  
        • dans l'onglet "Synthèse", cliquer sur "Générer la synthèse" pour obtenir une synthèse textuelle des activités de lobbying retenues, par un modèle de langage (LLM)  
        • utiliser les filtres pour ajuster la recherche: nombre de résultats demandés, budget, période  
        """)

DATA_DIR = Path("data")
DATA_RAW_DIR = DATA_DIR / "raw"

def _first_existing(paths):
    for p in paths:
        p = Path(p)
        if p.exists():
            return p
    return None

def _read_any_table(base_name: str) -> pd.DataFrame:
    """Lit data/raw/<base_name> en priorité, puis data/<base_name>, avec dtype=str pour préserver les IDs."""
    candidates = []
    for root in [DATA_RAW_DIR, DATA_DIR]:
        candidates.extend([
            root / f"{base_name}.xlsx",
            root / f"{base_name}.xls",
            root / f"{base_name}.csv",
            root / f"{base_name}.parquet",
        ])
    p = _first_existing(candidates)
    if p is None:
        return pd.DataFrame()
    if p.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(p, dtype=str)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p, dtype=str)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    return pd.DataFrame()

def _norm_id(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)

def _explode_id_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Explose une colonne d'IDs simples ou multi-IDs en gardant les autres colonnes."""
    out = df.copy()
    out[col] = (
        out[col].astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.split(r"\s*[;,|]\s*", regex=True)
    )
    out = out.explode(col)
    out[col] = out[col].astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    return out.dropna(subset=[col])

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    if df is None or df.empty:
        return None
    exact = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in exact:
            return exact[cand.lower()]
    normalized = {str(c).lower().replace(" ", "_").replace("-", "_"): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace(" ", "_").replace("-", "_")
        if key in normalized:
            return normalized[key]
    return None

def _join_unique(values, max_items=30):
    out = []
    seen = set()
    for x in pd.Series(values).dropna().astype(str):
        x = x.strip()
        if not x or x.lower() in {"nan", "none", "<na>"}:
            continue
        if x not in seen:
            seen.add(x)
            out.append(x)
        if len(out) >= max_items:
            break
    if len(seen) > max_items:
        out.append(f"… (+{len(seen) - max_items} autres)")
    return "; ".join(out)

@st.cache_data(show_spinner="Chargement des données de test sans FAISS…")
def load_all_test():
    # On réutilise les artefacts tabulaires s'ils existent, mais sans charger FAISS ni SentenceTransformer.
    try:
        df_acts = load_parquet(ART.df_activites_min)
        df_lois = load_parquet(ART.df_lois_min)
    except Exception:
        acts_path = _first_existing([DATA_DIR / "df_activites_min.parquet", DATA_DIR / "8_objets_activites.parquet"])
        lois_path = _first_existing([DATA_DIR / "df_lois_min.parquet"])
        if acts_path is None:
            st.error("Impossible de trouver les activités: artifacts/df_activites_min ou data/df_activites_min.parquet.")
            st.stop()
        df_acts = pd.read_parquet(acts_path)
        df_lois = pd.read_parquet(lois_path) if lois_path else pd.DataFrame()

    extra = {
        "affiliations": _read_any_table("5_affiliations"),
        "beneficiaires": _read_any_table("11_beneficiaires"),
        "observations": _read_any_table("14_observations"),
        "infos_generales": _read_any_table("1_informations_generales"),
    }

    if "activite_id" not in df_acts.columns:
        df_acts = df_acts.reset_index().rename(columns={"index": "activite_id"})
    df_acts["activite_id"] = _norm_id(df_acts["activite_id"])

    # Enrichissement organisationnel dès le chargement depuis 1_informations_generales.
    # Important : même si label_categorie_organisation existe déjà, on ajoute quand même
    # representants_id si absent, sinon org_search le perdra et les affiliations seront impossibles.
    infos = extra.get("infos_generales", pd.DataFrame())
    if not infos.empty and "denomination" in df_acts.columns:
        info_cols = []
        if "denomination" in infos.columns:
            info_cols.append("denomination")
        if "representants_id" in infos.columns:
            info_cols.append("representants_id")
        if "label_categorie_organisation" in infos.columns:
            info_cols.append("label_categorie_organisation")

        if "denomination" in info_cols and len(info_cols) > 1:
            right = infos[info_cols].drop_duplicates().copy()
            right["_denom_key"] = right["denomination"].astype("string").str.strip().str.lower()
            # une ligne par dénomination, avec concaténation des representants_id si besoin
            agg = {}
            if "representants_id" in right.columns:
                agg["representants_id"] = ("representants_id", lambda s: _join_unique(_norm_id(s), max_items=80))
            if "label_categorie_organisation" in right.columns:
                agg["label_categorie_organisation_from_infos"] = ("label_categorie_organisation", "first")
            right = right.groupby("_denom_key", dropna=False).agg(**agg).reset_index()

            df_acts["_denom_key"] = df_acts["denomination"].astype("string").str.strip().str.lower()
            df_acts = df_acts.merge(right, on="_denom_key", how="left")
            df_acts = df_acts.drop(columns=["_denom_key"], errors="ignore")

            if "representants_id" not in df_acts.columns and "representants_id" in right.columns:
                # déjà créé par le merge
                pass
            elif "representants_id_x" in df_acts.columns or "representants_id_y" in df_acts.columns:
                left = df_acts.get("representants_id_x")
                add = df_acts.get("representants_id_y")
                df_acts["representants_id"] = left.fillna(add) if left is not None else add
                df_acts = df_acts.drop(columns=["representants_id_x", "representants_id_y"], errors="ignore")

            if "label_categorie_organisation_from_infos" in df_acts.columns:
                if "label_categorie_organisation" not in df_acts.columns:
                    df_acts["label_categorie_organisation"] = df_acts["label_categorie_organisation_from_infos"]
                else:
                    df_acts["label_categorie_organisation"] = df_acts["label_categorie_organisation"].fillna(
                        df_acts["label_categorie_organisation_from_infos"]
                    )
                df_acts = df_acts.drop(columns=["label_categorie_organisation_from_infos"], errors="ignore")

    return df_acts, df_lois, extra

df_acts, df_lois, extra_tables = load_all_test()

def build_search_text_activites(df: pd.DataFrame) -> pd.Series:
    cols = [
        "objet_activite", "denomination", "domaines", "actions_menees",
        "decision_concernee", "observations", "label_categorie_organisation",
    ]
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.Series([""] * len(df), index=df.index)
    out = df[present].fillna("").astype(str).agg(" ".join, axis=1)
    return out

@st.cache_resource(show_spinner=False)
def build_bm25_activites_local(df_activites_min: pd.DataFrame):
    docs_acts = build_search_text_activites(df_activites_min).tolist()
    tokenized = [tokenize(t) for t in docs_acts]
    return BM25Okapi(tokenized)

bm25_acts = build_bm25_activites_local(df_acts)

def search_activites_local(query_text: str, top_n: int, year_range=None, min_budget: float = 0.0) -> pd.DataFrame:
    qtok = tokenize(query_text or "")
    cand = df_acts.copy()

    if year_range is not None and "date_publication_activite" in cand.columns:
        years = pd.to_datetime(cand["date_publication_activite"], errors="coerce").dt.year
        cand = cand[(years >= int(year_range[0])) & (years <= int(year_range[1]))].copy()

    if "budget_moyen_activite" in cand.columns:
        b = pd.to_numeric(cand["budget_moyen_activite"], errors="coerce").fillna(0)
        cand = cand[b >= float(min_budget)].copy()

    if cand.empty:
        return cand

    if not qtok:
        cand["bm25_score"] = 0.0
        cand["hybrid_score"] = 0.0
        return cand.head(int(top_n))

    scores_all = np.asarray(bm25_acts.get_scores(qtok), dtype=float)
    cand["bm25_score"] = scores_all[cand.index.to_numpy()] if np.issubdtype(cand.index.dtype, np.integer) else 0.0

    # Si l'index n'est pas RangeIndex, on remappe par position.
    if cand["bm25_score"].eq(0).all() and len(scores_all) == len(df_acts):
        pos_map = pd.Series(scores_all, index=df_acts.index)
        cand["bm25_score"] = cand.index.map(pos_map).astype(float)

    cand["hybrid_score"] = cand["bm25_score"]
    cand = cand.sort_values("hybrid_score", ascending=False).head(int(top_n))
    return cand

def build_beneficiaires_actions(selected_res: pd.DataFrame) -> pd.DataFrame:
    """Niveau activité/action : selected_res.activite_id -> 14_observations -> 11_beneficiaires."""
    if selected_res is None or selected_res.empty:
        return pd.DataFrame()

    acts = selected_res.copy()
    if "activite_id" not in acts.columns:
        acts = acts.reset_index().rename(columns={"index": "activite_id"})
    acts["activite_id"] = _norm_id(acts["activite_id"])

    obs = extra_tables.get("observations", pd.DataFrame()).copy()
    benef = extra_tables.get("beneficiaires", pd.DataFrame()).copy()
    if obs.empty or benef.empty:
        return pd.DataFrame()

    required_obs = {"activite_id", "action_representation_interet_id"}
    required_benef = {"action_representation_interet_id", "beneficiaire_action_menee"}
    if not required_obs.issubset(obs.columns) or not required_benef.issubset(benef.columns):
        return pd.DataFrame()

    obs["activite_id"] = _norm_id(obs["activite_id"])
    obs["action_representation_interet_id"] = _norm_id(obs["action_representation_interet_id"])
    benef["action_representation_interet_id"] = _norm_id(benef["action_representation_interet_id"])

    detail = (
        acts[["activite_id", "denomination", "objet_activite"]].drop_duplicates()
        .merge(obs[["activite_id", "action_representation_interet_id"]].drop_duplicates(), on="activite_id", how="inner")
        .merge(
            benef[["action_representation_interet_id", "beneficiaire_action_menee"]].drop_duplicates(),
            on="action_representation_interet_id",
            how="inner",
        )
    )
    detail["beneficiaire_action_menee"] = detail["beneficiaire_action_menee"].astype("string").str.strip()
    detail = detail[detail["beneficiaire_action_menee"].notna() & (detail["beneficiaire_action_menee"] != "")]
    if detail.empty:
        return pd.DataFrame()

    return (
        detail.groupby(["denomination", "activite_id", "objet_activite"], dropna=False)
        .agg(
            nb_beneficiaires=("beneficiaire_action_menee", "nunique"),
            beneficiaires_action_menee=("beneficiaire_action_menee", lambda s: _join_unique(s, max_items=80)),
        )
        .reset_index()
        .sort_values(["nb_beneficiaires", "denomination"], ascending=[False, True])
    )


def build_affiliations_organisations(org_search: pd.DataFrame) -> pd.DataFrame:
    """Niveau organisation : org_search.representants_id -> 5_affiliations.representants_id."""
    if org_search is None or org_search.empty or "representants_id" not in org_search.columns:
        return pd.DataFrame()

    aff = extra_tables.get("affiliations", pd.DataFrame()).copy()
    if aff.empty or not {"representants_id", "denomination_affiliation"}.issubset(aff.columns):
        return pd.DataFrame()

    org_ids = org_search[["denomination", "representants_id"]].dropna(subset=["representants_id"]).copy()
    org_ids = _explode_id_col(org_ids, "representants_id").drop_duplicates()

    aff_ids = aff[["representants_id", "denomination_affiliation"]].copy()
    aff_ids = _explode_id_col(aff_ids, "representants_id")
    aff_ids["denomination_affiliation"] = aff_ids["denomination_affiliation"].astype("string").str.strip()
    aff_ids = aff_ids.dropna(subset=["representants_id", "denomination_affiliation"]).drop_duplicates()

    detail = org_ids.merge(aff_ids, on="representants_id", how="inner")
    if detail.empty:
        return pd.DataFrame()

    return (
        detail.groupby("denomination", dropna=False)
        .agg(
            representants_id=("representants_id", lambda s: _join_unique(s, max_items=80)),
            nb_affiliations=("denomination_affiliation", "nunique"),
            denominations_affiliation=("denomination_affiliation", lambda s: _join_unique(s, max_items=120)),
        )
        .reset_index()
        .sort_values(["nb_affiliations", "denomination"], ascending=[False, True])
    )


# --- UI
st.markdown("### Recherche des activités de lobbying")

# year bounds
years = pd.to_datetime(df_acts["date_publication_activite"], errors="coerce").dropna().dt.year
y_min = int(years.min()) if not years.empty else 2018
y_max = int(years.max()) if not years.empty else 2026

query = st.text_input("Mots-clés / requête", value="")

c1, c2, c3 = st.columns(3)
with c1:
    topn = st.slider(
        "Nombre d'activités", 
        10, 100, 20, 10,
        help="Définit le nombre maximum d'activités de lobbying affichées dans les résultats."
    )

with c2:
    min_budget = st.slider(
        "Budget moyen/activité minimum (€)", 
        0, 5000, 0, 100,
        help="Filtre les activités de lobbying dont le budget moyen par activité (pour chaque organisation) est supérieur ou égal à ce montant. Cela permet de se concentrer sur les activités potentiellement plus significatives en termes d'influence."
    )

with c3:
    year_range = st.slider(
        "Période (année)", 
        y_min, y_max, (y_min, y_max),
        help="Filtre les activités de lobbying publiées dans cette plage d'années. Cela permet de se concentrer sur les activités récentes ou sur une période spécifique d'intérêt."
    )

min_bm25 = 8.0
min_vec  = 0.5
nprobe = 16
alpha_vec = 0.55
fusion_mode = "rrf"
rrf_k = 60

submitted = st.button("Lancer", type="primary")

if submitted:
    with st.spinner("Recherche locale BM25 sans FAISS…"):
        res = search_activites_local(
            query_text=query,
            top_n=int(topn),
            year_range=year_range,
            min_budget=float(min_budget),
        )
    st.session_state.results = res
    st.session_state.selected_results = None
    st.session_state.selected_lois = None
    st.session_state.last_query = query
    if res is None:
        st.warning("Aucun résultat (res = None). Lancez une recherche / vérifiez les filtres.")
    else:
        st.success(f"{len(res)} résultats")

# 3) READ: on lit toujours depuis session_state (jamais depuis une variable locale fragile)
res = st.session_state.get("results", None)
if not isinstance(res, pd.DataFrame) or res.empty:
    st.info("Lancez une recherche pour afficher des résultats.")
    st.stop()


def hybrid_search_lois_local(
    query_text: str,
    topn_laws: int = 50,
    k_bm25: int = 50,
    k_vec: int = 50,
    fusion_mode: str = "rrf",
    rrf_k: int = 60,
    alpha_vec: float = 0.55,
    nprobe: int = 16,
) -> pd.DataFrame:
    # Version locale test: BM25 uniquement, aucun appel FAISS / embedding.
    if df_lois is None or df_lois.empty:
        return pd.DataFrame()
    qtok = tokenize(query_text or "")
    if not qtok:
        return df_lois.iloc[0:0].copy()

    docs_lois = (
        df_lois.get("Titre", pd.Series([""] * len(df_lois))).fillna("").astype(str) + " " +
        df_lois.get("Thèmes", pd.Series([""] * len(df_lois))).fillna("").astype(str)
    ).tolist()
    bm25 = BM25Okapi([tokenize(t) for t in docs_lois])
    scores = np.asarray(bm25.get_scores(qtok), dtype=float)
    idx = np.argsort(scores)[::-1][:int(topn_laws)]
    out = df_lois.iloc[idx].copy()
    out["bm25_score"] = scores[idx]
    out["hybrid_score"] = out["bm25_score"]
    return out.sort_values("bm25_score", ascending=False)

def to_int64_safe(series: pd.Series) -> pd.Series:
    """Convertit en Int64 en neutralisant les infinis."""
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .round(0)
        .astype("Int64")
    )

res = st.session_state.get("results", None)

if not isinstance(res, pd.DataFrame) or res.empty:
    st.info("Lance une recherche pour afficher des résultats.")
    st.stop()

# =========================
# TABLEAU + SELECTION
# =========================
show = res.copy()

if "selected" not in show.columns:
    show.insert(0, "selected", True)

priority = ["selected", "objet_activite", "denomination"]
rest = [c for c in show.columns if c not in priority]
show = show.loc[:, [c for c in priority if c in show.columns] + rest]
for c in ["budget_total", "budget_moyen_activite"]:
    if c in show.columns:
        show[c] = (
            pd.to_numeric(show[c], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)  # ← neutralise les infinis
            .round(0)
            .astype("Int64")
        )



st.markdown("### Résultats des activités de lobbying")
st.markdown("Désélectionnez ce qui n'est pas pertinent, puis validez")

# formatage des dates
if "date_publication_activite" in show.columns:
    show["date_publication_activite"] = pd.to_datetime(
        show["date_publication_activite"], errors="coerce"
    )
    
edited = st.data_editor(
    show,
    use_container_width=True,
    column_config={
        "selected": st.column_config.CheckboxColumn("Sélection", default=True),
        "date_publication_activite": st.column_config.DatetimeColumn(
            "Date",
            format="YYYY-MM-DD"
        ),
    },
    disabled=[c for c in show.columns if c != "selected"],
    key="results_editor",
)

colv1, colv2 = st.columns([1, 4])
with colv1:
    validate = st.button("✅ Valider la sélection", type="primary")
with colv2:
    st.caption("La matrice, la frise chronologique et la synthèse des activités seront produites uniquement avec les lignes sélectionnées.")

if validate:
    st.session_state.selected_results = (
        edited[edited["selected"] == True]
        .drop(columns=["selected"], errors="ignore")
        .copy()
    )
    st.success(f"Sélection validée : {len(st.session_state.selected_results)} activités retenues.")

selected_res = st.session_state.get("selected_results", None)
if not isinstance(selected_res, pd.DataFrame) or selected_res.empty:
    # fallback : tout sélectionné tant qu'on n'a pas validé
    selected_res = res.copy()

# =========================
# BUDGET ESTIMÉ SUR LA RECHERCHE (org) - SUR selected_res
# =========================
st.markdown("#### Analyse du budget par organisation")

df_tmp = selected_res.copy()
if "activite_id" not in df_tmp.columns:
    df_tmp = df_tmp.reset_index().rename(columns={"index": "activite_id"})
if "activite_id" not in df_tmp.columns:
    df_tmp["activite_id"] = df_tmp.index.astype(str)

agg_org = {
    "nb_activites_matching": ("activite_id", "nunique"),
    "budget_moyen_activite": ("budget_moyen_activite", "first"),
    "budget_total_org": ("budget_total", "first"),
    "nb_activites_total_org": ("nb_activites_total", "first"),
}
if "representants_id" in df_tmp.columns:
    agg_org["representants_id"] = ("representants_id", lambda s: _join_unique(s, max_items=80))
if "label_categorie_organisation" in df_tmp.columns:
    agg_org["label_categorie_organisation"] = ("label_categorie_organisation", "first")

org_search = (
    df_tmp.groupby("denomination", dropna=False)
    .agg(**agg_org)
    .reset_index()
)

# Forcer les colonnes numériques
cols_num = [
    "nb_activites_matching",
    "budget_moyen_activite",
    "budget_total_org",
    "nb_activites_total_org",
]

for col in cols_num:
    org_search[col] = pd.to_numeric(org_search[col], errors="coerce")

# Calcul
org_search["budget_estime_recherche"] = (
    org_search["budget_moyen_activite"].fillna(0)
    * org_search["nb_activites_matching"].fillna(0)
)

# Arrondi à l'entier
cols_round = [
    "budget_moyen_activite",
    "budget_total_org",
    "budget_estime_recherche",
]

org_search[cols_round] = org_search[cols_round].round(0)

# Conversion en entier nullable pour garder un vrai type numérique
cols_int = [
    "nb_activites_matching",
    "budget_estime_recherche",    
    "budget_moyen_activite",
    "nb_activites_total_org",
    "budget_total_org",
]

for col in cols_int:
    org_search[col] = to_int64_safe(org_search[col])

# Affiliations au niveau organisation : calculées ici, après création d'org_search.
affiliations_par_org = build_affiliations_organisations(org_search)
if not affiliations_par_org.empty:
    org_search = org_search.merge(
        affiliations_par_org[["denomination", "nb_affiliations", "denominations_affiliation"]],
        on="denomination",
        how="left",
    )

priority_org_cols = [
    "denomination", "label_categorie_organisation", "representants_id",
    "nb_activites_matching", "budget_estime_recherche", "budget_moyen_activite",
    "budget_total_org", "nb_activites_total_org", "nb_affiliations", "denominations_affiliation",
]
org_search = org_search[[c for c in priority_org_cols if c in org_search.columns] + [c for c in org_search.columns if c not in priority_org_cols]]
org_search = org_search.sort_values("budget_estime_recherche", ascending=False)

with st.expander(
    "Pour chaque organisation: nombre d'activités et budget correspondant à la recherche, et budget global et moyen par activité",
    expanded=False,
):
    st.dataframe(org_search, use_container_width=True)

# =========================
# MATRICE BULLES org x domaines - SUR selected_res
# =========================
st.markdown("### Matrice organisations × domaines")

MIN_BUDGET = float(min_budget)
TOP_DOMAINS = 12  

df2 = selected_res.copy()
if "activite_id" not in df2.columns:
    df2 = df2.reset_index().rename(columns={"index": "activite_id"})
if "activite_id" not in df2.columns:
    df2["activite_id"] = df2.index.astype(str)

df2["domaines"] = df2.get("domaines", "").fillna("")
df2["domaines_list"] = df2["domaines"].astype(str).str.split(";")

def _count_domains(lst):
    if not isinstance(lst, list):
        return 0
    return len([d.strip() for d in lst if str(d).strip()])

df2["nb_domaines"] = df2["domaines_list"].apply(_count_domains)

df2 = df2.explode("domaines_list")
df2["domaines_list"] = df2["domaines_list"].astype(str).str.strip()
df2 = df2[df2["domaines_list"] != ""]

df2["budget_moyen_activite"] = pd.to_numeric(df2.get("budget_moyen_activite", 0), errors="coerce").fillna(0.0)

df2["budget_reparti"] = np.where(
    df2["nb_domaines"] > 0,
    df2["budget_moyen_activite"] / df2["nb_domaines"],
    0.0
)

def _join_objets(s, max_items=30):
    seen = set()
    out = []
    for x in s.dropna().astype(str):
        if x and x not in seen:
            seen.add(x)
            out.append(x)
        if len(out) >= max_items:
            break
    if len(seen) > max_items:
        out.append(f"... (+{len(seen)-max_items} autres)")
    return "<br>• " + "<br>• ".join(out) if out else ""

cell = (
    df2.groupby(["denomination", "domaines_list"], as_index=False)
    .agg(
        budget_total=("budget_reparti", "sum"),
        nb_activites=("activite_id", "nunique"),
        hybrid_moy=("hybrid_score", "mean"),
        objets=("objet_activite", lambda s: _join_objets(s, max_items=30)),
    )
)

cell = cell[cell["budget_total"] >= MIN_BUDGET].copy()

top_domains = (
    cell.groupby("domaines_list")["budget_total"]
    .sum().sort_values(ascending=False)
    .head(TOP_DOMAINS).index
)
cell = cell[cell["domaines_list"].isin(top_domains)].copy()

if cell.empty:
    st.warning("Aucune donnée après filtrage (MIN_BUDGET / TOP_DOMAINS).")
else:
    org_order = (
        cell.groupby("denomination")["budget_total"]
        .sum().sort_values(ascending=False).index.tolist()
    )
    dom_order = (
        cell.groupby("domaines_list")["budget_total"]
        .sum().sort_values(ascending=False).index.tolist()
    )

    cell["denomination"] = pd.Categorical(cell["denomination"], categories=org_order, ordered=True)
    cell["domaines_list"] = pd.Categorical(cell["domaines_list"], categories=dom_order, ordered=True)
    cell = cell.sort_values(["denomination", "domaines_list"])

    cell["hover"] = (
        "<b>Organisation :</b> " + cell["denomination"].astype(str) +
        "<br><b>Domaine :</b> " + cell["domaines_list"].astype(str) +
        "<br><b>Budget estimé sur la recherche (réparti) :</b> " + cell["budget_total"].fillna(0).replace([np.inf, -np.inf], 0).round(0).astype(int).astype(str) +
        "<br><b>Nb activités matching :</b> " + cell["nb_activites"].fillna(0).replace([np.inf, -np.inf], 0).astype(int).astype(str) +
        "<br><b>Score hybride moyen :</b> " + cell["hybrid_moy"].fillna(0).round(3).astype(str) +
        "<br><b>Objets :</b> " + cell["objets"].astype(str)
    )

    max_val = float(cell["budget_total"].max()) if len(cell) else 1.0
    desired_max_marker = 40
    sizeref = 2.0 * max_val / (desired_max_marker ** 2) if max_val > 0 else 1.0

    figm = go.Figure(
        data=go.Scatter(
            x=cell["domaines_list"].astype(str),
            y=cell["denomination"].astype(str),
            mode="markers",
            marker=dict(
                size=cell["budget_total"],
                sizemode="area",
                sizeref=sizeref,
                sizemin=3,
                opacity=0.75,
            ),
            text=cell["hover"],
            hovertemplate="%{text}<extra></extra>",
        )
    )

    max_len_org = max((len(s) for s in org_order), default=10)
    left_margin = min(520, max(180, 7 * max_len_org))

    figm.update_layout(
        title=f"Répartition du budget de lobbying par organisation et domaine.",
        xaxis=dict(title="Domaines", categoryorder="array", categoryarray=dom_order, tickangle=45),
        yaxis=dict(title="Organisations", categoryorder="array", categoryarray=org_order, autorange="reversed"),
        height=max(650, 28 * len(org_order) + 240),
        margin=dict(l=left_margin, r=10, t=80, b=120),
    )

    figm.update_yaxes(showgrid=True, automargin=True)
    figm.update_xaxes(showgrid=True)

    st.plotly_chart(figm, use_container_width=True)
    
    st.caption(
        f"Le budget par organisation et domaine est estimé à partir du budget moyen par activité, "
        f"réparti entre les domaines d'activité de chaque activité. "
        f"Seuls les domaines les plus fréquents sont affichés (ici {TOP_DOMAINS})."
        f"Les bulles sont proportionelles aux budgets estimés."
    )

# =========================
# BENEFICIAIRES + AFFILIATIONS - AVANT LES LOIS
# =========================
beneficiaires_par_activite = build_beneficiaires_actions(selected_res)

st.markdown("### Bénéficiaires des actions menées")
st.caption("Unité : activité/action. Jointures : activite_id → 14_observations → action_representation_interet_id → 11_beneficiaires.")
if beneficiaires_par_activite.empty:
    st.info("Aucun bénéficiaire trouvé pour les activités sélectionnées.")
else:
    st.dataframe(beneficiaires_par_activite, use_container_width=True)

st.markdown("### Affiliations des organisations")
st.caption("Unité : organisation. Jointure directe : org_search.representants_id → 5_affiliations.representants_id.")
if affiliations_par_org.empty:
    st.info("Aucune affiliation trouvée pour les organisations de la recherche.")
    if "representants_id" not in org_search.columns:
        st.warning("Diagnostic : org_search ne contient pas la colonne representants_id. Elle doit être présente dans selected_res/df_tmp avant le groupby.")
else:
    st.dataframe(affiliations_par_org, use_container_width=True)

# =========================
# LOIS CORRESPONDANT A LA RECHERCHE
# =========================

@st.cache_resource(show_spinner=False)
def build_bm25_lois(df_lois_min: pd.DataFrame):
    docs_lois = (
        df_lois_min.get("Titre", "").fillna("").astype(str) + " " +
        df_lois_min.get("Thèmes", "").fillna("").astype(str)
    ).tolist()
    tokenized = [tokenize(t) for t in docs_lois]
    return BM25Okapi(tokenized), docs_lois

bm25_lois, _docs_lois = build_bm25_lois(df_lois)

def search_lois(qtxt: str, top_n: int = 50):
    q = tokenize(qtxt)
    if not q:
        return df_lois.iloc[0:0].copy()
    scores = np.array(bm25_lois.get_scores(q), dtype=float)
    idx = np.argsort(scores)[::-1][:top_n]
    out = df_lois.iloc[idx].copy()
    out["bm25_score"] = scores[idx]
    return out.sort_values("bm25_score", ascending=False)

with st.expander("Paramètres pour la recherche sur les lois", expanded=False):
    topn_laws = st.slider(
        "Nombre de lois", 
        5, 200, 20, 5,
        help="Nombre maximum de lois affichées correspondant à la recherche."
    )
    
    apply_year_filter_laws = st.checkbox("Appliquer aux lois le filtre de la période déjà sélectionné pour les activités de lobbying", value=True)
    
    k_bm25_laws =  200
    k_vec_laws = 200
    min_bm25_laws =  8.0
    min_vec_laws = 0.5
    min_hybrid_laws = 0.0

query_used = st.session_state.get("last_query", "")
lois_res = hybrid_search_lois_local(
    query_text=query_used,
    topn_laws=int(topn_laws),
    k_bm25=int(k_bm25_laws),
    k_vec=int(k_vec_laws),
    fusion_mode=fusion_mode,
    rrf_k=int(rrf_k),
    alpha_vec=float(alpha_vec),
    nprobe=int(nprobe),
)

selected_lois = st.session_state.get("selected_lois", None)
if not isinstance(selected_lois, pd.DataFrame) or selected_lois.empty:
    selected_lois = lois_res.copy()
    st.session_state.selected_lois = selected_lois

show_laws = selected_lois.copy()

if "selected" not in show_laws.columns:
    show_laws.insert(0, "selected", True)

st.markdown("#### Lois correspondant à la recherche")
st.markdown("Désélectionnez les lois non pertinentes, puis validez")

if "date_publication_activite" in show.columns:
    show["date_publication_activite"] = pd.to_datetime(
        show["date_publication_activite"], errors="coerce"
    )

edited_laws = st.data_editor(
    show_laws,
    use_container_width=True,
    column_config={
        "selected": st.column_config.CheckboxColumn("Sélection", default=True),
        "date_publication_activite": st.column_config.DatetimeColumn(
            "Date",
            format="YYYY-MM-DD"
        ),
    },
    disabled=[c for c in show_laws.columns if c != "selected"],
    key="laws_editor",
)

validate_laws = st.button("✅ Valider la sélection des lois", type="primary")

if validate_laws:
    st.session_state.selected_lois = (
        edited_laws[edited_laws["selected"] == True]
        .drop(columns=["selected"], errors="ignore")
        .copy()
    )
    st.success(f"Sélection validée : {len(st.session_state.selected_lois)} lois retenues.")

selected_lois = st.session_state.get("selected_lois", None)
if not isinstance(selected_lois, pd.DataFrame) or selected_lois.empty:
    selected_lois = lois_res.copy()

if lois_res.empty:
    st.info("Aucune loi trouvée.")

# affichage de la frise chrronologique
st.markdown("### Chronologie activités & lois")

def first_theme(s):
    s = "" if pd.isna(s) else str(s)
    parts = [p.strip() for p in re.split(r"[;,/|]", s) if p.strip()]
    return parts[0] if parts else "Thème inconnu"

def stable_jitter(s: str, scale=0.25) -> float:
    h = zlib.crc32(s.encode("utf-8")) % 10_000
    return (h / 10_000 - 0.5) * 2 * scale

START_DATE = pd.Timestamp("2018-01-01")

res = st.session_state.get("results", None)
if not isinstance(res, pd.DataFrame) or res.empty:
    st.info("Lancez une recherche pour afficher des résultats.")
    st.stop()

acts = res.copy()
acts["date_evt"] = pd.to_datetime(acts.get("date_publication_activite"), errors="coerce")
acts = acts.dropna(subset=["date_evt"])
acts = acts[acts["date_evt"] >= START_DATE]

# theme/label/x pour activités
acts["theme"] = acts.get("domaines", "").fillna("").apply(first_theme)
acts["label_full"] = acts.get("denomination", "").astype(str)

m = pd.to_numeric(acts.get("budget_moyen_activite", 0), errors="coerce").fillna(0)
mmax = float(m.max()) if len(m) else 1.0
acts["x"] = (np.sqrt(m + 1) / np.sqrt(mmax + 1) * 6.0) + 1.0
acts["x"] = acts["label_full"].apply(lambda s: stable_jitter(str(s), 0.25)) + acts["x"]

laws = st.session_state.get("selected_lois", None)
if isinstance(laws, pd.DataFrame) and not laws.empty:
    laws = laws.copy()
    laws["date_evt"] = pd.to_datetime(laws.get("Date initiale"), errors="coerce", dayfirst=True)
    laws = laws.dropna(subset=["date_evt"])
    laws = laws[laws["date_evt"] >= START_DATE]
    laws["theme"] = laws.get("Thèmes", "").fillna("").apply(first_theme)
    laws["label_full"] = laws.get("Titre", "").astype(str)
    laws["x"] = -1.0 + laws["label_full"].apply(lambda s: stable_jitter(str(s), 0.15))
else:
    laws = pd.DataFrame(columns=["x", "date_evt", "theme", "label_full"])

themes = sorted(set(acts["theme"]).union(set(laws["theme"])))
palette = px.colors.qualitative.Safe
color_map = {t: palette[i % len(palette)] for i, t in enumerate(themes)}

figt = go.Figure()

if len(laws):
    figt.add_trace(go.Scatter(
        x=laws["x"],
        y=laws["date_evt"],
        mode="markers",
        name="Lois",
        marker=dict(size=10, symbol="square", color=[color_map[t] for t in laws["theme"]]),
        customdata=np.stack([
            laws["label_full"].to_numpy(),
            laws["theme"].to_numpy(),
            laws.get("Numéro de la loi", pd.Series([""]*len(laws))).astype(str).to_numpy(),
            laws.get("État du dossier", pd.Series([""]*len(laws))).astype(str).to_numpy(),
            laws.get("Date de promulgation", pd.Series([pd.NaT]*len(laws))).astype(str).to_numpy(),
            laws.get("URL du dossier", pd.Series([""]*len(laws))).astype(str).to_numpy(),
        ], axis=1),
        hovertemplate=(
            "<b>Loi</b><br>"
            "Titre: %{customdata[0]}<br>"
            "Thème: %{customdata[1]}<br>"
            "État: %{customdata[3]}<br>"
            "Promulgation: %{customdata[4]}<br>"
            "<extra></extra>"
        )
    ))

figt.add_trace(go.Scatter(
    x=acts["x"],
    y=acts["date_evt"],
    mode="markers",
    name="Activités de lobbying",
    marker=dict(size=9, symbol="circle", color=[color_map[t] for t in acts["theme"]]),
    customdata=np.stack([
        acts["label_full"].to_numpy(),
        acts["theme"].to_numpy(),
        acts.get("objet_activite", pd.Series([""]*len(acts))).astype(str).to_numpy(),
        acts.get("domaines", pd.Series([""]*len(acts))).astype(str).to_numpy(),
        pd.to_numeric(acts.get("budget_total", 0), errors="coerce").fillna(0).to_numpy(),
        pd.to_numeric(acts.get("budget_moyen_activite", 0), errors="coerce").fillna(0).to_numpy(),
    ], axis=1),
    hovertemplate=(
        "<b>Activité de lobbying</b><br>"
        "Organisation: %{customdata[0]}<br>"
        "Objet: %{customdata[2]}<br>"
        "Domaines: %{customdata[3]}<br>"
        "Budget total (org, tous exercices): %{customdata[4]:.0f} €<br>"
        "Budget moyen/activité (org): %{customdata[5]:.0f} €<br>"
        "<extra></extra>"
    )
))

figt.add_vline(x=0, line_width=1, opacity=0.3)

figt.update_layout(
    height=900,
    xaxis_title="Lois  ←   |   →  Organisations (budget_moyen_activite)",
    yaxis_title="Date",
    legend_title_text="",
    margin=dict(l=30, r=30, t=60, b=30),
)

st.plotly_chart(figt, use_container_width=True)


# =========================
# PREPARATION DONNEES LLM
# =========================

df_llm = selected_res.copy()

if "activite_id" not in df_llm.columns:
    df_llm = df_llm.reset_index().rename(columns={"index": "activite_id"})
if "activite_id" not in df_llm.columns:
    df_llm["activite_id"] = df_llm.index.astype(str)

# Colonnes numériques utiles
for col in ["budget_moyen_activite", "budget_total", "nb_activites_total", "hybrid_score"]:
    if col in df_llm.columns:
        df_llm[col] = pd.to_numeric(df_llm[col], errors="coerce")

# Domaines : on garde "Non renseigné" pour ne pas perdre de budget
df_llm["domaines"] = df_llm.get("domaines", "").fillna("").astype(str)
df_llm["domaines_list"] = df_llm["domaines"].apply(
    lambda s: [d.strip() for d in s.split(";") if d.strip()] if s.strip() else ["Non renseigné"]
)

df_llm["nb_domaines"] = df_llm["domaines_list"].apply(lambda x: max(len(x), 1))

# Budget réparti entre les domaines de l'activité
df_llm["budget_reparti"] = (
    df_llm["budget_moyen_activite"].fillna(0.0) / df_llm["nb_domaines"]
)

df_llm_exploded = df_llm.explode("domaines_list").rename(columns={"domaines_list": "domaine"})
df_llm_exploded["domaine"] = df_llm_exploded["domaine"].fillna("Non renseigné").astype(str)

# ---------- Tableau 1 : organisation (budget recherche + poids global)
org_llm = (
    df_llm_exploded.groupby("denomination", dropna=False)
    .agg(
        budget_cumule_recherche=("budget_reparti", "sum"),
        nb_activites_matching=("activite_id", "nunique"),
        budget_total_lobbying=("budget_total", "max"),
        nb_activites_total_org=("nb_activites_total", "max"),
    )
    .reset_index()
)

budget_total_recherche = float(org_llm["budget_cumule_recherche"].sum()) if not org_llm.empty else 0.0

org_llm["part_budget_recherche_pct"] = np.where(
    budget_total_recherche > 0,
    100 * org_llm["budget_cumule_recherche"] / budget_total_recherche,
    0.0
)

# Arrondis
for col in ["budget_cumule_recherche", "budget_total_lobbying"]:
    org_llm[col] = to_int64_safe(org_llm[col])

for col in ["nb_activites_matching", "nb_activites_total_org"]:
    org_llm[col] = pd.to_numeric(org_llm[col], errors="coerce").round(0).astype("Int64")

org_llm["part_budget_recherche_pct"] = org_llm["part_budget_recherche_pct"].round(1)

org_llm = org_llm.sort_values("budget_cumule_recherche", ascending=False)

# ---------- Tableau 2 : organisation x domaine
org_theme_llm = (
    df_llm_exploded.groupby(["denomination", "domaine"], dropna=False)
    .agg(
        budget_cumule_recherche_org_domaine=("budget_reparti", "sum"),
        nb_activites_matching_org_domaine=("activite_id", "nunique"),
        objets_activite=("objet_activite", lambda s: list(pd.Series(s).dropna().astype(str).unique())[:12]),
    )
    .reset_index()
)

# rattacher le budget recherche total de l'organisation
org_theme_llm = org_theme_llm.merge(
    org_llm[["denomination", "budget_cumule_recherche"]],
    on="denomination",
    how="left"
)

_denom = pd.to_numeric(org_theme_llm["budget_cumule_recherche"], errors="coerce").fillna(0)
org_theme_llm["part_du_budget_recherche_org_pct"] = np.where(
    _denom > 0,
    100 * pd.to_numeric(org_theme_llm["budget_cumule_recherche_org_domaine"], errors="coerce").fillna(0) / _denom,
    0.0
)

org_theme_llm["part_du_budget_total_recherche_pct"] = np.where(
    budget_total_recherche > 0,
    100 * org_theme_llm["budget_cumule_recherche_org_domaine"] / budget_total_recherche,
    0.0
)

org_theme_llm["budget_cumule_recherche_org_domaine"] = to_int64_safe(
    org_theme_llm["budget_cumule_recherche_org_domaine"]
)

org_theme_llm["nb_activites_matching_org_domaine"] = to_int64_safe(
    org_theme_llm["nb_activites_matching_org_domaine"]
)

org_theme_llm["part_du_budget_recherche_org_pct"] = org_theme_llm["part_du_budget_recherche_org_pct"].round(1)
org_theme_llm["part_du_budget_total_recherche_pct"] = org_theme_llm["part_du_budget_total_recherche_pct"].round(1)

org_theme_llm = org_theme_llm.sort_values(
    ["budget_cumule_recherche_org_domaine", "nb_activites_matching_org_domaine"],
    ascending=False
)

# ---------- Tableau 3 : domaine
theme_llm = (
    org_theme_llm.groupby("domaine", dropna=False)
    .agg(
        budget_cumule_recherche_domaine=("budget_cumule_recherche_org_domaine", "sum"),
        nb_organisations=("denomination", "nunique"),
        nb_activites_matching=("nb_activites_matching_org_domaine", "sum"),
    )
    .reset_index()
)

theme_llm["part_budget_recherche_pct"] = np.where(
    budget_total_recherche > 0,
    100 * pd.to_numeric(theme_llm["budget_cumule_recherche_domaine"], errors="coerce") / budget_total_recherche,
    0.0
)

theme_llm["budget_cumule_recherche_domaine"] = to_int64_safe(
    theme_llm["budget_cumule_recherche_domaine"]
)

theme_llm["part_budget_recherche_pct"] = theme_llm["part_budget_recherche_pct"].round(1)
theme_llm["nb_organisations"] = to_int64_safe(theme_llm["nb_organisations"])
theme_llm["nb_activites_matching"] = to_int64_safe(theme_llm["nb_activites_matching"])

theme_llm = theme_llm.sort_values("budget_cumule_recherche_domaine", ascending=False)


# synthèse par LLM        
st.markdown("### Synthèse des résultats")

res = st.session_state.get("selected_results", None)
if not isinstance(res, pd.DataFrame) or res.empty:
    # fallback : si l’utilisateur n’a pas validé la sélection, on prend les résultats bruts
    res = st.session_state.get("results", None)
    
if not isinstance(res, pd.DataFrame) or res.empty:
    st.warning("Aucun résultat. Lancez d'abord une recherche.")
    st.stop()

if st.button("Générer la synthèse (par LLM)", type="primary"):
    activites_llm = res[
        [c for c in ["denomination", "domaines", "objet_activite"] if c in res.columns]
    ].copy()

    info_llm = {
        "contexte": {
            "requete": st.session_state.get("last_query", ""),
            "nb_activites_retenues": int(res["activite_id"].nunique()) if "activite_id" in res.columns else int(len(res)),
            "budget_total_estime_recherche": round(budget_total_recherche),
        },
        "tableau_organisations": org_llm.to_dict(orient="records"),
        "tableau_organisations_domaines": org_theme_llm.to_dict(orient="records"),
        "tableau_domaines": theme_llm.to_dict(orient="records"),
        "activites_selectionnees": activites_llm.to_dict(orient="records"),
    }

    with st.spinner("Synthèse…"):
        txt = summarize_activites(info_llm)

    st.session_state.synth = txt

if st.session_state.get("synth"):
    st.text_area("Synthèse", st.session_state.synth, height=500)
