import streamlit as st
import pandas as pd
import numpy as np
from src.config import ART
from src.io_artifacts import load_parquet, load_joblib, load_faiss, read_manifest
from src.embed_index import FaissBundle
from src.embed_index import faiss_search
from src.hybrid_search import hybrid_search_activites
from src.llm_summarize import summarize_activites
import re
import zlib
import plotly.graph_objects as go
import plotly.express as px
from rank_bm25 import BM25Okapi
from src.textnorm import tokenize
from sentence_transformers import SentenceTransformer


# application Streamlit
st.set_page_config(page_title="Cartographie des influences", layout="wide")
st.title("Cartographie des influences")

if "results" not in st.session_state:
    st.session_state.results = None

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

@st.cache_resource(show_spinner="Chargement artefacts…")
def load_all():
    manifest = read_manifest()
    df_acts = load_parquet(ART.df_activites_min)
    df_lois = load_parquet(ART.df_lois_min)
    docs = load_parquet(ART.docs_activites)  # indexed by activite_id
    bm25_bundle = load_joblib(ART.bm25_activites)
    faiss_index = load_faiss(ART.faiss_activites)
    id_map = load_parquet(ART.id_map_activites)  # index=pos -> activite_id
    bm25_lois_bundle = load_joblib(ART.bm25_lois)
    faiss_lois_index = load_faiss(ART.faiss_lois)
    id_map_lois = load_parquet(ART.id_map_lois)
    df_affiliations = load_parquet("df_affiliations.parquet")
    df_beneficiaires = load_parquet("df_beneficiaires.parquet")
    df_observations = load_parquet("df_observations.parquet")
    try:
        df_benef_global = load_parquet("df_beneficiaires_activites_globales.parquet")
    except FileNotFoundError:
        df_benef_global = pd.DataFrame(columns=["activite_id", "beneficiaire", "budget_activite"])

    # rebuild FaissBundle doc_ids in index order
    doc_ids = id_map["activite_id"].astype(object).to_numpy()
    faiss_bundle = FaissBundle(index=faiss_index, doc_ids=doc_ids, normalize=True)
    loi_ids = id_map_lois["loi_id"].astype(object).to_numpy()
    faiss_lois_bundle = FaissBundle(index=faiss_lois_index, doc_ids=loi_ids, normalize=True)

    # model name from manifest (to stay consistent)
    st_model = (manifest.get("config", {}) or {}).get("st_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    embedder = SentenceTransformer(st_model)

    return (
        manifest, df_acts, df_lois, docs, bm25_bundle, faiss_bundle,
        bm25_lois_bundle, faiss_lois_bundle, embedder,
        df_affiliations, df_beneficiaires, df_observations, df_benef_global,
    )

(
    manifest, df_acts, df_lois, docs, bm25_bundle, faiss_bundle,
    bm25_lois_bundle, faiss_lois_bundle, embedder,
    df_affiliations, df_beneficiaires, df_observations, df_benef_global,
) = load_all()

def embed_query_fn(q: str) -> np.ndarray:
    q = "" if q is None else str(q)
    v = embedder.encode([q], convert_to_numpy=True)[0]
    return v.astype("float32", copy=False)


def _norm_id_series(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .mask(lambda x: x.str.lower().isin(["", "nan", "none", "<na>"]))
    )


def _explode_ids(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if df is None or df.empty or col not in df.columns:
        return pd.DataFrame(columns=[] if df is None else list(df.columns))
    out = df.copy()
    out[col] = _norm_id_series(out[col])
    out[col] = out[col].str.split(r"\s*[;,|]\s*", regex=True)
    out = out.explode(col)
    out[col] = _norm_id_series(out[col])
    return out.dropna(subset=[col])


def _join_unique(values, max_items: int = 80) -> str:
    out = []
    seen = set()
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


def build_affiliations_organisations(org_search: pd.DataFrame) -> pd.DataFrame:
    """Affiliations au niveau organisation: org_search.representants_id -> 5_affiliations."""
    if org_search is None or org_search.empty:
        return pd.DataFrame()
    if "representants_id" not in org_search.columns:
        return pd.DataFrame()
    required = {"representants_id", "denomination_affiliation"}
    if df_affiliations.empty or not required.issubset(df_affiliations.columns):
        return pd.DataFrame()

    org_ids = _explode_ids(org_search[["denomination", "representants_id"]].drop_duplicates(), "representants_id")
    aff = df_affiliations[["representants_id", "denomination_affiliation"]].copy()
    aff = _explode_ids(aff, "representants_id")
    aff["denomination_affiliation"] = aff["denomination_affiliation"].astype("string").str.strip()
    aff = aff.dropna(subset=["representants_id", "denomination_affiliation"]).drop_duplicates()

    detail = org_ids.merge(aff, on="representants_id", how="inner")
    if detail.empty:
        return pd.DataFrame()

    return (
        detail.groupby("denomination", dropna=False)
        .agg(
            nb_affiliations=("denomination_affiliation", "nunique"),
            denomination_affiliation=("denomination_affiliation", lambda s: _join_unique(s, max_items=120)),
        )
        .reset_index()
        .sort_values(["nb_affiliations", "denomination"], ascending=[False, True])
    )


def build_beneficiaires_activites(selected_res: pd.DataFrame) -> pd.DataFrame:
    """Bénéficiaires au niveau activité: activite_id -> observations -> beneficiaires."""
    if selected_res is None or selected_res.empty:
        return pd.DataFrame()
    required_obs = {"activite_id", "action_representation_interet_id"}
    required_ben = {"action_representation_interet_id", "beneficiaire_action_menee"}
    if df_observations.empty or df_beneficiaires.empty:
        return pd.DataFrame()
    if not required_obs.issubset(df_observations.columns) or not required_ben.issubset(df_beneficiaires.columns):
        return pd.DataFrame()

    acts = selected_res.copy()
    if "activite_id" not in acts.columns:
        acts = acts.reset_index().rename(columns={"index": "activite_id"})
    acts["activite_id"] = _norm_id_series(acts["activite_id"])

    obs = df_observations[["activite_id", "action_representation_interet_id"]].copy()
    obs["activite_id"] = _norm_id_series(obs["activite_id"])
    obs["action_representation_interet_id"] = _norm_id_series(obs["action_representation_interet_id"])
    obs = obs[obs["activite_id"].isin(set(acts["activite_id"].dropna()))].drop_duplicates()

    benef = df_beneficiaires[["action_representation_interet_id", "beneficiaire_action_menee"]].copy()
    benef["action_representation_interet_id"] = _norm_id_series(benef["action_representation_interet_id"])
    benef["beneficiaire_action_menee"] = benef["beneficiaire_action_menee"].astype("string").str.strip()

    detail = (
        obs.merge(benef, on="action_representation_interet_id", how="inner")
           .merge(acts[["activite_id", "denomination", "objet_activite"]].drop_duplicates(), on="activite_id", how="left")
    )
    if detail.empty:
        return pd.DataFrame()

    return (
        detail.groupby(["denomination", "objet_activite"], dropna=False)
        .agg(
            nb_beneficiaires=("beneficiaire_action_menee", "nunique"),
            beneficiaire_action_menee=("beneficiaire_action_menee", lambda s: _join_unique(s, max_items=80)),
        )
        .reset_index()
        .sort_values(["nb_beneficiaires", "denomination"], ascending=[False, True])
    )

def add_beneficiaire_column(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    out = df.copy()
    benef = build_beneficiaires_activites(out)

    if not benef.empty:
        out = out.merge(
            benef[["denomination", "objet_activite", "beneficiaire_action_menee", "nb_beneficiaires"]],
            on=["denomination", "objet_activite"],
            how="left",
        )
        out["beneficiaire"] = out["beneficiaire_action_menee"].fillna(out["denomination"])
        out = out.drop(columns=["beneficiaire_action_menee"], errors="ignore")
    else:
        out["beneficiaire"] = out["denomination"]
        out["nb_beneficiaires"] = pd.NA

    return out

def build_beneficiaire_stats(df_benef_global):
    return (
        df_benef_global.groupby("beneficiaire")
        .agg(
            nb_activites_total_beneficiaire=("activite_id", "nunique"),
            budget_total_beneficiaire=("budget_activite", "sum"),
        )
        .reset_index()
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
        10, 60, 20, 10,
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
    with st.spinner("Recherche…"):
        res = hybrid_search_activites(
            query=query,
            df_activites_min=df_acts,
            bm25_bundle=bm25_bundle,
            faiss_bundle=faiss_bundle,
            embed_query_fn=embed_query_fn,
            k_bm25=400,
            k_vec=400,
            nprobe=int(nprobe),
            alpha_vec=float(alpha_vec),
            topn=int(topn),
            year_range=year_range,
            min_budget=float(min_budget),
            fusion_mode=fusion_mode,
            rrf_k=int(rrf_k),
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
    # --- BM25 lois
    qtok = tokenize(query_text)
    if not qtok:
        return df_lois.iloc[0:0].copy()

    scores_bm = np.asarray(bm25_lois_bundle.bm25.get_scores(qtok), dtype=float)
    idx_bm = np.argsort(scores_bm)[::-1][:k_bm25]
    bm_ids = bm25_lois_bundle.doc_ids[idx_bm].astype(object)
    bm_scores = scores_bm[idx_bm]

    # --- Vector lois
    qvec = embed_query_fn(query_text)
    vec_ids, vec_scores = faiss_search(faiss_lois_bundle, qvec, topk=k_vec, nprobe=nprobe)

    # union ids
    all_ids = np.unique(np.concatenate([bm_ids.astype(object), vec_ids.astype(object)]))
    if len(all_ids) == 0:
        return df_lois.iloc[0:0].copy()

    # df_lois index = 0..N (strings ou ints). On force string pour matcher loi_id (string)
    laws_df = df_lois.copy().reset_index(drop=True)
    laws_df.index = laws_df.index.astype(str)

    cand = laws_df.loc[laws_df.index.intersection(pd.Index(all_ids.astype(str)))].copy()
    if cand.empty:
        return cand

    bm_map = {str(i): float(s) for i, s in zip(bm_ids, bm_scores)}
    vec_map = {str(i): float(s) for i, s in zip(vec_ids, vec_scores)}

    cand["bm25_score"] = cand.index.map(lambda x: bm_map.get(str(x), 0.0)).astype(float)
    cand["vec_score"] = cand.index.map(lambda x: vec_map.get(str(x), 0.0)).astype(float)

    if fusion_mode == "mean":
        def _minmax01(x):
            a, b = float(np.nanmin(x)), float(np.nanmax(x))
            if not np.isfinite(a) or not np.isfinite(b) or b-a < 1e-12:
                return np.zeros_like(x, dtype=float)
            return (x-a)/(b-a)

        cand["bm25_norm"] = _minmax01(cand["bm25_score"].to_numpy())
        cand["vec_norm"] = _minmax01(cand["vec_score"].to_numpy())
        cand["hybrid_score"] = alpha_vec * cand["vec_norm"] + (1.0-alpha_vec) * cand["bm25_norm"]
    else:
        # RRF
        def _rrf(ids, w, k=60):
            return {str(doc): w/(k+r) for r, doc in enumerate(ids, start=1)}

        bm_rrf = _rrf(bm_ids, w=(1.0-alpha_vec), k=rrf_k)
        vec_rrf = _rrf(vec_ids, w=alpha_vec, k=rrf_k)
        cand["hybrid_score"] = cand.index.map(lambda x: bm_rrf.get(str(x), 0.0) + vec_rrf.get(str(x), 0.0)).astype(float)

    cand = cand.sort_values("hybrid_score", ascending=False).head(int(topn_laws))
    return cand

def to_int64_safe(series: pd.Series) -> pd.Series:
    """Convertit en Int64 en neutralisant les infinis."""
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .round(0)
        .astype("Int64")
    )


def _activity_budget_column(df: pd.DataFrame) -> str:
    """Colonne budget au niveau activité: artefact dédié si présent, sinon colonne historique."""
    if "budget_activite" in df.columns:
        return "budget_activite"
    return "budget_moyen_activite"


def _parse_numeric_values(value) -> list[float]:
    """Extrait des nombres depuis des champs comme '23 ; 7'."""
    if pd.isna(value):
        return []
    values = []
    for part in re.split(r"[;,|]", str(value)):
        part = part.strip().replace("\u00a0", " ").replace(" ", "").replace(",", ".")
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError:
            continue
    return values


def _mean_distinct_numeric(values) -> float:
    """Moyenne des valeurs numériques distinctes connues, sans additionner des exercices."""
    nums = []
    seen = set()
    for value in values:
        for num in _parse_numeric_values(value):
            key = round(float(num), 6)
            if key not in seen:
                seen.add(key)
                nums.append(float(num))
    return float(np.mean(nums)) if nums else np.nan


def _format_date_yyyy_mm_dd(series: pd.Series) -> pd.Series:
    dates = pd.to_datetime(series, errors="coerce", dayfirst=True)
    return dates.dt.strftime("%Y-%m-%d").fillna("")

res = st.session_state.get("results", None)

if not isinstance(res, pd.DataFrame) or res.empty:
    st.info("Lance une recherche pour afficher des résultats.")
    st.stop()

# =========================
# TABLEAU ACTIVITÉS + SÉLECTION
# =========================
show_full = add_beneficiaire_column(res.copy())

if "selected" not in show_full.columns:
    show_full.insert(0, "selected", True)

# Nettoyage des budgets dans la version métier conservée pour la suite.
for c in ["budget_total", "budget_moyen_activite", "budget_activite"]:
    if c in show_full.columns:
        show_full[c] = to_int64_safe(show_full[c])

# Copie strictement dédiée à l'affichage: on ne modifie pas les noms métiers
# utilisés ensuite par la matrice, la frise et le payload LLM.
show_display = show_full.copy()
show_display = show_display.drop(
    columns=["label_categorie_organisation", "bm25_score", "vec_score"],
    errors="ignore",
)
show_display = show_display.rename(columns={"hybrid_score": "score de pertinence"})

priority = ["selected", "objet_activite", "denomination", "beneficiaire"]
if "score de pertinence" in show_display.columns:
    priority.append("score de pertinence")
rest = [c for c in show_display.columns if c not in priority]
show_display = show_display.loc[:, [c for c in priority if c in show_display.columns] + rest]

if "date_publication_activite" in show_display.columns:
    show_display["date_publication_activite"] = _format_date_yyyy_mm_dd(show_display["date_publication_activite"])

st.markdown("### Activités de lobbying correspondant à la recherche")
st.markdown("Tous les budgets sont exprimés en €")
st.markdown("Désélectionnez les activités que vous considérez non pertinentes pour votre recherche, puis validez")
st.caption(
    "Les activités de lobbying sont réalisées directement par l'organisation déclarante, "
    "ou par des cabinets de conseil ou d'avocats agissant pour le compte de l'organisation déclarante. "
    "Le bénéficiaire est l'entité qui bénéficie de l'activité de lobbying, "
    "et peut être différent de l'organisation déclarante : sociétés de conseil pour un client, "
    "groupe pour une filiale… "
    "Nous privilégions dans l'analyse les bénéficiaires plutôt que les déclarantes "
    '(« bénéficiaire » versus « dénomination »).'
)

edited = st.data_editor(
    show_display,
    use_container_width=True,
    column_config={
        "selected": st.column_config.CheckboxColumn("Sélection", default=True),
    },
    disabled=[c for c in show_display.columns if c != "selected"],
    key="results_editor",
)
st.caption("Le score de pertinence est un score hybride lexical et sémantique.")

colv1, colv2 = st.columns([1, 4])
with colv1:
    validate = st.button("✅ Valider la sélection", type="primary")
with colv2:
    st.caption("La matrice, la frise chronologique et la synthèse des activités seront produites uniquement avec les lignes sélectionnées.")

if validate:
    selected_index = edited.index[edited["selected"] == True]
    st.session_state.selected_results = (
        show_full.loc[selected_index]
        .drop(columns=["selected"], errors="ignore")
        .copy()
    )
    st.success(f"Sélection validée : {len(st.session_state.selected_results)} activités retenues.")

selected_res = st.session_state.get("selected_results", None)
if not isinstance(selected_res, pd.DataFrame) or selected_res.empty:
    # fallback : tout sélectionné tant qu'on n'a pas validé
    selected_res = show_full.drop(columns=["selected"], errors="ignore").copy()

# =========================
# TABLEAU ORGANISATIONS DÉCLARANTES
# =========================
st.markdown("#### Activités de lobbying par organisation déclarante")

df_tmp = selected_res.copy()
if "activite_id" not in df_tmp.columns:
    df_tmp = df_tmp.reset_index().rename(columns={"index": "activite_id"})
if "activite_id" not in df_tmp.columns:
    df_tmp["activite_id"] = df_tmp.index.astype(str)
df_tmp["activite_id"] = _norm_id_series(df_tmp["activite_id"])

budget_col = _activity_budget_column(df_tmp)
df_tmp[budget_col] = pd.to_numeric(df_tmp.get(budget_col, 0), errors="coerce").fillna(0)

agg_org = {
    "nb_activites_matching": ("activite_id", "nunique"),
    "budget_estime_recherche": (budget_col, "sum"),
}
if "label_categorie_organisation" in df_tmp.columns:
    agg_org["categorie"] = ("label_categorie_organisation", lambda s: _join_unique(s, max_items=20))
if "nb_activites_total" in df_tmp.columns:
    agg_org["nb_activites_total"] = ("nb_activites_total", "max")
if "budget_total" in df_tmp.columns:
    agg_org["budget_total"] = ("budget_total", "max")
if "nombre_salaries" in df_tmp.columns:
    agg_org["nombre_salaries"] = ("nombre_salaries", _mean_distinct_numeric)

org_search = (
    df_tmp.groupby("denomination", dropna=False)
    .agg(**agg_org)
    .reset_index()
)

if "representants_id" in df_tmp.columns:
    org_for_affiliations = (
        df_tmp[["denomination", "representants_id"]]
        .dropna(subset=["denomination"])
        .drop_duplicates()
    )
    affiliations_par_org = build_affiliations_organisations(org_for_affiliations)
    if not affiliations_par_org.empty:
        org_search = org_search.merge(affiliations_par_org, on="denomination", how="left")

for col in [
    "nb_activites_matching",
    "budget_estime_recherche",
    "nb_activites_total",
    "budget_total",
    "nombre_salaries",
    "nb_affiliations",
]:
    if col in org_search.columns:
        org_search[col] = to_int64_safe(org_search[col])

org_cols = [
    "denomination",
    "categorie",
    "nb_activites_matching",
    "budget_estime_recherche",
    "budget_total",
    "nb_activites_total",
    "nombre_salaries",
    "nb_affiliations",
    "denomination_affiliation",
]
org_search = org_search[[c for c in org_cols if c in org_search.columns]]
org_search = org_search.sort_values("budget_estime_recherche", ascending=False)
st.dataframe(org_search, use_container_width=True)

# =========================
# TABLEAU BÉNÉFICIAIRES
# =========================
st.markdown("#### Activités de lobbying par bénéficiaire")

# On reconstruit le tableau bénéficiaires depuis la sélection courante,
# avec la même base d'activités que les tableaux activités / organisations / matrice.
# df_benef_global reste utilisé uniquement pour les statistiques globales tous sujets.
obs_sel = df_observations[["activite_id", "action_representation_interet_id"]].copy()
obs_sel["activite_id"] = _norm_id_series(obs_sel["activite_id"])
obs_sel["action_representation_interet_id"] = _norm_id_series(obs_sel["action_representation_interet_id"])
obs_sel = obs_sel[obs_sel["activite_id"].isin(set(df_tmp["activite_id"].dropna()))].drop_duplicates()

benef_map = df_beneficiaires[["action_representation_interet_id", "beneficiaire_action_menee"]].copy()
benef_map["action_representation_interet_id"] = _norm_id_series(benef_map["action_representation_interet_id"])
benef_map["beneficiaire"] = benef_map["beneficiaire_action_menee"].astype("string").str.strip()
benef_map = benef_map.dropna(subset=["action_representation_interet_id", "beneficiaire"])
benef_map = benef_map.drop_duplicates(subset=["action_representation_interet_id", "beneficiaire"])

activity_budget = df_tmp[["activite_id", budget_col]].drop_duplicates("activite_id").copy()
activity_budget[budget_col] = pd.to_numeric(activity_budget[budget_col], errors="coerce").fillna(0)

benef_detail = (
    obs_sel.merge(
        benef_map[["action_representation_interet_id", "beneficiaire"]],
        on="action_representation_interet_id",
        how="inner",
    )
    .merge(activity_budget, on="activite_id", how="left")
    .drop_duplicates(subset=["activite_id", "beneficiaire"])
)

if not benef_detail.empty:
    benef_search = (
        benef_detail.groupby("beneficiaire", dropna=False)
        .agg(
            nb_activites_matching=("activite_id", "nunique"),
            budget_estime_recherche=(budget_col, "sum"),
        )
        .reset_index()
    )
else:
    # Fallback si aucune table bénéficiaire n'est exploitable : on retombe sur la colonne déjà enrichie.
    fallback = df_tmp.copy()
    benef_col = "beneficiaire" if "beneficiaire" in fallback.columns else "denomination"
    benef_search = (
        fallback.groupby(benef_col, dropna=False)
        .agg(
            nb_activites_matching=("activite_id", "nunique"),
            budget_estime_recherche=(budget_col, "sum"),
        )
        .reset_index()
        .rename(columns={benef_col: "beneficiaire"})
    )

benef_global = df_benef_global.copy()
if "activite_id" in benef_global.columns:
    benef_global["activite_id"] = _norm_id_series(benef_global["activite_id"])
if "budget_activite" in benef_global.columns:
    benef_global["budget_activite"] = pd.to_numeric(benef_global["budget_activite"], errors="coerce").fillna(0)

benef_stats = build_beneficiaire_stats(benef_global)
benef_search = benef_search.merge(benef_stats, on="beneficiaire", how="left")

for col in [
    "nb_activites_matching",
    "budget_estime_recherche",
    "budget_total_beneficiaire",
    "nb_activites_total_beneficiaire",
]:
    if col in benef_search.columns:
        benef_search[col] = to_int64_safe(benef_search[col])

benef_cols = [
    "beneficiaire",
    "nb_activites_matching",
    "budget_estime_recherche",
    "budget_total_beneficiaire",
    "nb_activites_total_beneficiaire",
]
benef_search = benef_search[[c for c in benef_cols if c in benef_search.columns]]
benef_search = benef_search.sort_values("budget_estime_recherche", ascending=False)
st.dataframe(benef_search, use_container_width=True)

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
    df2.groupby(["beneficiaire", "domaines_list"], as_index=False)
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
        cell.groupby("beneficiaire")["budget_total"]
        .sum().sort_values(ascending=False).index.tolist()
    )
    dom_order = (
        cell.groupby("domaines_list")["budget_total"]
        .sum().sort_values(ascending=False).index.tolist()
    )

    cell["beneficiaire"] = pd.Categorical(cell["beneficiaire"], categories=org_order, ordered=True)
    cell["domaines_list"] = pd.Categorical(cell["domaines_list"], categories=dom_order, ordered=True)
    cell = cell.sort_values(["beneficiaire", "domaines_list"])

    cell["hover"] = (
        "<b>Organisation :</b> " + cell["beneficiaire"].astype(str) +
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
            y=cell["beneficiaire"].astype(str),
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
        f"Le budget par organisation/bénéficiaire et domaine est estimé à partir du budget moyen par activité, "
        f"réparti entre les domaines d'activité de chaque activité. "
        f"Seuls les domaines les plus fréquents sont affichés (ici {TOP_DOMAINS})."
        f"Les bulles sont proportionelles aux budgets estimés."
    )

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

for date_col in ["Date initiale", "Date de promulgation", "date_evt", "date_publication_activite"]:
    if date_col in show_laws.columns:
        show_laws[date_col] = _format_date_yyyy_mm_dd(show_laws[date_col])

edited_laws = st.data_editor(
    show_laws,
    use_container_width=True,
    column_config={
        "selected": st.column_config.CheckboxColumn("Sélection", default=True),
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

acts = selected_res.copy()
acts["date_evt"] = pd.to_datetime(acts.get("date_publication_activite"), errors="coerce")
acts = acts.dropna(subset=["date_evt"])
acts = acts[acts["date_evt"] >= START_DATE]

# theme/label/x pour activités
acts["theme"] = acts.get("domaines", "").fillna("").apply(first_theme)
acts["label_full"] = acts.get("beneficiaire", acts["denomination"]).astype(str)

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
    df_llm_exploded.groupby("beneficiaire", dropna=False)
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
    df_llm_exploded.groupby(["beneficiaire", "domaine"], dropna=False)
    .agg(
        budget_cumule_recherche_beneficiaire_domaine=("budget_reparti", "sum"),
        nb_activites_matching_beneficiaire_domaine=("activite_id", "nunique"),
        organisations_declarantes=("denomination", lambda s: list(pd.Series(s).dropna().astype(str).unique())[:12]),
        objets_activite=("objet_activite", lambda s: list(pd.Series(s).dropna().astype(str).unique())[:12]),
    )
    .reset_index()
)

org_theme_llm = org_theme_llm.merge(
    org_llm[["beneficiaire", "budget_cumule_recherche"]],
    on="beneficiaire",
    how="left"
)

_denom = pd.to_numeric(org_theme_llm["budget_cumule_recherche"], errors="coerce").fillna(0)

org_theme_llm["part_du_budget_recherche_beneficiaire_pct"] = np.where(
    _denom > 0,
    100 * pd.to_numeric(
        org_theme_llm["budget_cumule_recherche_beneficiaire_domaine"],
        errors="coerce"
    ).fillna(0) / _denom,
    0.0
)

org_theme_llm["part_du_budget_total_recherche_pct"] = np.where(
    budget_total_recherche > 0,
    100 * org_theme_llm["budget_cumule_recherche_beneficiaire_domaine"]
    / budget_total_recherche,
    0.0
)

org_theme_llm["budget_cumule_recherche_beneficiaire_domaine"] = to_int64_safe(
    org_theme_llm["budget_cumule_recherche_beneficiaire_domaine"]
)

org_theme_llm["nb_activites_matching_beneficiaire_domaine"] = to_int64_safe(
    org_theme_llm["nb_activites_matching_beneficiaire_domaine"]
)

org_theme_llm["part_du_budget_recherche_beneficiaire_pct"] = (
    org_theme_llm["part_du_budget_recherche_beneficiaire_pct"].round(1)
)

org_theme_llm = org_theme_llm.sort_values(
    [
        "budget_cumule_recherche_beneficiaire_domaine",
        "nb_activites_matching_beneficiaire_domaine",
    ],
    ascending=False
)

# ---------- Tableau 3 : domaine
theme_llm = (
    org_theme_llm.groupby("domaine", dropna=False)
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
    0.0
)

theme_llm["budget_cumule_recherche_domaine"] = to_int64_safe(
    theme_llm["budget_cumule_recherche_domaine"]
)

theme_llm["part_budget_recherche_pct"] = theme_llm["part_budget_recherche_pct"].round(1)
theme_llm["nb_beneficiaires"] = to_int64_safe(
    theme_llm["nb_beneficiaires"]
)
theme_llm["nb_activites_matching"] = to_int64_safe(theme_llm["nb_activites_matching"])

theme_llm = theme_llm.sort_values("budget_cumule_recherche_domaine", ascending=False)


# synthèse par LLM        
st.markdown("### Synthèse des résultats")

res = selected_res.copy()

if not isinstance(res, pd.DataFrame) or res.empty:
    st.warning("Aucun résultat. Lancez d'abord une recherche.")
    st.stop()

if st.button("Générer la synthèse (par LLM)", type="primary"):
    activites_llm = res[
        [c for c in ["beneficiaire", "denomination","domaines", "objet_activite"] if c in res.columns]
    ].copy()

    info_llm = {
        "contexte": {
            "requete": st.session_state.get("last_query", ""),
            "nb_activites_retenues": int(res["activite_id"].nunique()) if "activite_id" in res.columns else int(len(res)),
            "budget_total_estime_recherche": round(budget_total_recherche),
        },
        "tableau_beneficiaires": org_llm.to_dict(orient="records"),
        "tableau_beneficiaires_domaines": org_theme_llm.to_dict(orient="records"),
        "tableau_domaines": theme_llm.to_dict(orient="records"),
        "activites_selectionnees": activites_llm.to_dict(orient="records"),
    }

    with st.spinner("Synthèse…"):
        txt = summarize_activites(info_llm)

    st.session_state.synth = txt

if st.session_state.get("synth"):
    st.text_area("Synthèse", st.session_state.synth, height=500)
