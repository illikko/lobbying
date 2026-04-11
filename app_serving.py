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


# --- Query embedding with sentence-transformers is NOT used on Render.
# We must embed the query using the same model space as offline embeddings.
# Simplest: also ship sentence-transformers on Render and encode just the query (light).
# If you want to avoid torch on Render, you'd need an external embedding service.
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Cartographie des influences", layout="wide")
st.title("Cartographie des influences")

if "results" not in st.session_state:
    st.session_state.results = None

with st.expander("ℹ️ À propos de l'application", expanded=False):
        st.write("""
        Cette application permet de cartographier les activités de lobbying en France.
        Elle utilise une recherche hybride lexical et sémantique pour trouver les activités les plus pertinentes par rapport à une requête comprenant plusieurs mots clé.
            ex.: transport, fret, fiscalité
        Les résultats de la recherche sont présentés sous forme de 
        - tableau avec des filtres interactifs
        - une matrice à bulles représentant les organisations et les domaines d'activité, 
        - une frise chronoligique des activités de lobbying et des lois
        - une synthèse par IA des activités de lobbying.
        
        Elle utilise les données ouvertes de la HATPV sur les activités de lobbying (environ 100K activités) et les lois du Sénat (envrion 6000 lois).
        L'application est à un stade d'expérimentation, et évolue en fonction des retours de ses utilisateurs.
        Elle est développée par Vincent Castaignet, un data analyst/scientist freelance.
        """)

with st.expander("ℹ️ Comment ça marche ?", expanded=False):
        st.write("""
        - écrire les mots clés recherchés dans le formulaire "mots-clés / requête", puis cliquer sur "Lancer" pour obtenir les résultats de recherche
        - lire les résultats (objet_activite), et dé-sélectionner les activités qui ne sont pas pertinentes (colonne "Sélection"), puis cliquer sur "Valider la sélection" pour confirmer les activités retenues
        - faire de même pour les lois correspondant à la recherche (dé-sélectionner les lois non pertinentes, puis valider)
        - explorer les différentes visualisations (matrice à bulles, frise chronologique)
        - dans l'onglet "Synthèse", cliquer sur "Générer la synthèse" pour obtenir une synthèse textuelle des activités de lobbying retenues, par un modèle de langage (LLM).
        - utiliser les filtres pour ajuster la recherche: nombre de résultats demandés, budget, période.
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

    # rebuild FaissBundle doc_ids in index order
    doc_ids = id_map["activite_id"].astype(object).to_numpy()
    faiss_bundle = FaissBundle(index=faiss_index, doc_ids=doc_ids, normalize=True)
    loi_ids = id_map_lois["loi_id"].astype(object).to_numpy()
    faiss_lois_bundle = FaissBundle(index=faiss_lois_index, doc_ids=loi_ids, normalize=True)

    # model name from manifest (to stay consistent)
    st_model = (manifest.get("config", {}) or {}).get("st_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    embedder = SentenceTransformer(st_model)

    return manifest, df_acts, df_lois, docs, bm25_bundle, faiss_bundle, bm25_lois_bundle, faiss_lois_bundle, embedder

manifest, df_acts, df_lois, docs, bm25_bundle, faiss_bundle, bm25_lois_bundle, faiss_lois_bundle, embedder = load_all()

def embed_query_fn(q: str) -> np.ndarray:
    q = "" if q is None else str(q)
    v = embedder.encode([q], convert_to_numpy=True)[0]
    return v.astype("float32", copy=False)

# --- UI
st.markdown("### Recherche des activités de lobbying")

# year bounds
years = pd.to_datetime(df_acts["date_publication_activite"], errors="coerce").dropna().dt.year
y_min = int(years.min()) if not years.empty else 2018
y_max = int(years.max()) if not years.empty else 2026

query = st.text_input("Mots-clés / requête", value="")

c1, c2, c3 = st.columns(3)
with c1:
    topn = st.slider("Top N", 10, 200, 20, 10)
with c2:
    min_budget = st.slider("Budget moyen/activité min (€)", 0, 5000, 0, 100)
with c3:
    year_range = st.slider("Période (année)", y_min, y_max, (y_min, y_max))

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
    st.info("Lance une recherche pour afficher des résultats.")
    st.stop()


def hybrid_search_lois_local(
    query_text: str,
    topn_laws: int = 50,
    k_bm25: int = 200,
    k_vec: int = 200,
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
        show[c] = pd.to_numeric(show[c], errors="coerce").round(0).astype("Int64")



st.markdown("### Résultats des activités de lobbying")
st.markdown("Désélectionnez ce qui n'est pas pertinent, puis validez")
    
edited = st.data_editor(
    show,
    use_container_width=True,
    column_config={"selected": st.column_config.CheckboxColumn("Sélection", default=True)},
    disabled=[c for c in show.columns if c != "selected"],
    key="results_editor",
)

colv1, colv2 = st.columns([1, 4])
with colv1:
    validate = st.button("✅ Valider la sélection", type="primary")
with colv2:
    st.caption("La matrice et les lois correspondantes seront calculées uniquement sur les lignes sélectionnées.")

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
st.markdown("#### Budget estimé sur la recherche (par organisation)")

df_tmp = selected_res.copy()
if "activite_id" not in df_tmp.columns:
    df_tmp = df_tmp.reset_index().rename(columns={"index": "activite_id"})
if "activite_id" not in df_tmp.columns:
    df_tmp["activite_id"] = df_tmp.index.astype(str)

org_search = (
    df_tmp.groupby("denomination", dropna=False)
    .agg(
        nb_activites_matching=("activite_id", "nunique"),
        budget_moyen_activite=("budget_moyen_activite", "first"),
        budget_total_org=("budget_total", "first"),
        nb_activites_total_org=("nb_activites_total", "first"),
    )
    .reset_index()
)
org_search["budget_estime_recherche"] = (
    pd.to_numeric(org_search["budget_moyen_activite"], errors="coerce").fillna(0.0)
    * pd.to_numeric(org_search["nb_activites_matching"], errors="coerce").fillna(0.0)
)
org_search = org_search.sort_values("budget_estime_recherche", ascending=False)

with st.expander("Détail budget estimé (budget_moyen_activite × nb_activites_matching)", expanded=False):
    st.dataframe(org_search, use_container_width=True)

# =========================
# MATRICE BULLES org x domaines - SUR selected_res
# =========================
st.markdown("### Matrice organisations × domaines")

MIN_BUDGET = float(min_budget)
TOP_DOMAINS = 12  # tu peux en faire un slider

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
        "<br><b>Budget estimé sur la recherche (réparti) :</b> " + cell["budget_total"].round(0).astype(int).astype(str) +
        "<br><b>Nb activités matching :</b> " + cell["nb_activites"].astype(int).astype(str) +
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
        title=f"Matrice org × domaines (bulles ∝ budget estimé réparti) — filtre ≥ {MIN_BUDGET:.0f} €, top {TOP_DOMAINS} domaines",
        xaxis=dict(title="Domaines", categoryorder="array", categoryarray=dom_order, tickangle=45),
        yaxis=dict(title="Organisations", categoryorder="array", categoryarray=org_order, autorange="reversed"),
        height=max(650, 28 * len(org_order) + 240),
        margin=dict(l=left_margin, r=10, t=80, b=120),
    )

    figm.update_yaxes(showgrid=True, automargin=True)
    figm.update_xaxes(showgrid=True)

    st.plotly_chart(figm, use_container_width=True)

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

with st.expander("Paramètres Lois", expanded=False):
    topn_laws = st.slider("Top N lois", 5, 200, 20, 5)
    apply_year_filter_laws = st.checkbox("Appliquer filtre période aux lois", value=True)
    
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
edited_laws = st.data_editor(
    show_laws,
    use_container_width=True,
    column_config={"selected": st.column_config.CheckboxColumn("Sélection", default=True)},
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

# synthèse par LLM        
st.markdown("### Synthèse LLM")

res = st.session_state.get("selected_results", None)
if not isinstance(res, pd.DataFrame) or res.empty:
    # fallback : si l’utilisateur n’a pas validé la sélection, on prend les résultats bruts
    res = st.session_state.get("results", None)
    
if not isinstance(res, pd.DataFrame) or res.empty:
    st.warning("Aucun résultat. Lance d'abord une recherche.")
    st.stop()

if st.button("Générer la synthèse", type="primary"):
    info_llm = res[[c for c in ["denomination", "budget_moyen_activite", "domaines", "objet_activite"] if c in res.columns]].to_dict(orient="records")
    with st.spinner("Synthèse…"):
        txt = summarize_activites(info_llm)
    st.session_state.synth = txt

if st.session_state.get("synth"):
    st.text_area("Synthèse", st.session_state.synth, height=500)
