# LobbySearch

LobbySearch est une application expérimentale de recherche et de visualisation autour des activités de représentation d'intérêts en France.

L'application croise deux sources principales :

- les données ouvertes HATVP sur les représentants d'intérêts et leurs activités ;
- des décisions publiques issues de Canutes/Légifrance.

Elle permet ensuite d'explorer ces données dans une interface Streamlit avec recherche, tableaux interactifs, visualisations et synthèse textuelle.

## Objectif

Le projet vise à :

- rechercher des activités de lobbying à partir de mots-clés ou de requêtes en langage naturel ;
- rapprocher ces activités de décisions publiques pertinentes ;
- explorer les résultats sous forme de tableaux, matrice bénéficiaires x domaines et chronologie ;
- produire, en option, une synthèse textuelle par LLM.

## Fonctionnement général

Le dépôt contient deux usages distincts.

### 1. Usage local

Le pipeline local est conçu pour rester simple :

- récupération/nettoyage HATVP via l'image Docker `tricoteuses-hatvp` ;
- import des décisions publiques depuis l'API Canutes/Légifrance ;
- transformation des données en tables Parquet ;
- construction des artefacts de recherche `BM25` ;
- lancement de l'interface Streamlit locale.

En local, le flux standard n'utilise ni embeddings ni FAISS.

### 2. Usage CI / reconstruction complète

Le pipeline complet construit en plus :

- les embeddings ;
- les index FAISS ;
- un snapshot d'artefacts cohérent côté publication.

Ce mode est réservé à la CI / Forgejo ou à des usages de maintenance explicites.

## Architecture simplifiée

```text
HATVP via Tricoteuses (Docker)      Décisions publiques via API Canutes/Légifrance
               |                                      |
               v                                      v
     data/tricoteuses/                        data/decisions/decisions.parquet
               |                                      |
               +-------------------+------------------+
                                   |
                                   v
                   scripts/import_tricoteuses.py
                                   |
                                   v
                        data/imported/*.parquet
                                   |
                                   v
                    build_artifacts.py --no-faiss
                                   |
                                   v
                              artifacts/
                                   |
                                   v
                         app_serving_test.py
```

## Structure du dépôt

```text
.
├── app_serving.py
├── app_serving_test.py
├── builder_app.py
├── build_artifacts.py
├── requirements.txt
├── requirements-local.txt
├── scripts/
│   ├── fetch_tricoteuses.py
│   ├── import_tricoteuses.py
│   ├── import_decisions_tricoteuses.py
│   ├── sync_local_bm25.py
│   ├── sync_all.py
│   └── verify_alignment.py
├── src/
│   ├── activity_analytics.py
│   ├── bm25_index.py
│   ├── config.py
│   ├── embed_index.py
│   ├── io_artifacts.py
│   ├── llm_summarize.py
│   ├── prep.py
│   ├── provenance.py
│   ├── search_backend.py
│   └── textnorm.py
├── data/
├── artifacts/
├── local_test_artifacts/
└── tests/
```

## Prérequis

Pour un usage local standard :

- Python 3.11 ou 3.12 ;
- Docker Desktop installé et démarré.

Aucun besoin d'installer Node.js ou npm sur Windows : ces étapes sont exécutées dans le conteneur Docker Tricoteuses.

## Installation

Créer un environnement virtuel puis installer les dépendances locales :

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements-local.txt
```

Pour une reconstruction complète avec FAISS et embeddings, utiliser `requirements.txt` à la place.

## Configuration

Le projet charge sa configuration depuis un fichier `.env` à la racine du dépôt.

Variables principales :

```env
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini

TRICOTEUSES_DATA_DIR=./data/tricoteuses
TRICOTEUSES_IMAGE=git.tricoteuses.fr/logiciels/tricoteuses-hatvp:latest

TRICOTEUSES_DECISIONS_API_URL=https://db.code4code.eu/canutes/texte_version
TRICOTEUSES_DECISIONS_API_TOKEN=
TRICOTEUSES_DECISIONS_FROM_YEAR=2018
TRICOTEUSES_DECISIONS_PAGE_SIZE=1000

LOBBYSEARCH_ARTIFACTS_DIR=./artifacts
TRICOTEUSES_DECISIONS_FILE=./data/decisions/decisions.parquet

```

Notes :

- `OPENAI_API_KEY` n'est pas nécessaire pour reconstruire les données ou les index BM25 ;
- elle devient utile pour la fonctionnalité de synthèse LLM ;
- l'URL par défaut des décisions publiques dans le code est `https://db.code4code.eu/canutes/texte_version`.

## Pipeline local recommandé

La commande standard pour reconstruire le pipeline local est :

```powershell
.\.venv\Scripts\python.exe scripts\sync_local_bm25.py
```

Cette commande effectue :

1. la récupération/nettoyage des données HATVP via Docker ;
2. l'import des décisions publiques ;
3. la transformation des JSON Tricoteuses en tables Parquet ;
4. la construction des artefacts applicatifs et des index BM25, sans FAISS.

Options utiles :

```powershell
.\.venv\Scripts\python.exe scripts\sync_local_bm25.py --no-fetch-hatvp
.\.venv\Scripts\python.exe scripts\sync_local_bm25.py --no-fetch-decisions
.\.venv\Scripts\python.exe scripts\sync_local_bm25.py --force-import
.\.venv\Scripts\python.exe scripts\sync_local_bm25.py --strict-hatvp
```

## Lancer l'application

### Interface locale BM25

```powershell
.\.venv\Scripts\python.exe -m streamlit run app_serving_test.py
```

### Interface complète

```powershell
.\.venv\Scripts\python.exe -m streamlit run app_serving.py
```

`app_serving.py` peut exploiter FAISS si les artefacts et dépendances correspondants sont disponibles.


## Déploiement de test sur Render (BM25, sans reconstruction)

Le fichier `render.yaml` configure un Web Service Python sur le plan **Free**,
avec `app_serving_test.py` et `requirements-render.txt` (sans Torch, FAISS ni
sentence-transformers). La synthèse LLM reste disponible au clic ; renseigner
`OPENAI_API_KEY` dans les variables secrètes Render. La clé n'est pas nécessaire
à la recherche BM25. Les appels de synthèse sont facturés séparément par le
fournisseur de l'API.

Avant publication, inclure dans le commit les fichiers de configuration, le code
et les huit artefacts préconstruits du **même snapshot** :

```text
artifacts/df_activites_min.parquet
artifacts/df_lois_min.parquet
artifacts/bm25_activites.joblib
artifacts/bm25_lois.joblib
artifacts/df_observations.parquet
artifacts/df_beneficiaires.parquet
artifacts/df_affiliations.parquet
artifacts/df_informations_generales.parquet
```

Les artefacts déjà suivis par Git doivent être publiés dans leur version locale
souhaitée ; `.gitignore` ne retire pas les fichiers déjà suivis. Une exception
autorise aussi l'ajout de `df_informations_generales.parquet`.
Les données sources, `docs_*.parquet`, les embeddings et les index FAISS ne sont
pas nécessaires à ce service.

Créer un Blueprint Render depuis le dépôt et sélectionner `render.yaml`.
Pour configurer un service manuellement, reprendre les commandes et variables
de ce fichier. Le build installe les dépendances puis vérifie la présence des
artefacts (`python -m scripts.check_serving_snapshot`) ; il n'exécute aucun
pipeline de données.

Avec `LOBBYSEARCH_PREBUILT_ONLY=1`, l'application charge uniquement
`LOBBYSEARCH_ARTIFACTS_DIR` et contrôle les identifiants des index. Un fichier
absent ou des identifiants incohérents arrêtent le chargement avec un message
d'erreur, sans repli vers les CSV, les données importées ou le snapshot historique,
et sans construction de BM25. Ce contrôle des identifiants ne remplace pas la
publication d'un snapshot cohérent (il ne compare pas le contenu textuel).
Le mode local habituel reste disponible sans cette variable.

Le plan gratuit dispose de 512 Mo de RAM : la taille compressée des Parquet ne
permet pas de prédire la mémoire consommée. Vérifier les métriques Render au
démarrage et lors des recherches avant de considérer le corpus complet comme
compatible. Aucun passage automatique à un plan payant n'est configuré.
Les artefacts livrés avec le déploiement ne nécessitent pas de disque persistant.
Voir les [limites Render Free](https://render.com/docs/free).

## Scripts principaux

### `scripts/fetch_tricoteuses.py`

Récupère les données HATVP via l'image Docker Tricoteuses et écrit les JSON nettoyés dans `data/tricoteuses/`.

### `scripts/import_tricoteuses.py`

Transforme les JSON HATVP en tables relationnelles Parquet dans `data/imported/`.

### `scripts/import_decisions_tricoteuses.py`

Interroge l'API Canutes/Légifrance et produit `data/decisions/decisions.parquet`, avec une colonne `Description` extraite en priorité depuis les notices des textes quand elle est disponible.

### `scripts/sync_local_bm25.py`

Pipeline local unique : fetch HATVP, import décisions, import Parquet, build BM25.

### `scripts/sync_all.py`

Pipeline complet pour CI / maintenance : reconstruction atomique Parquet + BM25 + FAISS. Ce script est volontairement bloqué en local sauf déverrouillage explicite.

### `build_artifacts.py`

Construit les artefacts utilisés par l'application :

- `df_activites_min.parquet`
- `df_lois_min.parquet`
- `bm25_activites.joblib`
- `bm25_lois.joblib`
- fichiers FAISS et mappings d'identifiants si activés
- `manifest.json`

## Tests

Exécuter les tests :

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## État du projet

Le projet est à un stade expérimental.

Points à garder en tête :

- l'interface et le pipeline évoluent encore ;
- la reconstruction locale standard privilégie la simplicité et évite FAISS ;
- la publication d'artefacts complets repose sur un pipeline CI distinct.

## Licence

Voir `LICENCE.txt`.
