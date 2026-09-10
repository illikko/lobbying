from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requests


CATALOG_API_URL = "https://db.code4code.eu/canutes/texte_version"
ARTICLES_API_URL = "https://db.code4code.eu/canutes/article"
HEADERS = {
    "Accept": "application/json",
    "Accept-Profile": "legifrance",
}
TEXT_TYPES = ("LOI", "DECRET", "ORDONNANCE", "ARRETE")


def _walk(obj: Any, path: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_path = f"{path}.{key}" if path else str(key)
            yield next_path, value
            yield from _walk(value, next_path)
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            next_path = f"{path}[{index}]"
            yield next_path, value
            yield from _walk(value, next_path)


def _scalar_candidates(obj: Any) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for path, value in _walk(obj):
        if isinstance(value, (str, int, float, bool)) and str(value).strip():
            out.append((path, str(value).strip()))
    return out


def _interesting_keys(obj: Any) -> list[tuple[str, str]]:
    patterns = ("id", "cid", "nor", "num", "texte", "article", "jorf", "legi")
    hits: list[tuple[str, str]] = []
    for path, value in _scalar_candidates(obj):
        low_path = path.lower()
        low_value = value.lower()
        if any(pattern in low_path for pattern in patterns) or any(
            token in low_value for token in ("jorf", "legi")
        ):
            hits.append((path, value))
    return hits


def _request(url: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    response = requests.get(url, headers=HEADERS, params=params, timeout=90)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError(f"Réponse inattendue pour {url}: {type(payload)!r}")
    return [item for item in payload if isinstance(item, dict)]


def fetch_catalog_sample(limit: int) -> list[dict[str, Any]]:
    return _request(
        CATALOG_API_URL,
        {
            "nature": f"in.({','.join(TEXT_TYPES)})",
            "data->META->META_SPEC->META_TEXTE_CHRONICLE->>DATE_PUBLI": "gte.2018-01-01",
            "data->META->META_COMMUN->>ORIGINE": "eq.JORF",
            "limit": limit,
        },
    )


def fetch_article_sample(limit: int) -> list[dict[str, Any]]:
    return _request(ARTICLES_API_URL, {"limit": limit})


def _postgrest_eq(field: str, value: str) -> dict[str, str]:
    return {field: f"eq.{value}"}


def build_article_queries(decision_row: dict[str, Any]) -> list[tuple[str, dict[str, str]]]:
    data = decision_row.get("data") if isinstance(decision_row.get("data"), dict) else decision_row
    candidates = _interesting_keys(decision_row)
    queries: list[tuple[str, dict[str, str]]] = []
    seen: set[tuple[str, str]] = set()

    explicit_values = []
    if isinstance(decision_row.get("id"), str):
        explicit_values.append(("row.id", str(decision_row["id"]).strip()))
    for path, value in candidates:
        explicit_values.append((path, value))

    article_fields = [
        "id",
        "cid",
        "cidtexte",
        "data->>id",
        "data->>cid",
        "data->>cidtexte",
        "data->META->META_COMMUN->>ID",
        "data->META->META_COMMUN->>CID",
        "data->META->META_COMMUN->>CIDTEXTE",
        "data->META->META_SPEC->META_ARTICLE->>ID",
        "data->META->META_SPEC->META_ARTICLE->>CID",
        "data->META->META_SPEC->META_ARTICLE->>CIDTEXTE",
        "data->META->META_SPEC->META_TEXTE_CHRONICLE->>CID",
        "data->META->META_SPEC->META_TEXTE_CHRONICLE->>CIDTEXTE",
        "data->META->META_SPEC->META_TEXTE_CHRONICLE->>NUM",
        "data->LIEN_TXT->LIEN_TXT->>id",
        "data->LIEN_TXT->LIEN_TXT->>cid",
    ]

    for source_path, source_value in explicit_values:
        if not source_value:
            continue
        for field in article_fields:
            key = (field, source_value)
            if key in seen:
                continue
            seen.add(key)
            label = f"{field} == {source_path} ({source_value})"
            queries.append((label, _postgrest_eq(field, source_value)))

    return queries


def probe_join(decision_row: dict[str, Any], max_queries: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for label, params in build_article_queries(decision_row)[:max_queries]:
        query_params = {"limit": 3, **params}
        try:
            rows = _request(ARTICLES_API_URL, query_params)
            count = len(rows)
            sample = rows[0] if rows else None
            results.append(
                {
                    "label": label,
                    "params": query_params,
                    "count": count,
                    "sample_keys": sorted(sample.keys())[:12] if isinstance(sample, dict) else [],
                }
            )
        except Exception as exc:
            results.append(
                {
                    "label": label,
                    "params": query_params,
                    "count": None,
                    "error": str(exc),
                }
            )
    return results


def summarize_decision_row(row: dict[str, Any]) -> dict[str, Any]:
    data = row.get("data") if isinstance(row.get("data"), dict) else row
    interesting = _interesting_keys(row)
    return {
        "top_level_keys": sorted(row.keys()),
        "interesting_identifiers": interesting[:30],
        "title_like_values": [
            (path, value)
            for path, value in _scalar_candidates(data)
            if "titre" in path.lower() or "title" in path.lower()
        ][:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnostique les clés de jointure probables entre texte_version et article."
    )
    parser.add_argument("--catalog-limit", type=int, default=5)
    parser.add_argument("--article-limit", type=int, default=1)
    parser.add_argument("--max-queries", type=int, default=30)
    args = parser.parse_args()

    catalog_rows = fetch_catalog_sample(limit=args.catalog_limit)
    article_rows = fetch_article_sample(limit=args.article_limit)

    report: dict[str, Any] = {
        "catalog_sample_size": len(catalog_rows),
        "article_sample_size": len(article_rows),
        "article_sample_identifiers": _interesting_keys(article_rows[0])[:30] if article_rows else [],
        "decisions": [],
    }

    for row in catalog_rows:
        entry = {
            "decision_summary": summarize_decision_row(row),
            "join_probes": probe_join(row, max_queries=args.max_queries),
        }
        report["decisions"].append(entry)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
