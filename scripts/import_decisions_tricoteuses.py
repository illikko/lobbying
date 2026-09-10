from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import requests

from src.config import (
    PATHS,
    TRICOTEUSES_DECISIONS_API_TOKEN,
    TRICOTEUSES_DECISIONS_API_URL,
    TRICOTEUSES_DECISIONS_FROM_YEAR,
    TRICOTEUSES_DECISIONS_PAGE_SIZE,
)

ALLOWED_TYPES = {"LOI", "DECRET", "ORDONNANCE", "ARRETE"}
EXCLUDED_TYPES = {"AMENDEMENT"}


def _walk(obj: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield str(key), value
            yield from _walk(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _walk(value)


def _first_value(obj: Any, keys: set[str]) -> Any:
    wanted = {k.casefold() for k in keys}
    for key, value in _walk(obj):
        if key.casefold() in wanted and value not in (None, "", [], {}):
            if isinstance(value, (str, int, float, bool)):
                return value
    return None


def _all_strings(obj: Any, keys: set[str]) -> list[str]:
    wanted = {k.casefold() for k in keys}
    out: list[str] = []
    for key, value in _walk(obj):
        if key.casefold() not in wanted:
            continue
        values = value if isinstance(value, list) else [value]
        for item in values:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
    return list(dict.fromkeys(out))


def _normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _extract_description(data: Any) -> str:
    candidates: list[str] = []

    notice_values = _all_strings(
        data,
        {
            "NOTICE",
            "notice",
            "description",
            "DESCRIPTION",
            "resume",
            "RESUME",
            "abstract",
            "ABSTRACT",
            "summary",
            "SUMMARY",
        },
    )
    candidates.extend(notice_values)

    meta_notice = _first_value(
        data,
        {
            "META_NOTICE",
            "metaNotice",
            "noticeTexte",
            "notice_texte",
            "texteNotice",
        },
    )
    if isinstance(meta_notice, str) and meta_notice.strip():
        candidates.append(meta_notice)

    unique = []
    seen: set[str] = set()
    for value in candidates:
        text = _normalize_space(value)
        if not text:
            continue
        if text not in seen:
            seen.add(text)
            unique.append(text)

    return " ".join(unique)


def normalize_type(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value).strip().upper()
    raw = raw.replace("É", "E").replace("È", "E").replace("Ê", "E").replace("À", "A")
    # Les natures Légifrance peuvent être plus détaillées : DECRET_LOI, ARRETE, etc.
    if "AMENDEMENT" in raw:
        return "AMENDEMENT"
    if "ORDONNANCE" in raw:
        return "ORDONNANCE"
    if "DECRET" in raw:
        return "DECRET"
    if "ARRETE" in raw:
        return "ARRETE"
    if re.search(r"(^|_)LOI($|_)", raw) or raw == "LOI":
        return "LOI"
    return raw or None


def _parse_date(value: Any) -> pd.Timestamp | pd.NaT:
    if value is None:
        return pd.NaT
    text = str(value).strip()
    if not text:
        return pd.NaT
    return pd.to_datetime(text, errors="coerce", format="%Y-%m-%d")


def parse_record(row: dict[str, Any]) -> dict[str, Any] | None:
    data = row.get("data") if isinstance(row.get("data"), dict) else row
    nature = normalize_type(
        _first_value(data, {"NATURE", "nature", "typeTexte", "type_texte", "type", "natureTexte"})
    )
    if nature in EXCLUDED_TYPES or nature not in ALLOWED_TYPES:
        return None

    title = _first_value(data, {"TITRE", "titre", "TITREFULL", "titreLong", "titreTexte"})
    if not title:
        return None

    publication_date = _parse_date(
        _first_value(data, {"DATE_PUBLI", "datePublication", "date_publication", "publicationDate", "DATE_TEXTE", "dateTexte"})
    )
    initial_date = _parse_date(
        _first_value(data, {"DATE_TEXTE", "dateTexte", "dateSignature", "dateDebut", "DATE_DEBUT"})
    )
    if pd.isna(initial_date):
        initial_date = publication_date

    identifier = row.get("id") or _first_value(data, {"ID", "id", "CID", "cid", "NOR", "nor", "numero", "NUM"})
    nor = _first_value(data, {"NOR", "nor"})
    number = _first_value(data, {"NUM", "numero", "num", "numeroTexte"})
    url = _first_value(data, {"URL", "url", "URL_SOURCE", "urlSource"})
    if not url and identifier:
        ident = str(identifier).strip()
        if ident.startswith("JORFTEXT") or ident.startswith("LEGITEXT"):
            url = f"https://www.legifrance.gouv.fr/loda/id/{ident}"

    themes = _all_strings(data, {"THEME", "themes", "motsCles", "mots_cles", "subject"})
    description = _extract_description(data)

    return {
        "decision_id": str(identifier or "").strip(),
        "type_decision": nature.lower(),
        "Date initiale": initial_date,
        "Date de promulgation": publication_date,
        "Titre": str(title).strip(),
        "Description": description,
        "Numéro de la loi": str(number or nor or identifier or "").strip(),
        "Thèmes": "; ".join(themes),
        "État du dossier": "publié",
        "URL du dossier": str(url or "").strip(),
        "nor": str(nor or "").strip(),
        "source": "tricoteuses-canutes-legifrance",
    }


def fetch_page(api_url: str, limit: int, offset: int, token: str = "", timeout: int = 90) -> list[dict[str, Any]]:
    headers = {
        "Accept": "application/json",
        "Accept-Profile": "legifrance",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = requests.get(
        api_url,
        params={
            "nature": "in.(LOI,DECRET,ORDONNANCE,ARRETE)",
            "data->META->META_SPEC->META_TEXTE_CHRONICLE->>DATE_PUBLI": "gte.2017-01-01",
            "data->META->META_COMMUN->>ORIGINE": "eq.JORF",
            "limit": limit, 
            "offset": offset},
        headers=headers,
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError("La réponse Canutes/Légifrance n'est pas une liste PostgREST.")
    return [x for x in payload if isinstance(x, dict)]


def fetch_all(api_url: str, page_size: int, token: str = "") -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = fetch_page(api_url, page_size, offset, token=token)
        if not page:
            break
        out.extend(page)
        offset += len(page)
        print(f"  décisions source lues: {offset:,}", flush=True)
        if len(page) < page_size:
            break
    return out


def build_dataframe(rows: list[dict[str, Any]], from_year: int) -> pd.DataFrame:
    parsed = [record for row in rows if (record := parse_record(row)) is not None]
    if not parsed:
        return pd.DataFrame(columns=[
            "decision_id", "type_decision", "Date initiale", "Date de promulgation", "Titre", "Description",
            "Numéro de la loi", "Thèmes", "État du dossier", "URL du dossier", "nor", "source"
        ])
    df = pd.DataFrame(parsed)
    df["Date initiale"] = pd.to_datetime(df["Date initiale"], errors="coerce")
    df["Date de promulgation"] = pd.to_datetime(df["Date de promulgation"], errors="coerce")
    date_ref = df["Date initiale"].fillna(df["Date de promulgation"])
    df = df[date_ref.dt.year.fillna(0).astype(int) >= int(from_year)].copy()
    df = df[~df["type_decision"].eq("amendement")]
    df = df.drop_duplicates(subset=["decision_id", "type_decision", "Titre"], keep="last")
    return df.sort_values(["Date initiale", "type_decision", "Titre"], na_position="last").reset_index(drop=True)


def run(api_url: str, output: Path, from_year: int, page_size: int, token: str = "") -> dict[str, Any]:
    rows = fetch_all(api_url, page_size=page_size, token=token)
    df = build_dataframe(rows, from_year=from_year)
    if df.empty:
        raise RuntimeError(
            "Aucune loi/décret/ordonnance/arrêté trouvé. Vérifiez TRICOTEUSES_DECISIONS_API_URL "
            "et la ressource PostgREST utilisée."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(output)
    counts = df["type_decision"].value_counts().sort_index().to_dict()
    manifest = {
        "source": api_url,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "from_year": from_year,
        "rows": int(len(df)),
        "counts_by_type": {str(k): int(v) for k, v in counts.items()},
        "amendements_inclus": False,
    }
    output.with_name("decisions_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Importe lois, décrets, ordonnances et arrêtés depuis Canutes/Légifrance")
    parser.add_argument("--api-url", default=TRICOTEUSES_DECISIONS_API_URL)
    parser.add_argument("--output", type=Path, default=PATHS.decisions_raw)
    parser.add_argument("--from-year", type=int, default=TRICOTEUSES_DECISIONS_FROM_YEAR)
    parser.add_argument("--page-size", type=int, default=TRICOTEUSES_DECISIONS_PAGE_SIZE)
    args = parser.parse_args()
    result = run(
        api_url=args.api_url,
        output=args.output.resolve(),
        from_year=args.from_year,
        page_size=args.page_size,
        token=TRICOTEUSES_DECISIONS_API_TOKEN,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
