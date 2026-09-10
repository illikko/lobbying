from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PATHS


TABLE_FILES = {
    "informations_generales": "informations_generales.parquet",
    "affiliations": "affiliations.parquet",
    "domaines_intervention": "domaines_intervention.parquet",
    "objets_activites": "objets_activites.parquet",
    "beneficiaires": "beneficiaires.parquet",
    "observations": "observations.parquet",
    "exercices": "exercices.parquet",
}


def source_files(data_dir: Path) -> list[Path]:
    root = data_dir / "repertoire_representants_interets"

    if not root.exists():
        raise FileNotFoundError(
            f"Données Tricoteuses introuvables : {root}. "
            "Exécutez d'abord scripts/fetch_tricoteuses.py."
        )

    files = sorted(root.glob("*.json"))

    if not files:
        raise FileNotFoundError(
            f"Aucun JSON représentant trouvé dans {root}"
        )

    return files


def fingerprint(files: list[Path]) -> str:
    h = hashlib.sha256()

    for path in files:
        h.update(path.name.encode("utf-8"))

        with path.open("rb") as f:
            for chunk in iter(
                lambda: f.read(1024 * 1024),
                b"",
            ):
                h.update(chunk)

    return h.hexdigest()


def rep_id(entity: dict) -> str:
    ident = str(
        entity.get("identifiantNational") or ""
    ).strip()

    typ = str(
        entity.get("typeIdentifiantNational") or "HATVP"
    ).strip() or "HATVP"

    if not ident:
        raise ValueError(
            "Un représentant Tricoteuses n'a pas "
            "d'identifiantNational."
        )

    return f"{typ}:{ident}"


def parse_money_range(
    value: object,
) -> tuple[float | None, float | None]:
    if value is None:
        return None, None

    text = str(value).lower().replace("€", "euros")

    nums = [
        float(n.replace(" ", ""))
        for n in re.findall(r"\d[\d ]*", text)
    ]

    if not nums:
        return None, None

    if text.lstrip().startswith("<"):
        return 0.0, nums[0]

    if len(nums) >= 2:
        return nums[0], nums[1]

    if ">" in text:
        return nums[0], nums[0]

    return nums[0], nums[0]


def parse_beneficiary(
    value: object,
) -> tuple[str, bool]:
    text = str(value or "").strip()

    en_propre = bool(
        re.search(
            r"\s*\(en\s+propre\)\s*$",
            text,
            flags=re.I,
        )
    )

    if en_propre:
        text = re.sub(
            r"\s*\(en\s+propre\)\s*$",
            "",
            text,
            flags=re.I,
        ).strip()

    return text, en_propre


def _build_informations_generales(
    infos: list[dict],
) -> pd.DataFrame:
    """
    Construit une seule ligne d'identité par representants_id.

    Cas normal :
      - une seule ligne -> conservée telle quelle.

    Doublons strictement identiques :
      - supprimés.

    Même identifiant / même organisation mais catégories HATVP différentes :
      - fusion en une seule ligne ;
      - toutes les catégories sont conservées dans
        `label_categorie_organisation`, séparées par " | " ;
      - `categorie_organisation_multiple` vaut True.

    Vraie collision (même representants_id mais SIREN/type/dénomination
    différents) :
      - erreur explicite, afin de ne jamais masquer un conflit métier.
    """
    df = (
        pd.DataFrame(infos)
        .drop_duplicates()
        .reset_index(drop=True)
    )

    if df.empty:
        return df

    rows: list[dict] = []

    for rid, group in df.groupby(
        "representants_id",
        sort=True,
        dropna=False,
    ):
        def unique_values(column: str) -> list[str]:
            if column not in group.columns:
                return []

            values: list[str] = []
            for value in group[column].tolist():
                if pd.isna(value):
                    continue

                text = str(value).strip()
                if text and text not in values:
                    values.append(text)

            return values

        identifiers = unique_values(
            "identifiant_national"
        )
        identifier_types = unique_values(
            "type_identifiant_national"
        )
        denominations = unique_values(
            "denomination"
        )

        if (
            len(identifiers) > 1
            or len(identifier_types) > 1
            or len(denominations) > 1
        ):
            columns = [
                col
                for col in [
                    "representants_id",
                    "identifiant_national",
                    "type_identifiant_national",
                    "denomination",
                    "label_categorie_organisation",
                ]
                if col in group.columns
            ]

            raise ValueError(
                "Vraie collision de representants_id :\n"
                + group[columns].to_string(index=False)
            )

        categories = sorted(
            unique_values(
                "label_categorie_organisation"
            )
        )

        row = group.iloc[0].to_dict()

        # Ne pas choisir arbitrairement une catégorie lorsque la HATVP
        # fournit plusieurs classifications pour le même SIREN.
        row["label_categorie_organisation"] = (
            " | ".join(categories)
            if categories
            else None
        )

        row["categorie_organisation_multiple"] = (
            len(categories) > 1
        )

        rows.append(row)

    result = pd.DataFrame(rows).reset_index(
        drop=True
    )

    if result["representants_id"].duplicated().any():
        raise AssertionError(
            "representants_id doit être unique après fusion."
        )

    return result


def build_tables(
    files: list[Path],
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    infos: list[dict] = []
    affiliations: list[dict] = []
    domaines: list[dict] = []
    objets: list[dict] = []
    beneficiaires: list[dict] = []
    observations: list[dict] = []
    exercices: list[dict] = []

    n_actions = 0

    for path in files:
        entity = json.loads(
            path.read_text(encoding="utf-8")
        )

        rid = rep_id(entity)
        cat = entity.get("categorieOrganisation") or {}

        infos.append(
            {
                "representants_id": rid,
                "identifiant_national": (
                    entity.get("identifiantNational")
                ),
                "type_identifiant_national": (
                    entity.get("typeIdentifiantNational")
                ),
                "denomination": entity.get("denomination"),
                "label_categorie_organisation": (
                    cat.get("label")
                ),
            }
        )

        for aff in entity.get("affiliations") or []:
            affiliations.append(
                {
                    "representants_id": rid,
                    "denomination_affiliation": (
                        aff.get("denomination")
                    ),
                }
            )

        for ex_pos, ex_node in enumerate(
            entity.get("exercices") or []
        ):
            ex = (
                (ex_node or {})
                .get("publicationCourante")
                or {}
            )

            ex_raw_id = ex.get("exerciceId")

            ex_id = (
                str(ex_raw_id)
                if ex_raw_id is not None
                else f"{rid}:ex:{ex_pos}"
            )

            dep_inf, dep_sup = parse_money_range(
                ex.get("montantDepense")
            )

            exercices.append(
                {
                    "exercices_id": ex_id,
                    "representants_id": rid,
                    "date_debut": ex.get("dateDebut"),
                    "date_fin": ex.get("dateFin"),
                    "montant_depense": (
                        ex.get("montantDepense")
                    ),
                    "montant_depense_inf": dep_inf,
                    "montant_depense_sup": dep_sup,
                    "nombre_activites": (
                        ex.get("nombreActivite")
                    ),
                    "nombre_salaries": (
                        ex.get("nombreSalaries")
                    ),
                }
            )

            for act_pos, act_node in enumerate(
                ex.get("activites") or []
            ):
                act = (
                    (act_node or {})
                    .get("publicationCourante")
                    or {}
                )

                fiche = str(
                    act.get("identifiantFiche") or ""
                ).strip()

                act_id = (
                    fiche
                    or f"{ex_id}:act:{act_pos}"
                )

                objets.append(
                    {
                        "activite_id": act_id,
                        "exercices_id": ex_id,
                        "representants_id": rid,
                        "date_publication_activite": (
                            act.get("publicationDate")
                        ),
                        "objet_activite": (
                            act.get("objet")
                        ),
                    }
                )

                for domain in (
                    act.get("domainesIntervention")
                    or []
                ):
                    domaines.append(
                        {
                            "activite_id": act_id,
                            (
                                "domaines_intervention_"
                                "actions_menees"
                            ): domain,
                        }
                    )

                for action_pos, action in enumerate(
                    act.get(
                        "actionsRepresentationInteret"
                    )
                    or []
                ):
                    action = action or {}

                    action_id = (
                        f"{act_id}:action:{action_pos}"
                    )

                    n_actions += 1

                    observations.append(
                        {
                            "activite_id": act_id,
                            (
                                "action_representation_"
                                "interet_id"
                            ): action_id,
                            "observation": (
                                action.get("observation")
                            ),
                        }
                    )

                    for tier in (
                        action.get("tiers") or []
                    ):
                        name, en_propre = (
                            parse_beneficiary(tier)
                        )

                        if name:
                            beneficiaires.append(
                                {
                                    (
                                        "action_representation_"
                                        "interet_id"
                                    ): action_id,
                                    (
                                        "beneficiaire_"
                                        "action_menee"
                                    ): name,
                                    (
                                        "action_menee_"
                                        "en_propre"
                                    ): en_propre,
                                    "raw_tiers": tier,
                                }
                            )

    df_infos = _build_informations_generales(
        infos
    )

    tables = {
        "informations_generales": df_infos,
        "affiliations": pd.DataFrame(
            affiliations
        ),
        "domaines_intervention": pd.DataFrame(
            domaines
        ),
        "objets_activites": pd.DataFrame(
            objets
        ),
        "beneficiaires": pd.DataFrame(
            beneficiaires
        ),
        "observations": pd.DataFrame(
            observations
        ),
        "exercices": pd.DataFrame(
            exercices
        ),
    }

    counts = {
        name: int(len(df))
        for name, df in tables.items()
    }

    counts["actions"] = n_actions

    return tables, counts


def load_manifest(
    directory: Path,
) -> dict:
    path = directory / "import_manifest.json"

    if not path.exists():
        return {}

    try:
        return json.loads(
            path.read_text(encoding="utf-8")
        )
    except Exception:
        return {}


def atomic_replace_dir(
    source: Path,
    destination: Path,
) -> None:
    backup = destination.with_name(
        destination.name + ".previous"
    )

    if backup.exists():
        shutil.rmtree(backup)

    if destination.exists():
        destination.rename(backup)

    try:
        source.rename(destination)

    except Exception:
        if destination.exists():
            shutil.rmtree(destination)

        if backup.exists():
            backup.rename(destination)

        raise

    else:
        if backup.exists():
            shutil.rmtree(backup)


def run(
    data_dir: Path,
    output_dir: Path,
    force: bool = False,
) -> dict:
    files = source_files(data_dir)
    fp = fingerprint(files)

    previous = load_manifest(output_dir)

    if (
        not force
        and previous.get("source_fingerprint") == fp
    ):
        return {
            **previous,
            "changed": False,
        }

    tables, counts = build_tables(files)

    output_dir.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = Path(
        tempfile.mkdtemp(
            prefix="lobbysearch-import-",
            dir=output_dir.parent,
        )
    )

    try:
        for key, filename in TABLE_FILES.items():
            tables[key].to_parquet(
                temp / filename,
                index=False,
            )

        manifest = {
            "schema_version": 1,
            "source": (
                "@tricoteuses/hatvp:"
                "repertoire_representants_interets"
            ),
            "source_directory": str(
                data_dir.resolve()
            ),
            "source_fingerprint": fp,
            "source_files": len(files),
            "imported_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "counts": counts,
        }

        (
            temp / "import_manifest.json"
        ).write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        atomic_replace_dir(
            temp,
            output_dir,
        )

        return {
            **manifest,
            "changed": True,
        }

    finally:
        if temp.exists():
            shutil.rmtree(
                temp,
                ignore_errors=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Importe les JSON nettoyés de "
            "tricoteuses-hatvp en tables LobbySearch."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PATHS.tricoteuses_data,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PATHS.data_imported,
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    result = run(
        args.data_dir.resolve(),
        args.output_dir.resolve(),
        force=args.force,
    )

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        "Import Tricoteuses terminé. "
        "Les tables Parquet LobbySearch "
        "sont disponibles dans data/imported.",
        flush=True,
    )


if __name__ == "__main__":
    main()
