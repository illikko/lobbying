import json
from pathlib import Path

from scripts.import_tricoteuses import build_tables, parse_beneficiary, parse_money_range


def test_beneficiary_en_propre():
    assert parse_beneficiary("ASSOCIATION FRANCAISE DU RAIL (en propre)") == (
        "ASSOCIATION FRANCAISE DU RAIL", True
    )
    assert parse_beneficiary("YARA FRANCE") == ("YARA FRANCE", False)


def test_money_ranges():
    assert parse_money_range("< 10 000 euros") == (0.0, 10000.0)
    assert parse_money_range("> = 100 000 euros et < 200 000 euros") == (100000.0, 200000.0)


def test_import_creates_stable_tables(tmp_path: Path):
    data = tmp_path / "trico"
    reps = data / "repertoire_representants_interets"
    reps.mkdir(parents=True)
    entity = {
        "identifiantNational": "123456789",
        "typeIdentifiantNational": "SIREN",
        "denomination": "TEST ORG",
        "categorieOrganisation": {"label": "Entreprise"},
        "affiliations": [{"denomination": "FEDERATION TEST"}],
        "exercices": [{
            "publicationCourante": {
                "exerciceId": 42,
                "montantDepense": "> = 10 000 euros et < 25 000 euros",
                "nombreActivite": 1,
                "nombreSalaries": 2,
                "activites": [{
                    "publicationCourante": {
                        "identifiantFiche": "ABC123",
                        "objet": "Objet test",
                        "publicationDate": "2025-02-03",
                        "domainesIntervention": ["Energie"],
                        "actionsRepresentationInteret": [{
                            "observation": "Observation test",
                            "tiers": ["CLIENT TEST", "TEST ORG (en propre)"]
                        }]
                    }
                }]
            }
        }]
    }
    source = reps / "123456789.json"
    source.write_text(json.dumps(entity), encoding="utf-8")
    tables, counts = build_tables([source])
    assert counts["beneficiaires"] == 2
    benef = tables["beneficiaires"]
    assert set(benef["action_menee_en_propre"].tolist()) == {False, True}
    assert benef["action_representation_interet_id"].iloc[0] == "ABC123:action:0"
    assert tables["objets_activites"]["activite_id"].iloc[0] == "ABC123"
    assert tables["exercices"]["exercices_id"].iloc[0] == "42"

