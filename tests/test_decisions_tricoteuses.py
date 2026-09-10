from scripts.import_decisions_tricoteuses import build_dataframe, normalize_type, parse_record


def test_normalize_type():
    assert normalize_type("LOI") == "LOI"
    assert normalize_type("Décret") == "DECRET"
    assert normalize_type("ORDONNANCE") == "ORDONNANCE"
    assert normalize_type("Arrêté") == "ARRETE"
    assert normalize_type("AMENDEMENT") == "AMENDEMENT"


def test_parse_and_exclude_amendements():
    rows = [
        {
            "id": "JORFTEXT1",
            "data": {
                "META": {"META_COMMUN": {"NATURE": "LOI"}},
                "TITRE": "Loi test",
                "DATE_TEXTE": "01/02/2025",
            },
        },
        {
            "id": "JORFTEXT2",
            "data": {
                "META": {"META_COMMUN": {"NATURE": "DECRET"}},
                "TITRE": "Décret test",
                "DATE_TEXTE": "02/02/2025",
                "NOTICE": [{"texte": "Description du décret"}],
            },
        },
        {"id": "AMD1", "data": {"META": {"META_COMMUN": {"NATURE": "AMENDEMENT"}}, "TITRE": "Amendement test", "DATE_TEXTE": "03/02/2025"}},
    ]
    assert parse_record(rows[0])["type_decision"] == "loi"
    assert parse_record(rows[1])["Description"] == "Description du décret"
    assert parse_record(rows[2]) is None
    df = build_dataframe(rows, from_year=2018)
    assert set(df["type_decision"]) == {"loi", "decret"}
    assert "amendement" not in set(df["type_decision"])
    assert "Description" in df.columns
