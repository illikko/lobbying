from __future__ import annotations

import pandas as pd

from src.activity_analytics import (
    build_search_analytics,
    enrich_activities_with_beneficiaries,
    validate_beneficiary_budget_not_overcounted,
    validate_no_beneficiary_join_on_text_keys,
    validate_selected_ids_exist_in_observations_when_expected,
    validate_three_tables_share_same_enriched_source,
)


def run() -> None:
    selected = pd.DataFrame(
        {
            "activite_id": ["a1", "a2"],
            "denomination": ["Org A", "Org B"],
            "objet_activite": ["Objet 1", "Objet 2"],
            "budget_moyen_activite": [100.0, 90.0],
            "domaines": ["Energie; Fiscalite", "Sante"],
            "hybrid_score": [0.9, 0.6],
        }
    )
    observations = pd.DataFrame(
        {
            "activite_id": ["a1", "a1", "a2"],
            "action_representation_interet_id": ["x1", "x2", "x3"],
        }
    )
    beneficiaires = pd.DataFrame(
        {
            "action_representation_interet_id": ["x1", "x2"],
            "beneficiaire_action_menee": ["Benef 1", "Benef 2"],
        }
    )

    enriched = enrich_activities_with_beneficiaries(selected, observations, beneficiaires)
    assert set(enriched["activite_id"]) == {"a1", "a2"}
    assert float(enriched.loc[enriched["activite_id"] == "a1", "budget_beneficiaire"].sum()) == 100.0
    assert float(enriched.loc[enriched["activite_id"] == "a2", "budget_beneficiaire"].sum()) == 0.0

    analytics = build_search_analytics(selected, observations, beneficiaires)
    validate_no_beneficiary_join_on_text_keys()
    validate_selected_ids_exist_in_observations_when_expected(analytics.enriched, observations)
    validate_beneficiary_budget_not_overcounted(analytics.enriched)
    validate_three_tables_share_same_enriched_source(analytics)

    assert analytics.table_beneficiaires["beneficiaire"].tolist() == ["Benef 1", "Benef 2"]
    assert int(analytics.table_organisations["nb_activites_matching"].sum()) == 2
    print("tests_minimal.py: OK")


if __name__ == "__main__":
    run()
