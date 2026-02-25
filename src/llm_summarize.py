from __future__ import annotations
import os
from openai import OpenAI

def summarize_activites(info_llm: list[dict], model: str = "gpt-4o-mini") -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "OPENAI_API_KEY manquant. Impossible de générer la synthèse."

    info_llm = info_llm

    client = OpenAI(api_key=api_key)

    system_msg = {
        "role": "system",
        "content": (
            "Vous êtes un expert en affaires publiques.\n"
            "On vous fournit une sélection d'activités de lobbying (HATVP) filtrées.\n"
            "Le budget_moyen_activite est une estimation du budget pour une activité.\n"
            "Produisez :\n"
            "1) Synthèse globale (domaines + objet_activite)\n"
            "2) Thématiques récurrentes + organisations les plus actives + oppositions\n"
            "3) Analyse budgets + thématiques d'investissement\n"
            "Réponse structurée, détaillée, avec puces, chiffres clés, et exemples. Evitez les généralités.\n"
        )
    }
    user_msg = {"role": "user", "content": f"Données: {info_llm}"}

    resp = client.chat.completions.create(
        model=model,
        messages=[system_msg, user_msg],
        temperature=0,
        max_tokens=1500
    )
    txt = (resp.choices[0].message.content or "").strip()
    return txt or "Le modèle n'a pas renvoyé de texte."