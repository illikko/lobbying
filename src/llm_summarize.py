from __future__ import annotations
import os
from openai import OpenAI
import json

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
        "On vous fournit une sélection d'activités de lobbying correspondant à une thématique recherchée, "
        "ainsi que plusieurs tableaux agrégés par bénéficiaire des actions menées.\n\n"

        "Définitions impératives :\n"
        "- Le bénéficiaire désigne l'entité au nom ou au bénéfice de laquelle une action de lobbying est déclarée.\n"
        "- L'organisation déclarante désigne l'organisation qui déclare l'activité de lobbying.\n"
        "- Ne confondez jamais bénéficiaire et organisation déclarante.\n"
        "- Si le bénéficiaire est identique à l'organisation déclarante, vous pouvez le signaler, mais sans supposer que c'est toujours le cas.\n\n"
        
        "Règles d'interprétation impératives :\n"
        "- N'utilisez pas budget_moyen_activite : ce calcul intermédiaire n'est pas fourni et ne doit pas être reconstitué.\n"
        "- Le champ budget_estime_recherche correspond au budget estimé associé à la sélection issue de la thématique recherchée.\n"
        "- Le champ budget_total correspond au budget global lobbying de l'organisation déclarante, au-delà de la seule thématique, quand il est fourni.\n"
        "- Les pourcentages décrivent des parts du budget de la thématique, pas du budget global lobbying.\n"
        "- Appuyez-vous d'abord sur les textes de objet_activite.\n"
        "- Ne remplacez pas les objets d'activité précis par des formulations vagues si les données permettent d'être concret.\n"
        "- Citez au minimum 8 objets d'activité distincts dans la réponse, en les reformulant le moins possible.\n"
        "- Regroupez les objets par domaine quand c'est pertinent, mais sans perdre les formulations concrètes.\n"
        "- N'évoquez une convergence ou une opposition que si elle ressort d'au moins deux objets d'activité explicites.\n\n"
        "Produisez une réponse structurée avec :\n"
        "1) Vue d'ensemble de la thématique : budget global, principaux domaines, poids budgétaire de chaque domaine, et description détaillée des objets d'activité réellement présents.\n"
        "2) Bénéficiaires les plus présents dans la thématique recherchée.\n"
        "3) Poids relatif des bénéficiaires dans la thématique recherchée, "
        "en distinguant les organisations déclarantes associées lorsque c'est utile.\n"
        "4) Convergences, oppositions ou lignes de fracture visibles, uniquement lorsqu'elles sont étayées par les objets d'activité.\n\n"
        "Dans la section 1, consacrez au moins la moitié du texte à décrire concrètement les objets d'activité. "
        "Évitez les généralités. "
        "N'inventez pas de causalité. "
        "Signalez explicitement les incertitudes."
        "Pas de conclusion vague."
        )
    }
    
    user_msg = {
        "role": "user",
        "content": (
            "Analyse les données suivantes.\n"
            "Le bloc 'tableau_beneficiaires' sert à décrire les bénéficiaires des actions les plus présents.\n"
            "Le bloc 'tableau_organisations' sert à décrire les organisations déclarantes associées, sans les confondre avec les bénéficiaires.\n"
            "Le bloc 'tableau_beneficiaires_domaines' sert à relier bénéficiaires, domaines et organisations déclarantes.\n"
            "Le bloc 'tableau_domaines' sert à décrire les domaines et leur poids budgétaire.\n"
            "Le bloc 'activites_selectionnees' sert à détailler précisément les objets d'activité.\n\n"
            "Données JSON :\n"
            + json.dumps(info_llm, ensure_ascii=False, indent=2)
        )
    }

    resp = client.chat.completions.create(
        model=model,
        messages=[system_msg, user_msg],
        temperature=0,
        max_tokens=1500
    )
    txt = (resp.choices[0].message.content or "").strip()
    return txt or "Le modèle n'a pas renvoyé de texte."
