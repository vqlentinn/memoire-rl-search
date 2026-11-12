"""HotpotQA, cadre « distracteurs » : chaque question vient avec 10 paragraphes, dont 2 contiennent
les faits nécessaires (supporting facts) et 8 sont des distracteurs proches.

Source : hotpotqa/hotpot_qa (Hugging Face), configuration distractor, partie validation (7 405 questions,
toutes de niveau « hard » ; la partie test officielle n'a pas de réponses publiques).
Choix fixés par le plan d'analyse (docs/partie2/plan_analyse.md) :
- seules les questions avec exactement 10 paragraphes sont gardées (7 345 sur 7 405), pour que
  l'environnement soit identique pour toutes ;
- les réponses « yes » / « no » sont exclues (447 comparaisons) : avec 32 tirages, une réponse binaire
  est trouvée au moins une fois même au hasard, ce qui gonflerait pass@k sans rien dire des capacités ;
- tirage sans remise avec une graine fixe, séparément pour les questions de pont et de comparaison.
"""
from pathlib import Path

import numpy as np

REPO, FICHIER = "hotpotqa/hotpot_qa", "distractor/validation-00000-of-00001.parquet"
TYPES = ("bridge", "comparison")


def telecharger(dest="data/raw/hotpot_distractor"):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO, FICHIER, repo_type="dataset", local_dir=dest)


def paragraphes(context):
    """Les 10 paragraphes au format "Titre\\nTexte", comme le corpus Wikipédia de Search-R1."""
    return [f"{t}\n{''.join(s).strip()}" for t, s in zip(context["title"], context["sentences"])]


def preparer(df):
    """DataFrame brut -> liste de questions éligibles, dans l'ordre du fichier."""
    out = []
    for r in df.itertuples(index=False):
        if len(r.context["title"]) != 10 or str(r.answer).strip().lower() in ("yes", "no"):
            continue
        out.append({"id": r.id, "question": r.question, "golds": [r.answer], "type": r.type,
                    "paragraphs": paragraphes(r.context),
                    "gold_titles": sorted(set(r.supporting_facts["title"]))})
    return out


def charger(chemin=None, type_question="bridge", n=500, seed=42, exclure=(), ids=None):
    """n questions d'un type, tirées avec la graine `seed` ; `exclure` : ids à écarter (mise au point).
    `ids` : liste explicite (sous-ensemble difficile) ; renvoie ces questions dans cet ordre, sans tirage."""
    import pandas as pd
    if type_question not in TYPES:
        raise ValueError(f"type inconnu : {type_question}")
    chemin = chemin or telecharger()
    eligibles = [q for q in preparer(pd.read_parquet(chemin))
                 if q["type"] == type_question and q["id"] not in set(exclure)]
    if ids is not None:
        par_id = {q["id"]: q for q in eligibles}
        manquants = [i for i in ids if i not in par_id]
        if manquants:
            raise ValueError(f"{len(manquants)} ids absents ou d'un autre type, par exemple {manquants[:3]}")
        return [par_id[i] for i in ids]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(eligibles), size=min(n, len(eligibles)), replace=False)
    return [eligibles[i] for i in sorted(idx)]
