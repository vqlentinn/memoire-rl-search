"""2WikiMultiHopQA, cadre « distracteurs » : chaque question vient avec 10 paragraphes, dont ceux qui contiennent les
faits nécessaires (supporting facts) ; même interface que data/hotpot_distractor.py (amendement du 24 nov. 2026).

Source : RUC-NLPIR/FlashRAG_datasets, 2wikimultihopqa/dev.jsonl (12 576 questions ; contexte et faits utiles dans
« metadata »). Search-R1 n'a pas été entraîné sur ce jeu (NQ et HotpotQA seulement) : hors distribution.
Choix, comme pour HotpotQA :
- seules les questions avec exactement 10 paragraphes distincts sont gardées (115 contextes « compositional » ont un
  paragraphe en double) ;
- les réponses « yes » / « no » sont exclues (pass@k les trouverait au hasard) ;
- type principal : « compositional » (deux étapes, l'entité du premier paragraphe mène au second : l'équivalent des
  ponts de HotpotQA ; 2 paragraphes utiles par question). Mesuré avant toute donnée : avec la question comme requête,
  les deux paragraphes utiles sont dans le top 3 du BM25 fermé pour 29 % des questions (45 % pour les ponts HotpotQA).
"""
import json
from pathlib import Path

import numpy as np

REPO, FICHIER = "RUC-NLPIR/FlashRAG_datasets", "2wikimultihopqa/dev.jsonl"
TYPES = ("compositional", "inference", "comparison", "bridge_comparison")


def telecharger(dest="data/raw/wiki2"):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO, FICHIER, repo_type="dataset", local_dir=dest)


def preparer(lignes):
    out = []
    for x in lignes:
        m = x["metadata"]
        ctx = m["context"]
        paras = [f"{t}\n{''.join(s).strip()}" for t, s in zip(ctx["title"], ctx["content"])]
        if len(paras) != 10 or len(set(paras)) != 10:
            continue
        if str(x["golden_answers"][0]).strip().lower() in ("yes", "no"):
            continue
        out.append({"id": x["id"], "question": x["question"], "golds": list(x["golden_answers"]), "type": m["type"],
                    "paragraphs": paras, "gold_titles": sorted(set(m["supporting_facts"]["title"]))})
    return out


def charger(chemin=None, type_question="compositional", n=500, seed=42, exclure=(), ids=None):
    """Même contrat que data.hotpot_distractor.charger."""
    if type_question not in TYPES:
        raise ValueError(f"type inconnu : {type_question}")
    chemin = chemin or telecharger()
    lignes = [json.loads(l) for l in Path(chemin).read_text().split("\n") if l.strip()]
    eligibles = [q for q in preparer(lignes) if q["type"] == type_question and q["id"] not in set(exclure)]
    if ids is not None:
        par_id = {q["id"]: q for q in eligibles}
        manquants = [i for i in ids if i not in par_id]
        if manquants:
            raise ValueError(f"{len(manquants)} ids absents ou d'un autre type, par exemple {manquants[:3]}")
        return [par_id[i] for i in ids]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(eligibles), size=min(n, len(eligibles)), replace=False)
    return [eligibles[i] for i in sorted(idx)]
