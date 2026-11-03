"""Chargement des jeux de QA au format FlashRAG (celui de Search-R1).

Chaque exemple a au moins "question" et "golden_answers".
"""
from datasets import load_dataset

DATASETS = ["nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa", "musique", "bamboogle"]


def load_qa(name: str, split: str = "test", n: int | None = 500, seed: int = 42):
    """Renvoie une liste de dicts {id, question, golds}, échantillonnée avec graine fixe.

    Les splits disponibles varient selon le jeu (test ou dev) : en cas d'erreur,
    vérifier la fiche du dataset RUC-NLPIR/FlashRAG_datasets sur le Hub.
    """
    ds = load_dataset("RUC-NLPIR/FlashRAG_datasets", name, split=split)
    if n is not None and n < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n))
    out = []
    for i, ex in enumerate(ds):
        q = ex["question"].strip()
        if not q.endswith("?"):
            q += "?"  # même traitement que Search-R1
        out.append({"id": f"{name}-{ex.get('id', i)}", "question": q, "golds": ex["golden_answers"]})
    return out
