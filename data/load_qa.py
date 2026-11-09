"""Chargement des jeux de QA au format FlashRAG (celui de Search-R1).

Chaque exemple a au moins "question" et "golden_answers".
"""
from datasets import load_dataset

DATASETS = ["nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa", "musique", "bamboogle"]

# Split d'évaluation : test s'il existe, sinon dev (vérifié sur RUC-NLPIR/FlashRAG_datasets).
EVAL_SPLITS = {"nq": "test", "triviaqa": "test", "popqa": "test", "hotpotqa": "dev",
               "2wikimultihopqa": "dev", "musique": "dev", "bamboogle": "test"}


def load_qa(name: str, split: str | None = None, n: int | None = 500, seed: int = 42):
    """Renvoie une liste de dicts {id, question, golds}, échantillonnée avec graine fixe."""
    split = split or EVAL_SPLITS[name]
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
