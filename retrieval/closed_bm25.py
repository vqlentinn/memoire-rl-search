"""Recherche fermée : un petit index BM25 par question, sur ses 10 paragraphes seulement.

Reproduit le cadre de Zhai et al. (arXiv 2604.14877) : BM25 Okapi, k1 = 1,5, b = 0,75, comme rank_bm25,
y compris le plancher des idf négatifs (epsilon * idf moyen, epsilon = 0,25), qui compte ici car un mot
présent dans plus de la moitié des 10 paragraphes aurait sinon un idf négatif.
Implémenté en NumPy, sans dépendance, et déterministe : les ex aequo sont départagés par l'ordre des
paragraphes dans le fichier, comme dans retrieval/bm25_index.py.
"""
import math
import re
from collections import Counter

import numpy as np

_MOT = re.compile(r"\w+", re.UNICODE)


def tokeniser(texte):
    return _MOT.findall(texte.lower())


class BM25Okapi:
    def __init__(self, docs, k1=1.5, b=0.75, epsilon=0.25):
        self.k1, self.b = k1, b
        self.toks = [tokeniser(d) for d in docs]
        self.longueurs = np.array([len(t) for t in self.toks], dtype=float)
        self.lmoy = self.longueurs.mean() if len(docs) else 0.0
        self.tf = [Counter(t) for t in self.toks]
        df = Counter(m for t in self.toks for m in set(t))
        N = len(docs)
        idf = {m: math.log(N - n + 0.5) - math.log(n + 0.5) for m, n in df.items()}
        plancher = epsilon * (sum(idf.values()) / len(idf)) if idf else 0.0
        self.idf = {m: (v if v >= 0 else plancher) for m, v in idf.items()}

    def scores(self, requete):
        s = np.zeros(len(self.toks))
        for m in tokeniser(requete):
            if m not in self.idf:
                continue
            tf = np.array([c[m] for c in self.tf], dtype=float)
            s += self.idf[m] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.longueurs / self.lmoy))
        return s


class ClosedBM25Retriever:
    """Un index par question. La boucle Search-R1 appelle search_in(id, requête, k)."""

    def __init__(self, questions):
        self.docs = {q["id"]: q["paragraphs"] for q in questions}
        self.index = {q["id"]: BM25Okapi(q["paragraphs"]) for q in questions}

    def search_in(self, qid, requete, k=3):
        scores = self.index[qid].scores(requete)
        ordre = np.lexsort((np.arange(len(scores)), -scores))  # score décroissant, puis ordre du fichier
        return [self.docs[qid][i] for i in ordre[:k]]

    @staticmethod
    def format_searchr1(passages) -> str:
        """Même format que BM25Retriever.format_searchr1 (search() de infer.py dans Search-R1)."""
        out = ""
        for idx, content in enumerate(passages):
            title, _, text = content.partition("\n")
            out += f"Doc {idx + 1}(Title: {title}) {text}\n"
        return out
