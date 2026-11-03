"""Retriever BM25 (bm25s) au format Search-R1.

Corpus : JSONL, une ligne par passage, {"id": ..., "contents": "Titre\\nTexte"}.
Le formatage des résultats reproduit search() de infer.py dans le dépôt Search-R1.
"""
import json
from pathlib import Path

import bm25s

try:
    import Stemmer
    _STEMMER = Stemmer.Stemmer("english")
except ImportError:  # le stemming est optionnel
    _STEMMER = None


def load_corpus(path):
    with open(path) as f:
        return [json.loads(line)["contents"] for line in f]


class BM25Retriever:
    def __init__(self, corpus, index_dir=None):
        self.corpus = corpus
        if index_dir and Path(index_dir).exists():
            self.retriever = bm25s.BM25.load(index_dir, mmap=True)
        else:
            self.retriever = bm25s.BM25()
            tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=_STEMMER)
            self.retriever.index(tokens)
            if index_dir:
                self.retriever.save(index_dir)

    def search(self, query: str, k: int = 3):
        tokens = bm25s.tokenize([query], stopwords="en", stemmer=_STEMMER)
        ids, scores = self.retriever.retrieve(tokens, k=k)
        return [self.corpus[i] for i in ids[0]]

    @staticmethod
    def format_searchr1(passages) -> str:
        out = ""
        for idx, content in enumerate(passages):
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            out += f"Doc {idx + 1}(Title: {title}) {text}\n"
        return out
