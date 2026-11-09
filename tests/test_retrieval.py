import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from retrieval.bm25_index import BM25Retriever  # noqa: E402

CORPUS = ["A\nchat noir", "B\nchat noir", "C\nchat noir", "D\nchien gris", "E\noiseau bleu"]


def _titles(passages):
    return [p.split("\n")[0] for p in passages]


def test_ties_broken_by_passage_index():
    retriever = BM25Retriever(CORPUS)
    assert _titles(retriever.search("chat", k=2)) == ["A", "B"]


def test_best_score_comes_first():
    retriever = BM25Retriever(CORPUS)
    assert _titles(retriever.search("chien", k=1)) == ["D"]


def test_stopwords_only_and_unknown_words_do_not_crash():
    retriever = BM25Retriever(CORPUS)
    assert len(retriever.search("the and of", k=3)) == 3
    assert len(retriever.search("zzqxvy", k=3)) == 3


def test_k_larger_than_corpus():
    retriever = BM25Retriever(CORPUS[:2])
    assert len(retriever.search("chat", k=3)) == 2
