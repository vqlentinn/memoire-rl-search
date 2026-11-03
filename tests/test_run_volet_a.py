"""Tirage des questions de scripts/run_volet_a.py, sans téléchargement."""
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import run_volet_a as r  # noqa: E402


def fake_load_qa(name, n=None, seed=42):
    # Graines différentes : deux ordres qui se recoupent, comme deux mélanges du même jeu.
    order = range(1000) if seed == 42 else range(50, 1050)
    return [{"id": str(i), "question": f"q{i}?", "golds": ["x"]} for i in order][:n]


def test_graine_42_donne_le_tirage_d_evaluation():
    with mock.patch.object(r, "load_qa", fake_load_qa):
        assert [q["question"] for q in r.select_questions("nq", 3, 42)] == ["q0?", "q1?", "q2?"]


def test_mise_au_point_sans_question_d_evaluation():
    with mock.patch.object(r, "load_qa", fake_load_qa):
        dev = r.select_questions("nq", 10, 7)
        evaluation = {q["question"] for q in fake_load_qa("nq", n=r.EVAL_N, seed=42)}
    assert len(dev) == 10 and not {q["question"] for q in dev} & evaluation
