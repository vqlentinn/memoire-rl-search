import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.metrics import exact_match, extract_answer, f1_score  # noqa: E402
from eval.stats import bootstrap_ci, mcnemar  # noqa: E402


def test_exact_match_normalization():
    assert exact_match("The Beatles.", ["beatles"]) == 1
    assert exact_match("Paris", ["London", "paris"]) == 1
    assert exact_match("Rome", ["Paris"]) == 0
    assert exact_match(None, ["Paris"]) == 0


def test_f1_partial():
    assert f1_score("Barack Obama", ["Obama"]) == 2 / 3
    assert f1_score("Paris", ["Paris"]) == 1.0


def test_extract_answer_takes_last():
    text = "<think>x</think><answer> draft </answer> ... <answer> Beijing </answer>"
    assert extract_answer(text) == "Beijing"
    assert extract_answer("no answer here") is None


def test_stats():
    mean, lo, hi = bootstrap_ci([1, 0, 1, 1, 0, 1, 1, 0, 1, 1])
    assert lo <= mean <= hi
    b, c, p = mcnemar([1, 1, 1, 0], [0, 1, 1, 0])
    assert (b, c) == (1, 0) and p == 1.0
