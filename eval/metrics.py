"""Métriques EM et F1, normalisation SQuAD.

La normalisation et l'exact match reprennent celles de Search-R1
(verl/utils/reward_score/qa_em.py) pour que les scores soient comparables.
"""
import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _as_list(golds):
    return [golds] if isinstance(golds, str) else list(golds)


def exact_match(prediction, golds) -> int:
    if prediction is None:
        return 0
    p = normalize_answer(prediction)
    return int(any(p == normalize_answer(g) for g in _as_list(golds)))


def _f1(prediction: str, gold: str) -> float:
    p_tokens = normalize_answer(prediction).split()
    g_tokens = normalize_answer(gold).split()
    common = Counter(p_tokens) & Counter(g_tokens)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(p_tokens)
    recall = n_same / len(g_tokens)
    return 2 * precision * recall / (precision + recall)


def f1_score(prediction, golds) -> float:
    if prediction is None:
        return 0.0
    return max(_f1(prediction, g) for g in _as_list(golds))


def extract_answer(text: str):
    """Dernière réponse entre <answer> et </answer>, ou None."""
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return matches[-1].strip() if matches else None
