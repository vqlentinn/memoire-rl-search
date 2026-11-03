"""Statistiques : intervalles bootstrap, test de McNemar apparié, kappa de Cohen."""
import numpy as np
from scipy.stats import binomtest, chi2
from sklearn.metrics import cohen_kappa_score


def bootstrap_ci(scores, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0):
    """Moyenne et IC à (1 - alpha) par rééchantillonnage des questions."""
    x = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def mcnemar(correct_a, correct_b):
    """Test de McNemar sur deux systèmes évalués sur les mêmes questions.

    Exact (binomial) si moins de 25 désaccords, sinon chi2 avec correction de continuité.
    Renvoie (b, c, p_value) : b = A juste et B faux, c = A faux et B juste.
    """
    a = np.asarray(correct_a, dtype=bool)
    b_ = np.asarray(correct_b, dtype=bool)
    b = int(np.sum(a & ~b_))
    c = int(np.sum(~a & b_))
    n = b + c
    if n == 0:
        return b, c, 1.0
    if n < 25:
        p = binomtest(b, n, 0.5).pvalue
    else:
        stat = (abs(b - c) - 1) ** 2 / n
        p = float(chi2.sf(stat, df=1))
    return b, c, float(p)


def kappa(labels_human, labels_judge) -> float:
    """Accord juge LLM contre annotation humaine (volet C)."""
    return float(cohen_kappa_score(labels_human, labels_judge))
