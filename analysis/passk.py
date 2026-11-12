"""Inférence appariée sur les courbes pass@k : le renforcement élargit-il ce que l'agent sait faire ?

Pour chaque question i, chaque modèle produit n générations indépendantes, dont c_i sont justes.
- pass@k d'une question : probabilité qu'au moins une de k générations tirées sans remise parmi les n
  soit juste. Estimateur sans biais (Chen et al., 2021) : 1 - C(n - c, k) / C(n, k).
- Courbe appariée : D(k) = moyenne sur les questions de [pass@k(RL, i) - pass@k(base, i)]. Les deux
  modèles voient les mêmes questions : la variabilité entre questions s'annule dans la différence.
- Incertitude : bootstrap à multiplicateurs par question (Chernozhukov et al. ; méthode reprise par
  Yang, arXiv 2609.22547). On tire des poids gaussiens xi_i, sans recalculer les pass@k.
- Bandes simultanées : D(k) ± q * sigma(k) pour tous les k à la fois, q étant le quantile du
  maximum sur k de |t*(k)|. Une bande entièrement au-dessus de 0 sur une plage de k certifie un gain
  sur toute cette plage ; une bande sous 0 en un k certifie une perte (un « croisement »).
- Test de dominance (intersection-union) : « D(k) > 0 pour tout k de la grille » est soutenu si le plus
  petit t(k) dépasse le quantile unilatéral 1 - alpha de la loi normale. Ce test d'intersection-union
  est valide sans correction pour comparaisons multiples (Berger, 1982), au prix d'être conservateur.

Tout est déterministe à graine fixée. Les fonctions prennent des tableaux NumPy : (questions,) pour c.
"""
import numpy as np
from scipy.special import gammaln
from scipy.stats import binomtest, norm


def pass_at_k(n, c, k):
    """Estimateur sans biais de pass@k, vectorisé sur c (et n). Requiert 1 <= k <= n."""
    n, c = np.broadcast_arrays(np.asarray(n, dtype=float), np.asarray(c, dtype=float))
    if np.any(k > n) or k < 1:
        raise ValueError("il faut 1 <= k <= n")
    out = np.ones_like(c)
    ok = (n - c) >= k  # sinon, toute sélection de k contient une génération juste : pass@k = 1
    # C(n-c, k) / C(n, k) en log pour éviter les dépassements (n jusqu'à quelques centaines)
    log_ratio = (gammaln(n[ok] - c[ok] + 1) - gammaln(n[ok] - c[ok] - k + 1)
                 - gammaln(n[ok] + 1) + gammaln(n[ok] - k + 1))
    out[ok] = 1.0 - np.exp(log_ratio)
    return out


def courbes(c, n, ks):
    """Matrice (questions, len(ks)) des pass@k par question."""
    c = np.asarray(c)
    return np.stack([pass_at_k(n, c, k) for k in ks], axis=1)


def difference_appariee(c_rl, c_base, n, ks):
    """Contributions par question d_i(k) et courbe moyenne D(k)."""
    d = courbes(c_rl, n, ks) - courbes(c_base, n, ks)
    return d, d.mean(axis=0)


def bootstrap_multiplicateur(d, n_boot=2000, alpha=0.05, seed=0):
    """Bandes simultanées et tests à partir des contributions d (questions x k).

    Renvoie D(k), sigma(k), bandes ponctuelles et simultanées, quantile simultané, t(k),
    et le résultat du test de dominance.
    """
    d = np.asarray(d, dtype=float)
    m, K = d.shape
    D = d.mean(axis=0)
    centre = d - D
    rng = np.random.default_rng(seed)
    xi = rng.standard_normal((n_boot, m))
    boot = xi @ centre / m  # (n_boot, K) : D*(k) - D(k)
    sigma = boot.std(axis=0, ddof=1)
    sigma_sur = np.where(sigma > 0, sigma, np.inf)  # k où toutes les questions sont identiques
    t_boot = np.abs(boot) / sigma_sur
    q_simul = np.quantile(t_boot.max(axis=1), 1 - alpha)
    z = norm.ppf(1 - alpha / 2)
    t = np.where(sigma > 0, D / sigma_sur, 0.0)
    z_uni = norm.ppf(1 - alpha)
    max_boot = t_boot.max(axis=1)
    return {
        # p-valeurs : ponctuelle unilatérale (D(k) > 0) et simultanée bilatérale (ajustée sur tous les k)
        "p_uni": norm.sf(t), "p_simul": np.array([np.mean(max_boot >= abs(x)) for x in t]),
        # Test d'intersection-union : p = max des p unilatérales (Berger, 1982)
        "p_dominance": float(norm.sf(t).max()),
        "D": D, "sigma": sigma, "t": t,
        "bande_ponct": (D - z * sigma, D + z * sigma),
        "bande_simul": (D - q_simul * sigma, D + q_simul * sigma),
        "q_simul": q_simul,
        # Dominance (intersection-union) : chaque D(k) significativement > 0 au niveau alpha unilatéral.
        "dominance": bool(np.all(t > z_uni)), "t_min": float(t.min()),
        # Croisement certifié : au moins un k où la bande simultanée est entièrement sous 0.
        "perte_certifiee": bool(np.any(D + q_simul * sigma < 0)),
        "gain_certifie": bool(np.any(D - q_simul * sigma > 0)),
    }


def frontiere(c_rl, c_base):
    """Questions résolues au moins une fois (pass@n) : gagnées, perdues, test exact de McNemar."""
    a, b = np.asarray(c_rl) > 0, np.asarray(c_base) > 0
    gagnees, perdues = int(np.sum(a & ~b)), int(np.sum(~a & b))
    p = binomtest(gagnees, gagnees + perdues, 0.5).pvalue if gagnees + perdues else 1.0
    return {"rl": int(a.sum()), "base": int(b.sum()), "gagnees": gagnees, "perdues": perdues, "p_mcnemar": float(p)}


def grille_k(n):
    """Puissances de 2 jusqu'à n, et n lui-même."""
    ks = [1]
    while ks[-1] * 2 <= n:
        ks.append(ks[-1] * 2)
    if ks[-1] != n:
        ks.append(n)
    return ks


def sous_ensemble_difficile(c_presel):
    """Ids des questions sans aucune réussite du modèle de départ à la présélection.

    `c_presel` : {id: réussites} obtenues sur des tirages de présélection. Les tirages d'évaluation doivent être
    NOUVEAUX : réutiliser ceux-ci donnerait au modèle de départ un pass@k nul par construction.
    """
    return sorted(i for i, c in c_presel.items() if c == 0)
