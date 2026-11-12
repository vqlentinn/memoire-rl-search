"""Inférence pass@k sur données simulées : estimateur, bandes simultanées, dominance, frontière."""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.passk import (bootstrap_multiplicateur, difference_appariee, frontiere,  # noqa: E402
                            grille_k, pass_at_k, sous_ensemble_difficile)


@pytest.mark.parametrize("n,c,k", [(5, 2, 1), (5, 2, 3), (6, 0, 4), (6, 6, 2), (8, 3, 5), (8, 1, 8)])
def test_estimateur_egal_au_denombrement(n, c, k):
    """pass@k = part des sous-ensembles de k générations qui contiennent au moins une réponse juste."""
    gens = [1] * c + [0] * (n - c)
    attendu = np.mean([any(s) for s in combinations(gens, k)])
    assert pass_at_k(n, c, k) == pytest.approx(attendu)


def test_cas_limites():
    assert pass_at_k(32, 0, 32) == 0.0
    assert pass_at_k(32, 32, 1) == 1.0
    assert pass_at_k(32, 8, 1) == pytest.approx(8 / 32)
    assert pass_at_k(200, 3, 150) == pytest.approx(1 - np.prod([(197 - j) / (200 - j) for j in range(150)]))
    with pytest.raises(ValueError):
        pass_at_k(4, 1, 5)


def test_grille():
    assert grille_k(32) == [1, 2, 4, 8, 16, 32]
    assert grille_k(24) == [1, 2, 4, 8, 16, 24]


def _simuler(rng, m, n, p_base, effet):
    """Probabilités de réussite par question ; le RL ajoute `effet` (borné dans [0, 1])."""
    p = rng.beta(0.6, 1.4, size=m) if p_base is None else np.full(m, p_base)
    return rng.binomial(n, np.clip(p + effet, 0, 1)), rng.binomial(n, p)


def test_pas_de_dominance_sous_l_hypothese_nulle():
    rng = np.random.default_rng(1)
    fausses = 0
    for r in range(40):
        c_rl, c_base = _simuler(rng, 300, 16, None, 0.0)
        d, _ = difference_appariee(c_rl, c_base, 16, grille_k(16))
        fausses += bootstrap_multiplicateur(d, n_boot=400, seed=r)["dominance"]
    assert fausses <= 4  # niveau nominal 5 % unilatéral, test conservateur


def test_dominance_detectee_quand_l_effet_est_net():
    rng = np.random.default_rng(2)
    c_rl, c_base = _simuler(rng, 400, 16, None, 0.15)
    d, D = difference_appariee(c_rl, c_base, 16, grille_k(16))
    res = bootstrap_multiplicateur(d, n_boot=800)
    assert np.all(D > 0) and res["dominance"] and res["gain_certifie"] and not res["perte_certifiee"]


def test_couverture_des_bandes_simultanees():
    """La vraie courbe (calculée sur une très grande population) est dans la bande environ 95 % du temps."""
    rng = np.random.default_rng(3)
    n, ks = 16, grille_k(16)
    p = rng.beta(0.6, 1.4, size=200_000)
    vraie = np.mean([1 - (1 - np.clip(p + 0.05, 0, 1)) ** k - (1 - (1 - p) ** k) for k in ks], axis=1)
    couvert = 0
    for r in range(60):
        idx = rng.integers(0, len(p), 300)
        c_rl, c_base = rng.binomial(n, np.clip(p[idx] + 0.05, 0, 1)), rng.binomial(n, p[idx])
        # L'estimateur est sans biais : son espérance sur la population est 1 - (1 - p)^k
        d, _ = difference_appariee(c_rl, c_base, n, ks)
        lo, hi = bootstrap_multiplicateur(d, n_boot=500, seed=r)["bande_simul"]
        couvert += bool(np.all((lo <= vraie) & (vraie <= hi)))
    assert couvert >= 51  # 85 % au moins sur 60 tirages, pour un niveau nominal de 95 %


def test_frontiere():
    f = frontiere([0, 3, 1, 0, 2], [1, 0, 1, 0, 0])
    assert (f["rl"], f["base"], f["gagnees"], f["perdues"]) == (3, 2, 2, 1)


def test_selection_sur_les_memes_tirages_fabrique_un_elargissement():
    """Sans aucun effet réel (agent = départ), sélectionner les questions ratées par le départ puis réutiliser les
    mêmes tirages fait croire à un élargissement ; des tirages nouveaux le font disparaître."""
    rng = np.random.default_rng(4)
    n, ks = 16, grille_k(16)
    p = rng.beta(0.3, 3.0, size=3000)  # beaucoup de questions difficiles
    presel = rng.binomial(n, p)
    ids = sous_ensemble_difficile(dict(enumerate(presel)))
    agent = rng.binomial(n, p[ids])  # même modèle : aucun effet
    _, D_biaise = difference_appariee(agent, presel[ids], n, ks)
    _, D_juste = difference_appariee(agent, rng.binomial(n, p[ids]), n, ks)
    assert D_biaise[-1] > 0.05          # élargissement fabriqué
    assert abs(D_juste[-1]) < 0.03      # disparaît avec des tirages nouveaux
