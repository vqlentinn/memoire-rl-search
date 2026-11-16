"""Partie 2 : validation de la règle de décision par simulation (16 novembre 2026, sans GPU).

Les probabilités de réussite par question sont tirées du modèle ajusté au modèle de départ sur les ponts
(bêta-binomiale à inflation de zéros, analysis/extrapolation_partie2.py) : un monde simulé qui ressemble aux données.
Chaque réplication : N = 500 questions, n = 32 tirages par modèle, verdict H2 par la règle du plan
(analysis/confirmatoire_partie2.analyser_paire, même bootstrap). Scénarios pour l'« agent » :
- identique : mêmes p_i (tirages indépendants) ;
- fiabilisation : p'_i = 1 − (1 − p_i)^g pour p_i > 0, les questions hors de portée le restent ;
- élargissement : une part delta des questions hors de portée devient résoluble avec p' = 0,05 ;
- redistribution : même loi marginale, mais profils corrélés à rho = 0,7 seulement (copule gaussienne) : même
  frontière attendue, pas les mêmes questions.
Pour la redistribution, on simule aussi le sous-ensemble difficile du plan (présélection du départ à 16 tirages sur
un vivier de 1 000 questions, évaluation sur 32 tirages nouveaux) et son miroir (présélection de l'agent).

    python analysis/simulation_partie2.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta as loi_beta, norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.confirmatoire_partie2 import analyser_paire  # noqa: E402


def quantile_zibb(u, a, b, pi0):
    u = np.asarray(u, float)
    p = np.zeros_like(u)
    m = u > pi0
    p[m] = loi_beta.ppf((u[m] - pi0) / (1 - pi0), a, b)
    return p


def couple(scenario, m, par, rng, x=None):
    a, b, pi0 = par
    if scenario == "redistribution":
        z = rng.multivariate_normal([0, 0], [[1, 0.7], [0.7, 1]], size=m)
        return quantile_zibb(norm.cdf(z[:, 0]), a, b, pi0), quantile_zibb(norm.cdf(z[:, 1]), a, b, pi0)
    p = quantile_zibb(rng.random(m), a, b, pi0)
    if scenario == "identique":
        return p, p.copy()
    if scenario == "fiabilisation":
        return p, np.where(p > 0, 1 - (1 - p) ** x, 0.0)
    if scenario == "elargissement":
        q = p.copy()
        zeros = np.flatnonzero(p == 0)
        k = int(round(x * m))
        q[rng.choice(zeros, size=min(k, len(zeros)), replace=False)] = 0.05
        return p, q
    raise ValueError(scenario)


def verdict(p_dep, p_ag, rng, n=32, n_boot=1000):
    c_b, c_a = rng.binomial(n, p_dep), rng.binomial(n, p_ag)
    r = analyser_paire(c_a, c_b, n, n_boot=n_boot)
    return r["H2"], r["D"][-1]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--genere", default="docs/partie2/genere")
    ap.add_argument("--reps", type=int, default=200)
    args = ap.parse_args(argv)
    par = json.loads(Path(args.genere, "extrapolation.json").read_text())["it-em-ppo_bridge"]["ZIBB"]["parametres_depart"][:3]
    print(f"monde simulé : a {par[0]:.3f}, b {par[1]:.3f}, pi0 {par[2]:.3f}")
    rng = np.random.default_rng(2026)
    res = {"parametres": par, "reps": args.reps, "scenarios": {}}
    plan = [("identique", None), ("fiabilisation", 1.5), ("fiabilisation", 2.0)] + \
           [("elargissement", d) for d in (0.02, 0.04, 0.06, 0.08)] + [("redistribution", None)]
    for sc, x in plan:
        vs, ds = [], []
        for _ in range(args.reps):
            p_d, p_a = couple(sc, 500, par, rng, x)
            v, d = verdict(p_d, p_a, rng)
            vs.append(v)
            ds.append(d)
        taux = {k: vs.count(k) / len(vs) for k in ("élargissement", "fiabilisation seule", "rétrécissement", "indéterminé")}
        cle = sc + ("" if x is None else f"_{x}")
        res["scenarios"][cle] = {"taux": taux, "D32_moyen": float(np.mean(ds))}
        print(f"  {cle:22} D(32) moyen {np.mean(ds):+.3f} | " + " ".join(f"{k} {v:.0%}" for k, v in taux.items()))
    # Sous-ensemble difficile et miroir sous redistribution pure
    for sens in ("difficile", "miroir"):
        vs = []
        for _ in range(args.reps):
            p_d, p_a = couple("redistribution", 1000, par, rng)
            sel_p = p_d if sens == "difficile" else p_a
            garde = rng.binomial(16, sel_p) == 0
            v, _ = verdict(p_d[garde], p_a[garde], rng)
            vs.append(v)
        taux = {k: vs.count(k) / len(vs) for k in ("élargissement", "fiabilisation seule", "rétrécissement", "indéterminé")}
        res["scenarios"][f"redistribution_{sens}"] = {"taux": taux}
        print(f"  redistribution, {sens:9} | " + " ".join(f"{k} {v:.0%}" for k, v in taux.items()))
    Path(args.genere, "simulation.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return res


if __name__ == "__main__":
    main()
