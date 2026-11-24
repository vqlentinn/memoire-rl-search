"""Partie 2 : sensibilité au budget de recherche, T = 8 contre T = 4 (session 7, paire PPO ; EXPLORATOIRE, 24 nov. 2026).

Le modèle de départ se comporte de même à T = 4 et T = 8 (moins d'une recherche par essai, limite presque jamais
atteinte) : ses deux runs sont deux tirages du même comportement. On compare donc :
- les paires « croisées » (PPO T = 8 contre départ T = 4, départ T = 8 contre départ T = 4) par la règle du plan ;
- l'agent à chaque budget contre les 64 essais réunis du départ (pass@k estimé sur 64 tirages, plus précis),
  IC ponctuel à 95 % par bootstrap des questions ;
- pass@budget à T = 8 (mêmes fonctions que analysis/cout_partie2.py).

    python analysis/budget_partie2.py --indir results/partie2
"""
import argparse
import json
import sys
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.confirmatoire_partie2 import DEPART, analyser_paire, charger, court, reussites  # noqa: E402
from analysis.cout_partie2 import COUTS, courbe_cout  # noqa: E402
from analysis.passk import bootstrap_multiplicateur  # noqa: E402


def pass_k(c, n, k):
    return np.array([1 - comb(n - x, k) / comb(n, k) if n - x >= k else 1.0 for x in c])


def ecart_ic(c_a, n_a, c_b, n_b, k, n_boot=5000, seed=0):
    d = pass_k(c_a, n_a, k) - pass_k(c_b, n_b, k)
    rng = np.random.default_rng(seed)
    b = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return float(d.mean()), [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/partie2")
    ap.add_argument("--genere", default="docs/partie2/genere")
    args = ap.parse_args(argv)
    runs = charger(args.indir)
    dep, ppo = court(DEPART["it"]), "it-em-ppo"
    lignes = {(court(m), c[3]): L for (m, c), L in runs.items()
              if c[0] == "bridge" and c[6] == "42" and c[2] == 32 and c[3] in (4, 8)}
    R = {k: reussites(L, 32) for k, L in lignes.items()}
    ids = sorted(set.intersection(*[set(R[(m, t)]) for m in (dep, ppo) for t in (4, 8)]))
    C = {k: np.array([R[k][i] for i in ids]) for k in R if k[0] in (dep, ppo)}
    res = {"questions": len(ids), "paires": {}, "contre_depart_64": {}, "cout_T8": {}}
    for nom, a, b in (("PPO T8 / départ T8", (ppo, 8), (dep, 8)), ("PPO T8 / départ T4", (ppo, 8), (dep, 4)),
                      ("PPO T4 / départ T4", (ppo, 4), (dep, 4)), ("départ T8 / départ T4", (dep, 8), (dep, 4)),
                      ("PPO T8 / PPO T4", (ppo, 8), (ppo, 4))):
        r = analyser_paire(C[a], C[b], 32)
        res["paires"][nom] = {"D1": r["D"][0], "D32": r["D"][-1], "simul32": [r["simul_bas"][-1], r["simul_haut"][-1]],
                              "H2": r["H2"], "frontiere": [r["frontiere"]["rl"], r["frontiere"]["base"]]}
        x = res["paires"][nom]
        print(f"  {nom:22} D(1) {x['D1']:+.3f} D(32) {x['D32']:+.3f} bande [{x['simul32'][0]:+.3f}, {x['simul32'][1]:+.3f}]"
              f" « {x['H2']} » frontière {x['frontiere'][0]}/{x['frontiere'][1]}")
    c64 = C[(dep, 4)] + C[(dep, 8)]
    for t in (4, 8):
        for k in (1, 32):
            v, ic = ecart_ic(C[(ppo, t)], 32, c64, 64, k)
            res["contre_depart_64"][f"PPO T{t} k{k}"] = {"ecart": v, "ic": ic}
            print(f"  PPO T{t} contre départ (64 essais), k = {k:2} : {v:+.3f} [{ic[0]:+.3f}, {ic[1]:+.3f}]")
    for nom, champ in COUTS.items():
        xa = float(np.mean([t[champ] for t in lignes[(ppo, 8)]]))
        xb = float(np.mean([t[champ] for t in lignes[(dep, 8)]]))
        budgets, d, _, _ = courbe_cout(C[(ppo, 8)], C[(dep, 8)], xa, xb)
        r = bootstrap_multiplicateur(d, n_boot=4000)
        res["cout_T8"][nom] = {"rapport": xa / xb, "D_petit": float(r["D"][0]), "D_grand": float(r["D"][-1]),
                               "bande_petit": [float(r["bande_simul"][0][0]), float(r["bande_simul"][1][0])],
                               "bande_grand": [float(r["bande_simul"][0][-1]), float(r["bande_simul"][1][-1])],
                               "gain_certifie": bool(r["gain_certifie"]), "perte_certifiee": bool(r["perte_certifiee"])}
        x = res["cout_T8"][nom]
        print(f"  coût ({nom}) ×{x['rapport']:.2f} : petit budget {x['D_petit']:+.3f} {x['bande_petit']}, "
              f"grand {x['D_grand']:+.3f} | gain certifié {x['gain_certifie']}, perte certifiée {x['perte_certifiee']}")
    Path(args.genere, "budget.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return res


if __name__ == "__main__":
    main()
