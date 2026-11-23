"""Partie 2 : H4, le fine-tuning supervisé sur les bonnes traces récupère-t-il le gain du renforcement ?
(amendement du 23 novembre 2026, règles écrites avant les données d'évaluation des modèles entraînés).

Pour chaque modèle entraîné (it-sft-ppo, principal ; it-sft-depart, secondaire), sur les 500 ponts (T = 4, n = 32) :
- contre le modèle de départ : règle H2 du plan et H1 (gain au premier essai), via analyser_paire ;
- contre l'agent PPO : E(k) = pass@k(SFT) − pass@k(PPO), IC à 95 % par bootstrap des questions (10 000 tirages) ;
  H4 « soutenue » si H1 tient et si les bornes basses de E(1) et E(32) dépassent −0,03 ; « partiellement » si H1 tient
  sans cette non-infériorité ; « non soutenue » sinon ;
- part récupérée au premier essai ρ = (pass@1(SFT) − pass@1(départ)) / (pass@1(PPO) − pass@1(départ)), IC bootstrap.
Rapporté en plus : statuts (sans réponse), recherches et jetons par trajectoire.

    python analysis/sft_partie2.py --indir results/partie2
    python analysis/sft_partie2.py --essai it-em-grpo   # vérification de la chaîne avec un modèle déjà évalué
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.confirmatoire_partie2 import DEPART, analyser_paire, charger, court, reussites  # noqa: E402
from analysis.passk import courbes  # noqa: E402

N = 32
KS = [1, 2, 4, 8, 16, 32]
MARGE = 0.03
ENTRAINES = ("it-sft-ppo", "it-sft-depart")


def ecart_bootstrap(c_x, c_y, n=N, ks=KS, n_boot=10000, seed=0):
    """E(k) = pass@k(x) − pass@k(y), appariés par question ; IC percentile à 95 % (rééchantillonnage des questions)."""
    d = courbes(c_x, n, ks) - courbes(c_y, n, ks)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    boot = d[idx].mean(1)
    return d.mean(0), np.percentile(boot, 2.5, 0), np.percentile(boot, 97.5, 0)


def part_recuperee(c_sft, c_ppo, c_dep, n=N, n_boot=10000, seed=0):
    """ρ au premier essai et son IC percentile (bootstrap des questions ; ratio recalculé à chaque tirage)."""
    a, p, b = c_sft / n, c_ppo / n, c_dep / n
    rho = (a.mean() - b.mean()) / (p.mean() - b.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n_boot, len(a)))
    num, den = (a[idx] - b[idx]).mean(1), (p[idx] - b[idx]).mean(1)
    r = num / den
    return float(rho), [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def verdict_h4(h1, e_bas):
    if not h1:
        return "non soutenue"
    if e_bas[0] > -MARGE and e_bas[-1] > -MARGE:
        return "soutenue"
    return "partiellement"


def descriptif(lignes):
    st = Counter(l["status"] for l in lignes)
    return {"em": float(np.mean([l["em"] for l in lignes])),
            "statuts": {k: v / len(lignes) for k, v in st.items()},
            "recherches": float(np.mean([l["n_search"] for l in lignes])),
            "jetons": float(np.mean([l["gen_tokens"] for l in lignes]))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/partie2")
    ap.add_argument("--genere", default="docs/partie2/genere")
    ap.add_argument("--essai", default=None, help="analyser ce modèle déjà évalué à la place des modèles entraînés")
    args = ap.parse_args(argv)
    runs = {court(m): L for (m, c), L in charger(args.indir).items()
            if c[0] == "bridge" and c[3] == 4 and c[6] == "42" and c[2] == N}
    dep, ppo = court(DEPART["it"]), "it-em-ppo"
    cibles = [args.essai] if args.essai else [m for m in ENTRAINES if m in runs]
    if not cibles:
        raise SystemExit(f"aucun modèle entraîné dans {args.indir} (attendus : {ENTRAINES})")
    r_dep, r_ppo = reussites(runs[dep], N), reussites(runs[ppo], N)
    res = {"marge": MARGE, "ks": KS, "depart": descriptif(runs[dep]), "ppo": descriptif(runs[ppo])}
    for m in cibles:
        r_m = reussites(runs[m], N)
        ids = sorted(set(r_m) & set(r_dep) & set(r_ppo))
        c_m, c_d, c_p = (np.array([r[i] for i in ids]) for r in (r_m, r_dep, r_ppo))
        paire = analyser_paire(c_m, c_d, N)
        e, lo, hi = ecart_bootstrap(c_m, c_p)
        rho, rho_ic = part_recuperee(c_m, c_p, c_d)
        x = {"questions": len(ids), "H1": paire["H1"], "H2": paire["H2"], "D_depart": paire["D"],
             "simul_bas": paire["simul_bas"], "simul_haut": paire["simul_haut"], "ks_paire": paire["ks"],
             "passk": paire["passk_agent"], "passk_depart": paire["passk_depart"],
             "passk_ppo": courbes(c_p, N, KS).mean(0).tolist(), "E_ppo": e.tolist(), "E_bas": lo.tolist(),
             "E_haut": hi.tolist(), "rho": rho, "rho_ic": rho_ic, "H4": verdict_h4(paire["H1"], lo),
             "frontiere": paire["frontiere"], "transitions": paire["transitions"], **descriptif(runs[m])}
        res[m] = x
        print(f"== {m} ({len(ids)} q.) : H1 {x['H1']}, H2 « {x['H2']} », H4 « {x['H4']} »")
        print("   pass@k " + " ".join(f"{k}:{v:.3f}" for k, v in zip(paire["ks"], paire["passk_agent"]) if k in KS))
        print("   E(k) contre PPO " + " ".join(f"{k}:{v:+.3f}[{a:+.3f},{b:+.3f}]" for k, v, a, b in zip(KS, e, lo, hi)))
        print(f"   ρ {rho:.2f} [{rho_ic[0]:.2f} ; {rho_ic[1]:.2f}] | recherches {x['recherches']:.2f}, "
              f"jetons {x['jetons']:.0f}, statuts " + ", ".join(f"{k} {v:.1%}" for k, v in x["statuts"].items()))
    if not args.essai:
        Path(args.genere, "sft.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return res


if __name__ == "__main__":
    main()
