"""Partie 2 : comparaison à coût égal (EXPLORATOIRE, 16 novembre 2026, sans GPU).

pass@k compare les modèles à nombre d'essais égal. Mais un essai de l'agent coûte plus cher : il génère plus de jetons
et lance plus de recherches. À budget égal, le modèle de départ peut tenter davantage d'essais. On compare donc
pass@budget : pour un budget B (jetons générés, ou recherches), chaque modèle dispose de k_m = B / c_m essais, où c_m
est son coût moyen par essai sur l'ensemble des questions de la condition. pass@k pour k non entier : interpolation
linéaire en log k entre les estimateurs sans biais aux entiers voisins (par question, donc les contributions d_i(B)
restent appariées). Incertitude : même bootstrap à multiplicateurs que les tests principaux, bandes simultanées sur la
grille de budgets. Le coût moyen est traité comme connu (il est estimé sur 16 000 trajectoires).

Mesures de coût exactes : jetons générés (décodage, ce que paie un GPU) et nombre de recherches (ce que paie un moteur
de recherche facturé à l'appel). Les jetons lus (passages insérés dans le contexte) ne sont pas comptés.

    python analysis/cout_partie2.py --indir results/partie2
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.confirmatoire_partie2 import DEPART, charger, court, reussites  # noqa: E402
from analysis.passk import bootstrap_multiplicateur, pass_at_k  # noqa: E402

COUTS = {"jetons": "gen_tokens", "recherches": "n_search"}


def pass_reel(c, n, k):
    """pass@k par question pour k réel dans [1, n] : interpolation linéaire en log k entre entiers."""
    k = float(np.clip(k, 1, n))
    lo, hi = int(np.floor(k)), int(np.ceil(k))
    a = pass_at_k(n, c, lo)
    if hi == lo:
        return a
    w = (np.log(k) - np.log(lo)) / (np.log(hi) - np.log(lo))
    return (1 - w) * a + w * pass_at_k(n, c, hi)


def courbe_cout(c_a, c_b, cout_a, cout_b, n=32, n_points=24):
    """Grille de budgets où les deux modèles ont entre 1 et n essais ; contributions par question."""
    b_min = max(cout_a, cout_b)          # au moins un essai chacun
    b_max = n * min(cout_a, cout_b)      # au plus n essais chacun
    budgets = np.geomspace(b_min, b_max, n_points)
    d = np.stack([pass_reel(c_a, n, b / cout_a) - pass_reel(c_b, n, b / cout_b) for b in budgets], axis=1)
    pa = np.array([pass_reel(c_a, n, b / cout_a).mean() for b in budgets])
    pb = np.array([pass_reel(c_b, n, b / cout_b).mean() for b in budgets])
    return budgets, d, pa, pb


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/partie2")
    ap.add_argument("--genere", default="docs/partie2/genere")
    args = ap.parse_args(argv)
    runs = charger(args.indir)
    dep = court(DEPART["it"])
    res = {}
    for typ in ("bridge", "comparison"):
        sel = {court(m): L for (m, c), L in runs.items() if c[0] == typ and c[3] == 4 and c[6] == "42"}
        for ag in ("it-em-ppo", "it-em-grpo"):
            a, b = reussites(sel[ag], 32), reussites(sel[dep], 32)
            ids = sorted(set(a) & set(b))
            c_a, c_b = np.array([a[i] for i in ids]), np.array([b[i] for i in ids])
            for nom, champ in COUTS.items():
                cout_a = float(np.mean([t[champ] for t in sel[ag]]))
                cout_b = float(np.mean([t[champ] for t in sel[dep]]))
                budgets, d, pa, pb = courbe_cout(c_a, c_b, cout_a, cout_b)
                r = bootstrap_multiplicateur(d, n_boot=4000)
                cle = f"{ag}_{typ}_{nom}"
                res[cle] = {"cout_agent": cout_a, "cout_depart": cout_b, "rapport": cout_a / cout_b,
                            "budgets": budgets.tolist(), "essais_agent": (budgets / cout_a).tolist(),
                            "essais_depart": (budgets / cout_b).tolist(), "pass_agent": pa.tolist(),
                            "pass_depart": pb.tolist(), "D": r["D"].tolist(),
                            "simul_bas": r["bande_simul"][0].tolist(), "simul_haut": r["bande_simul"][1].tolist(),
                            "perte_certifiee": r["perte_certifiee"], "gain_certifie": r["gain_certifie"]}
                i_max = int(np.argmax(r["D"]))
                print(f"  {cle:32} coût/essai {cout_a:7.2f} contre {cout_b:7.2f} (×{cout_a / cout_b:.2f}) | "
                      f"D au plus petit budget {r['D'][0]:+.3f} [{r['bande_simul'][0][0]:+.3f}, {r['bande_simul'][1][0]:+.3f}]"
                      f" | au plus grand {r['D'][-1]:+.3f} [{r['bande_simul'][0][-1]:+.3f}, {r['bande_simul'][1][-1]:+.3f}]"
                      f" (agent {budgets[-1] / cout_a:.1f} essais, départ {budgets[-1] / cout_b:.1f})"
                      f" | gain certifié {r['gain_certifie']}, perte certifiée {r['perte_certifiee']}")
    Path(args.genere, "cout.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    figure(res, Path(args.genere, "cout_ponts.pdf"))
    return res


def figure(res, chemin):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), sharey=True)
    for ax, (nom, etiquette) in zip(axes, (("jetons", "budget : jetons générés par question"),
                                          ("recherches", "budget : recherches par question"))):
        for ag, coul, mk, lab in (("it-em-ppo", "#2563eb", "o", "agent PPO − départ"),
                                  ("it-em-grpo", "#0891b2", "s", "agent GRPO − départ")):
            r = res[f"{ag}_bridge_{nom}"]
            ax.fill_between(r["budgets"], r["simul_bas"], r["simul_haut"], color=coul, alpha=0.13, lw=0)
            ax.plot(r["budgets"], r["D"], "-", marker=mk, markevery=4, color=coul, lw=2, ms=4,
                    label=f"{lab} (coût ×{r['rapport']:.2f})")
        ax.axhline(0, color="#111827", lw=0.8)
        ax.set(xscale="log", xlabel=etiquette)
        ticks = [300, 500, 1000, 2000, 5000] if nom == "jetons" else [2, 3, 5, 10, 20]
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e7eb", lw=0.6)
        ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    axes[0].set_ylabel("écart de pass@budget")
    fig.tight_layout()
    fig.savefig(chemin)
    plt.close(fig)


if __name__ == "__main__":
    main()
