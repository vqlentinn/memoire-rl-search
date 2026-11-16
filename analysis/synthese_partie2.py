"""Partie 2 : figure de synthèse des courbes D(k) sur les questions de pont (16 novembre 2026).

Un panneau par agent ; dans chaque panneau, l'écart apparié D(k) = pass@k(agent) − pass@k(départ) et sa bande
simultanée à 95 %, pour chaque ensemble de questions : échantillon principal (500 q., T = 4), réplication sur le vivier
(1 000 q., 16 tirages), questions ratées par le départ en présélection (session 4), questions ratées par l'agent
(session 9, si disponible), et sans recherche (T = 0). Lit docs/partie2/genere/resultats.json
(écrit par analysis/confirmatoire_partie2.py).

    python analysis/synthese_partie2.py
"""
import argparse
import json
from pathlib import Path

# (étiquette, filtre sur la condition, couleur, marqueur, style)
SERIES = [
    ("échantillon principal (500 q.)", lambda c, a: c[3] == 4 and c[6] == "42", "#2563eb", "o", "-"),
    ("réplication (1 000 q., 16 essais)", lambda c, a: c[3] == 4 and c[6] == "43", "#0d9488", "s", "-"),
    ("questions ratées par le départ", lambda c, a: c[3] == 4 and c[6] == "ids-difficiles-ponts", "#d97706", "^", "-"),
    ("questions ratées par l'agent", lambda c, a: c[3] == 4 and c[6] == f"ids-difficiles-{a}", "#7c3aed", "v", "-"),
    ("sans recherche (T = 0, 500 q.)", lambda c, a: c[3] == 0 and c[6] == "42", "#6b7280", "D", "--"),
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--genere", default="docs/partie2/genere")
    args = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = Path(args.genere)
    paires = json.loads((g / "resultats.json").read_text())["paires"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=True)
    for ax, (court, titre) in zip(axes, (("ppo", "agent PPO"), ("grpo", "agent GRPO"))):
        for nom, filtre, coul, mk, ls in SERIES:
            p = [p for p in paires if p["agent"].endswith(f"it-em-{court}-v0.3") and p["cond"][0] == "bridge"
                 and filtre(p["cond"], court)]
            if not p:
                continue
            r = p[0]["res"]
            ax.fill_between(r["ks"], r["simul_bas"], r["simul_haut"], color=coul, alpha=0.12, lw=0)
            ax.plot(r["ks"], r["D"], ls, marker=mk, color=coul, lw=2, ms=4.5, label=nom)
        ax.axhline(0, color="#111827", lw=0.8)
        ks = [1, 2, 4, 8, 16, 32]
        ax.set(xscale="log", xticks=ks, xticklabels=ks, xlabel="k (essais)", title=titre)
        ax.title.set_fontsize(10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e7eb", lw=0.6)
    axes[0].set_ylabel("D(k) = pass@k agent − départ")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(g / "synthese_dk.pdf")
    plt.close(fig)
    print(f"écrit : {g / 'synthese_dk.pdf'}")


if __name__ == "__main__":
    main()
