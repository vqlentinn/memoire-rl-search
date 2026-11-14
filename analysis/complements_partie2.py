"""Partie 2 : compléments sans GPU (EXPLORATOIRE, 14 novembre 2026).

1. Effet exclu : borne haute de la bande simultanée de D(32) (ce que les données excluent comme élargissement).
2. Redistribution de la frontière : à 16 tirages, l'agent change-t-il l'ensemble des questions résolues plus que le
   hasard ? Référence : le modèle de départ contre lui-même (tirages 0-15 contre 16-31). Comparaison : l'agent sur une
   moitié contre le départ sur l'autre (moyenne des deux croisements). Sous l'hypothèse « même probabilité de réussite
   par question », les deux nombres de questions qui changent ont la même espérance (tirages tous indépendants).
   IC à 95 % par bootstrap des questions.
3. Figures : entonnoir (cherche, trouve les deux paragraphes utiles, répond, répond juste) et typologie des échecs.

    python analysis/complements_partie2.py --indir results/partie2
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.confirmatoire_partie2 import DEPART, charger, court  # noqa: E402

COULEURS = {"Qwen2.5-3B-Instruct": "#6b7280", "it-em-ppo": "#2563eb", "it-em-grpo": "#0891b2"}
NOMS = {"Qwen2.5-3B-Instruct": "départ (Instruct)", "it-em-ppo": "agent PPO", "it-em-grpo": "agent GRPO"}


def moities(lignes, n):
    h = [defaultdict(int), defaultdict(int)]
    for t in lignes:
        h[int(t["sample"]) >= n // 2][t["id"]] += int(t["em"])
        h[0][t["id"]] += 0
        h[1][t["id"]] += 0
    return h


def redistribution(agent, depart, n=32, n_boot=5000, seed=0):
    ids = sorted(set(depart[0]) & set(agent[0]))
    f = lambda x: np.array([x[i] > 0 for i in ids])  # noqa: E731
    dA, dB, aA, aB = f(depart[0]), f(depart[1]), f(agent[0]), f(agent[1])
    ref = (dA != dB).astype(float)
    croise = ((aB != dA).astype(float) + (aA != dB)) / 2
    x = croise - ref
    rng = np.random.default_rng(seed)
    bs = [x[rng.integers(0, len(x), len(x))].sum() for _ in range(n_boot)]
    return {"questions": len(ids), "change_agent_depart": float(croise.sum()), "change_depart_depart": float(ref.sum()),
            "gagnees": float(((aB & ~dA).sum() + (aA & ~dB).sum()) / 2),
            "perdues": float(((dA & ~aB).sum() + (dB & ~aA).sum()) / 2),
            "difference": float(x.sum()), "ic95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def figure_entonnoir(rech, chemin):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    etapes = ["cherche", "trouve les 2\nparagraphes utiles", "trouve et\nrépond", "trouve, répond\net juste", "juste\n(toutes trajectoires)"]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    larg = 0.26
    for j, m in enumerate(["Qwen2.5-3B-Instruct", "it-em-ppo", "it-em-grpo"]):
        r = rech[f"{m}_bridge"]
        v = [r["a_cherche"], r["trouve_2"], r["trouve_2"] * r["repond_si_trouve_2"],
             r["trouve_2"] * r["repond_si_trouve_2"] * r["em_si_trouve_2_et_repond"], r["em"]]
        x = np.arange(len(v)) + (j - 1) * larg
        ax.bar(x, v, width=larg - 0.02, color=COULEURS[m], label=NOMS[m])
        for xi, vi in zip(x, v):
            ax.text(xi, vi + 0.012, f"{vi:.0%}".replace("%", ""), ha="center", fontsize=6.5, color="#374151")
    ax.set_xticks(range(len(etapes)), etapes, fontsize=7.5)
    ax.set_ylabel("part des trajectoires", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", color="#e5e7eb", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(chemin)
    plt.close(fig)


def figure_echecs(ech, chemin):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cats = [("sans réponse : budget", ["sans_reponse_budget"], "#9ca3af", ""),
            ("sans réponse : format", ["sans_reponse_format"], "#d1d5db", ""),
            ("répond sans chercher", ["sans_recherche"], "#f59e0b", ""),
            ("recherche ratée", ["recherche_ratee"], "#dc2626", ""),
            ("lecture ratée, presque juste (F1 ≥ 0,5)", ["lecture_ratee_presque"], "#93c5fd", "////"),
            ("lecture ratée", ["lecture_ratee"], "#2563eb", "")]
    modeles = ["it-em-grpo", "it-em-ppo", "Qwen2.5-3B-Instruct"]
    fig, ax = plt.subplots(figsize=(7.2, 2.3))
    gauche = np.zeros(len(modeles))
    for nom, cles, coul, hach in cats:
        v = np.array([sum(ech[m]["part_des_trajectoires"][k] for k in cles) for m in modeles])
        ax.barh(range(len(modeles)), v, left=gauche, color=coul, hatch=hach, edgecolor="white", lw=1, label=nom)
        for i, (g, vi) in enumerate(zip(gauche, v)):
            if vi >= 0.05:
                ax.text(g + vi / 2, i, f"{vi:.0%}".replace("%", ""), ha="center", va="center", fontsize=7,
                        color="white" if coul in ("#dc2626", "#2563eb") else "#111827")
        gauche += v
    for i, m in enumerate(modeles):
        ax.text(gauche[i] + 0.01, i, f"échecs : {gauche[i]:.0%}", va="center", fontsize=7, color="#374151")
    ax.set_yticks(range(len(modeles)), [NOMS[m] for m in modeles], fontsize=8)
    ax.set_xlim(0, 0.9)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlabel("part des trajectoires (ponts, T = 4)", fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=6.5, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.45, -0.38))
    fig.tight_layout()
    fig.savefig(chemin, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/partie2")
    ap.add_argument("--genere", default="docs/partie2/genere")
    ap.add_argument("--data", default="data/raw/hotpot_distractor/validation.parquet")
    args = ap.parse_args(argv)
    g = Path(args.genere)
    res = {"effet_exclu": {}, "redistribution": {}}
    for p in json.loads((g / "resultats.json").read_text())["paires"]:
        c = p["cond"]
        if c[3] == 4 and c[6] == "42":
            r = p["res"]
            res["effet_exclu"][f"{court(p['agent'])}_{c[0]}"] = {
                "D32": r["D"][-1], "bande_simultanee_k32": [r["simul_bas"][-1], r["simul_haut"][-1]],
                "ic_ponctuel_k32": [r["ponct_bas"][-1], r["ponct_haut"][-1]], "D1": r["D"][0]}
    runs = charger(args.indir)
    for typ in ("bridge", "comparison"):
        h = {court(m): moities(L, c[2]) for (m, c), L in runs.items() if c[0] == typ and c[3] == 4 and c[6] == "42"}
        dep = court(DEPART["it"])
        for ag in sorted(k for k in h if k != dep):
            r = redistribution(h[ag], h[dep])
            res["redistribution"][f"{ag}_{typ}"] = r
            print(f"  {ag:12} {typ:10} questions qui changent : agent/départ {r['change_agent_depart']:.1f} "
                  f"(+{r['gagnees']:.1f} / −{r['perdues']:.1f}), départ/départ {r['change_depart_depart']:.0f} ; "
                  f"différence {r['difference']:+.1f} [{r['ic95'][0]:+.1f}, {r['ic95'][1]:+.1f}]")
    # Composition (ponts) : d'où viennent les questions gagnées et perdues (agent tirages 16-31, départ tirages 0-15).
    from analysis.recherche_partie2 import mesurer
    from data.hotpot_distractor import charger as charger_q
    from retrieval.closed_bm25 import ClosedBM25Retriever
    qs = {q["id"]: q for q in charger_q(args.data, "bridge", n=500, seed=42)}
    retr = ClosedBM25Retriever(list(qs.values()))
    ponts = {court(m): L for (m, c), L in runs.items() if c[0] == "bridge" and c[3] == 4 and c[6] == "42"}
    dep = ponts[court(DEPART["it"])]
    front = lambda L, moitie: {t["id"] for t in L if (int(t["sample"]) >= 16) == moitie and t["em"]}  # noqa: E731
    for ag, L in sorted(ponts.items()):
        if ag == court(DEPART["it"]):
            continue
        B = [(t, m) for t, m in zip(L, mesurer(L, qs, retr)) if int(t["sample"]) >= 16]
        perdues, gagnees = front(dep, False) - front(L, True), front(L, True) - front(dep, False)
        tp = [(t, m) for t, m in B if t["id"] in perdues]
        res["composition"] = res.get("composition", {})
        res["composition"][ag] = {
            "perdues": len(perdues), "gagnees": len(gagnees),
            "sans_reponse_sur_perdues": float(np.mean([t["status"] != "answer" for t, _ in tp])),
            "sans_reponse_partout": float(np.mean([t["status"] != "answer" for t, _ in B])),
            "trouve_2_sur_perdues": float(np.mean([m["trouve_2"] for _, m in tp])),
            "trouve_2_reussites_gagnees": float(np.mean([m["trouve_2"] for t, m in B if t["id"] in gagnees and t["em"]])),
            "trouve_2_reussites_partout": float(np.mean([m["trouve_2"] for t, m in B if t["em"]])),
        }
        print(f"  composition {ag} : " + ", ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}"
                                                  for k, v in res["composition"][ag].items()))
    for k, v in res["effet_exclu"].items():
        print(f"  {k:22} D(32) {v['D32']:+.3f}, bande simultanée [{v['bande_simultanee_k32'][0]:+.3f}, "
              f"{v['bande_simultanee_k32'][1]:+.3f}], IC ponctuel [{v['ic_ponctuel_k32'][0]:+.3f}, {v['ic_ponctuel_k32'][1]:+.3f}]")
    figure_entonnoir(json.loads((g / "recherche.json").read_text()), g / "entonnoir_ponts.pdf")
    figure_echecs(json.loads((g / "echecs.json").read_text()), g / "echecs_ponts.pdf")
    (g / "complements.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return res


if __name__ == "__main__":
    main()
