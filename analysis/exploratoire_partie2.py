"""Partie 2 : analyses EXPLORATOIRES (non prévues par le plan gelé, écrites après avoir vu les résultats des sessions
1 et 2, le 13 novembre 2026). Elles décrivent ; elles ne testent aucune hypothèse et ne changent aucun verdict.

1. Gain par difficulté, sur tirages séparés. Classer les questions selon la réussite du modèle de départ, puis mesurer
   l'écart sur les MÊMES tirages, crée une régression vers la moyenne (les questions « faciles » le sont en partie par
   chance, et l'agent semble y perdre même s'il est identique). On classe donc sur les tirages 0 à 15 du modèle de
   départ, et l'on mesure les deux modèles sur les tirages 16 à 31 seulement. IC à 95 % par bootstrap par question.
   La classe « 0/16 » est une version interne du sous-ensemble difficile (présélection et mesure indépendantes, mais
   16 tirages de mesure et pas de vivier séparé) : elle annonce la session 4 sans la remplacer.
2. Causes des sorties sans réponse, par classe : format (sortie mal formée), budget épuisé (recherche refusée),
   troncature ; et EM parmi les trajectoires qui répondent.

    python analysis/exploratoire_partie2.py --indir results/partie2 --outdir docs/partie2/genere
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.confirmatoire_partie2 import DEPART, charger, court  # noqa: E402

CLASSES = [("0/16", 0, 0), ("1-3/16", 1, 3), ("4-8/16", 4, 8), ("9-15/16", 9, 15), ("16/16", 16, 16)]
SEL, MES = range(0, 16), range(16, 32)


def par_question(lignes):
    m = defaultdict(dict)
    for l in lignes:
        m[l["id"]][l["sample"]] = l
    return m


def ic_moyenne(v, n_boot=2000, seed=0):
    v = np.asarray(v, dtype=float)
    if len(v) < 2:
        return float(v.mean()) if len(v) else float("nan"), float("nan"), float("nan")
    xi = np.random.default_rng(seed).standard_normal((n_boot, len(v)))
    s = (xi @ (v - v.mean()) / len(v)).std(ddof=1)
    return float(v.mean()), float(v.mean() - 1.96 * s), float(v.mean() + 1.96 * s)


def par_difficulte(base, agent):
    ids = sorted(i for i in set(base) & set(agent) if all(s in base[i] for s in range(32))
                 and all(s in agent[i] for s in MES))
    out = []
    for nom, lo, hi in CLASSES:
        q = [i for i in ids if lo <= sum(base[i][s]["em"] for s in SEL) <= hi]
        if not q:
            continue
        pb = np.array([np.mean([base[i][s]["em"] for s in MES]) for i in q])
        pa = np.array([np.mean([agent[i][s]["em"] for s in MES]) for i in q])
        traj_a = [agent[i][s] for i in q for s in MES]
        st = Counter(t["status"] for t in traj_a)
        rep = [t["em"] for t in traj_a if t["status"] == "answer"]
        rep_b = [base[i][s]["em"] for i in q for s in MES if base[i][s]["status"] == "answer"]
        out.append({
            "classe": nom, "questions": len(q),
            "reussite_depart": float(pb.mean()), "reussite_agent": float(pa.mean()), "ecart": ic_moyenne(pa - pb),
            "jamais_depart": float(np.mean(pb == 0)), "jamais_agent": float(np.mean(pa == 0)),
            "agent_sans_reponse": 1 - st["answer"] / len(traj_a), "agent_format": st["format"] / len(traj_a),
            "agent_budget": st["budget_exhausted"] / len(traj_a), "agent_tronquees": st["truncated"] / len(traj_a),
            "em_si_reponse_agent": float(np.mean(rep)) if rep else float("nan"),
            "em_si_reponse_depart": float(np.mean(rep_b)) if rep_b else float("nan"),
        })
    return out


def figure(resultats, chemin):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cles = list(resultats)
    fig, axes = plt.subplots(1, len(cles), figsize=(3.3 * len(cles), 3.2), sharey=True)
    for ax, cle in zip(np.atleast_1d(axes), cles):
        R = resultats[cle]
        x = np.arange(len(R))
        m = [r["ecart"][0] for r in R]
        err = np.array([[r["ecart"][0] - r["ecart"][1] for r in R], [r["ecart"][2] - r["ecart"][0] for r in R]])
        ax.axhline(0, color="#111827", lw=0.8)
        ax.errorbar(x, m, yerr=err, fmt="o", color="#2563eb", ms=5, capsize=3, lw=1.5)
        ax.set_xticks(x, [f"{r['classe']}\n({r['questions']} q)" for r in R], fontsize=7)
        ax.set_title(cle, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e7eb", lw=0.6)
    np.atleast_1d(axes)[0].set_ylabel("réussite agent − départ\n(tirages 16 à 31)")
    fig.supxlabel("réussites du modèle de départ sur les tirages 0 à 15", fontsize=9)
    fig.tight_layout()
    fig.savefig(chemin)
    plt.close(fig)


def latex(resultats):
    L = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
         r"Agent, type & Classe (départ, tirages 0--15) & $N$ & Départ & Agent & Écart [IC 95\,\%] & "
         r"Jamais résolues (départ / agent) & Sans réponse (agent) \\", r"\midrule"]
    for cle, R in resultats.items():
        for r in R:
            e = r["ecart"]
            L.append(f"{cle} & {r['classe']} & {r['questions']} & {r['reussite_depart']:.3f} & {r['reussite_agent']:.3f}"
                     f" & {e[0]:+.3f} [{e[1]:+.3f} ; {e[2]:+.3f}] & {r['jamais_depart']:.0%} / {r['jamais_agent']:.0%}"
                     f" & {r['agent_sans_reponse']:.0%} \\\\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L.append(r"\end{tabular}")
    return "\n".join(L).replace("%", r"\%").replace(r"\,\%", r"\,\%")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/partie2")
    ap.add_argument("--outdir", default="docs/partie2/genere")
    args = ap.parse_args(argv)
    runs = charger(args.indir)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    resultats = {}
    for (modele, cond), lignes in sorted(runs.items()):
        if modele in DEPART.values() or cond[2] != 32 or cond[6] != "42" or cond[7] != 0:
            continue
        cle_dep = (DEPART["it" if "-it-" in modele else "base"], cond)
        if cle_dep not in runs:
            continue
        cle = f"{court(modele)}, {cond[0]}, T={cond[3]}"
        resultats[cle] = par_difficulte(par_question(runs[cle_dep]), par_question(lignes))
        print(f"== {cle}")
        for r in resultats[cle]:
            e = r["ecart"]
            print(f"   {r['classe']:8} {r['questions']:3} q | départ {r['reussite_depart']:.3f} agent "
                  f"{r['reussite_agent']:.3f} écart {e[0]:+.3f} [{e[1]:+.3f}, {e[2]:+.3f}] | jamais "
                  f"{r['jamais_depart']:.0%} / {r['jamais_agent']:.0%} | agent sans réponse "
                  f"{r['agent_sans_reponse']:.1%} (format {r['agent_format']:.1%}, budget {r['agent_budget']:.1%}) | "
                  f"EM si réponse agent {r['em_si_reponse_agent']:.3f} départ {r['em_si_reponse_depart']:.3f}")
    if resultats:
        (out / "exploratoire.json").write_text(json.dumps(resultats, indent=1, ensure_ascii=False))
        (out / "tableau_difficulte.tex").write_text(latex(resultats) + "\n")
        figure(resultats, out / "difficulte_tirages_separes.pdf")
    return resultats


if __name__ == "__main__":
    main()
