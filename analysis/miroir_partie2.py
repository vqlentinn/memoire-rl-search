"""Partie 2 : sous-ensemble miroir (amendement du 15 novembre 2026, EXPLORATOIRE ; sessions 8 et 9).

S_D : questions du vivier sans réussite du modèle de départ en présélection (session 3), évaluées en session 4.
S_A : questions du vivier sans réussite de l'agent A en présélection (session 8), évaluées en session 9.
D_diff(k) = pass@k(agent) − pass@k(départ) sur S_D ; D_miroir(k) = pass@k(départ) − pass@k(agent) sur S_A.
Asymétrie Δ(k) = D_diff(k) − D_miroir(k) : seule trace d'un élargissement net, car une redistribution pure
(même frontière, profils différents) avantage chaque modèle sur les questions que l'autre rate.
IC à 95 % par bootstrap des questions du VIVIER (une question peut être dans S_D et dans S_A).
Règle (plan, amendement du 15 novembre), à k = 32 et par paire : « élargissement net » si la borne basse de l'IC de
Δ(32) > 0 ; « rétrécissement net » si la borne haute < 0 ; sinon « redistribution » si la bande simultanée de
D_miroir(32) est au-dessus de 0, « indéterminé » sinon.

    python analysis/miroir_partie2.py --indir results/partie2
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.confirmatoire_partie2 import DEPART, analyser_paire, charger, court, reussites  # noqa: E402
from analysis.passk import courbes, grille_k  # noqa: E402

AGENTS = {"it-em-ppo": "ppo", "it-em-grpo": "grpo"}


def asymetrie(vivier, d_diff, d_miroir, ks, n_boot=10000, seed=0):
    """vivier : liste d'ids ; d_diff, d_miroir : {id: contributions par k (agent − départ, resp. départ − agent)}."""
    K = len(ks)
    num_d = np.array([d_diff.get(i, np.zeros(K)) for i in vivier])
    num_m = np.array([d_miroir.get(i, np.zeros(K)) for i in vivier])
    in_d = np.array([i in d_diff for i in vivier], float)
    in_m = np.array([i in d_miroir for i in vivier], float)

    def stat(j):
        return num_d[j].sum(0) / in_d[j].sum() - num_m[j].sum(0) / in_m[j].sum()

    tout = np.arange(len(vivier))
    rng = np.random.default_rng(seed)
    bs = np.array([stat(rng.integers(0, len(vivier), len(vivier))) for _ in range(n_boot)])
    return {"ks": ks, "delta": stat(tout).tolist(),
            "ic_bas": np.percentile(bs, 2.5, axis=0).tolist(), "ic_haut": np.percentile(bs, 97.5, axis=0).tolist(),
            "D_diff": (num_d.sum(0) / in_d.sum()).tolist(), "D_miroir": (num_m.sum(0) / in_m.sum()).tolist(),
            "n_diff": int(in_d.sum()), "n_miroir": int(in_m.sum())}


def verdict(asym, miroir):
    if asym["ic_bas"][-1] > 0:
        return "élargissement net"
    if asym["ic_haut"][-1] < 0:
        return "rétrécissement net"
    if miroir["simul_bas"][-1] > 0:
        return "redistribution"
    return "indéterminé"


def figure(res, chemin):
    """Par paire : avantage de l'agent sur S_D, avantage du départ sur S_A, et leur différence Δ(k) avec son IC."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    noms = {"it-em-ppo": "paire PPO", "it-em-grpo": "paire GRPO"}
    fig, axes = plt.subplots(1, len(res), figsize=(4.6 * len(res), 3.3), sharey=True, squeeze=False)
    for a, (ag, r) in zip(axes[0], res.items()):
        x = r["asymetrie"]
        ks = x["ks"]
        a.plot(ks, x["D_diff"], "-o", color="#2563eb", lw=2, ms=4,
               label="agent − départ, sur les questions ratées par le départ")
        a.plot(ks, x["D_miroir"], "-o", color="#6b7280", lw=2, ms=4,
               label="départ − agent, sur les questions ratées par l'agent")
        a.fill_between(ks, x["ic_bas"], x["ic_haut"], color="#111827", alpha=0.12, lw=0, label="IC 95 % de Δ(k)")
        a.plot(ks, x["delta"], "-", color="#111827", lw=1.5, label="Δ(k) : différence des deux")
        a.axhline(0, color="#111827", lw=0.8)
        a.set(xscale="log", xticks=ks, xticklabels=ks, xlabel="k (essais)", title=f"{noms[ag]} ({x['n_diff']} et {x['n_miroir']} questions) : {r['verdict']}")
        a.title.set_fontsize(9)
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", color="#e5e7eb", lw=0.6)
    axes[0][0].set_ylabel("écart de pass@k")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(chemin)
    plt.close(fig)


def contributions(c_a, c_b, n, ks):
    """{id: pass@k(a) − pass@k(b)} sur les ids communs et complets."""
    ids = sorted(set(c_a) & set(c_b))
    d = courbes([c_a[i] for i in ids], n, ks) - courbes([c_b[i] for i in ids], n, ks)
    return dict(zip(ids, d))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/partie2")
    ap.add_argument("--out", default="docs/partie2/genere/miroir.json")
    args = ap.parse_args(argv)
    runs = charger(args.indir)
    dep = court(DEPART["it"])

    def run(modele, ids):
        L = [L for (m, c), L in runs.items() if court(m) == modele and c[0] == "bridge" and c[3] == 4 and c[6] == ids]
        return reussites(sum(L, []), 32) if L else None

    vivier = sorted({t["id"] for (m, c), L in runs.items()
                     if c[6] == "43" and c[2] == 16 and c[0] == "bridge" for t in L})
    print(f"vivier : {len(vivier)} questions")
    n, ks = 32, grille_k(32)
    res = {}
    for ag, nom in AGENTS.items():
        a_d, b_d = run(ag, "ids-difficiles-ponts"), run(dep, "ids-difficiles-ponts")
        a_m, b_m = run(ag, f"ids-difficiles-{nom}"), run(dep, f"ids-difficiles-{nom}")
        if a_m is None or b_m is None:
            print(f"  {ag} : session 9 absente, rien à faire")
            continue
        ids_m = sorted(set(a_m) & set(b_m))
        miroir = analyser_paire(np.array([b_m[i] for i in ids_m]), np.array([a_m[i] for i in ids_m]), n)
        asym = asymetrie(vivier, contributions(a_d, b_d, n, ks), contributions(b_m, a_m, n, ks), ks)
        res[ag] = {"asymetrie": asym, "verdict": verdict(asym, miroir),
                   "miroir": {k: miroir[k] for k in ("questions", "D", "simul_bas", "simul_haut", "frontiere",
                                                      "passk_agent", "passk_depart")}}
        # dans « miroir », « agent » désigne le modèle de départ (gagnant attendu) : voir D_miroir
        print(f"  {ag} : S_D {asym['n_diff']} q., S_A {asym['n_miroir']} q. ; D_diff(32) {asym['D_diff'][-1]:+.3f}, "
              f"D_miroir(32) {asym['D_miroir'][-1]:+.3f} (bande [{miroir['simul_bas'][-1]:+.3f}, "
              f"{miroir['simul_haut'][-1]:+.3f}]) ; Δ(32) {asym['delta'][-1]:+.3f} "
              f"[{asym['ic_bas'][-1]:+.3f}, {asym['ic_haut'][-1]:+.3f}] -> {res[ag]['verdict']}")
        print(f"    frontière sur S_A : départ {miroir['frontiere']['rl']}, agent {miroir['frontiere']['base']}")
    if res:
        Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
        figure(res, Path(args.out).with_name("miroir.pdf"))
    return res


if __name__ == "__main__":
    main()
