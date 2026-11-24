"""Partie 2 : essais équivalents, prédiction de la fiabilisation pure, puissance (EXPLORATOIRE, 24 novembre 2026).

Le modèle de départ se comporte de même à T = 4 et T = 8 (session 7) : ses deux runs réunis donnent 64 tirages par
question, donc un pass@k sans biais jusqu'à k = 64, sans modèle paramétrique.

1. Essais équivalents. Pour un agent et un k, k'(k) est le nombre d'essais du départ qui donne la même réussite :
   pass_départ(k') = pass_agent(k) (pass_départ interpolé en log k entre entiers, k' dans [1, 64]). m(k) = k'/k.
   Un essai de l'agent « vaut » m(k) essais du départ quand on en fait k.
2. Fiabilisation pure. Si l'entraînement ne faisait qu'affûter chaque question, p' = 1 − (1 − p)^g (g > 1, même g
   partout), alors pass_agent(k) = E[1 − (1 − p)^(g k)] = pass_départ(g k) : m(k) = g pour tout k. On prend g = k'(1)
   (calé sur le premier essai) et l'on prédit, sans modèle, pass_agent(32) = pass_départ(32 g). L'écart observé moins
   prédit à k = 32 mesure ce que la fiabilisation pure n'explique pas (négatif : l'agent perd plus de questions
   qu'une simple concentration ne le voudrait).
3. Puissance. La règle « élargissement » exige à k = 32 : D(32) / sigma > q (q quantile simultané). Avec sigma =
   s / sqrt(N) (s : écart-type par question de d_i(32), mesuré), la puissance pour un écart vrai delta est
   Phi(delta sqrt(N) / s − q), et le N nécessaire à 80 % est ((q + z_0,8) s / delta)^2.
IC à 95 % par bootstrap des questions (1 000 tirages).

    python analysis/equivalents_partie2.py --indir results/partie2
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.confirmatoire_partie2 import analyser_paire, charger, court, reussites, taille_de  # noqa: E402
from analysis.cout_partie2 import pass_reel  # noqa: E402
from analysis.passk import pass_at_k  # noqa: E402

KS = [1, 2, 4, 8, 16, 32]
AGENTS = (("it-em-ppo", 4), ("it-em-grpo", 4), ("it-em-ppo", 8))
SFT = (("it-sft-ppo", 4), ("it-sft-depart", 4))  # H4 (session 11) : calculés, hors figure
COUL = {("it-em-ppo", 4): "#2563eb", ("it-em-grpo", 4): "#d97706", ("it-em-ppo", 8): "#0d9488"}
NOM = {("it-em-ppo", 4): "agent PPO, $T = 4$", ("it-em-grpo", 4): "agent GRPO, $T = 4$",
       ("it-em-ppo", 8): "agent PPO, $T = 8$"}


def courbe_depart(c64, n_dep=64):
    grille = np.geomspace(1, n_dep, 400)
    return grille, np.array([pass_reel(c64, n_dep, k).mean() for k in grille])


def equivalent(cible, grille, courbe):
    """k' tel que courbe(k') = cible ; nan si hors de [1, n_dep]."""
    if cible < courbe[0] or cible > courbe[-1]:
        return np.nan
    return float(np.exp(np.interp(cible, courbe, np.log(grille))))


def analyser(c_a, c64, n_dep=64):
    grille, cb = courbe_depart(c64, n_dep)
    pa = np.array([pass_at_k(32, c_a, k).mean() for k in KS])
    kp = np.array([equivalent(v, grille, cb) for v in pa])
    g = kp[0]
    pred32 = float(pass_reel(c64, n_dep, 32 * g).mean()) if np.isfinite(g) and 32 * g <= n_dep else np.nan
    return pa, kp, g, pred32


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/partie2")
    ap.add_argument("--genere", default="docs/partie2/genere")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--type", default="bridge", choices=["bridge", "compositional"],
                    help="compositional : 2WikiMultiHopQA, départ à 64 essais = graines de génération 0 et 1 à T = 4")
    ap.add_argument("--taille", type=int, default=3, choices=[3, 7],
                    help="7 : paire 7B GRPO (amendement du 24 nov.) ; départ à 64 essais si le run de graine 1 existe, "
                         "sinon 32 (m(k) seulement là où k m(k) <= 32, pas de prédiction de pass@32)")
    args = ap.parse_args(argv)
    dep = f"Qwen2.5-{args.taille}B-Instruct"
    # runs de la taille demandée seulement : l'intersection des questions ne dépend pas des runs d'une autre taille
    R = {(court(m), c[3], c[7]): reussites(L, 32) for (m, c), L in charger(args.indir).items()
         if c[0] == args.type and c[6] == "42" and c[2] == 32 and c[3] in (4, 8) and taille_de(m) == args.taille}
    ids = sorted(set.intersection(*[set(v) for v in R.values()]))
    C3 = {k: np.array([v[i] for i in ids]) for k, v in R.items()}
    n_dep = 64
    if (dep, 4, 1) in C3:  # 2Wiki, 7B : départ à T = 4, graines de génération 0 et 1
        c64 = C3[(dep, 4, 0)] + C3[(dep, 4, 1)]
    elif args.type == "bridge" and (dep, 8, 0) in C3:  # ponts 3B : départ à T = 4 et T = 8 (même comportement)
        c64 = C3[(dep, 4, 0)] + C3[(dep, 8, 0)]
    else:  # un seul run du départ
        c64, n_dep = C3[(dep, 4, 0)], 32
    C = {(m, t): v for (m, t, g), v in C3.items() if g == 0}
    agents = tuple(a for a in (AGENTS + SFT if args.taille == 3 else (("7b-it-em-grpo", 4),)) if a in C)
    rng = np.random.default_rng(0)
    res = {"questions": len(ids), "ks": KS, "essais_depart": n_dep,
           "depart_64": [float(pass_at_k(n_dep, c64, k).mean()) for k in (1, 32, n_dep)], "agents": {}, "puissance": {}}
    print(f"départ, {n_dep} tirages : pass@1 {res['depart_64'][0]:.3f}, pass@32 {res['depart_64'][1]:.3f}, "
          f"pass@{n_dep} {res['depart_64'][2]:.3f}")
    for ag in agents:
        pa, kp, g, pred = analyser(C[ag], c64, n_dep)
        boot = []
        for _ in range(args.n_boot):
            j = rng.integers(0, len(ids), len(ids))
            b = analyser(C[ag][j], c64[j], n_dep)
            boot.append(np.concatenate([b[1] / KS, [b[2], b[0][-1] - b[3]]]))
        boot = np.array(boot)
        lo, hi = np.nanpercentile(boot, 2.5, 0), np.nanpercentile(boot, 97.5, 0)
        cle = f"{ag[0]}_T{ag[1]}"
        res["agents"][cle] = {"pass": pa.tolist(), "k_equivalent": kp.tolist(), "m": (kp / KS).tolist(),
                              "m_bas": lo[:6].tolist(), "m_haut": hi[:6].tolist(), "g": g, "g_ic": [lo[6], hi[6]],
                              "pass32_predit": pred, "pass32_observe": float(pa[-1]),
                              "ecart_observe_moins_predit": float(pa[-1] - pred), "ecart_ic": [lo[7], hi[7]],
                              "part_hors_64": float(np.mean(np.isnan(boot[:, 5])))}
        r = res["agents"][cle]
        print(f"== {cle} : m(k) " + " ".join(f"{k}:{m:.2f}[{a:.2f},{b:.2f}]" for k, m, a, b in
                                            zip(KS, r["m"], r["m_bas"], r["m_haut"])))
        print(f"   g = {g:.2f} [{lo[6]:.2f}, {hi[6]:.2f}] | pass@32 observé {pa[-1]:.3f}, prédit par la fiabilisation pure "
              f"{pred:.3f} : écart {pa[-1] - pred:+.3f} [{lo[7]:+.3f}, {hi[7]:+.3f}]")
    # Puissance de la règle à k = 32 (paires T = 4 et T = 8, écart-type mesuré par question)
    z80 = norm.ppf(0.8)
    for ag in agents:
        dep_t = (dep, ag[1])
        r = analyser_paire(C[ag], C[dep_t], 32)
        d32 = (C[ag] > 0).astype(float) - (C[dep_t] > 0).astype(float)
        s, q = float(d32.std(ddof=1)), float(r["q_simul"])
        res["puissance"][f"{ag[0]}_T{ag[1]}"] = {
            "s": s, "q_simul": q,
            "puissance_N500": {str(d): float(norm.cdf(d * np.sqrt(500) / s - q)) for d in (0.02, 0.03, 0.04, 0.05)},
            "N_80": {str(d): int(np.ceil(((q + z80) * s / d) ** 2)) for d in (0.02, 0.03, 0.04, 0.05)}}
        p = res["puissance"][f"{ag[0]}_T{ag[1]}"]
        print(f"   puissance {ag} : s {s:.3f}, q {q:.2f} | à N = 500 : " +
              ", ".join(f"δ {d} -> {v:.0%}" for d, v in p["puissance_N500"].items()) +
              " | N pour 80 % : " + ", ".join(f"δ {d} -> {v}" for d, v in p["N_80"].items()))
    suffixe = ("" if args.type == "bridge" else "_2wiki") + ("" if args.taille == 3 else f"_{args.taille}b")
    Path(args.genere, f"equivalents{suffixe}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    if suffixe == "":
        figure(res, C, c64, Path(args.genere, "equivalents_ponts.pdf"))
    return res


def figure(res, C, c64, chemin):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for ag in AGENTS:
        r = res["agents"][f"{ag[0]}_T{ag[1]}"]
        a1.fill_between(KS, r["m_bas"], r["m_haut"], color=COUL[ag], alpha=0.12, lw=0)
        a1.plot(KS, r["m"], "-o", ms=4, color=COUL[ag], lw=2, label=NOM[ag])
    a1.axhline(1, color="#111827", lw=0.8)
    a1.set(xscale="log", xlabel="k (essais de l'agent)", ylabel="m(k) = essais équivalents du départ / k")
    a1.set_title("Combien d'essais du départ vaut un essai de l'agent", fontsize=10)
    # Droite : D(k) contre le départ à 64 tirages, PPO T = 4 et T = 8, IC ponctuels
    rng = np.random.default_rng(1)
    for ag in (("it-em-ppo", 4), ("it-em-ppo", 8)):
        d = np.stack([pass_at_k(32, C[ag], k) - pass_at_k(64, c64, k) for k in KS], 1)
        b = np.array([d[rng.integers(0, len(d), len(d))].mean(0) for _ in range(2000)])
        a2.fill_between(KS, np.percentile(b, 2.5, 0), np.percentile(b, 97.5, 0), color=COUL[ag], alpha=0.15, lw=0)
        a2.plot(KS, d.mean(0), "-o", ms=4, color=COUL[ag], lw=2, label=NOM[ag])
    a2.axhline(0, color="#111827", lw=0.8)
    a2.set(xscale="log", xlabel="k (essais)", ylabel="D(k) contre le départ (64 tirages)")
    a2.set_title("Budget de recherche : 4 ou 8", fontsize=10)
    for ax in (a1, a2):
        ax.set_xticks(KS, [str(k) for k in KS])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e7eb", lw=0.6)
        ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(chemin)
    plt.close(fig)


if __name__ == "__main__":
    main()
