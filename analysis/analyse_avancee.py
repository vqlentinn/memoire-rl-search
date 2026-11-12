"""Analyses avancées (exploratoires) : modèle logistique, courbe de gain des recherches, sensibilité à la
métrique, tailles d'effet, décomposition par jeu, complémentarité des systèmes.

Comme analyse_approfondie.py, ces analyses ont été choisies après avoir vu les résultats : l'article les
présente comme exploratoires. Elles répondent à des objections précises (voir les commentaires de chaque bloc).

    python analysis/analyse_avancee.py --indir results/eval_v4
Sorties dans docs/article/genere/ : avance_valeurs.tex (macros \\valAv...), avance_glm.tex,
avance_sensibilite.tex, avance_jeux.tex, fig_fourragement.pdf.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.analyse_approfondie import FAM, HARNAIS, NOMS, RL, _parts, famille, trouvee  # noqa: E402
from analysis.analyse_mini import ENCRE, ENCRE2, GRILLE, JEUX, SYSTEMES, charger, fr, holm, style  # noqa: E402
from eval.metrics import normalize_answer  # noqa: E402
from eval.stats import mcnemar  # noqa: E402

CHERCHENT = HARNAIS + RL
N_BOOT, GRAINE = 1000, 0


def souple(l):
    """Correspondance souple : une réponse attendue apparaît en mots entiers dans la prédiction.
    Répond à l'objection « l'EM pénalise « Dame Judi Dench » pour « Judi Dench » »."""
    p = l.get("prediction")
    if not p:
        return 0
    p = f" {normalize_answer(p)} "
    return int(any(g and f" {g} " in p for g in (normalize_answer(x) for x in l["golds"])))


def tableau_long(D):
    import pandas as pd
    rows = []
    for (s, j), lignes in D.items():
        for i, l in lignes.items():
            rows.append({"q": f"{j}:{i}", "jeu": j, "sys": s, "rl": int(s in RL), "em": l["em"], "f1": l["f1"],
                         "souple": souple(l), "trouvee": int(trouvee(l)) if s != "A0" else 0,
                         "connue": D[("A0", j)][i]["em"], "n_search": l["n_search"]})
    return pd.DataFrame(rows)


def glm(df):
    """Régression logistique EM ~ RL × trouvée + connue + jeu, erreurs-types groupées par question.
    Répond à « l'écart vient-il seulement de questions plus faciles ou de passages plus utiles ? »."""
    import statsmodels.formula.api as smf
    d = df[df.sys.isin(CHERCHENT)].copy()
    m = smf.logit("em ~ rl * trouvee + connue + C(jeu)", data=d).fit(
        disp=False, cov_type="cluster", cov_kwds={"groups": d["q"].astype("category").cat.codes})
    ci = m.conf_int()
    noms = {"rl": "Agent entraîné (RL)", "trouvee": "Réponse dans les passages", "rl:trouvee": "RL $\\times$ réponse dans les passages",
            "connue": "Question connue du modèle seul (A0 juste)", "C(jeu)[T.triviaqa]": "Jeu TriviaQA (contre NQ)"}
    res = {k: (np.exp(m.params[k]), np.exp(ci.loc[k, 0]), np.exp(ci.loc[k, 1]), m.pvalues[k]) for k in noms}
    return res, noms, int(m.nobs), d["q"].nunique()


def courbe_gain(D):
    """Part cumulée des questions où la réponse a été trouvée après k recherches, et gain marginal de la
    k-ième recherche. C'est la courbe de rendement décroissant de la théorie du fourragement."""
    out = {}
    for s in CHERCHENT:
        L = [l for j in JEUX for l in D[(s, j)].values()]
        cum = [np.mean([trouvee(l, k) for l in L]) for k in range(1, 5)]
        # part des questions qui ont fait au moins k recherches (le gain ne peut venir que de celles-là)
        atteint = [np.mean([l["n_search"] >= k for l in L]) for k in range(1, 5)]
        out[s] = {"cum": cum, "atteint": atteint, "marg": [cum[0]] + [cum[k] - cum[k - 1] for k in range(1, 4)]}
    return out


def boot_paire(a, b, n_boot=N_BOOT, seed=GRAINE):
    rng = np.random.default_rng(seed)
    d = np.asarray(a, float) - np.asarray(b, float)
    m = rng.choice(d, size=(n_boot, len(d))).mean(axis=1)
    return d.mean(), *np.quantile(m, [0.025, 0.975])


def sensibilite(df):
    """HA1 recalculée avec trois métriques. Répond à « le résultat dépend-il de la stricte EM ? »."""
    out = {}
    for met in ("em", "souple", "f1"):
        piv = df.pivot_table(index="q", columns="sys", values=met)
        rl, ha = piv[RL].mean(axis=1), piv[HARNAIS].mean(axis=1)
        delta, lo, hi = boot_paire(rl, ha)
        ps, nb = [], 0
        for a in RL:
            for b in HARNAIS:
                if met == "f1":
                    continue
                _, _, p = mcnemar(piv[a].values, piv[b].values)
                ps.append((p, piv[a].mean() - piv[b].mean()))
        if ps:
            adj = holm([p for p, _ in ps])
            nb = sum(1 for (p, dlt), q in zip(ps, adj) if dlt > 0 and q < 0.05)
        out[met] = {"delta": delta, "lo": lo, "hi": hi, "nb": nb if met != "f1" else None}
    return out


def tailles_effet(df):
    """Odds ratio de McNemar (c / b : questions que seul l'agent réussit / que seul le harnais réussit),
    avec intervalle exact (Clopper-Pearson sur c / (b + c))."""
    piv = df.pivot_table(index="q", columns="sys", values="em")
    out = {}
    for a in RL:
        for h in HARNAIS:
            c = int(((piv[a] == 1) & (piv[h] == 0)).sum()); b = int(((piv[a] == 0) & (piv[h] == 1)).sum())
            lo = beta.ppf(0.025, c, b + 1) if c else 0.0
            hi = beta.ppf(0.975, c + 1, b) if b else np.inf
            out[(a, h)] = (c / b if b else np.inf, lo / (1 - lo), hi / (1 - hi) if hi < 1 else np.inf, c, b)
    return out


def decomp_par_jeu(D):
    """La décomposition trouver / lire / sans appui, recalculée sur chaque jeu, avec bootstrap."""
    out = {}
    for j in JEUX:
        ids = sorted(D[("A0", j)])
        cache = {s: np.array([[trouvee(D[(s, j)][i]), D[(s, j)][i]["em"]] for i in ids], float) for s in CHERCHENT}
        pile = lambda m, idx: np.vstack([cache[s][idx] for s in m])  # noqa: E731
        base = _parts(pile(HARNAIS, np.arange(len(ids))), pile(RL, np.arange(len(ids))))
        rng = np.random.default_rng(GRAINE)
        boots = np.array([_parts(pile(HARNAIS, b), pile(RL, b))
                          for b in (rng.integers(0, len(ids), len(ids)) for _ in range(N_BOOT))])
        lo, hi = np.quantile(boots, [0.025, 0.975], axis=0)
        out[j] = (base, lo, hi)
    return out


def complementarite(df):
    """Les systèmes réussissent-ils les mêmes questions ? Borne « oracle » : une question est réussie si au
    moins un système de la famille la réussit."""
    piv = df.pivot_table(index="q", columns="sys", values="em")
    jac = lambda a, b: ((piv[a] == 1) & (piv[b] == 1)).sum() / max(1, ((piv[a] == 1) | (piv[b] == 1)).sum())  # noqa: E731
    return {"oracle_h": piv[HARNAIS].max(axis=1).mean(), "oracle_rl": piv[RL].max(axis=1).mean(),
            "oracle_tous": piv[SYSTEMES].max(axis=1).mean(), "jac_rl": jac("A4", "A5"),
            "jac_h": np.mean([jac("A1", "A2"), jac("A1", "A3"), jac("A2", "A3")]),
            "aucun": (piv[SYSTEMES].max(axis=1) == 0).mean()}


def ecrire(G, C, S, T, J, K, out, n_obs, n_q):
    v = ["% Généré par analysis/analyse_avancee.py : ne pas modifier à la main."]
    m = lambda k, x: v.append(f"\\expandafter\\def\\csname valAv-{k}\\endcsname{{{x}}}")  # noqa: E731
    pval = lambda p: "$<$\\,0{,}001" if p < 0.001 else fr(p, 3)  # noqa: E731
    res, noms = G
    t = ["\\begin{tabular}{@{}lccc@{}}", "\\toprule", "Variable & Odds ratio & IC 95\\,\\% & $p$\\\\", "\\midrule"]
    for k, nom in noms.items():
        o, lo, hi, p = res[k]
        t.append(f"{nom} & {fr(o)} & [{fr(lo)}\\,; {fr(hi)}] & {pval(p)}\\\\")
        cle = k.replace(":", "x").replace("C(jeu)[T.triviaqa]", "tqa")
        m(f"glm-{cle}", fr(o)); m(f"glm-{cle}-lo", fr(lo)); m(f"glm-{cle}-hi", fr(hi)); m(f"glm-{cle}-p", pval(p))
    t += ["\\bottomrule", "\\end{tabular}"]
    (out / "avance_glm.tex").write_text("\n".join(t) + "\n")
    m("glm-nobs", f"{n_obs:,}".replace(",", "\\,")); m("glm-nq", n_q)

    for s, d in C.items():
        for k in range(4):
            m(f"{s}-cum{k + 1}", fr(100 * d["cum"][k], 0)); m(f"{s}-marg{k + 1}", fr(100 * d["marg"][k], 1))
            m(f"{s}-atteint{k + 1}", fr(100 * d["atteint"][k], 0))

    noms_met = {"em": "Correspondance exacte (EM)", "souple": "Correspondance souple", "f1": "F1"}
    t = ["\\begin{tabular}{@{}lcc@{}}", "\\toprule", "Métrique & Écart RL $-$ harnais [IC 95\\,\\%] & Paires significatives (Holm)\\\\", "\\midrule"]
    for met, d in S.items():
        nb = "—" if d["nb"] is None else f"{d['nb']} sur 6"
        t.append(f"{noms_met[met]} & {fr(100 * d['delta'], 1)} pt [{fr(100 * d['lo'], 1)}\\,; {fr(100 * d['hi'], 1)}] & {nb}\\\\")
        m(f"sens-{met}", fr(100 * d["delta"], 1)); m(f"sens-{met}-lo", fr(100 * d["lo"], 1)); m(f"sens-{met}-hi", fr(100 * d["hi"], 1))
        if d["nb"] is not None:
            m(f"sens-{met}-nb", d["nb"])
    t.append("\\midrule")
    t.append("\\multicolumn{3}{@{}l}{\\emph{Tailles d'effet (EM) : odds ratio de McNemar, questions réussies par le seul agent / par le seul harnais}}\\\\")
    for (a, h), (o, lo, hi, c, b) in T.items():
        t.append(f"{a} contre {h} & OR = {fr(o, 1)} [{fr(lo, 1)}\\,; {fr(hi, 1)}] & {c} contre {b} questions\\\\")
    t += ["\\bottomrule", "\\end{tabular}"]
    (out / "avance_sensibilite.tex").write_text("\n".join(t) + "\n")
    ors = [x[0] for x in T.values()]
    m("or-min", fr(min(ors), 1)); m("or-max", fr(max(ors), 1))

    t = ["\\begin{tabular}{@{}lcccc@{}}", "\\toprule", "Jeu & Trouver & Lire & Sans appui & Écart total\\\\", "\\midrule"]
    for j, (base, lo, hi) in J.items():
        cells = [f"{fr(100 * base[k], 1)} [{fr(100 * lo[k], 1)}\\,; {fr(100 * hi[k], 1)}]" for k in range(4)]
        t.append(f"{JEUX[j]} & " + " & ".join(cells) + "\\\\")
        for k, n in enumerate(["trouver", "lire", "sans", "total"]):
            m(f"jeu-{j}-{n}", fr(100 * base[k], 1))
    t += ["\\bottomrule", "\\end{tabular}"]
    (out / "avance_jeux.tex").write_text("\n".join(t) + "\n")

    for k, x in K.items():
        m(f"comp-{k}", fr(100 * x, 0) if k.startswith(("oracle", "aucun")) else fr(x))
    (out / "avance_valeurs.tex").write_text("\n".join(v) + "\n")


def figure_gain(C, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "pdf.fonttype": 42})
    marqueurs = {"A1": "s", "A2": "o", "A3": "^", "A4": "D", "A5": "o"}
    traits = {"A1": "-", "A2": "--", "A3": ":", "A4": "-", "A5": "--"}
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.7))
    k = np.arange(1, 5)
    for s in HARNAIS + RL:
        d = C[s]
        n_max = 1 if s == "A1" else 4
        axes[0].plot(k[:n_max], np.array(d["cum"][:n_max]) * 100, traits[s], marker=marqueurs[s], ms=5, lw=1.8,
                     color=FAM[famille(s)], mec="white", mew=0.8)
        axes[0].annotate(s, (n_max, d["cum"][n_max - 1] * 100), xytext=(5, 0), textcoords="offset points",
                         fontsize=7.5, va="center", color=ENCRE)
        if s != "A1":
            axes[1].plot(k, np.array(d["marg"]) * 100, traits[s], marker=marqueurs[s], ms=5, lw=1.8,
                         color=FAM[famille(s)], mec="white", mew=0.8, label=NOMS[s])
    axes[0].set_ylim(0, 55); axes[0].set_xticks(k); axes[0].set_xlim(0.7, 4.6)
    axes[0].set_xlabel("Nombre de recherches", color=ENCRE2, fontsize=8)
    axes[0].set_title("Réponse trouvée (part cumulée, %)", fontsize=8.5, color=ENCRE)
    axes[1].set_ylim(0, 40); axes[1].set_xticks(k)
    axes[1].set_xlabel("Rang de la recherche", color=ENCRE2, fontsize=8)
    axes[1].set_title("Gain de la k-ième recherche (points)", fontsize=8.5, color=ENCRE)
    axes[1].legend(frameon=False, fontsize=7, loc="upper right")
    for ax in axes:
        style(ax); ax.grid(axis="y", color=GRILLE, lw=0.6)
    fig.tight_layout()
    fig.savefig(out / "fig_fourragement.pdf"); fig.savefig(out / "fig_fourragement.png", dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/eval_v4")
    ap.add_argument("--outdir", default="docs/article/genere")
    args = ap.parse_args()
    D, fictif = charger(args.indir)
    df = tableau_long(D)
    res, noms, n_obs, n_q = glm(df)
    C, S, T, J, K = courbe_gain(D), sensibilite(df), tailles_effet(df), decomp_par_jeu(D), complementarite(df)
    out = Path(args.outdir)
    ecrire((res, noms), C, S, T, J, K, out, n_obs, n_q)
    figure_gain(C, out)
    print(("DONNÉES FICTIVES — " if fictif else "") + f"GLM ({n_obs} obs., {n_q} questions) :")
    for k in noms:
        o, lo, hi, p = res[k]
        print(f"  {k:22} OR {o:.2f} [{lo:.2f}; {hi:.2f}] p={p:.4f}")
    print("Gain cumulé (k=1..4) :", {s: [round(100 * x) for x in d["cum"]] for s, d in C.items()})
    print("Gain marginal :", {s: [round(100 * x, 1) for x in d["marg"]] for s, d in C.items()})
    print("Atteint k recherches :", {s: [round(100 * x) for x in d["atteint"]] for s, d in C.items()})
    print("Sensibilité :", {k: (round(100 * d["delta"], 1), round(100 * d["lo"], 1), round(100 * d["hi"], 1), d["nb"]) for k, d in S.items()})
    print("Tailles d'effet :", {f"{a}-{h}": (round(o, 1), c, b) for (a, h), (o, lo, hi, c, b) in T.items()})
    print("Par jeu :", {j: [round(100 * x, 1) for x in base] for j, (base, lo, hi) in J.items()})
    print("Complémentarité :", {k: round(float(x), 2) for k, x in K.items()})


if __name__ == "__main__":
    main()
