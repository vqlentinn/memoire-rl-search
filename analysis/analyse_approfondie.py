"""Analyses complémentaires (exploratoires) de la mini-expérience : trouver ou lire, quoi chercher,
gains et pertes, typologie des erreurs, comparaisons appariées, coût en jetons.

Ces analyses n'étaient pas dans le plan d'analyse fixé avant le run : l'article les présente comme
exploratoires, sans verdict. Elles partent uniquement des fichiers JSONL.

« Réponse trouvée » : une des réponses attendues, normalisée comme pour l'EM, apparaît telle quelle dans un
passage lu (normalisé de même). C'est une estimation : un passage peut contenir la chaîne sans la justifier.

    python analysis/analyse_approfondie.py --indir results/eval_v4
Sorties dans docs/article/genere/ : approf_valeurs.tex (macros \\valAp...), approf_requetes.tex,
fig_transitions.pdf, fig_decomposition.pdf, fig_erreurs.pdf, fig_matrice.pdf.
"""
import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.analyse_mini import ENCRE, ENCRE2, GRILLE, JEUX, SYSTEMES, charger, fr, style  # noqa: E402
from eval.metrics import normalize_answer  # noqa: E402
from eval.stats import bootstrap_ci, mcnemar  # noqa: E402

CHERCHENT = ["A1", "A2", "A3", "A4", "A5"]
HARNAIS, RL = ["A1", "A2", "A3"], ["A4", "A5"]
NOMS = {"A0": "A0 direct", "A1": "A1 RAG", "A2": "A2 ReAct", "A3": "A3 IRCoT", "A4": "A4 PPO", "A5": "A5 GRPO"}
# Couleurs : palette de référence. Familles comme dans les autres figures ; catégories d'erreurs sur les
# emplacements 3 à 6 (aqua, jaune, magenta, vert), validés par paires adjacentes dans cet ordre.
FAM = {"A0": "#8a8984", "harnais": "#2a78d6", "rl": "#eb6834"}
CAT = ["#1baf7a", "#eda100", "#e87ba4", "#008300"]
ABST = ("not provided", "not mentioned", "not available", "not given", "cannot", "unknown", "no information")


def famille(s):
    return "A0" if s == "A0" else ("rl" if s in RL else "harnais")


def passages(l):
    """Liste des passages lus, recherche par recherche (liste de listes de chaînes)."""
    return [[p if isinstance(p, str) else str(p) for p in r] for r in (l.get("docs") or [])]


def trouvee(l, k=None):
    """La réponse attendue figure-t-elle dans les passages des k premières recherches (toutes si None) ?"""
    golds = [normalize_answer(g) for g in l["golds"] if normalize_answer(g)]
    rech = passages(l)[: k if k else None]
    texte = " " + " ".join(normalize_answer(p) for r in rech for p in r) + " "
    return any(f" {g} " in texte for g in golds)


def jaccard(a, b):
    a, b = set(normalize_answer(a).split()), set(normalize_answer(b).split())
    return len(a & b) / len(a | b) if a | b else 0.0


def categorie(l):
    """Typologie automatique d'une réponse fausse (volet C en miniature)."""
    p = l.get("prediction")
    if l["status"] != "answer" or not p or any(a in p.lower() for a in ABST):
        return 0  # sans réponse ou abstention
    if l["f1"] >= 0.5:
        return 1  # réponse proche, formulation différente
    if l["system"] != "A0" and trouvee(l):
        return 2  # la bonne réponse était dans les passages lus
    return 3  # réponse absente des passages (ou pas de passages) et fausse


def analyser(D, ids):
    R = {"trans": {}, "decomp": {}, "req": {}, "err": {}, "jetons": {}}
    toutes = [(j, i) for j in JEUX for i in ids[j]]
    for s in SYSTEMES:
        L = [D[(s, j)][i] for j, i in toutes]
        R["jetons"][s] = np.mean([l.get("gen_tokens", 0) for l in L])
        fausses = [l for l in L if l["em"] == 0]
        c = np.bincount([categorie(l) for l in fausses], minlength=4)
        R["err"][s] = c
        if s == "A0":
            continue
        a0 = [D[("A0", j)][i]["em"] for j, i in toutes]
        em = [l["em"] for l in L]
        R["trans"][s] = {
            "gagnees": sum(1 for x, y in zip(a0, em) if not x and y),
            "perdues": sum(1 for x, y in zip(a0, em) if x and not y),
            "gardees": sum(1 for x, y in zip(a0, em) if x and y),
        }
        t = np.array([trouvee(l) for l in L])
        t1 = np.array([trouvee(l, 1) for l in L])
        e = np.array(em)
        R["decomp"][s] = {
            "trouvee": t.mean(), "trouvee1": t1.mean(), "n_t": int(t.sum()), "n_nt": int((~t).sum()),
            "em_t": e[t].mean() if t.any() else 0.0, "em_nt": e[~t].mean() if (~t).any() else 0.0,
            "apres_reform": (t & ~t1).mean(),
            "ic_em_t": bootstrap_ci(e[t])[1:] if t.sum() > 1 else (0, 0),
            "ic_em_nt": bootstrap_ci(e[~t])[1:] if (~t).sum() > 1 else (0, 0),
        }
        q = [(l["question"], l["queries"]) for l in L]
        toutes_req = [(qq, r) for qq, rs in q for r in rs]
        R["req"][s] = {
            "mots": np.mean([len(r.split()) for _, r in toutes_req]) if toutes_req else 0,
            "jacc": np.mean([jaccard(qq, r) for qq, r in toutes_req]) if toutes_req else 0,
            "copie1": np.mean([bool(rs) and normalize_answer(rs[0]) == normalize_answer(qq) for qq, rs in q]),
            "repetees": np.mean([len(rs) != len({normalize_answer(r) for r in rs}) for _, rs in q]),
            "distinctes": np.mean([len({normalize_answer(r) for r in rs}) for _, rs in q]),
        }
    # Comparaisons appariées sur les 200 questions (15 paires), sans correction : description seulement.
    R["matrice"] = {}
    for a, b in combinations(SYSTEMES, 2):
        ea = [D[(a, j)][i]["em"] for j, i in toutes]
        eb = [D[(b, j)][i]["em"] for j, i in toutes]
        _, _, p = mcnemar(ea, eb)
        R["matrice"][(a, b)] = (np.mean(ea) - np.mean(eb), p)
    # Agrégats par famille pour l'arbre de décision.
    for nom, membres in (("harnais", HARNAIS), ("rl", RL)):
        Lf = [D[(s, j)][i] for s in membres for j, i in toutes]
        t = np.array([trouvee(l) for l in Lf]); e = np.array([l["em"] for l in Lf])
        R["decomp"][nom] = {"trouvee": t.mean(), "em_t": e[t].mean(), "em_nt": e[~t].mean(),
                            "n": len(Lf), "n_t": int(t.sum()), "n_nt": int((~t).sum())}
    R["oaxaca"] = decomposer(D, toutes)
    return R


def _parts(rows_h, rows_r):
    """Écart d'EM RL - harnais coupé en trois parts qui s'additionnent exactement."""
    th, eh = rows_h[:, 0].astype(bool), rows_h[:, 1]
    tr, er = rows_r[:, 0].astype(bool), rows_r[:, 1]
    p_h, p_r = th.mean(), tr.mean()
    e_th, e_nth = eh[th].mean(), eh[~th].mean()
    e_tr, e_ntr = er[tr].mean(), er[~tr].mean()
    trouver = (p_r - p_h) * (e_th - e_nth)
    lire = p_r * (e_tr - e_th)
    sans = (1 - p_r) * (e_ntr - e_nth)
    return np.array([trouver, lire, sans, er.mean() - eh.mean()])


def decomposer(D, toutes, n_boot=1000, seed=0):
    """Décomposition contrefactuelle, avec intervalles bootstrap (rééchantillonnage des questions)."""
    cache = {s: np.array([[trouvee(D[(s, j)][i]), D[(s, j)][i]["em"]] for j, i in toutes], dtype=float)
             for s in HARNAIS + RL}
    pile = lambda membres, idx: np.vstack([cache[s][idx] for s in membres])  # noqa: E731
    base = _parts(pile(HARNAIS, np.arange(len(toutes))), pile(RL, np.arange(len(toutes))))
    rng = np.random.default_rng(seed)
    boots = np.array([_parts(pile(HARNAIS, b), pile(RL, b))
                      for b in (rng.integers(0, len(toutes), len(toutes)) for _ in range(n_boot))])
    lo, hi = np.quantile(boots, [0.025, 0.975], axis=0)
    return {"parts": base, "lo": lo, "hi": hi}


def ecrire(R, out):
    pc = lambda x: fr(100 * x, 0)  # noqa: E731
    v = ["% Généré par analysis/analyse_approfondie.py : ne pas modifier à la main."]
    m = lambda k, x: v.append(f"\\expandafter\\def\\csname valAp-{k}\\endcsname{{{x}}}")  # noqa: E731
    for s, d in R["decomp"].items():
        for k in ("trouvee", "em_t", "em_nt"):
            m(f"{s}-{k}", pc(d[k]))
        for k in ("n_t", "n_nt"):
            m(f"{s}-{k}", d[k])
        if "trouvee1" in d:
            m(f"{s}-trouvee1", pc(d["trouvee1"])); m(f"{s}-reform", pc(d["apres_reform"]))
    for s, d in R["trans"].items():
        for k, x in d.items():
            m(f"{s}-{k}", x)
    for s, d in R["req"].items():
        m(f"{s}-copie1", pc(d["copie1"])); m(f"{s}-jacc", fr(d["jacc"])); m(f"{s}-repetees", pc(d["repetees"]))
    for s, c in R["err"].items():
        for k, x in enumerate(c):
            m(f"{s}-err{k}", int(x))
    for s, x in R["jetons"].items():
        m(f"{s}-jetons", fr(x, 0))
    for k, nom in enumerate(["trouver", "lire", "sans", "total"]):
        o = R["oaxaca"]
        m(f"ox-{nom}", fr(100 * o["parts"][k], 1)); m(f"ox-{nom}-lo", fr(100 * o["lo"][k], 1))
        m(f"ox-{nom}-hi", fr(100 * o["hi"][k], 1))
        if nom != "total":
            m(f"ox-{nom}-part", fr(100 * o["parts"][k] / o["parts"][3], 0))
    (out / "approf_valeurs.tex").write_text("\n".join(v) + "\n")

    t = ["\\begin{tabular}{@{}lcccccc@{}}", "\\toprule",
         "Système & Mots par & Recouvrement & 1\\textsuperscript{re} requête & Requêtes & Requête & Jetons\\\\",
         " & requête & avec la question & = question & distinctes & répétée & générés\\\\", "\\midrule"]
    for s in CHERCHENT:
        d = R["req"][s]
        t.append(f"{NOMS[s]} & {fr(d['mots'], 1)} & {fr(d['jacc'])} & {pc(d['copie1'])}\\,\\% & {fr(d['distinctes'], 1)} "
                 f"& {pc(d['repetees'])}\\,\\% & {fr(R['jetons'][s], 0)}\\\\")
        if s == "A3":
            t.append("\\midrule")
    t += ["\\bottomrule", "\\end{tabular}"]
    (out / "approf_requetes.tex").write_text("\n".join(t) + "\n")


def figures(R, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "pdf.fonttype": 42})

    # 1. Gains et pertes par rapport au modèle seul (barres divergentes, en nombre de questions sur 200).
    fig, ax = plt.subplots(figsize=(6.3, 2.5))
    y = np.arange(len(CHERCHENT))[::-1]
    for yy, s in zip(y, CHERCHENT):
        g, p = R["trans"][s]["gagnees"], R["trans"][s]["perdues"]
        c = FAM[famille(s)]
        ax.barh(yy, g, color=c, height=0.55)
        ax.barh(yy, -p, color=c, alpha=0.45, height=0.55)
        ax.text(g + 1, yy, f"+{g}", va="center", fontsize=8, color=ENCRE)
        ax.text(-p - 1, yy, f"−{p}", va="center", ha="right", fontsize=8, color=ENCRE)
    ax.axvline(0, color=ENCRE2, lw=0.8)
    ax.set_yticks(y, [NOMS[s] for s in CHERCHENT])
    ax.set_xlim(-45, 70)
    ax.set_xlabel("Questions perdues (à gauche) et gagnées (à droite) par rapport au modèle seul (A0), sur 200",
                  color=ENCRE2, fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(out / "fig_transitions.pdf"); fig.savefig(out / "fig_transitions.png", dpi=200); plt.close(fig)

    # 2. Trouver ou lire : part des questions où la réponse est dans les passages, EM selon ce cas.
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.5), sharey=True)
    titres = ["Réponse dans les passages lus", "EM si la réponse y est", "EM si elle n'y est pas"]
    for ax, k, ti in zip(axes, ["trouvee", "em_t", "em_nt"], titres):
        for yy, s in zip(y, CHERCHENT):
            d = R["decomp"][s]
            ax.barh(yy, d[k], color=FAM[famille(s)], height=0.55)
            fin = d[k]
            if k in ("em_t", "em_nt"):
                lo, hi = d["ic_" + k]
                ax.plot([lo, hi], [yy, yy], color=ENCRE2, lw=0.9)
                fin = max(fin, hi)
            ax.text(fin + 0.03, yy, f"{100 * d[k]:.0f}", va="center", fontsize=7.5, color=ENCRE)
        ax.set_xlim(0, 1); ax.set_title(ti, fontsize=8, color=ENCRE); style(ax)
        ax.set_xticks([0, 0.5, 1], ["0", "50", "100 %"])
    axes[0].set_yticks(y, [NOMS[s] for s in CHERCHENT])
    fig.tight_layout()
    fig.savefig(out / "fig_decomposition.pdf"); fig.savefig(out / "fig_decomposition.png", dpi=200); plt.close(fig)

    # 3. Typologie des erreurs (barres empilées, nombre de réponses fausses sur 200).
    noms_cat = ["Sans réponse ou abstention", "Réponse proche (F1 ≥ 0,5)",
                "Bonne réponse dans les passages, non donnée", "Réponse absente des passages"]
    fig, ax = plt.subplots(figsize=(6.3, 2.7))
    yy_all = np.arange(len(SYSTEMES))[::-1]
    for yy, s in zip(yy_all, SYSTEMES):
        left = 0
        for k, n in enumerate(R["err"][s]):
            if n:
                ax.barh(yy, n, left=left, color=CAT[k], height=0.55, edgecolor="white", linewidth=1)
                if n >= 8:
                    ax.text(left + n / 2, yy, str(n), ha="center", va="center", fontsize=7,
                            color="white" if k == 3 else ENCRE)
            left += n
        ax.text(left + 2, yy, f"{left}", va="center", fontsize=7.5, color=ENCRE2)
    ax.set_yticks(yy_all, [NOMS[s] for s in SYSTEMES]); ax.set_xlim(0, 200)
    ax.set_xlabel("Réponses fausses sur 200 questions, par type", color=ENCRE2, fontsize=8)
    style(ax)
    ax.legend(handles=[Patch(color=c, label=n) for c, n in zip(CAT, noms_cat)], loc="upper center",
              bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / "fig_erreurs.pdf"); fig.savefig(out / "fig_erreurs.png", dpi=200); plt.close(fig)

    # 4. Matrice des écarts d'EM appariés (ligne moins colonne, en points), divergente bleu / orange.
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#eb6834"])
    n = len(SYSTEMES)
    M = np.full((n, n), np.nan)
    for (a, b), (dlt, p) in R["matrice"].items():
        i, j = SYSTEMES.index(a), SYSTEMES.index(b)
        M[i, j], M[j, i] = 100 * dlt, -100 * dlt
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    im = ax.imshow(M, cmap=cmap, vmin=-25, vmax=25)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a, b = (SYSTEMES[i], SYSTEMES[j]) if (SYSTEMES[i], SYSTEMES[j]) in R["matrice"] else (SYSTEMES[j], SYSTEMES[i])
            p = R["matrice"][(a, b)][1]
            etoile = "*" if p < 0.05 / 15 else ""
            ax.text(j, i, f"{M[i, j]:+.0f}{etoile}", ha="center", va="center", fontsize=7.5, color=ENCRE)
    ax.set_xticks(range(n), SYSTEMES); ax.set_yticks(range(n), SYSTEMES)
    ax.tick_params(length=0, labelsize=8, colors=ENCRE2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("EM ligne − EM colonne (points)", fontsize=7.5, color=ENCRE2); cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_matrice.pdf"); fig.savefig(out / "fig_matrice.png", dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/eval_v4")
    ap.add_argument("--outdir", default="docs/article/genere")
    args = ap.parse_args()
    D, fictif = charger(args.indir)
    ids = {j: sorted(D[("A0", j)]) for j in JEUX}
    R = analyser(D, ids)
    out = Path(args.outdir)
    ecrire(R, out)
    figures(R, out)
    print(("DONNÉES FICTIVES — " if fictif else "") + "Trouver ou lire (sur 200 questions) :")
    for s in CHERCHENT + ["harnais", "rl"]:
        d = R["decomp"][s]
        extra = f" | trouvée à la 1re recherche {d['trouvee1']:.0%}" if "trouvee1" in d else ""
        print(f"  {s:8} trouvée {d['trouvee']:.0%} (n={d['n_t']}) | EM si trouvée {d['em_t']:.0%} | EM sinon {d['em_nt']:.0%}{extra}")
    o = R["oaxaca"]
    print("Décomposition RL - harnais (points d'EM) :", {n: f"{100*o['parts'][k]:.1f} [{100*o['lo'][k]:.1f}; {100*o['hi'][k]:.1f}]"
          for k, n in enumerate(["trouver", "lire", "sans passage", "total"])})
    print("Gains / pertes vs A0 :", {s: (d["gagnees"], d["perdues"]) for s, d in R["trans"].items()})
    print("Erreurs [sans rép., proche, lue non donnée, absente] :", {s: list(map(int, c)) for s, c in R["err"].items()})
    print("Requêtes :", {s: {k: round(float(x), 2) for k, x in d.items()} for s, d in R["req"].items()})
    print("Jetons :", {s: round(float(x)) for s, x in R["jetons"].items()})


if __name__ == "__main__":
    main()
