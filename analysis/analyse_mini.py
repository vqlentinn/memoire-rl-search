"""Analyse de la mini-expérience : tableaux LaTeX, valeurs citées dans le texte et figures de l'article.

Part uniquement des fichiers JSONL (règle du projet). Applique les règles de décision du tableau
« tab:test » de docs/article/main.tex, fixées avant le run complet.

    python analysis/analyse_mini.py --indir results/fictif     # données fictives (test de la chaîne)
    python analysis/analyse_mini.py --indir results            # vrais runs, quand ils existeront

Sorties dans docs/article/genere/ : resultats.tex (tableau), tests.tex (tableau des hypothèses),
valeurs.tex (macros \\val... pour le texte), fig_em.pdf, fig_cout.pdf.
Si une seule ligne porte "fictif": true, tout est marqué DONNÉES FICTIVES (macro \\fictif vraie,
filigrane sur les figures).
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.stats import bootstrap_ci, mcnemar  # noqa: E402

SYSTEMES = ["A0", "A1", "A2", "A3", "A4", "A5"]
HARNAIS = ["A1", "A2", "A3"]
RL = ["A4", "A5"]
JEUX = {"nq": "NQ (vu)", "triviaqa": "TriviaQA (non vu)"}
N_BOOT, ALPHA, GRAINE = 1000, 0.05, 0
MOTIF = re.compile(r"^(A[0-5])_(nq|triviaqa)_e5_1m_n(\d+)\.jsonl$")  # sans suffixe _seed : évaluation seule

# Couleurs : trois premiers emplacements de la palette de référence (validée, CVD), gris pour A0.
COULEUR = {"A0": "#8a8984", "harnais": "#2a78d6", "rl": "#eb6834"}
ENCRE, ENCRE2, GRILLE = "#0b0b0b", "#52514e", "#e4e3df"


def famille(s):
    return "A0" if s == "A0" else ("rl" if s in RL else "harnais")


def charger(indir):
    """{(système, jeu): {id: ligne}} ; le jeu est lu dans le nom du fichier (absent des lignes réelles)."""
    donnees, fictif = {}, False
    for f in sorted(Path(indir).glob("*.jsonl")):
        m = MOTIF.match(f.name)
        if not m:
            continue
        s, jeu = m.group(1), m.group(2)
        lignes = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        fictif |= any(l.get("fictif") for l in lignes)
        donnees[(s, jeu)] = {l["id"]: l for l in lignes}
    manquants = [(s, j) for s in SYSTEMES for j in JEUX if (s, j) not in donnees]
    if manquants:
        sys.exit(f"Fichiers manquants dans {indir} : {manquants}")
    for j in JEUX:  # mêmes questions pour tous les systèmes, sinon les tests appariés n'ont pas de sens
        ids = {s: set(donnees[(s, j)]) for s in SYSTEMES}
        if len({frozenset(v) for v in ids.values()}) != 1:
            sys.exit(f"{j} : les systèmes n'ont pas les mêmes questions")
    return donnees, fictif


def colonne(donnees, s, jeu, champ, ids):
    return np.array([float(donnees[(s, jeu)][i][champ]) for i in ids])


def ic_diff_independants(x, y, n_boot=N_BOOT, seed=GRAINE):
    """IC bootstrap de mean(x) - mean(y), x et y rééchantillonnés séparément (deux jeux distincts)."""
    rng = np.random.default_rng(seed)
    mx = rng.choice(x, size=(n_boot, len(x))).mean(axis=1)
    my = rng.choice(y, size=(n_boot, len(y))).mean(axis=1)
    d = mx - my
    lo, hi = np.quantile(d, [ALPHA / 2, 1 - ALPHA / 2])
    return float(x.mean() - y.mean()), float(lo), float(hi)


def holm(pvals):
    """p-valeurs ajustées par la méthode de Holm (même ordre qu'en entrée)."""
    p = np.asarray(pvals)
    ordre = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    courant = 0.0
    for rang, k in enumerate(ordre):
        courant = max(courant, min(1.0, (m - rang) * p[k]))
        adj[k] = courant
    return adj


def fr(x, d=2):
    """Nombre au format français pour LaTeX (virgule décimale)."""
    return f"{x:.{d}f}".replace(".", "{,}").replace("-", "$-$")


def pval(p):
    return "$<$\\,0{,}001" if p < 0.001 else fr(p, 3)


def analyser(donnees):
    ids = {j: sorted(donnees[("A0", j)]) for j in JEUX}
    r = {"desc": {}, "tests": {}}
    # Description : EM, F1, recherches, part des budgets épuisés.
    for s in SYSTEMES:
        for j in JEUX:
            em = colonne(donnees, s, j, "em", ids[j])
            ns = colonne(donnees, s, j, "n_search", ids[j])
            statut = [donnees[(s, j)][i]["status"] for i in ids[j]]
            r["desc"][(s, j)] = {
                "em": bootstrap_ci(em, N_BOOT, ALPHA, GRAINE),
                "f1": colonne(donnees, s, j, "f1", ids[j]).mean(),
                "rech": bootstrap_ci(ns, N_BOOT, ALPHA, GRAINE),
                "epuise": sum(x == "budget_exhausted" for x in statut),
                "n": len(em),
            }
    # Descriptifs pour la discussion : part des questions sans recherche, pertes sur les questions
    # « connues » (A0 juste), abstentions explicites. Calculés ici pour que le texte ne cite que des sorties.
    connues = [(j, i) for j in JEUX for i in ids[j] if donnees[("A0", j)][i]["em"] == 1]
    abst = re.compile(r"not (provided|mentioned|available|given)|cannot|unknown|no information", re.I)
    r["connues"] = len(connues)
    r["extra"] = {}
    for s in SYSTEMES[1:]:
        lignes = [donnees[(s, j)][i] for j in JEUX for i in ids[j]]
        r["extra"][s] = {
            "sans_rech": sum(l["n_search"] == 0 for l in lignes) / len(lignes),
            "perdues": sum(donnees[(s, j)][i]["em"] == 0 for j, i in connues),
            "abst": sum(bool(l.get("prediction")) and bool(abst.search(l["prediction"] or "")) for l in lignes),
            "sans_rep": sum(l["status"] != "answer" for l in lignes),
        }
    # HA1 : McNemar sur les deux jeux réunis, 6 paires RL x harnais, Holm.
    paires = [(a, b) for a in RL for b in HARNAIS]
    brut = []
    for a, b in paires:
        ca = np.concatenate([colonne(donnees, a, j, "em", ids[j]) for j in JEUX])
        cb = np.concatenate([colonne(donnees, b, j, "em", ids[j]) for j in JEUX])
        bb, cc, p = mcnemar(ca, cb)
        brut.append((a, b, bb, cc, p, ca.mean() - cb.mean()))
    adj = holm([x[4] for x in brut])
    ha1 = [dict(a=a, b=b, b_=bb, c=cc, p=p, p_holm=q, ecart=e) for (a, b, bb, cc, p, e), q in zip(brut, adj)]
    nb_ok = sum(x["ecart"] > 0 and x["p_holm"] < ALPHA for x in ha1)
    r["tests"]["HA1"] = {"paires": ha1, "nb_ok": nb_ok,
                         "verdict": "soutenue" if nb_ok == 6 else ("partiellement soutenue" if nb_ok else "non soutenue")}
    # HA2 : (RL - harnais) sur NQ moins (RL - harnais) sur TriviaQA.
    ecarts = {}
    for j in JEUX:
        rl = np.mean([colonne(donnees, s, j, "em", ids[j]) for s in RL], axis=0)
        ha = np.mean([colonne(donnees, s, j, "em", ids[j]) for s in HARNAIS], axis=0)
        ecarts[j] = rl - ha
    d, lo, hi = ic_diff_independants(ecarts["nq"], ecarts["triviaqa"])
    r["tests"]["HA2"] = {"ecart_nq": ecarts["nq"].mean(), "ecart_tqa": ecarts["triviaqa"].mean(),
                         "d": d, "lo": lo, "hi": hi, "verdict": "soutenue" if lo > 0 else "non soutenue"}
    # HA3a et HA3b : recherches (moyenne A4, A5) - recherches A2, appariées par question.
    diff, diff_connues = [], []
    for j in JEUX:
        rl = np.mean([colonne(donnees, s, j, "n_search", ids[j]) for s in RL], axis=0)
        a2 = colonne(donnees, "A2", j, "n_search", ids[j])
        connue = colonne(donnees, "A0", j, "em", ids[j]) == 1
        diff.append(rl - a2)
        diff_connues.append((rl - a2)[connue])
    diff, diff_connues = np.concatenate(diff), np.concatenate(diff_connues)
    m, lo, hi = bootstrap_ci(diff, N_BOOT, ALPHA, GRAINE)
    r["tests"]["HA3a"] = {"d": m, "lo": lo, "hi": hi, "n": len(diff),
                          "verdict": "soutenue" if lo > 0 else "non soutenue"}
    m, lo, hi = bootstrap_ci(diff_connues, N_BOOT, ALPHA, GRAINE)
    explo = len(diff_connues) < 20
    v = "soutenue" if lo > 0 else "non soutenue"
    r["tests"]["HA3b"] = {"d": m, "lo": lo, "hi": hi, "n": len(diff_connues),
                          "verdict": v + (" (exploratoire)" if explo else "")}
    return r


def ecrire_tex(r, outdir, fictif):
    d = r["desc"]
    marque = "\\fictifmarque" if fictif else ""
    # Tableau descriptif
    t = ["\\begin{tabular}{@{}lcccccc@{}}", "\\toprule",
         "& \\multicolumn{3}{c}{NQ (vu)} & \\multicolumn{3}{c}{TriviaQA (non vu)}\\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
         "Système & EM [IC 95\\,\\%] & F1 & Rech. & EM [IC 95\\,\\%] & F1 & Rech.\\\\", "\\midrule"]
    for s in SYSTEMES:
        if s == "A4":
            t.append("\\midrule")
        cells = []
        for j in JEUX:
            em, lo, hi = d[(s, j)]["em"]
            cells += [f"{fr(em)} [{fr(lo)}\\,; {fr(hi)}]", fr(d[(s, j)]["f1"]), fr(d[(s, j)]["rech"][0])]
        t.append(f"{s} & " + " & ".join(cells) + "\\\\")
    t += ["\\bottomrule", "\\end{tabular}"]
    (outdir / "resultats.tex").write_text(marque + "\n".join(t) + "\n")

    # Tableau des tests
    T = r["tests"]
    t = ["\\begin{tabular}{@{}lp{7.6cm}l@{}}", "\\toprule", "Hyp. & Résultat du test & Verdict\\\\", "\\midrule"]
    paires = "; ".join(f"{x['a']}--{x['b']} {fr(100 * x['ecart'], 1)} pt ($p_{{\\text{{Holm}}}}$ {pval(x['p_holm'])})"
                       for x in T["HA1"]["paires"])
    t.append(f"HA1 & Écarts d'EM RL $-$ harnais, NQ et TriviaQA réunis (200 questions) : {paires}. "
             f"{T['HA1']['nb_ok']} paire(s) sur 6 positives et significatives. & {T['HA1']['verdict']}\\\\")
    h = T["HA2"]
    t.append(f"HA2 & Écart RL $-$ harnais : {fr(100 * h['ecart_nq'], 1)} pt sur NQ, {fr(100 * h['ecart_tqa'], 1)} pt "
             f"sur TriviaQA ; différence {fr(100 * h['d'], 1)} pt [{fr(100 * h['lo'], 1)}\\,; {fr(100 * h['hi'], 1)}]. "
             f"& {h['verdict']}\\\\")
    h = T["HA3a"]
    t.append(f"HA3a & Recherches par question, RL $-$ A2 : {fr(h['d'])} [{fr(h['lo'])}\\,; {fr(h['hi'])}], "
             f"{h['n']} questions. & {h['verdict']}\\\\")
    h = T["HA3b"]
    t.append(f"HA3b & Même écart, questions où A0 répond juste : {fr(h['d'])} [{fr(h['lo'])}\\,; {fr(h['hi'])}], "
             f"{h['n']} questions. & {h['verdict']}\\\\")
    t += ["\\bottomrule", "\\end{tabular}"]
    (outdir / "tests.tex").write_text(marque + "\n".join(t) + "\n")

    # Macros pour le texte : \valEM{A4}{nq}, \valRech{A2}{triviaqa}, \valHA{HA3a}{d}...
    v = ["% Généré par analysis/analyse_mini.py : ne pas modifier à la main.",
         f"\\{'fictiftrue' if fictif else 'fictiffalse'}"]
    for (s, j), x in d.items():
        v.append(f"\\expandafter\\def\\csname valEM-{s}-{j}\\endcsname{{{fr(100 * x['em'][0], 0)}}}")
        v.append(f"\\expandafter\\def\\csname valRech-{s}-{j}\\endcsname{{{fr(x['rech'][0])}}}")
        v.append(f"\\expandafter\\def\\csname valEpuise-{s}-{j}\\endcsname{{{x['epuise']}}}")
    v.append(f"\\def\\valConnues{{{r['connues']}}}")
    for sy, x in r["extra"].items():
        v.append(f"\\expandafter\\def\\csname valSansRech-{sy}\\endcsname{{{fr(100 * x['sans_rech'], 0)}}}")
        v.append(f"\\expandafter\\def\\csname valPerdues-{sy}\\endcsname{{{x['perdues']}}}")
        v.append(f"\\expandafter\\def\\csname valAbst-{sy}\\endcsname{{{x['abst']}}}")
        v.append(f"\\expandafter\\def\\csname valSansRep-{sy}\\endcsname{{{x['sans_rep']}}}")
    for j in JEUX:  # meilleur système de chaque famille, pour le résumé : \valBest{rl}{nq}, \valBestNom{rl}{nq}
        for fam, membres in (("rl", RL), ("harnais", HARNAIS)):
            s = max(membres, key=lambda m: d[(m, j)]["em"][0])
            v.append(f"\\expandafter\\def\\csname valBest-{fam}-{j}\\endcsname{{{fr(100 * d[(s, j)]['em'][0], 0)}}}")
            v.append(f"\\expandafter\\def\\csname valBestNom-{fam}-{j}\\endcsname{{{s}}}")
    for k, h in T.items():
        for c in ("d", "lo", "hi", "n", "nb_ok", "ecart_nq", "ecart_tqa"):
            if c in h:
                x = h[c]
                txt = str(x) if c in ("n", "nb_ok") else (fr(100 * x, 1) if k in ("HA2",) else fr(x))
                v.append(f"\\expandafter\\def\\csname valHA-{k}-{c}\\endcsname{{{txt}}}")
        v.append(f"\\expandafter\\def\\csname valHA-{k}-verdict\\endcsname{{{h['verdict']}}}")
        if "d" in h and k != "HA2":  # version anglaise (point décimal) pour l'abstract
            v.append(f"\\expandafter\\def\\csname valHAen-{k}-d\\endcsname{{{h['d']:.2f}}}")
    (outdir / "valeurs.tex").write_text("\n".join(v) + "\n")


def style(ax):
    for c in ("top", "right"):
        ax.spines[c].set_visible(False)
    for c in ("left", "bottom"):
        ax.spines[c].set_color(GRILLE)
    ax.tick_params(colors=ENCRE2, labelsize=8, length=0)
    ax.grid(axis="x", color=GRILLE, lw=0.6)
    ax.set_axisbelow(True)


def filigrane(fig, fictif):
    if fictif:
        fig.text(0.5, 0.58, "DONNÉES FICTIVES", ha="center", va="center", rotation=15,
                 fontsize=24, color="#e34948", alpha=0.16, weight="bold")


def figures(r, outdir, fictif):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "pdf.fonttype": 42})
    d = r["desc"]
    legende = [Line2D([], [], marker="o", ls="", ms=6, color=COULEUR[k], label=l)
               for k, l in (("A0", "Sans recherche"), ("harnais", "Harnais fixe"), ("rl", "Entraîné par RL"))]

    # Figure 1 : EM et IC bootstrap, un panneau par jeu, même axe x (de 0 à 1).
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.6), sharey=True)
    y = np.arange(len(SYSTEMES))[::-1]
    for ax, j in zip(axes, JEUX):
        for yy, s in zip(y, SYSTEMES):
            em, lo, hi = d[(s, j)]["em"]
            c = COULEUR[famille(s)]
            ax.plot([lo, hi], [yy, yy], color=c, lw=2, solid_capstyle="round")
            ax.plot(em, yy, "o", ms=6, color=c, mec="white", mew=1.2)
        ax.set_xlim(0, 1)
        ax.set_title(JEUX[j], fontsize=9, color=ENCRE)
        ax.set_xlabel("Correspondance exacte (EM)", color=ENCRE2, fontsize=8)
        style(ax)
    axes[0].set_yticks(y, SYSTEMES)
    fig.legend(handles=legende, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    filigrane(fig, fictif)
    fig.savefig(outdir / "fig_em.pdf")
    fig.savefig(outdir / "fig_em.png", dpi=200)
    plt.close(fig)

    # Figure 2 : précision selon le coût (recherches par question), un panneau par jeu.
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.7), sharey=True)
    for ax, j in zip(axes, JEUX):
        for s in SYSTEMES:
            em = d[(s, j)]["em"][0]
            rech = d[(s, j)]["rech"][0]
            ax.plot(rech, em, "o", ms=7, color=COULEUR[famille(s)], mec="white", mew=1.2)
            ax.annotate(s, (rech, em), xytext=(5, 3), textcoords="offset points", fontsize=8, color=ENCRE)
        ax.set_xlim(-0.2, 4.2)
        ax.set_ylim(0, 1)
        ax.set_title(JEUX[j], fontsize=9, color=ENCRE)
        ax.set_xlabel("Recherches par question (moyenne)", color=ENCRE2, fontsize=8)
        style(ax)
        ax.grid(axis="y", color=GRILLE, lw=0.6)
    axes[0].set_ylabel("EM", color=ENCRE2, fontsize=8)
    fig.legend(handles=legende, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    filigrane(fig, fictif)
    fig.savefig(outdir / "fig_cout.pdf")
    fig.savefig(outdir / "fig_cout.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results")
    ap.add_argument("--outdir", default="docs/article/genere")
    args = ap.parse_args()
    donnees, fictif = charger(args.indir)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    r = analyser(donnees)
    ecrire_tex(r, out, fictif)
    figures(r, out, fictif)
    # Tableau de contrôle (étape 9.3 de la trame) : ce qui a servi à dessiner.
    print(("DONNÉES FICTIVES\n" if fictif else "") + "système  jeu        n   EM     IC95            F1    rech  épuisés")
    for (s, j), x in r["desc"].items():
        em, lo, hi = x["em"]
        print(f"{s:7}  {j:9} {x['n']:3}  {em:.3f}  [{lo:.3f}; {hi:.3f}]  {x['f1']:.3f}  {x['rech'][0]:.2f}  {x['epuise']}")
    for k, h in r["tests"].items():
        print(k, {c: (round(v, 4) if isinstance(v, float) else v) for c, v in h.items() if c != "paires"})
    print(f"Sorties : {out}/")


if __name__ == "__main__":
    main()
