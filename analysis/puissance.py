"""Puissance du test de McNemar apparié, par simulation (analyse de conception, pas un résultat).

Question : avec n questions, quelle chance a-t-on de déclarer significatif un vrai écart d'EM de d points
entre deux systèmes ? Hypothèses (déclarées dans l'article) :
- EM du système de référence : 25 % (ordre de grandeur de la couverture de NQ dans le corpus 1m) ;
- réponses corrélées entre systèmes par une « facilité » commune de la question (même modèle que
  analysis/donnees_fictives.py, bruit propre 0,7) ;
- seuil 5 % (une seule comparaison) et 5 % / 6 (borne prudente de Holm pour les six paires de HA1).

    python analysis/puissance.py            # écrit docs/article/genere/puissance.tex
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.stats import mcnemar  # noqa: E402

BASE, BRUIT, ALPHA, PAIRES = 0.25, 0.7, 0.05, 6
TAILLES = [100, 200, 500, 1000]
ECARTS = [0.05, 0.10, 0.15]


def puissance(n, ecart, reps, rng):
    s = np.sqrt(1 + BRUIT ** 2)
    seuil_a, seuil_b = norm.ppf(BASE) * s, norm.ppf(BASE + ecart) * s
    ok_simple = ok_holm = 0
    for _ in range(reps):
        z = rng.standard_normal(n)
        a = (z + BRUIT * rng.standard_normal(n)) < seuil_a
        b = (z + BRUIT * rng.standard_normal(n)) < seuil_b
        _, _, p = mcnemar(b, a)
        ok_simple += p < ALPHA
        ok_holm += p < ALPHA / PAIRES
    return ok_simple / reps, ok_holm / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--outdir", default="docs/article/genere")
    args = ap.parse_args()
    rng = np.random.default_rng(42)
    res = {(n, d): puissance(n, d, args.reps, rng) for n in TAILLES for d in ECARTS}

    pct = lambda x: f"{100 * x:.0f}\\,\\%"  # noqa: E731
    t = ["\\begin{tabular}{@{}r" + "cc" * len(ECARTS) + "@{}}", "\\toprule",
         "& " + " & ".join(f"\\multicolumn{{2}}{{c}}{{Écart réel {int(100 * d)} points}}" for d in ECARTS) + "\\\\",
         "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(ECARTS))),
         "Questions & " + " & ".join(["5\\,\\%", "Holm"] * len(ECARTS)) + "\\\\", "\\midrule"]
    for n in TAILLES:
        t.append(f"{n} & " + " & ".join(f"{pct(res[(n, d)][0])} & {pct(res[(n, d)][1])}" for d in ECARTS) + "\\\\")
    t += ["\\bottomrule", "\\end{tabular}"]
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "puissance.tex").write_text("\n".join(t) + "\n")
    v = [f"\\expandafter\\def\\csname valPuiss-{n}-{int(100 * d)}-{k}\\endcsname{{{pct(res[(n, d)][i])}}}"
         for n in TAILLES for d in ECARTS for i, k in enumerate(("simple", "holm"))]
    (out / "puissance_valeurs.tex").write_text("% Généré par analysis/puissance.py\n" + "\n".join(v) + "\n")
    print(f"Puissance (EM de référence {BASE:.0%}, {args.reps} simulations)")
    for n in TAILLES:
        print(f"n={n:5}", "  ".join(f"d={int(100 * d):2}: {res[(n, d)][0]:.2f}/{res[(n, d)][1]:.2f}" for d in ECARTS))


if __name__ == "__main__":
    main()
