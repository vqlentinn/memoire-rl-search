"""Partie 2 : quelles questions l'agent réussit-il mieux que son modèle de départ ? (EXPLORATOIRE, 17 novembre 2026)

Caractéristiques de chaque question de pont, mesurables sans modèle de langage :
- rep_mots : nombre de mots de la (première) réponse de référence ; rep_nombre : elle contient un chiffre ;
- q_mots : nombre de mots de la question ;
- q_suffit : la question, utilisée comme requête, ramène les deux paragraphes utiles dans le top 3 (BM25 fermé) ;
- rang_pire : rang (1 à 10) du paragraphe utile le moins bien classé avec la question comme requête ;
- titres_nommes : nombre de titres des paragraphes utiles (0 à 2) qui apparaissent dans la question.
Deux modèles, échantillon principal (ponts, T = 4, 500 questions), pour chaque agent :
1. régression linéaire de d_i = réussite de l'agent − réussite du départ (sur 32 tirages chacun) sur les
   caractéristiques standardisées, erreurs-types robustes HC3 : effet en points de réussite par écart-type ;
2. modèle logistique au niveau des trajectoires, em ~ agent × caractéristiques, erreurs-types regroupées par question
   (GEE, corrélation échangeable) : l'interaction agent × caractéristique dit comment l'avantage varie, en rapport de
   cotes. Les deux doivent concorder ; aucune correction pour comparaisons multiples (6 caractéristiques × 2 agents) :
   description, pas test.

    python analysis/predicteurs_partie2.py --indir results/partie2
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.confirmatoire_partie2 import DEPART, charger, court  # noqa: E402
from analysis.recherche_partie2 import titres  # noqa: E402

VARS = ["rep_mots", "rep_nombre", "q_mots", "q_suffit", "rang_pire", "titres_nommes"]
NOMS = {"rep_mots": "mots de la réponse", "rep_nombre": "réponse numérique", "q_mots": "mots de la question",
        "q_suffit": "la question suffit comme requête", "rang_pire": "rang du paragraphe utile le plus loin",
        "titres_nommes": "titres utiles nommés dans la question"}


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"\(.*?\)", "", t.lower())).strip()


def caracteristiques(q, retr):
    golds = q["golds"] if isinstance(q["golds"], list) else [q["golds"]]
    g = str(golds[0])
    classement = [p.split("\n", 1)[0] for p in retr.search_in(q["id"], q["question"], k=10)]
    gold = list(q["gold_titles"])
    rangs = [classement.index(t) + 1 if t in classement else 11 for t in gold]
    return {"rep_mots": len(g.split()), "rep_nombre": int(bool(re.search(r"\d", g))),
            "q_mots": len(q["question"].split()), "q_suffit": int(set(gold) <= set(classement[:3])),
            "rang_pire": max(rangs), "titres_nommes": sum(norm(t) in q["question"].lower() for t in gold)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/partie2")
    ap.add_argument("--data", default="data/raw/hotpot_distractor/validation.parquet")
    ap.add_argument("--out", default="docs/partie2/genere/predicteurs.json")
    args = ap.parse_args(argv)
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from data.hotpot_distractor import charger as charger_q
    from retrieval.closed_bm25 import ClosedBM25Retriever
    qs = {q["id"]: q for q in charger_q(args.data, "bridge", n=500, seed=42)}
    retr = ClosedBM25Retriever(list(qs.values()))
    X = pd.DataFrame({i: caracteristiques(q, retr) for i, q in qs.items()}).T.astype(float)
    runs = {court(m): L for (m, c), L in charger(args.indir).items() if c[0] == "bridge" and c[3] == 4 and c[6] == "42"}
    succes = {}
    for m, L in runs.items():
        s = defaultdict(list)
        for t in L:
            s[t["id"]].append(int(t["em"]))
        succes[m] = {i: np.mean(v) for i, v in s.items()}
    dep = court(DEPART["it"])
    Z = (X - X.mean()) / X.std()
    res = {"description": X.describe().to_dict(), "correlations": X.corr().round(3).to_dict()}
    print("Caractéristiques (moyenne) : " + ", ".join(f"{v} {X[v].mean():.2f}" for v in VARS))
    for ag in ("it-em-ppo", "it-em-grpo"):
        ids = [i for i in X.index if i in succes[ag] and i in succes[dep]]
        d = pd.Series({i: succes[ag][i] - succes[dep][i] for i in ids})
        ols = sm.OLS(d, sm.add_constant(Z.loc[ids, VARS])).fit(cov_type="HC3")
        # Trajectoires : em ~ agent * caractéristiques, GEE groupé par question
        lignes = [{"id": t["id"], "em": int(t["em"]), "agent": int(m == ag)} for m in (ag, dep) for t in runs[m]]
        T = pd.DataFrame(lignes).join(Z[VARS], on="id")
        formule = "em ~ agent * (" + " + ".join(VARS) + ")"
        gee = smf.gee(formule, groups="id", data=T, family=sm.families.Binomial(),
                      cov_struct=sm.cov_struct.Exchangeable()).fit()
        r = {}
        for v in VARS:
            it = f"agent:{v}"
            r[v] = {"ols_points": float(ols.params[v]), "ols_ic": [float(x) for x in ols.conf_int().loc[v]],
                    "ols_p": float(ols.pvalues[v]), "gee_or_interaction": float(np.exp(gee.params[it])),
                    "gee_ic": [float(np.exp(x)) for x in gee.conf_int().loc[it]], "gee_p": float(gee.pvalues[it])}
        res[ag] = {"questions": len(ids), "r2": float(ols.rsquared), "constante": float(ols.params["const"]),
                   "effets": r}
        print(f"== {ag} ({len(ids)} q.) : écart moyen {ols.params['const']:+.3f}, R² {ols.rsquared:.3f}")
        for v in VARS:
            x = r[v]
            print(f"  {NOMS[v]:40} {x['ols_points']:+.3f} [{x['ols_ic'][0]:+.3f}, {x['ols_ic'][1]:+.3f}] p {x['ols_p']:.3f}"
                  f" | GEE RC interaction {x['gee_or_interaction']:.2f} [{x['gee_ic'][0]:.2f}, {x['gee_ic'][1]:.2f}]"
                  f" p {x['gee_p']:.3f}")
    Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
    return res


if __name__ == "__main__":
    main()
