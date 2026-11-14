"""Partie 2 : typologie des échecs (EXPLORATOIRE, 14 novembre 2026, sans GPU).

Chaque trajectoire fausse (EM = 0) reçoit UNE cause, dans cet ordre de priorité (règles mécaniques, sans juge) :
1. sans_reponse_budget : quatre recherches faites, toujours pas de réponse (statut budget_exhausted) ;
2. sans_reponse_format : ni réponse ni recherche bien balisée (statut format), ou sortie tronquée / contexte dépassé ;
3. sans_recherche : réponse donnée sans aucune recherche ;
4. recherche_ratee : réponse donnée, mais les deux paragraphes utiles n'ont pas tous deux été ramenés ;
5. lecture_ratee : les deux paragraphes utiles ont été ramenés, et la réponse est fausse ;
   on distingue « presque juste » (F1 >= 0,5 : la notation stricte pèse) du reste.
Pour recherche_ratee, on note aussi si la première requête recopie la question, et si une requête est répétée.
Sur les ponts, T = 4, questions graine 42 (étude principale), pour les trois modèles. Des exemples tirés au hasard
(graine 0) sont écrits pour la lecture qualitative.

    python analysis/echecs_partie2.py --indir results/partie2 --data data/raw/hotpot_distractor/validation.parquet
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.confirmatoire_partie2 import charger, court  # noqa: E402
from analysis.recherche_partie2 import mesurer  # noqa: E402

CAUSES = ["sans_reponse_budget", "sans_reponse_format", "sans_recherche", "recherche_ratee",
          "lecture_ratee_presque", "lecture_ratee"]


def cause(t, m):
    if t["status"] == "budget_exhausted":
        return "sans_reponse_budget"
    if t["status"] != "answer":
        return "sans_reponse_format"
    if t["n_search"] == 0:
        return "sans_recherche"
    if not m["trouve_2"]:
        return "recherche_ratee"
    return "lecture_ratee_presque" if float(t["f1"]) >= 0.5 else "lecture_ratee"


def typologie(lignes, mes):
    c, sous = Counter(), Counter()
    exemples = {k: [] for k in CAUSES}
    for t, m in zip(lignes, mes):
        if t["em"]:
            continue
        k = cause(t, m)
        c[k] += 1
        if k == "recherche_ratee":
            sous["q1_est_question"] += m["q1_est_question"]
            sous["requete_repetee"] += m["repetees"] > 0
            sous["un_seul_utile"] += m["trouve_1"]
        exemples[k].append(t)
    n = len(lignes)
    return {"trajectoires": n, "echecs": sum(c.values()),
            "part_des_trajectoires": {k: c[k] / n for k in CAUSES},
            "part_des_echecs": {k: c[k] / max(sum(c.values()), 1) for k in CAUSES},
            "recherche_ratee_detail": {k: v / max(c["recherche_ratee"], 1) for k, v in sous.items()}}, exemples


def extrait(t, longueur=700):
    fin = t["trajectory"][-longueur:]
    return {"id": t["id"], "question": t["question"], "golds": t["golds"], "prediction": t.get("prediction", ""),
            "requetes": t["queries"], "fin_de_trajectoire": fin}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/partie2")
    ap.add_argument("--data", default="data/raw/hotpot_distractor/validation.parquet")
    ap.add_argument("--out", default="docs/partie2/genere/echecs.json")
    ap.add_argument("--exemples", default="results/partie2_echecs_exemples.json")  # non versionné (textes longs)
    args = ap.parse_args(argv)
    from data.hotpot_distractor import charger as charger_q
    from retrieval.closed_bm25 import ClosedBM25Retriever

    qs = {q["id"]: q for q in charger_q(args.data, "bridge", n=500, seed=42)}
    retr = ClosedBM25Retriever(list(qs.values()))
    res, ex = {}, {}
    rng = random.Random(0)
    for (modele, cond), lignes in sorted(charger(args.indir).items()):
        if cond[0] != "bridge" or cond[3] != 4 or cond[6] != "42":
            continue
        r, exemples = typologie(lignes, mesurer(lignes, qs, retr))
        res[court(modele)] = r
        ex[court(modele)] = {k: [extrait(t) for t in rng.sample(v, min(5, len(v)))] for k, v in exemples.items()}
        p = r["part_des_trajectoires"]
        print(f"== {court(modele)} : {r['echecs']} échecs sur {r['trajectoires']} trajectoires")
        for k in CAUSES:
            print(f"  {k:24} {p[k]:6.1%} des trajectoires, {r['part_des_echecs'][k]:6.1%} des échecs")
        d = r["recherche_ratee_detail"]
        print("  recherche ratée : " + ", ".join(f"{k} {v:.0%}" for k, v in d.items()))
    Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    Path(args.exemples).write_text(json.dumps(ex, indent=1, ensure_ascii=False))
    return res


if __name__ == "__main__":
    main()
