"""Partie 2 : quelles questions l'agent gagne-t-il, lesquelles perd-il ? (EXPLORATOIRE, 15 novembre 2026, sans GPU)

Pour chaque question de pont, deux caractéristiques mesurables sans modèle :
- q_suffit : la question elle-même, utilisée comme requête (BM25 fermé, top 3), ramène les deux paragraphes utiles ;
- l'écart de recherche : part des trajectoires de l'agent qui ramènent les deux paragraphes utiles, moins celle du
  modèle de départ (mêmes 32 tirages que la mesure de la frontière).
On compare les questions gagnées (résolues par l'agent au moins une fois en 32 tirages, jamais par le départ), perdues
(l'inverse) et les autres. Échantillon principal (ponts, T = 4, 500 questions) et sous-ensemble difficile (session 4).
Les questions gagnées et perdues sont en partie du bruit d'échantillonnage (voir la redistribution) : description, pas test.

    python analysis/mecanisme_partie2.py --indir results/partie2
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.confirmatoire_partie2 import DEPART, charger, court  # noqa: E402
from analysis.recherche_partie2 import mesurer, resumer, titres  # noqa: E402


def par_question(lignes, mes):
    em, t2 = defaultdict(int), defaultdict(list)
    for t, m in zip(lignes, mes):
        em[t["id"]] += int(t["em"])
        t2[t["id"]].append(m["trouve_2"])
    return em, {i: float(np.mean(v)) for i, v in t2.items()}


def decrire(ids, q_suffit, ecart_t2, em_a, em_d):
    ids = list(ids)
    if not ids:
        return {"questions": 0}
    return {"questions": len(ids), "q_suffit": float(np.mean([q_suffit[i] for i in ids])),
            "ecart_trouve_2": float(np.mean([ecart_t2[i] for i in ids])),
            "em_agent": float(np.mean([em_a[i] / 32 for i in ids])), "em_depart": float(np.mean([em_d[i] / 32 for i in ids]))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/partie2")
    ap.add_argument("--data", default="data/raw/hotpot_distractor/validation.parquet")
    ap.add_argument("--out", default="docs/partie2/genere/mecanisme.json")
    args = ap.parse_args(argv)
    from data.hotpot_distractor import charger as charger_q
    from retrieval.closed_bm25 import ClosedBM25Retriever
    runs = charger(args.indir)
    dep = court(DEPART["it"])
    res = {}
    for nom, filtre in (("principal", "42"), ("difficile", "ids-difficiles-ponts")):
        sel = {court(m): L for (m, c), L in runs.items() if c[0] == "bridge" and c[3] == 4 and c[6] == filtre}
        if dep not in sel:
            continue
        ids_q = sorted({t["id"] for t in sel[dep]})
        qs = {q["id"]: q for q in charger_q(args.data, "bridge", ids=ids_q)}
        retr = ClosedBM25Retriever(list(qs.values()))
        q_suffit = {i: set(qs[i]["gold_titles"]) <= titres(retr.search_in(i, qs[i]["question"], k=3)) for i in ids_q}
        mes_d = mesurer(sel[dep], qs, retr)
        em_d, t2_d = par_question(sel[dep], mes_d)
        res[f"entonnoir_{nom}"] = {dep: resumer(mes_d)}
        refs = [sel[dep][j]["golds"] for j in range(0, len(sel[dep]), 32)]
        res[f"entonnoir_{nom}"]["mots_par_reference"] = float(np.mean(
            [len((g if isinstance(g, str) else g[0]).split()) for g in refs]))
        for ag in ("it-em-ppo", "it-em-grpo"):
            mes_a = mesurer(sel[ag], qs, retr)
            em_a, t2_a = par_question(sel[ag], mes_a)
            res[f"entonnoir_{nom}"][ag] = resumer(mes_a)
            ids = sorted(set(em_a) & set(em_d))
            ecart = {i: t2_a[i] - t2_d[i] for i in ids}
            gagnees = [i for i in ids if em_a[i] > 0 and em_d[i] == 0]
            perdues = [i for i in ids if em_a[i] == 0 and em_d[i] > 0]
            autres = [i for i in ids if i not in set(gagnees) | set(perdues)]
            # Écart d'EM au premier essai selon que la question suffit ou non comme requête
            d1 = {v: float(np.mean([(em_a[i] - em_d[i]) / 32 for i in ids if q_suffit[i] == v])) for v in (True, False)}
            n_qs = {v: sum(q_suffit[i] == v for i in ids) for v in (True, False)}
            r = {"gagnees": decrire(gagnees, q_suffit, ecart, em_a, em_d),
                 "perdues": decrire(perdues, q_suffit, ecart, em_a, em_d),
                 "autres": decrire(autres, q_suffit, ecart, em_a, em_d),
                 "D1_si_question_suffit": d1[True], "D1_si_question_ne_suffit_pas": d1[False],
                 "questions_ou_la_question_suffit": n_qs[True], "questions_ou_elle_ne_suffit_pas": n_qs[False]}
            res[f"{ag}_{nom}"] = r
            print(f"== {ag}, {nom} ({len(ids)} questions ; la question suffit pour {n_qs[True]})")
            for k in ("gagnees", "perdues", "autres"):
                x = r[k]
                if x["questions"]:
                    print(f"  {k:8} {x['questions']:4} q. | question suffit {x['q_suffit']:.0%} | écart trouve_2 "
                          f"{x['ecart_trouve_2']:+.2f} | EM agent {x['em_agent']:.3f}, départ {x['em_depart']:.3f}")
            print(f"  D(1) si la question suffit {d1[True]:+.3f}, sinon {d1[False]:+.3f}")
        for m, e in res[f"entonnoir_{nom}"].items():
            if isinstance(e, dict):
                print(f"  entonnoir {m:20} trouve_2 {e['trouve_2']:.2f}, répond si trouve {e['repond_si_trouve_2']:.2f}, "
                      f"EM si trouve et répond {e['em_si_trouve_2_et_repond']:.3f}, EM {e['em']:.3f}")
    Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return res


if __name__ == "__main__":
    main()
