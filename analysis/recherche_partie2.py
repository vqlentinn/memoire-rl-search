"""Partie 2 : qualité de la recherche, mesurée avec les paragraphes utiles de HotpotQA (EXPLORATOIRE, 13 nov. 2026).

Chaque question HotpotQA désigne ses deux paragraphes utiles (« supporting facts »). Pour chaque trajectoire, on
recalcule les passages ramenés par ses requêtes (BM25 fermé, déterministe : identiques à ceux de la trajectoire dans
99,4 % des cas, cf. journal du 13 nov.) et l'on note :
- trouve_2 : les deux paragraphes utiles ont été ramenés au moins une fois au cours de la trajectoire ;
- trouve_1 : au moins un des deux ;
- requetes_repetees : part des requêtes identiques (après normalisation) à une requête précédente.
Puis, par modèle : P(trouve_2), EM | trouve_2, EM | non trouve_2. Décomposition « trouver / lire » de l'écart d'EM entre
agent et départ (contrefactuelle, à la Oaxaca, comme la partie 1, mais avec les vrais paragraphes utiles) :
    EM = P(t) · EM|t + (1 − P(t)) · EM|¬t
    écart = [P_a(t) − P_d(t)] · (EM_d|t − EM_d|¬t)      « trouver »  (l'agent trouve plus souvent)
          + P_a(t) · (EM_a|t − EM_d|t) + (1 − P_a(t)) · (EM_a|¬t − EM_d|¬t)   « lire »  (mieux à trouvaille égale)
La décomposition est exacte (les deux termes somment à l'écart). « Lire » inclut ici le fait de répondre : un agent qui
trouve tout mais ne répond pas perd dans « lire ».

    python analysis/recherche_partie2.py --indir results/partie2 --data data/raw/hotpot_distractor/validation.parquet
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


def norm(q):
    return re.sub(r"\W+", " ", q.lower()).strip()


def titres(passages):
    return {p.split("\n", 1)[0] for p in passages}


def mesurer(lignes, questions, retriever, topk=3):
    out = []
    for l in lignes:
        q = questions[l["id"]]
        vus, deja, rep = set(), set(), 0
        for r in l["queries"]:
            vus |= titres(retriever.search_in(l["id"], r, k=topk))
            rep += norm(r) in deja
            deja.add(norm(r))
        gold = set(q["gold_titles"])
        out.append({"id": l["id"], "sample": l["sample"], "em": l["em"], "status": l["status"],
                    "n_search": l["n_search"], "trouve_2": gold <= vus, "trouve_1": bool(gold & vus),
                    "repetees": rep / len(l["queries"]) if l["queries"] else 0.0,
                    "q1_est_question": bool(l["queries"]) and norm(l["queries"][0]) == norm(q["question"])})
    return out


def resumer(m):
    em = np.array([x["em"] for x in m])
    t2 = np.array([x["trouve_2"] for x in m])
    a_cherche = np.array([x["n_search"] > 0 for x in m])
    rep = np.array([x["status"] == "answer" for x in m])
    return {
        "trajectoires": len(m), "em": float(em.mean()), "a_cherche": float(a_cherche.mean()),
        "trouve_2": float(t2.mean()), "trouve_1": float(np.mean([x["trouve_1"] for x in m])),
        "trouve_2_si_cherche": float(t2[a_cherche].mean()) if a_cherche.any() else float("nan"),
        "em_si_trouve_2": float(em[t2].mean()) if t2.any() else float("nan"),
        "em_si_non_trouve_2": float(em[~t2].mean()) if (~t2).any() else float("nan"),
        "repond_si_trouve_2": float(rep[t2].mean()) if t2.any() else float("nan"),
        "em_si_trouve_2_et_repond": float(em[t2 & rep].mean()) if (t2 & rep).any() else float("nan"),
        "requetes_repetees": float(np.mean([x["repetees"] for x in m if x["n_search"] > 0])),
        "q1_est_question": float(np.mean([x["q1_est_question"] for x in m if x["n_search"] > 0])),
        "recherches": float(np.mean([x["n_search"] for x in m])),
    }


def decomposer(a, d):
    """Écart d'EM agent − départ = trouver + lire (exact)."""
    trouver = (a["trouve_2"] - d["trouve_2"]) * (d["em_si_trouve_2"] - d["em_si_non_trouve_2"])
    lire = (a["trouve_2"] * (a["em_si_trouve_2"] - d["em_si_trouve_2"])
            + (1 - a["trouve_2"]) * (a["em_si_non_trouve_2"] - d["em_si_non_trouve_2"]))
    return {"ecart": a["em"] - d["em"], "trouver": trouver, "lire": lire}


def bootstrap_decomposition(ma, md, n_boot=500, seed=0):
    """IC à 95 % par rééchantillonnage des questions (toutes leurs trajectoires ensemble)."""
    par_q_a, par_q_d = defaultdict(list), defaultdict(list)
    for x in ma:
        par_q_a[x["id"]].append(x)
    for x in md:
        par_q_d[x["id"]].append(x)
    ids = sorted(set(par_q_a) & set(par_q_d))
    rng = np.random.default_rng(seed)
    tirages = []
    for _ in range(n_boot):
        tir = rng.choice(len(ids), len(ids), replace=True)
        sa = [x for i in tir for x in par_q_a[ids[i]]]
        sd = [x for i in tir for x in par_q_d[ids[i]]]
        tirages.append(decomposer(resumer(sa), resumer(sd)))
    return {k: (float(np.percentile([t[k] for t in tirages], 2.5)), float(np.percentile([t[k] for t in tirages], 97.5)))
            for k in ("ecart", "trouver", "lire")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/partie2")
    ap.add_argument("--data", default="data/raw/hotpot_distractor/validation.parquet")
    ap.add_argument("--outdir", default="docs/partie2/genere")
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args(argv)
    from data.hotpot_distractor import charger as charger_q
    from retrieval.closed_bm25 import ClosedBM25Retriever

    runs = charger(args.indir)
    res = {}
    for t in ("bridge", "comparison"):
        qs = {q["id"]: q for q in charger_q(args.data, t, n=500, seed=42)}
        retr = ClosedBM25Retriever(list(qs.values()))
        mes = {}
        for (modele, cond), lignes in runs.items():
            if cond[0] != t or cond[6] != "42" or cond[3] != 4:
                continue
            mes[modele] = mesurer(lignes, qs, retr)
            res[f"{court(modele)}_{t}"] = resumer(mes[modele])
        dep = DEPART["it"]
        print(f"== {t}")
        for modele, m in sorted(mes.items()):
            r = res[f"{court(modele)}_{t}"]
            print(f"  {court(modele):22} EM {r['em']:.3f} | cherche {r['a_cherche']:.0%} | 2 utiles trouvés {r['trouve_2']:.1%}"
                  f" (si cherche {r['trouve_2_si_cherche']:.1%}) | EM si trouvés {r['em_si_trouve_2']:.3f}, sinon "
                  f"{r['em_si_non_trouve_2']:.3f} | répond si trouvés {r['repond_si_trouve_2']:.0%}, EM si trouvés et "
                  f"répond {r['em_si_trouve_2_et_repond']:.3f} | requêtes répétées {r['requetes_repetees']:.1%} | "
                  f"1re requête = question {r['q1_est_question']:.0%}")
        if dep in mes:
            for modele in sorted(mes):
                if modele == dep:
                    continue
                d = decomposer(res[f"{court(modele)}_{t}"], res[f"{court(dep)}_{t}"])
                ic = bootstrap_decomposition(mes[modele], mes[dep], args.n_boot)
                res[f"decomposition_{court(modele)}_{t}"] = {"point": d, "ic95": ic}
                print(f"  décomposition {court(modele)} − départ : écart {d['ecart']:+.3f} {ic['ecart']} = trouver "
                      f"{d['trouver']:+.3f} {tuple(round(x, 3) for x in ic['trouver'])} + lire {d['lire']:+.3f} "
                      f"{tuple(round(x, 3) for x in ic['lire'])}")
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    (Path(args.outdir) / "recherche.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return res


if __name__ == "__main__":
    main()
