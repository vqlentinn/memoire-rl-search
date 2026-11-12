"""Partie 2, H3 : analyse de la lecture croisée (scripts/run_lecture.py).

Pour une paire (départ, agent), 4 cases : lecteur × chercheur. Par question, EM moyenne de chaque case sur ses
lectures, puis trois effets par question, moyennés sur les questions présentes dans les 4 cases :
- avantage de lecture   = moyenne sur les chercheurs de [EM(agent lit X) − EM(départ lit X)] ;
- avantage de recherche = moyenne sur les lecteurs de [EM(L lit l'agent) − EM(L lit le départ)] ;
- interaction            = [EM(a, a) − EM(d, a)] − [EM(a, d) − EM(d, d)] (la lecture de l'agent vaut-elle plus
                           sur ses propres plans ?).
IC à 95 % par bootstrap à multiplicateurs par question (comme analysis/passk.py). H3 (secondaire, exploratoire) est
soutenue si l'IC de l'avantage de lecture est au-dessus de 0 ; on rapporte aussi la part de l'écart total
EM(a, a) − EM(d, d) due à chaque effet.

    python analysis/lecture_croisee.py --indir results/partie2_lecture
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import norm

NOM = re.compile(r"^lecteur-(?P<lecteur>.+?)_chercheur-(?P<chercheur>.+?)_(?P<type>bridge|comparison)_m(?P<m>\d+)$")
DEPART = "Qwen2.5-3B-Instruct"


def charger(indir):
    """{(type, lecteur, chercheur): {id: [em…]}}"""
    cases = {}
    for f in sorted(Path(indir).glob("*.jsonl")):
        m = NOM.match(f.stem)
        if not m:
            continue
        par_q = defaultdict(list)
        for l in f.read_text().split("\n"):
            if l.strip():
                x = json.loads(l)
                par_q[x["id"]].append(x["em"])
        cases[(m["type"], m["lecteur"], m["chercheur"])] = par_q
    return cases


def ic(v, n_boot=2000, seed=0):
    v = np.asarray(v, dtype=float)
    xi = np.random.default_rng(seed).standard_normal((n_boot, len(v)))
    s = (xi @ (v - v.mean()) / len(v)).std(ddof=1)
    z = norm.ppf(0.975)
    return float(v.mean()), float(v.mean() - z * s), float(v.mean() + z * s)


def effets(cases, type_q, agent, depart=DEPART):
    cle = {(lec, che): (type_q, lec, che) for lec in (agent, depart) for che in (agent, depart)}
    if not all(c in cases for c in cle.values()):
        return None
    ids = sorted(set.intersection(*(set(cases[c]) for c in cle.values())))
    em = {k: np.array([np.mean(cases[c][i]) for i in ids]) for k, c in cle.items()}
    a, d = agent, depart
    lecture = ((em[(a, a)] - em[(d, a)]) + (em[(a, d)] - em[(d, d)])) / 2
    recherche = ((em[(a, a)] - em[(a, d)]) + (em[(d, a)] - em[(d, d)])) / 2
    interaction = (em[(a, a)] - em[(d, a)]) - (em[(a, d)] - em[(d, d)])
    total = em[(a, a)] - em[(d, d)]
    out = {"questions": len(ids), "cases": {f"{l} lit {c}": float(v.mean()) for (l, c), v in em.items()},
           "lecture": ic(lecture), "recherche": ic(recherche), "interaction": ic(interaction), "total": ic(total)}
    # total = lecture + recherche exactement (plan 2 × 2 équilibré) : parts de l'écart
    if abs(out["total"][0]) > 1e-9:
        out["part_lecture"] = out["lecture"][0] / out["total"][0]
    out["H3"] = out["lecture"][1] > 0
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/partie2_lecture")
    ap.add_argument("--out", default=None, help="JSON de sortie (option)")
    args = ap.parse_args(argv)
    cases = charger(args.indir)
    agents = sorted({k[1] for k in cases} - {DEPART})
    res = {}
    for t in ("bridge", "comparison"):
        for a in agents:
            r = effets(cases, t, a)
            if r is None:
                continue
            res[f"{a}_{t}"] = r
            print(f"== {a} contre {DEPART}, {t}, {r['questions']} questions")
            for k, v in r["cases"].items():
                print(f"   EM {k:55} {v:.3f}")
            for k in ("total", "lecture", "recherche", "interaction"):
                m, lo, hi = r[k]
                print(f"   {k:12} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
            print(f"   H3 (avantage de lecture > 0) : {'soutenue' if r['H3'] else 'non soutenue'}"
                  + (f" ; part de l'écart due à la lecture {r['part_lecture']:.0%}" if "part_lecture" in r else ""))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return res


if __name__ == "__main__":
    main()
