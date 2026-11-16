"""Partie 2 : réplication sur le vivier (EXPLORATOIRE, 16 novembre 2026, sans GPU).

Après la session 8, les trois modèles ont 16 tirages (graine de génération 1) sur les 1 000 questions de pont du
vivier (graine 43), disjointes de l'échantillon principal. C'est un second échantillon indépendant, plus grand mais à
16 tirages : on y applique les règles de H2 à k <= 16, puis on réunit les deux échantillons (contributions d_i(k) pour
k <= 16, l'estimateur sans biais utilisant les 32 tirages de l'échantillon principal) avec le même bootstrap.
Le vivier n'a pas été conçu comme un test : cette analyse est exploratoire.

    python analysis/replication_partie2.py --indir results/partie2
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.confirmatoire_partie2 import DEPART, charger, court, reussites  # noqa: E402
from analysis.passk import bootstrap_multiplicateur, difference_appariee, frontiere  # noqa: E402

KS = [1, 2, 4, 8, 16]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/partie2")
    ap.add_argument("--out", default="docs/partie2/genere/replication.json")
    args = ap.parse_args(argv)
    runs = charger(args.indir)
    dep = court(DEPART["it"])

    def c(m, sq, n):
        L = [L for (mm, cc), L in runs.items()
             if court(mm) == m and cc[0] == "bridge" and cc[3] == 4 and cc[6] == sq and cc[2] == n]
        return reussites(sum(L, []), n)

    res = {}
    for ag in ("it-em-ppo", "it-em-grpo"):
        parts, r_ag = [], {}
        for nom, sq, n in (("principal", "42", 32), ("vivier", "43", 16)):
            a, b = c(ag, sq, n), c(dep, sq, n)
            ids = sorted(set(a) & set(b))
            if not ids:
                continue
            ca, cb = np.array([a[i] for i in ids]), np.array([b[i] for i in ids])
            d, _ = difference_appariee(ca, cb, n, KS)
            parts.append(d)
            r = bootstrap_multiplicateur(d, n_boot=4000)
            r_ag[nom] = {"questions": len(ids), "D": r["D"].tolist(), "simul_bas": r["bande_simul"][0].tolist(),
                         "simul_haut": r["bande_simul"][1].tolist(), "dominance": r["dominance"],
                         "p_dominance": r["p_dominance"]}
            if n == 16:
                r_ag[nom]["frontiere"] = frontiere(ca, cb)
        if len(parts) == 2:
            r = bootstrap_multiplicateur(np.vstack(parts), n_boot=4000)
            d1, d2 = parts[0][:, -1], parts[1][:, -1]
            se = np.sqrt(d1.var(ddof=1) / len(d1) + d2.var(ddof=1) / len(d2))
            r_ag["reunis"] = {"questions": int(sum(len(p) for p in parts)), "D": r["D"].tolist(),
                              "simul_bas": r["bande_simul"][0].tolist(), "simul_haut": r["bande_simul"][1].tolist(),
                              "dominance": r["dominance"], "p_dominance": r["p_dominance"],
                              "z_heterogeneite_k16": float((d2.mean() - d1.mean()) / se)}
        res[ag] = r_ag
        for nom, x in r_ag.items():
            print(f"  {ag:11} {nom:9} ({x['questions']} q.) D(k) " + " ".join(f"{v:+.3f}" for v in x["D"])
                  + f" | bande à k=16 [{x['simul_bas'][-1]:+.3f}, {x['simul_haut'][-1]:+.3f}] | dominance {x['dominance']}"
                  + (f" | frontière {x['frontiere']['rl']} contre {x['frontiere']['base']}" if "frontiere" in x else "")
                  + (f" | z hétérogénéité {x['z_heterogeneite_k16']:.2f}" if "z_heterogeneite_k16" in x else ""))
    Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return res


if __name__ == "__main__":
    main()
