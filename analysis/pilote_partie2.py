"""Analyse de la session pilote de la partie 2 : débit, format, troncature, courbes indicatives, budget et puissance.

Le pilote (50 questions de mise au point) ne teste aucune hypothèse : il sert à fixer N (questions) et n (tirages)
dans le budget Kaggle, avant de geler le plan d'analyse.

    python analysis/pilote_partie2.py --indir results/partie2_pilote --resume resume_pilote.json
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.passk import bootstrap_multiplicateur, courbes, difference_appariee, frontiere, grille_k  # noqa: E402

DEPART = {"it": "Qwen2.5-3B-Instruct_bridge", "base": "Qwen2.5-3B_bridge"}


def charger(indir):
    out = {}
    for f in sorted(Path(indir).glob("*.jsonl")):
        L = [json.loads(l) for l in f.read_text().split("\n") if l.strip()]
        if L:
            out[f.stem] = L
    return out


def reussites(L, n):
    """c par question (seulement les questions complètes), dans l'ordre des ids."""
    par_q = defaultdict(list)
    for l in L:
        par_q[l["id"]].append(l["em"])
    return {i: sum(v) for i, v in par_q.items() if len(v) >= n}


def decrire(runs, secondes):
    lignes = []
    for nom, L in runs.items():
        st = Counter(l["status"] for l in L)
        lignes.append({
            "run": nom, "trajectoires": len(L),
            "em": np.mean([l["em"] for l in L]),
            "bon_format": 1 - (st["format"] + st["truncated"] + st["context_exceeded"]) / len(L),
            "tronquees": np.mean([l["truncated"] for l in L]),
            "budget_epuise": st["budget_exhausted"] / len(L),
            "recherches": np.mean([l["n_search"] for l in L]),
            "jetons": np.mean([l["gen_tokens"] for l in L]),
            "traj_par_s": len(L) / secondes[nom] if secondes.get(nom) else None,
        })
    return lignes


def paires(runs, n):
    """Associe chaque checkpoint à son modèle de départ (même gabarit de prompt)."""
    out = []
    for nom in runs:
        if not nom.startswith("qwen2.5-3b-"):
            continue
        famille = "it" if "-it-" in nom else "base"
        tag = nom.split("_temp")[1]  # "0.7_chat_seed7" : même température, même gabarit
        depart = [k for k in runs if k.startswith(DEPART[famille] + "_") and k.endswith(tag)]
        if depart:
            out.append((nom, depart[0]))
    return out


def puissance(c_rl, c_base, n, N, n_sim=300, seed=0):
    """Probabilité de conclure à l'« élargissement » (règle du plan) si la population ressemblait au pilote :
    on tire N questions avec remise parmi les paires (c_rl, c_base) observées."""
    rng = np.random.default_rng(seed)
    ks = grille_k(n)
    paires_obs = np.stack([c_rl, c_base], axis=1)
    succes = 0
    for s in range(n_sim):
        idx = rng.integers(0, len(paires_obs), N)
        d, _ = difference_appariee(paires_obs[idx, 0], paires_obs[idx, 1], n, ks)
        r = bootstrap_multiplicateur(d, n_boot=400, seed=s)
        succes += r["dominance"] and r["bande_simul"][0][-1] > 0
    return succes / n_sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/partie2_pilote")
    ap.add_argument("--resume", default=None, help="resume_pilote.json du kernel (durées des runs)")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--quota-heures", type=float, default=30.0)
    args = ap.parse_args()
    runs = charger(args.indir)
    secondes = {}
    if args.resume:
        res = json.loads(Path(args.resume).read_text())["resume"]
        secondes = {k: v.get("secondes") for k, v in res.items()}

    print("== Description par run")
    desc = decrire(runs, secondes)
    for d in desc:
        tps = f"{d['traj_par_s']:.2f} traj/s" if d["traj_par_s"] else "durée inconnue"
        print(f"  {d['run'][:60]:60} n={d['trajectoires']:5} EM {d['em']:.3f} | bon format {d['bon_format']:.1%} | "
              f"tronquées {d['tronquees']:.1%} | budget épuisé {d['budget_epuise']:.1%} | "
              f"rech {d['recherches']:.2f} | jetons {d['jetons']:.0f} | {tps}")

    print("\n== Paires (indicatif : 50 questions de mise au point, aucune conclusion)")
    ks = grille_k(args.n)
    for rl, dep in paires(runs, args.n):
        a, b = reussites(runs[rl], args.n), reussites(runs[dep], args.n)
        ids = sorted(set(a) & set(b))
        if not ids:
            continue
        c_rl, c_b = np.array([a[i] for i in ids]), np.array([b[i] for i in ids])
        d, D = difference_appariee(c_rl, c_b, args.n, ks)
        pk_rl, pk_b = courbes(c_rl, args.n, ks).mean(0), courbes(c_b, args.n, ks).mean(0)
        f = frontiere(c_rl, c_b)
        print(f"  {rl[:40]} contre {dep[:30]} ({len(ids)} questions)")
        print("    k      " + " ".join(f"{k:>6}" for k in ks))
        print("    agent  " + " ".join(f"{x:6.3f}" for x in pk_rl))
        print("    départ " + " ".join(f"{x:6.3f}" for x in pk_b))
        print("    D(k)   " + " ".join(f"{x:+6.3f}" for x in D))
        print(f"    frontière : agent {f['rl']}, départ {f['base']}, gagnées {f['gagnees']}, perdues {f['perdues']}")
        # Marge d'élargissement : un gain de frontière n'est possible que sur les questions que le départ ne
        # résout jamais en n essais. Si cette part est proche de 0, pass@n sature et l'élargissement est
        # indétectable quelle que soit la taille de l'étude (plafond).
        print(f"    hors de portée du départ (0 réussite sur {args.n}) : {np.mean(c_b == 0):.0%} ; "
              f"de l'agent : {np.mean(c_rl == 0):.0%}")
        for N in (250, 500, 1000):
            print(f"    puissance « élargissement » si la population ressemblait au pilote, N = {N} : "
                  f"{puissance(c_rl, c_b, args.n, N):.0%}")

    print("\n== Budget GPU (deux T4 en parallèle, débit moyen observé)")
    debits = [d["traj_par_s"] for d in desc if d["traj_par_s"]]
    if debits:
        debit = 2 * float(np.mean(debits))  # deux files en parallèle
        # 3 modèles = les deux paires Instruct (le modèle de départ est commun) ; 6 = les quatre paires
        for modeles, N, n, T in [(3, 500, 32, 2), (3, 500, 16, 2), (3, 1000, 32, 2), (6, 500, 32, 2), (6, 500, 16, 2)]:
            traj = modeles * N * 2 * n * T  # 2 types de questions (ponts, comparaisons) ; T valeurs de T
            h = traj / debit / 3600
            print(f"  {modeles} modèles, {N} questions par type, n = {n}, {T} valeurs de T : {traj:,} trajectoires, "
                  f"environ {h:.1f} h ({h / args.quota_heures:.1f} semaine(s) de quota)")


if __name__ == "__main__":
    main()
