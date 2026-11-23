"""H4 : jeux d'entraînement supervisé à partir des trajectoires réussies du vivier (amendement du 23 novembre 2026).

Sources : trajectoires du vivier (1 000 ponts, graine 43, 16 tirages, T = 4), disjoint des 500 questions d'évaluation
et du pilote. L'agent PPO (session 8) et le modèle de départ (session 3). Règles (plan d'analyse) :
- gardées : EM = 1, non tronquées, sans dépassement de contexte ; doublons exacts (même question, même texte) retirés ;
- au plus `--par-question` trajectoires par question (tirage aléatoire, graine 0) ;
- la source la plus grande est sous-échantillonnée à la taille de l'autre : même nombre d'exemples.
Sortie : un JSONL par source, une ligne par exemple {id, question, trajectory, source}. Le prompt n'y est pas : il est
reconstruit à l'entraînement avec le même gabarit que l'évaluation (systems/searchr1.py).

    python scripts/sft_donnees.py --indir results/partie2 --out data/sft
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

SOURCES = {"ppo": "qwen2.5-3b-it-em-ppo-v0.3_bridge_q1000_n16_T4_temp0.7_chat_seed43_g1",
           "depart": "Qwen2.5-3B-Instruct_bridge_q1000_n16_T4_temp0.7_chat_seed43_g1"}


def lire(indirs, prefixe):
    lignes = []
    for d in indirs:
        for f in sorted(Path(d).rglob(prefixe + "*.jsonl")):
            lignes += [json.loads(l) for l in f.read_text().split("\n") if l.strip()]
    return lignes


def garder(lignes):
    """Trajectoires réussies, complètes, sans doublon exact ; groupées par question (ordre de lecture conservé)."""
    vus, par_q = set(), defaultdict(list)
    for t in lignes:
        if not t["em"] or t.get("truncated") or t.get("context_exceeded"):
            continue
        cle = (t["id"], t["trajectory"])
        if cle in vus:
            continue
        vus.add(cle)
        par_q[t["id"]].append(t)
    return par_q


def selection(par_q, par_question, rng):
    ex = []
    for i in sorted(par_q):
        L = par_q[i]
        ex += rng.sample(L, par_question) if len(L) > par_question else list(L)
    return ex


def construire(lignes_par_source, par_question=4, seed=0):
    rng = random.Random(seed)
    jeux = {s: selection(garder(L), par_question, rng) for s, L in lignes_par_source.items()}
    taille = min(len(v) for v in jeux.values())
    for s in jeux:
        if len(jeux[s]) > taille:
            jeux[s] = sorted(rng.sample(jeux[s], taille), key=lambda t: (t["id"], t.get("sample", 0)))
    return jeux


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", action="append", default=None, help="répétable ; défaut results/partie2")
    ap.add_argument("--out", default="data/sft")
    ap.add_argument("--par-question", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    indirs = args.indir or ["results/partie2"]
    lignes = {s: lire(indirs, p) for s, p in SOURCES.items()}
    for s, L in lignes.items():
        if not L:
            raise SystemExit(f"aucune trajectoire pour la source {s} ({SOURCES[s]}) dans {indirs}")
        if len({t['id'] for t in L}) != 1000:
            raise SystemExit(f"source {s} : {len({t['id'] for t in L})} questions au lieu de 1 000")
    jeux = construire(lignes, args.par_question, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    resume = {}
    for s, ex in jeux.items():
        with open(out / f"sft_{s}.jsonl", "w") as f:
            for t in ex:
                f.write(json.dumps({"id": t["id"], "question": t["question"], "trajectory": t["trajectory"],
                                    "source": s}, ensure_ascii=False) + "\n")
        resume[s] = {"exemples": len(ex), "questions": len({t["id"] for t in ex}),
                     "recherches": round(sum(t["n_search"] for t in ex) / len(ex), 2),
                     "reussites_avant_selection": sum(t["em"] for t in lignes[s])}
    (out / "resume.json").write_text(json.dumps(resume, indent=1))
    print(json.dumps(resume, indent=1))
    return resume


if __name__ == "__main__":
    main()
