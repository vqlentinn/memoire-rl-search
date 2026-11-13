"""Partie 2, H3, variante « réponse forcée » (amendement du 13 novembre 2026, APRÈS avoir vu la session 5).

Constat de la session 5 : sans son raisonnement, l'agent PPO lecteur veut presque toujours chercher encore (il ne
répond que dans 3 à 5 % des lectures), GRPO dans 37 à 55 % des cas. Le protocole prévu comptait ces tentatives comme
des échecs : il mesure alors la propension à chercher, pas la lecture. Cette variante force une réponse.

Pour chaque lecture SANS réponse de la session 5 : on garde ce que le lecteur a écrit avant sa tentative de recherche
(coupé au dernier « <search> »), on ajoute « <answer> », et le lecteur complète sa réponse. Les lectures qui avaient déjà
une réponse sont recopiées telles quelles. Toutes les lectures finissent donc par une réponse, dans les quatre cases.
Résultat rapporté comme exploratoire ; l'analyse prévue reste rapportée telle quelle.

    VLLM_WORKER_MULTIPROC_METHOD=spawn python scripts/run_lecture_forcee.py \\
        --lecteur PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-ppo-v0.3 \\
        --lecture results/partie2_lecture/lecteur-qwen2.5-3b-it-em-ppo-v0.3_chercheur-…_bridge_m8.jsonl \\
        --chercheur results/partie2/qwen2.5-3b-it-em-ppo-v0.3_bridge_q500_n32_T4_temp0.7_chat.jsonl
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def couper(trajectoire):
    """Ce que le lecteur a écrit avant sa (dernière) tentative de recherche, sans espace final."""
    i = trajectoire.rfind("<search>")
    return (trajectoire[:i] if i >= 0 else trajectoire).rstrip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lecteur", required=True)
    ap.add_argument("--lecture", required=True, help="sortie de scripts/run_lecture.py (session 5)")
    ap.add_argument("--chercheur", required=True, nargs="+")
    ap.add_argument("--type", default="bridge", choices=["bridge", "comparison"])
    ap.add_argument("--n-questions", type=int, default=500)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=64, help="la réponse seule : quelques mots")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--lot", type=int, default=64)
    ap.add_argument("--data", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="results/partie2_lecture")
    args = ap.parse_args()

    from data.hotpot_distractor import charger
    from eval.metrics import exact_match, extract_answer, f1_score
    from retrieval.closed_bm25 import ClosedBM25Retriever
    from scripts.run_lecture import construire_prefixe
    from systems.searchr1 import SearchR1System

    questions = {q["id"]: q for q in charger(args.data, args.type, n=args.n_questions, seed=42)}
    retriever = ClosedBM25Retriever(list(questions.values()))
    requetes = {}
    for chemin in args.chercheur:
        for l in Path(chemin).read_text().split("\n"):
            if l.strip():
                t = json.loads(l)
                requetes[(t["id"], t["sample"])] = t["queries"]
    lectures = [json.loads(l) for l in Path(args.lecture).read_text().split("\n") if l.strip()]
    out = Path(args.outdir) / (Path(args.lecture).stem + "_force.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    garder = [dict(x, force=False) for x in lectures if x["status"] == "answer"]
    a_forcer = [x for x in lectures if x["status"] != "answer"]
    print(f"{Path(args.lecture).name} : {len(garder)} lectures avec réponse recopiées, {len(a_forcer)} à forcer",
          flush=True)
    taches = []
    for x in a_forcer:
        coupe = couper(x["trajectory"])
        pref = construire_prefixe(x["id"], requetes[(x["id"], x["sample"])], retriever, args.topk)
        taches.append(dict(questions[x["id"]], sample=x["sample"], lecture=x, coupe=coupe,
                           prefixe=pref + coupe + ("\n" if coupe else "") + "<answer>"))
    resultats = list(garder)
    if taches:
        systeme = SearchR1System(args.lecteur, retriever, topk=args.topk, max_turns=0, max_tokens=args.max_tokens,
                                 temperature=args.temperature, max_model_len=args.max_model_len, seed=args.seed)
        t0 = time.time()
        for i in range(0, len(taches), args.lot):
            lot = taches[i:i + args.lot]
            for tache, r in zip(lot, systeme.run(lot)):
                complet = tache["coupe"] + ("\n" if tache["coupe"] else "") + "<answer>" + r["trajectory"]
                if "</answer>" not in r["trajectory"]:
                    complet += "</answer>"  # réponse coupée par max_tokens : on garde ce qui a été écrit
                pred = extract_answer(complet)
                resultats.append(dict(tache["lecture"], trajectory=complet, prediction=pred, status="answer",
                                      em=exact_match(pred, tache["golds"]), f1=f1_score(pred, tache["golds"]),
                                      force=True, statut_initial=tache["lecture"]["status"]))
            print(f"  {i + len(lot)}/{len(taches)} forcées, {time.time() - t0:.0f} s", flush=True)
    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in resultats))
    f = [x for x in resultats if x["force"]]
    print(f"EM {sum(x['em'] for x in resultats) / len(resultats):.3f} | forcées {len(f)}, EM des forcées "
          f"{sum(x['em'] for x in f) / max(len(f), 1):.3f} | statuts initiaux {dict(Counter(x['status'] for x in lectures))}")


if __name__ == "__main__":
    main()
