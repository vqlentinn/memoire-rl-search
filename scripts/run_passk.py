"""Partie 2 : n générations par question pour un checkpoint, dans l'environnement fermé HotpotQA.

Une ligne JSONL par génération (question, numéro de tirage, trajectoire, requêtes, statut, EM, F1,
troncature). Reprise : les questions déjà complètes (n tirages présents) sont sautées.

    VLLM_WORKER_MULTIPROC_METHOD=spawn python scripts/run_passk.py \\
        --model PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo-v0.3 \\
        --type bridge --n-questions 50 --n-samples 32 --max-turns 4 --temperature 0.7

T (--max-turns) : nombre maximal de recherches. T = 0 : aucune recherche permise ; une trajectoire qui
tente quand même de chercher se termine sans réponse (statut no_search_allowed), ce qui est rapporté.
"""
import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import exact_match, extract_answer, f1_score  # noqa: E402

SEED_EVAL, N_EVAL = 42, 500  # tirage d'évaluation : 500 questions par type, graine 42


def statut(t, max_turns):
    if extract_answer(t["trajectory"]) is not None:
        return "answer"
    if t.get("truncated"):
        return "truncated"
    if re.search(r"<search>", t["trajectory"]):
        return "no_search_allowed" if max_turns == 0 else "budget_exhausted"
    return "format"  # ni réponse ni recherche : le modèle n'a pas suivi le format


def noter(t, max_turns):
    t["prediction"] = extract_answer(t["trajectory"])
    t["status"] = statut(t, max_turns)
    t["em"] = exact_match(t["prediction"], t["golds"])
    t["f1"] = f1_score(t["prediction"], t["golds"])
    t.pop("docs", None)  # les passages se retrouvent depuis l'id de la question : fichiers plus légers
    return t


def nom_court(model):
    return model.split("/")[-1].replace("SearchR1-nq_hotpotqa_train-", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--type", required=True, choices=["bridge", "comparison"])
    ap.add_argument("--n-questions", type=int, default=50)
    ap.add_argument("--n-samples", type=int, default=32)
    ap.add_argument("--max-turns", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--questions-par-lot", type=int, default=8)
    ap.add_argument("--no-chat-template", action="store_true")
    ap.add_argument("--data", default=None, help="parquet HotpotQA distractor (téléchargé sinon)")
    ap.add_argument("--seed", type=int, default=0, help="graine de génération (vLLM)")
    ap.add_argument("--seed-questions", type=int, default=SEED_EVAL,
                    help="autre que 42 : tirage de mise au point, disjoint des questions d'évaluation")
    ap.add_argument("--outdir", default="results/partie2")
    args = ap.parse_args()

    from data.hotpot_distractor import charger
    from retrieval.closed_bm25 import ClosedBM25Retriever
    from systems.searchr1 import SearchR1System

    tag = "brut" if args.no_chat_template else "chat"
    out = Path(args.outdir) / (f"{nom_court(args.model)}_{args.type}_q{args.n_questions}_n{args.n_samples}"
                               f"_T{args.max_turns}_temp{args.temperature}_{tag}"
                               f"{'' if args.seed_questions == SEED_EVAL else f'_seed{args.seed_questions}'}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    faits = Counter()
    if out.exists():
        faits = Counter(json.loads(l)["id"] for l in out.read_text().splitlines() if l.strip())
    exclure = []
    if args.seed_questions != SEED_EVAL:  # pilote : aucune question d'évaluation
        exclure = [q["id"] for q in charger(args.data, args.type, n=N_EVAL, seed=SEED_EVAL)]
    questions = charger(args.data, args.type, n=args.n_questions, seed=args.seed_questions, exclure=exclure)
    a_faire = [q for q in questions if faits[q["id"]] < args.n_samples]
    print(f"{nom_court(args.model)} {args.type} T={args.max_turns} : {len(questions) - len(a_faire)} questions "
          f"déjà complètes, {len(a_faire)} à faire -> {out}", flush=True)
    if not a_faire:
        return
    # Une question incomplète est refaite entièrement : on retire ses lignes partielles.
    if faits:
        garder = [l for l in out.read_text().splitlines() if l.strip() and faits[json.loads(l)["id"]] >= args.n_samples]
        out.write_text("".join(x + "\n" for x in garder))

    systeme = SearchR1System(args.model, ClosedBM25Retriever(questions), topk=args.topk, max_turns=args.max_turns,
                             max_tokens=args.max_tokens, temperature=args.temperature,
                             use_chat_template=not args.no_chat_template, seed=args.seed)
    t0, n_traj = time.time(), 0
    for i in range(0, len(a_faire), args.questions_par_lot):
        lot = a_faire[i:i + args.questions_par_lot]
        trajs = [noter(t, args.max_turns) for t in systeme.run_samples(lot, args.n_samples)]
        with open(out, "a") as f:
            for t in trajs:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        n_traj += len(trajs)
        dt = time.time() - t0
        print(f"  {i + len(lot)}/{len(a_faire)} questions, {n_traj} trajectoires, {dt:.0f} s "
              f"({n_traj / dt:.1f} trajectoires/s)", flush=True)

    lignes = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    st = Counter(l["status"] for l in lignes)
    print(f"EM moyenne {sum(l['em'] for l in lignes) / len(lignes):.3f} | statuts {dict(st)} | "
          f"tronquées {sum(l['truncated'] for l in lignes) / len(lignes):.1%} | "
          f"recherches/trajectoire {sum(l['n_search'] for l in lignes) / len(lignes):.2f}")


if __name__ == "__main__":
    main()
