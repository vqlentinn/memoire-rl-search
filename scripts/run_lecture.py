"""Partie 2, H3 : lecture croisée. Un modèle « lecteur » reçoit le plan de recherche d'un modèle « chercheur »
(ses requêtes et les passages qu'elles ramènent, sans son raisonnement) et doit répondre sans chercher davantage.

Plan de 2 × 2 : lecteur ∈ {départ, agent} × chercheur ∈ {départ, agent}, sur les mêmes questions et les mêmes
trajectoires de chercheur. À chercheur fixé, l'écart entre lecteurs mesure l'avantage de lecture ; à lecteur fixé,
l'écart entre chercheurs mesure l'avantage de recherche. Le raisonnement du chercheur est retiré dans les 4 cases,
y compris quand le lecteur est le chercheur lui-même : seul change qui a choisi les requêtes.

Le début imposé reproduit exactement ce que la boucle insère après chaque recherche :
    <search> requête </search>\\n\\n<information>passages</information>\\n\\n
Les passages sont recalculés depuis les requêtes (BM25 fermé, déterministe, ex aequo départagés) : ce sont ceux que
le chercheur a lus. Une trajectoire de chercheur sans recherche est écartée (rien à lire) ; leur nombre est rapporté.

    VLLM_WORKER_MULTIPROC_METHOD=spawn python scripts/run_lecture.py \\
        --lecteur Qwen/Qwen2.5-3B-Instruct \\
        --chercheur results/partie2/qwen2.5-3b-it-em-ppo-v0.3_bridge_q500_n32_T4_temp0.7_chat.jsonl --m 8
"""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def construire_prefixe(qid, requetes, retriever, topk=3):
    """Début de trajectoire imposé : les requêtes du chercheur et leurs passages, au format de la boucle."""
    blocs = []
    for r in requetes:
        info = retriever.format_searchr1(retriever.search_in(qid, r, k=topk))
        blocs.append(f"<search> {r} </search>\n\n<information>{info}</information>\n\n")
    return "".join(blocs)


def lire_chercheur(chemins, m):
    """{id: [trajectoires]} : les m premiers tirages (numéros 0..m-1) avec au moins une recherche."""
    par_q, ecartees = defaultdict(list), 0
    for chemin in chemins:
        for l in Path(chemin).read_text().split("\n"):
            if not l.strip():
                continue
            t = json.loads(l)
            if t["sample"] >= m:
                continue
            if not t["queries"]:
                ecartees += 1
                continue
            par_q[t["id"]].append(t)
    return par_q, ecartees


def nom_chercheur(chemin):
    return Path(chemin).name.split("_")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lecteur", required=True)
    ap.add_argument("--chercheur", required=True, nargs="+", help="JSONL du chercheur (les moitiés « shard » ensemble)")
    ap.add_argument("--type", default="bridge", choices=["bridge", "comparison"])
    ap.add_argument("--m", type=int, default=8, help="trajectoires de chercheur par question (tirages 0..m-1)")
    ap.add_argument("--n-questions", type=int, default=500)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--questions-par-lot", type=int, default=16)
    ap.add_argument("--data", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="results/partie2_lecture")
    args = ap.parse_args()

    from data.hotpot_distractor import charger
    from eval.metrics import exact_match, extract_answer, f1_score
    from retrieval.closed_bm25 import ClosedBM25Retriever
    from scripts.run_passk import nom_court, statut
    from systems.searchr1 import SearchR1System

    questions = {q["id"]: q for q in charger(args.data, args.type, n=args.n_questions, seed=42)}
    retriever = ClosedBM25Retriever(list(questions.values()))
    traj, ecartees = lire_chercheur(args.chercheur, args.m)
    chercheur = nom_chercheur(args.chercheur[0])
    out = Path(args.outdir) / f"lecteur-{nom_court(args.lecteur)}_chercheur-{chercheur}_{args.type}_m{args.m}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    faits = set()
    if out.exists():
        faits = {(x["id"], x["sample"]) for x in map(json.loads, filter(str.strip, out.read_text().split("\n")))}
    taches = [dict(questions[qid], sample=t["sample"], requetes=t["queries"], em_chercheur=t["em"],
                   prefixe=construire_prefixe(qid, t["queries"], retriever, args.topk))
              for qid, L in traj.items() if qid in questions for t in L if (qid, t["sample"]) not in faits]
    print(f"lecteur {nom_court(args.lecteur)}, chercheur {chercheur} : {len(taches)} lectures à faire, "
          f"{ecartees} trajectoires sans recherche écartées -> {out}", flush=True)
    if not taches:
        return
    # T = 0 : aucune recherche de plus ; une tentative de recherche est comptée sans réponse (statut rapporté).
    systeme = SearchR1System(args.lecteur, retriever, topk=args.topk, max_turns=0, max_tokens=args.max_tokens,
                             temperature=args.temperature, max_model_len=args.max_model_len, seed=args.seed)
    t0 = time.time()
    for i in range(0, len(taches), args.questions_par_lot):
        lot = taches[i:i + args.questions_par_lot]
        res = systeme.run(lot)
        with open(out, "a") as f:
            for tache, r in zip(lot, res):
                pred = extract_answer(r["trajectory"])
                f.write(json.dumps({
                    "id": r["id"], "sample": tache["sample"], "lecteur": nom_court(args.lecteur),
                    "chercheur": chercheur, "n_requetes": len(tache["requetes"]), "em_chercheur": tache["em_chercheur"],
                    "trajectory": r["trajectory"], "prediction": pred, "status": statut(r, 0),
                    "em": exact_match(pred, r["golds"]), "f1": f1_score(pred, r["golds"]),
                    "truncated": r["truncated"], "context_exceeded": r["context_exceeded"],
                    "gen_tokens": r["gen_tokens"], "n_search": r["n_search"],
                }, ensure_ascii=False) + "\n")
        print(f"  {i + len(lot)}/{len(taches)} lectures, {time.time() - t0:.0f} s", flush=True)
    L = [json.loads(l) for l in out.read_text().split("\n") if l.strip()]
    print(f"EM {sum(x['em'] for x in L) / len(L):.3f} | statuts {dict(Counter(x['status'] for x in L))}")


if __name__ == "__main__":
    main()
