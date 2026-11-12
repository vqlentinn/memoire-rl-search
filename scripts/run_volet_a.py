"""Point d'entrée du volet A : un système (A0 à A5) sur un jeu de QA, résultats écrits par lots.

Un run coupé reprend là où il s'est arrêté : les identifiants déjà présents dans le fichier de sortie
sont sautés. Une ligne JSONL par question (trajectoire, requêtes, documents, réponse, EM, F1, jetons).

Exemple (Kaggle, GPU T4) :
    VLLM_WORKER_MULTIPROC_METHOD=spawn python scripts/run_volet_a.py --system A1 --dataset nq --n 100 --corpus 1m --retriever e5
L'index e5_Flat-1m est téléchargé au premier lancement. Spawn est nécessaire : E5 initialise CUDA avant vLLM.

Avec une graine autre que 42 (mise au point), les questions d'évaluation sont exclues du tirage.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.load_qa import load_qa  # noqa: E402
from eval.metrics import exact_match, extract_answer, f1_score  # noqa: E402

A4_MODEL = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-ppo-v0.3"
EVAL_SEED = 42
EVAL_N = 100


def select_questions(dataset, n, seed):
    """Tirage d'évaluation (graine 42), ou tirage de mise au point sans aucune question d'évaluation."""
    if seed == EVAL_SEED:
        return load_qa(dataset, n=n, seed=seed)
    evaluation = {q["question"] for q in load_qa(dataset, n=EVAL_N, seed=EVAL_SEED)}
    candidates = load_qa(dataset, n=n + EVAL_N, seed=seed)
    return [q for q in candidates if q["question"] not in evaluation][:n]


def build_retriever(kind, corpus_size, e5_index):
    from scripts.smoke_test import load_subset_corpus
    corpus = load_subset_corpus(corpus_size)
    if kind == "bm25":
        from retrieval.bm25_index import BM25Retriever
        return BM25Retriever(corpus)
    from retrieval.e5_index import E5Retriever
    if not e5_index:  # même fichier que scripts/measure_coverage.py
        from huggingface_hub import hf_hub_download
        e5_index = hf_hub_download("maknee/wiki-18-subsets", f"e5_Flat-{corpus_size}.index", repo_type="dataset")
    return E5Retriever(corpus, e5_index)


def build_system(name, retriever, model, topk):
    if name in ("A4", "A5"):
        from systems.searchr1 import SearchR1System
        if name == "A5" and not model:
            sys.exit("A5 : préciser --model (identifiant du checkpoint GRPO v0.3, collection "
                     "PeterJinGo/search-r1-v03). Il n'est volontairement pas deviné.")
        return SearchR1System(model or A4_MODEL, retriever, topk=topk)
    from systems import baselines as b
    llm = b.load_llm(model or b.DEFAULT_MODEL)
    cls = {"A0": b.DirectSystem, "A1": b.RAGSystem, "A2": b.ReActSystem, "A3": b.IRCoTSystem}[name]
    return cls(llm, retriever, topk=topk)


def finish(t, name):
    """Ajoute prédiction, statut, EM et F1 (A4/A5 : réponse extraite de la trajectoire)."""
    if "prediction" not in t:
        t["prediction"] = extract_answer(t["trajectory"])
        t["status"] = "answer" if t["prediction"] is not None else (
            "budget_exhausted" if t["trajectory"].rstrip().endswith("</search>") else "anomaly")
    t["system"] = name
    t["em"] = exact_match(t["prediction"], t["golds"])
    t["f1"] = f1_score(t["prediction"], t["golds"])
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=["A0", "A1", "A2", "A3", "A4", "A5"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--corpus", default="100k")
    ap.add_argument("--retriever", default="bm25", choices=["bm25", "e5"])
    ap.add_argument("--e5-index", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--seed", type=int, default=EVAL_SEED)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    suffix = "" if args.seed == EVAL_SEED else f"_seed{args.seed}"
    out = Path(args.outdir) / f"{args.system}_{args.dataset}_{args.retriever}_{args.corpus}_n{args.n}{suffix}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["id"] for line in out.read_text().split("\n") if line.strip()}
    questions = [q for q in select_questions(args.dataset, args.n, args.seed) if q["id"] not in done]
    print(f"{args.system} sur {args.dataset} : {len(done)} déjà faites, {len(questions)} à faire -> {out}")
    if not questions:
        return

    # A0 ne cherche pas : inutile de charger le corpus et l'index.
    retriever = None if args.system == "A0" else build_retriever(args.retriever, args.corpus, args.e5_index)
    system = build_system(args.system, retriever, args.model, args.topk)
    t0 = time.time()
    for i in range(0, len(questions), args.batch):
        trajs = [finish(t, args.system) for t in system.run(questions[i:i + args.batch])]
        with open(out, "a") as f:
            for t in trajs:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        print(f"  lot {i // args.batch + 1} écrit ({len(trajs)} questions, {time.time() - t0:.0f} s)")

    rows = [json.loads(line) for line in out.read_text().split("\n") if line.strip()]
    n = len(rows)
    print(f"EM {sum(r['em'] for r in rows) / n:.3f} | F1 {sum(r['f1'] for r in rows) / n:.3f} | "
          f"recherches par question {sum(r['n_search'] for r in rows) / n:.2f} | n = {n}")


if __name__ == "__main__":
    main()
