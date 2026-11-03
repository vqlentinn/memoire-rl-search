"""Smoke test (semaine 2) : un checkpoint Search-R1 tourne-t-il sur Kaggle ?

Critère de réussite : au moins 18 questions sur 20 terminées avec <answer>.

Usage (Kaggle, GPU T4, internet activé) :
    python scripts/smoke_test.py --n 20
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.load_qa import load_qa  # noqa: E402
from eval.metrics import exact_match, extract_answer, f1_score  # noqa: E402
from retrieval.bm25_index import BM25Retriever  # noqa: E402
from systems.searchr1 import SearchR1System  # noqa: E402

DEFAULT_MODEL = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-ppo-v0.3"


def load_subset_corpus(size: str = "100k"):
    """Sous-corpus wiki-18 (100k passages = 24 Mo) : dataset maknee/wiki-18-subsets."""
    path = hf_hub_download("maknee/wiki-18-subsets", f"wiki-18-{size}.jsonl.gz", repo_type="dataset")
    with gzip.open(path, "rt") as f:
        return [json.loads(line)["contents"] for line in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--corpus", default="100k")
    ap.add_argument("--out", default="results/smoke_test.jsonl")
    args = ap.parse_args()

    questions = load_qa("nq", split="test", n=args.n)
    corpus = load_subset_corpus(args.corpus)
    print(f"{len(questions)} questions, {len(corpus)} passages")

    retriever = BM25Retriever(corpus)
    system = SearchR1System(args.model, retriever)
    trajs = system.run(questions)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_answer, em_total = 0, 0
    with open(args.out, "w") as f:
        for t in trajs:
            pred = extract_answer(t["trajectory"])
            t["prediction"] = pred
            t["em"] = exact_match(pred, t["golds"])
            t["f1"] = f1_score(pred, t["golds"])
            n_answer += pred is not None
            em_total += t["em"]
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    n = len(trajs)
    print(f"Réponses <answer> : {n_answer}/{n}")
    print(f"EM : {em_total / n:.2f} (indicatif : corpus réduit, 20 questions)")
    print(f"Recherches par question : {sum(t['n_search'] for t in trajs) / n:.2f}")
    print(f"Temps total : {trajs[0]['batch_seconds']:.0f} s")
    print("SMOKE TEST " + ("RÉUSSI" if n_answer >= 0.9 * n else "ÉCHOUÉ"))


if __name__ == "__main__":
    main()
