"""Couverture du corpus : la réponse attendue est-elle retrouvable dans un sous-corpus ?

Pour chaque question, la question sert de requête (BM25 ou E5). On note le rang du premier passage qui contient
une réponse attendue (mots entiers, après la normalisation des métriques).

Limites : sans reformulation de requête. Une réponse courte peut apparaître dans
un passage sans rapport (faux positif) : c'est une estimation haute de l'utilité des passages.

Usage (CPU) :
    python scripts/measure_coverage.py --corpus 100k
    python scripts/measure_coverage.py --corpus 1m --retriever e5
Le fichier de sortie est écrit par lots : relancer la commande reprend là où le run s'est arrêté.
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.load_qa import DATASETS, load_qa  # noqa: E402
from eval.metrics import normalize_answer  # noqa: E402
from eval.stats import bootstrap_ci  # noqa: E402
from retrieval.bm25_index import BM25Retriever  # noqa: E402

TOPK = 100


def load_subset_corpus(size: str):
    path = hf_hub_download("maknee/wiki-18-subsets", f"wiki-18-{size}.jsonl.gz", repo_type="dataset")
    with gzip.open(path, "rt") as f:
        return [json.loads(line)["contents"] for line in f]


def load_e5(corpus, size: str):
    from retrieval.e5_index import E5Retriever
    index_path = hf_hub_download("maknee/wiki-18-subsets", f"e5_Flat-{size}.index", repo_type="dataset")
    retriever = E5Retriever(corpus, index_path)
    # Jeu de données tiers sans fiche descriptive : vérifie que l'index suit l'ordre du corpus.
    step = len(corpus) // 50
    similarities = retriever.check_alignment(list(range(0, len(corpus), step))[:50])
    print(f"Alignement index/corpus : min {min(similarities):.4f}, moyenne {sum(similarities) / len(similarities):.4f}",
          flush=True)
    if min(similarities) < 0.98:
        raise ValueError("L'index ne correspond pas au corpus")
    return retriever


def first_hit_rank(passages, golds):
    """Rang (à partir de 1) du premier passage contenant une réponse attendue, sinon None."""
    targets = [f" {t} " for t in (normalize_answer(str(g)) for g in golds) if t]
    for rank, passage in enumerate(passages, start=1):
        text = f" {normalize_answer(passage)} "
        if any(t in text for t in targets):
            return rank
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="100k")
    ap.add_argument("--retriever", choices=["bm25", "e5"], default="bm25")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    default_name = f"coverage_{args.corpus}.jsonl" if args.retriever == "bm25" else f"coverage_e5_{args.corpus}.jsonl"
    out = Path(args.out or f"results/{default_name}")
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in out.open()] if out.exists() else []
    done = {r["id"] for r in rows}

    corpus = load_subset_corpus(args.corpus)
    print(f"{len(corpus)} passages", flush=True)
    if args.retriever == "bm25":
        retriever = BM25Retriever(corpus)
    else:
        retriever = load_e5(corpus, args.corpus)

    with out.open("a") as f:
        for name in DATASETS:
            for q in load_qa(name, n=args.n):
                if q["id"] in done:
                    continue
                rank = first_hit_rank(retriever.search(q["question"], k=TOPK), q["golds"])
                row = {"dataset": name, "id": q["id"], "corpus": args.corpus, "retriever": args.retriever,
                       "hit_rank": rank}
                f.write(json.dumps(row) + "\n")
                rows.append(row)
            f.flush()
            print(f"{name} terminé", flush=True)

    print(f"\nCorpus {args.corpus}, {args.retriever} | jeu | n | réponse dans le top-3 [IC 95 %] | réponse dans le top-{TOPK} [IC 95 %]")
    for name in DATASETS:
        ranks = [r["hit_rank"] for r in rows if r["dataset"] == name]
        top3 = bootstrap_ci([r is not None and r <= 3 for r in ranks])
        top100 = bootstrap_ci([r is not None for r in ranks])
        print(f"{name:16} {len(ranks):4}  {top3[0]:.3f} [{top3[1]:.3f}, {top3[2]:.3f}]"
              f"  {top100[0]:.3f} [{top100[1]:.3f}, {top100[2]:.3f}]")


if __name__ == "__main__":
    main()
