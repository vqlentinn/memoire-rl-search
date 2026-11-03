"""Retriever dense E5 : même encodage que search_r1/search/retrieval_server.py de Search-R1.

Modèle intfloat/e5-base-v2, préfixes "query: " et "passage: ", moyenne des jetons puis normalisation,
index FAISS plat (recherche exacte) du dataset maknee/wiki-18-subsets.
"""
from retrieval.bm25_index import BM25Retriever


class E5Retriever:
    format_searchr1 = staticmethod(BM25Retriever.format_searchr1)

    def __init__(self, corpus, index_path, model_id="intfloat/e5-base-v2", max_length=256):
        # Imports locaux : les autres systèmes n'ont besoin ni de FAISS ni de Transformers.
        import faiss
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.corpus = corpus
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.index = faiss.read_index(str(index_path))
        if self.index.ntotal != len(corpus):
            raise ValueError(f"Index de {self.index.ntotal} vecteurs pour {len(corpus)} passages")

    def _encode(self, texts):
        batch = self.tokenizer(texts, max_length=self.max_length, padding=True, truncation=True,
                               return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            hidden = self.model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1)
        mean = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        return self.torch.nn.functional.normalize(mean, dim=-1).cpu().numpy()

    def search(self, query: str, k: int = 3):
        _, ids = self.index.search(self._encode([f"query: {query}"]), k)
        return [self.corpus[i] for i in ids[0] if i >= 0]

    def check_alignment(self, indices):
        """Similarité entre les vecteurs de l'index et les passages réencodés : proche de 1 si l'ordre concorde."""
        vectors = self._encode([f"passage: {self.corpus[i]}" for i in indices])
        return [float(vectors[n] @ self.index.reconstruct(int(i))) for n, i in enumerate(indices)]
