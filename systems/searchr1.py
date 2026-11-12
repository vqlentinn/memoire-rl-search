"""Systèmes A4 et A5 : checkpoints Search-R1, boucle de recherche avec vLLM.

Suit la boucle de search_r1/llm_agent/generation.py du dépôt Search-R1 (prompt copié de infer.py) :
- prompt = prompts/searchr1.txt passé dans le chat template ;
- génération jusqu'à </search> ou fin de séquence ;
- après chaque recherche, ajout de "\\n\\n<information>{docs}</information>\\n\\n" ;
- max_turns tours avec recherche, puis une dernière génération sans recherche. Si le modèle y
  demande encore une recherche, la trajectoire se termine sans <answer>.
Écarts connus avec generation.py : décodage glouton (choix du protocole), 512 jetons par tour au lieu
de 500, pas de troncature des résultats ni de message pour action invalide, pas de strip() sur les docs.
Les questions sont traitées par lots : chaque tour génère pour toutes les questions encore actives.

Partie 2 (pass@k) : run_samples() tire n générations par question (temperature > 0), un retriever
« fermé » peut exposer search_in(id, requête, k) pour chercher dans les seuls paragraphes de la question,
et chaque trajectoire signale si un tour a été coupé par la limite de jetons (troncature).
"""
import re
import time
from pathlib import Path

from vllm import LLM, SamplingParams

PROMPT = (Path(__file__).resolve().parent.parent / "prompts" / "searchr1.txt").read_text()
SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)


class SearchR1System:
    def __init__(self, model_id: str, retriever, topk: int = 3, max_turns: int = 4,
                 max_tokens: int = 512, temperature: float = 0.0, max_model_len: int = 4096,
                 use_chat_template: bool = True, seed: int = 0):
        self.llm = LLM(model_id, dtype="float16", max_model_len=max_model_len,
                       gpu_memory_utilization=0.85, seed=seed)
        self.use_chat_template = use_chat_template
        self.max_tokens, self.max_model_len = max_tokens, max_model_len
        self.tok = self.llm.get_tokenizer()
        self.retriever = retriever
        self.topk = topk
        self.max_turns = max_turns
        self.params = SamplingParams(temperature=temperature, max_tokens=max_tokens,
                                     stop=["</search>"], include_stop_str_in_output=True)

    def _initial_prompt(self, question: str) -> str:
        texte = PROMPT.format(question=question)
        if not self.use_chat_template:  # modèle de base : prompt brut, à comparer au pilote
            return texte
        msgs = [{"role": "user", "content": texte}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def run(self, questions):
        """questions : liste de dicts {id, question, golds}. Renvoie une trajectoire par question."""
        t0 = time.time()
        states = [{"q": q, "text": self._initial_prompt(q["question"]), "gen": "",
                   "queries": [], "docs": [], "tokens": 0, "done": False, "truncated": False}
                  for q in questions]
        for turn in range(self.max_turns + 1):
            active = [s for s in states if not s["done"]]
            # Garde-fou : une trajectoire dont le contexte ne laisse plus la place d'un tour complet est arrêtée et
            # marquée, au lieu de faire rejeter tout le lot par vLLM (paragraphes longs de HotpotQA, pilote du 12 nov.).
            for s in active:
                if len(self.tok.encode(s["text"])) + self.max_tokens > self.max_model_len:
                    s["done"], s["context_exceeded"], s["truncated"] = True, True, True
            active = [s for s in active if not s["done"]]
            if not active:
                break
            outs = self.llm.generate([s["text"] for s in active], self.params, use_tqdm=False)
            for s, o in zip(active, outs):
                out = o.outputs[0]
                s["text"] += out.text
                s["gen"] += out.text
                s["tokens"] += len(out.token_ids)
                s["truncated"] |= getattr(out, "finish_reason", None) == "length"
                ended_on_search = out.stop_reason == "</search>"
                # Sans <search> ouvrant, pas de requête : fin de la trajectoire sans faire planter le lot.
                matches = SEARCH_RE.findall(out.text)
                if not ended_on_search or turn == self.max_turns or not matches:
                    s["done"] = True
                    continue
                query = matches[-1].strip()
                if hasattr(self.retriever, "search_in"):  # recherche fermée, propre à la question
                    passages = self.retriever.search_in(s["q"]["id"], query, k=self.topk)
                else:
                    passages = self.retriever.search(query, k=self.topk)
                info = self.retriever.format_searchr1(passages)
                s["queries"].append(query)
                s["docs"].append(passages)
                block = f"\n\n<information>{info}</information>\n\n"
                s["text"] += block
                s["gen"] += block
        elapsed = time.time() - t0
        return [{
            "id": s["q"]["id"], "question": s["q"]["question"], "golds": s["q"]["golds"],
            "trajectory": s["gen"], "queries": s["queries"], "docs": s["docs"],
            "n_search": len(s["queries"]), "gen_tokens": s["tokens"],
            "batch_seconds": elapsed, "truncated": s["truncated"], "sample": s["q"].get("sample", 0),
            "context_exceeded": s.get("context_exceeded", False),
        } for s in states]

    def run_samples(self, questions, n):
        """n générations indépendantes par question (pass@k). Chaque copie garde l'id de sa question ;
        le champ "sample" les distingue. L'aléa vient de temperature > 0 et de la graine de vLLM."""
        copies = [dict(q, sample=j) for q in questions for j in range(n)]
        return self.run(copies)
