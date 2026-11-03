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
"""
import re
import time
from pathlib import Path

from vllm import LLM, SamplingParams

PROMPT = (Path(__file__).resolve().parent.parent / "prompts" / "searchr1.txt").read_text()
SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)


class SearchR1System:
    def __init__(self, model_id: str, retriever, topk: int = 3, max_turns: int = 4,
                 max_tokens: int = 512, temperature: float = 0.0, max_model_len: int = 4096):
        self.llm = LLM(model_id, dtype="float16", max_model_len=max_model_len,
                       gpu_memory_utilization=0.85, seed=0)
        self.tok = self.llm.get_tokenizer()
        self.retriever = retriever
        self.topk = topk
        self.max_turns = max_turns
        self.params = SamplingParams(temperature=temperature, max_tokens=max_tokens,
                                     stop=["</search>"], include_stop_str_in_output=True)

    def _initial_prompt(self, question: str) -> str:
        msgs = [{"role": "user", "content": PROMPT.format(question=question)}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def run(self, questions):
        """questions : liste de dicts {id, question, golds}. Renvoie une trajectoire par question."""
        t0 = time.time()
        states = [{"q": q, "text": self._initial_prompt(q["question"]), "gen": "",
                   "queries": [], "docs": [], "tokens": 0, "done": False} for q in questions]
        for turn in range(self.max_turns + 1):
            active = [s for s in states if not s["done"]]
            if not active:
                break
            outs = self.llm.generate([s["text"] for s in active], self.params, use_tqdm=False)
            for s, o in zip(active, outs):
                out = o.outputs[0]
                s["text"] += out.text
                s["gen"] += out.text
                s["tokens"] += len(out.token_ids)
                ended_on_search = out.stop_reason == "</search>"
                if not ended_on_search or turn == self.max_turns:
                    s["done"] = True
                    continue
                query = SEARCH_RE.findall(out.text)[-1].strip()
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
            "batch_seconds": elapsed,
        } for s in states]
