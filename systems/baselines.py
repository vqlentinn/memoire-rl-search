"""Systèmes A0 à A3 : réponse directe, RAG, ReAct par prompt, IRCoT.

Mêmes questions, même corpus, même retriever et mêmes 3 passages par recherche que A4 et A5.
Le modèle (Qwen2.5-3B-Instruct par défaut) est le même pour les quatre : seul le harnais change.
Décodage glouton. Budget de recherche : au plus `max_searches` appels (4, comme les 4 tours de A4/A5).
Un vLLM chargé une fois (load_llm) peut servir à plusieurs systèmes.

Chaque système renvoie une trajectoire par question, avec les clés de SearchR1System
(id, question, golds, trajectory, queries, docs, n_search, gen_tokens, batch_seconds) plus
`prediction` (réponse extraite, ou None) et `status` (answer, budget_exhausted, anomaly).

Écarts connus avec les articles d'origine :
- ReAct : 2 exemples écrits pour ce projet (le ReAct d'origine utilise des exemples de HotpotQA) ;
  recherche dans le corpus du projet et non l'API Wikipédia.
- IRCoT : 2 exemples écrits pour ce projet (le papier utilise des exemples du jeu évalué) ; première
  recherche avec la question, puis une recherche par phrase de raisonnement ; passages cumulés sans
  re-classement ; pas d'ajustement des hyperparamètres.
"""
import re
import time
from pathlib import Path

from eval.metrics import extract_answer

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"
DIRECT = (PROMPTS / "direct.txt").read_text()
RAG = (PROMPTS / "rag.txt").read_text()
REACT = (PROMPTS / "react.txt").read_text()
IRCOT = (PROMPTS / "ircot.txt").read_text()

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ACTION_RE = re.compile(r"Action:\s*(Search|Finish)\[(.*?)\]", re.DOTALL | re.IGNORECASE)
ANSWER_IS_RE = re.compile(r"answer is:?\s*(.*)", re.DOTALL | re.IGNORECASE)
SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


def load_llm(model_id: str = DEFAULT_MODEL, max_model_len: int = 6144):
    from vllm import LLM  # import local : les tests n'ont pas besoin de vLLM
    return LLM(model_id, dtype="float16", max_model_len=max_model_len,
               gpu_memory_utilization=0.85, seed=0)


def _sampling_params(**kwargs):
    from vllm import SamplingParams
    return SamplingParams(**kwargs)


def parse_react_action(text: str):
    """Première action du texte : ("search" | "finish", argument), ou (None, None)."""
    m = ACTION_RE.search(text)
    return (m.group(1).lower(), m.group(2).strip()) if m else (None, None)


def parse_ircot_answer(cot: str):
    """Réponse après « the answer is: », sans le point final, ou None."""
    m = ANSWER_IS_RE.search(cot)
    if not m:
        return None
    return m.group(1).strip().rstrip(".").strip() or None


def first_sentence(text: str) -> str:
    """Première phrase : IRCoT cherche après chaque phrase, même si le modèle a écrit toute sa chaîne.
    Si cette phrase donne la réponse, tout le texte est gardé : le point de « St. Louis » ne doit pas la couper."""
    text = text.strip()
    first = SENTENCE_END_RE.split(text, maxsplit=1)[0].strip()
    return text if ANSWER_IS_RE.search(first) else first


def _record(q, text, queries, docs, tokens, prediction, status, elapsed):
    return {"id": q["id"], "question": q["question"], "golds": q["golds"], "trajectory": text,
            "queries": queries, "docs": docs, "n_search": len(queries), "gen_tokens": tokens,
            "batch_seconds": elapsed, "prediction": prediction, "status": status}


class _Harness:
    name = ""

    def __init__(self, llm, retriever=None, topk: int = 3, max_tokens: int = 256,
                 temperature: float = 0.0):
        self.llm = llm
        self.tok = llm.get_tokenizer()
        self.retriever = retriever
        self.topk = topk
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _chat(self, content: str) -> str:
        msgs = [{"role": "user", "content": content}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def _params(self, stop=None, max_tokens=None):
        return _sampling_params(temperature=self.temperature, max_tokens=max_tokens or self.max_tokens,
                                stop=stop)

    def _format(self, passages) -> str:
        return self.retriever.format_searchr1(passages)


class DirectSystem(_Harness):
    """A0 : réponse directe, sans recherche."""
    name = "A0"

    def run(self, questions):
        t0 = time.time()
        prompts = [self._chat(DIRECT.format(question=q["question"])) for q in questions]
        outs = self.llm.generate(prompts, self._params(), use_tqdm=False)
        elapsed = time.time() - t0
        res = []
        for q, o in zip(questions, outs):
            out = o.outputs[0]
            pred = extract_answer(out.text)
            res.append(_record(q, out.text, [], [], len(out.token_ids), pred,
                               "answer" if pred is not None else "anomaly", elapsed))
        return res


class RAGSystem(_Harness):
    """A1 : une recherche avec la question, puis réponse."""
    name = "A1"

    def run(self, questions):
        t0 = time.time()
        passages = [self.retriever.search(q["question"], k=self.topk) for q in questions]
        prompts = [self._chat(RAG.format(docs=self._format(p), question=q["question"]))
                   for q, p in zip(questions, passages)]
        outs = self.llm.generate(prompts, self._params(), use_tqdm=False)
        elapsed = time.time() - t0
        res = []
        for q, p, o in zip(questions, passages, outs):
            out = o.outputs[0]
            pred = extract_answer(out.text)
            res.append(_record(q, out.text, [q["question"]], [p], len(out.token_ids), pred,
                               "answer" if pred is not None else "anomaly", elapsed))
        return res


class ReActSystem(_Harness):
    """A2 : ReAct par prompt (Thought, Action Search[...] ou Finish[...], Observation)."""
    name = "A2"

    def __init__(self, *args, max_searches: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_searches = max_searches

    def run(self, questions):
        t0 = time.time()
        states = [{"q": q, "text": self._chat(REACT.format(question=q["question"])), "gen": "",
                   "queries": [], "docs": [], "tokens": 0, "done": False, "pred": None,
                   "status": "anomaly"} for q in questions]
        for turn in range(self.max_searches + 1):
            active = [s for s in states if not s["done"]]
            if not active:
                break
            outs = self.llm.generate([s["text"] for s in active], self._params(stop=["Observation:"]),
                                     use_tqdm=False)
            for s, o in zip(active, outs):
                out = o.outputs[0]
                s["tokens"] += len(out.token_ids)
                m = ACTION_RE.search(out.text)
                gen = out.text[:m.end()] if m else out.text  # on garde jusqu'à la première action
                s["text"] += gen
                s["gen"] += gen
                kind, arg = parse_react_action(out.text)
                if kind == "finish":
                    s["pred"], s["status"], s["done"] = arg or None, "answer", True
                    if not arg:
                        s["status"] = "anomaly"
                    continue
                if kind != "search":
                    s["done"] = True  # ni Search ni Finish : anomalie
                    continue
                if turn == self.max_searches:
                    s["status"], s["done"] = "budget_exhausted", True
                    continue
                passages = self.retriever.search(arg, k=self.topk)
                block = f"\nObservation: {self._format(passages)}\n"
                s["queries"].append(arg)
                s["docs"].append(passages)
                s["text"] += block
                s["gen"] += block
        elapsed = time.time() - t0
        return [_record(s["q"], s["gen"], s["queries"], s["docs"], s["tokens"], s["pred"],
                        s["status"], elapsed) for s in states]


class IRCoTSystem(_Harness):
    """A3 : IRCoT. Recherche avec la question, puis une phrase de raisonnement à la fois ; chaque
    phrase sert de requête pour la recherche suivante ; les passages s'accumulent."""
    name = "A3"

    def __init__(self, *args, max_searches: int = 4, max_steps: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_searches = max_searches
        self.max_steps = max_steps

    @staticmethod
    def _add(state, passages):
        for p in passages:
            if p not in state["paras"]:
                state["paras"].append(p)

    def _search(self, state, query):
        passages = self.retriever.search(query, k=self.topk)
        state["queries"].append(query)
        state["docs"].append(passages)
        self._add(state, passages)

    def run(self, questions):
        t0 = time.time()
        states = [{"q": q, "paras": [], "cot": "", "queries": [], "docs": [], "tokens": 0,
                   "done": False, "pred": None, "status": "anomaly"} for q in questions]
        for s in states:
            self._search(s, s["q"]["question"])
        for _ in range(self.max_steps):
            active = [s for s in states if not s["done"]]
            if not active:
                break
            prompts = [self._chat(IRCOT.format(docs=self._format(s["paras"]), question=s["q"]["question"]))
                       + "Thought:" + s["cot"] for s in active]
            outs = self.llm.generate(prompts, self._params(stop=["\n"], max_tokens=96), use_tqdm=False)
            for s, o in zip(active, outs):
                out = o.outputs[0]
                s["tokens"] += len(out.token_ids)
                sent = first_sentence(out.text)
                if not sent:
                    s["done"] = True  # sortie vide : anomalie
                    continue
                s["cot"] += " " + sent
                pred = parse_ircot_answer(sent)
                if pred is not None:
                    s["pred"], s["status"], s["done"] = pred, "answer", True
                elif len(s["queries"]) < self.max_searches:
                    self._search(s, sent)
        for s in states:
            if not s["done"]:
                s["status"] = "budget_exhausted"  # max_steps atteint sans réponse
        elapsed = time.time() - t0
        return [_record(s["q"], "Thought:" + s["cot"], s["queries"], s["docs"], s["tokens"], s["pred"],
                        s["status"], elapsed) for s in states]
