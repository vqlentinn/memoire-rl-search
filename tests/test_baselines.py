"""Systèmes A0 à A3 avec un faux modèle et un faux retriever : ni vLLM ni GPU."""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from systems import baselines as b  # noqa: E402


def out(text):
    return types.SimpleNamespace(outputs=[types.SimpleNamespace(text=text, token_ids=[0] * 3)])


class FakeLLM:
    """Pour chaque appel de generate, renvoie la sortie suivante du scénario (une par prompt)."""
    def __init__(self, script):
        self.script, self.calls, self.prompts = script, 0, []

    def get_tokenizer(self):
        return types.SimpleNamespace(apply_chat_template=lambda msgs, **kw: msgs[0]["content"])

    def generate(self, prompts, params, use_tqdm=False):
        self.prompts.append(prompts)
        texts = self.script[self.calls]
        self.calls += 1
        return [out(t) for t in texts[:len(prompts)]]


class FakeRetriever:
    def __init__(self):
        self.queries = []

    def search(self, query, k=3):
        self.queries.append(query)
        return [f"Titre {len(self.queries)}\nTexte du passage pour {query}"]

    @staticmethod
    def format_searchr1(passages):
        return "".join(f"Doc {i + 1}(Title: {p.split(chr(10))[0]}) {p.split(chr(10))[1]}\n"
                       for i, p in enumerate(passages))


Q = [{"id": "t-1", "question": "Qui a écrit Don Quichotte?", "golds": ["Cervantes"]}]


def setup_module(module):
    b._sampling_params = lambda **kw: kw  # pas de vLLM


def test_parse_react_action():
    assert b.parse_react_action("Thought: x\nAction: Search[a b]") == ("search", "a b")
    assert b.parse_react_action("Action: Finish[Paris]") == ("finish", "Paris")
    assert b.parse_react_action("rien") == (None, None)


def test_parse_ircot_answer_et_phrase():
    assert b.parse_ircot_answer("Il est né en Espagne. So the answer is: Spain.") == "Spain"
    assert b.parse_ircot_answer("Pas de réponse encore.") is None
    assert b.first_sentence("Première phrase. Deuxième phrase.") == "Première phrase."
    assert b.first_sentence("A. So the answer is: St. Louis.") == "A. So the answer is: St. Louis."


def test_a0_direct():
    res = b.DirectSystem(FakeLLM([["<answer> Cervantes </answer>"]])).run(Q)[0]
    assert res["prediction"] == "Cervantes" and res["n_search"] == 0 and res["status"] == "answer"


def test_a1_rag_une_recherche():
    r = FakeRetriever()
    res = b.RAGSystem(FakeLLM([["<answer>Cervantes</answer>"]]), r).run(Q)[0]
    assert r.queries == [Q[0]["question"]] and res["n_search"] == 1 and res["prediction"] == "Cervantes"


def test_a2_react_recherche_puis_reponse():
    r = FakeRetriever()
    llm = FakeLLM([["Thought: je cherche.\nAction: Search[auteur Don Quichotte]\nObservation:"],
                   ["Thought: trouvé.\nAction: Finish[Cervantes]"]])
    res = b.ReActSystem(llm, r).run(Q)[0]
    assert res["queries"] == ["auteur Don Quichotte"] and res["prediction"] == "Cervantes"
    assert res["status"] == "answer" and "Observation: Doc 1" in res["trajectory"]


def test_a2_budget_epuise():
    r = FakeRetriever()
    llm = FakeLLM([["Action: Search[x]"]] * 3)
    res = b.ReActSystem(llm, r, max_searches=2).run(Q)[0]
    assert res["n_search"] == 2 and res["prediction"] is None and res["status"] == "budget_exhausted"


def test_a2_sans_action():
    res = b.ReActSystem(FakeLLM([["Je ne sais pas."]]), FakeRetriever()).run(Q)[0]
    assert res["prediction"] is None and res["status"] == "anomaly"


def test_a3_ircot_une_recherche_par_phrase():
    r = FakeRetriever()
    llm = FakeLLM([["Don Quichotte est de Cervantes. Autre phrase."],
                   ["So the answer is: Cervantes."]])
    res = b.IRCoTSystem(llm, r).run(Q)[0]
    # recherche initiale (question), puis une recherche avec la première phrase
    assert r.queries == [Q[0]["question"], "Don Quichotte est de Cervantes."]
    assert res["prediction"] == "Cervantes" and res["n_search"] == 2 and res["status"] == "answer"
    assert "Doc 2(Title: Titre 2)" in llm.prompts[1][0]


def test_a3_budget_de_recherche_et_etapes():
    r = FakeRetriever()
    llm = FakeLLM([["Phrase un."], ["Phrase deux."], ["Phrase trois."]])
    res = b.IRCoTSystem(llm, r, max_searches=2, max_steps=3).run(Q)[0]
    assert res["n_search"] == 2 and res["prediction"] is None and res["status"] == "budget_exhausted"
