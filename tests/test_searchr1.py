"""Boucle Search-R1 avec un faux modèle : ces tests n'ont besoin ni de vLLM ni de GPU."""
import sys
import types
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class FakeOutput:
    def __init__(self, text, stop_reason):
        self.outputs = [types.SimpleNamespace(text=text, stop_reason=stop_reason, token_ids=[0])]


class FakeLLM:
    """Renvoie à la suite les sorties prévues."""
    script = []

    def __init__(self, *args, **kwargs):
        self.calls = 0

    def get_tokenizer(self):
        return types.SimpleNamespace(apply_chat_template=lambda msgs, **kwargs: msgs[0]["content"],
                                     encode=lambda t: t.split())

    def generate(self, prompts, params, use_tqdm=False):
        text, stop_reason = self.script[self.calls]
        self.calls += 1
        return [FakeOutput(text, stop_reason)]


class FakeRetriever:
    def search(self, query, k=3):
        return ["Titre\nTexte"]

    @staticmethod
    def format_searchr1(passages):
        return "Doc 1(Title: Titre) Texte\n"


def _run(script, max_turns=2):
    fake_vllm = types.SimpleNamespace(LLM=FakeLLM, SamplingParams=lambda **kwargs: None)
    with mock.patch.dict(sys.modules, {"vllm": fake_vllm}):
        sys.modules.pop("systems.searchr1", None)
        from systems.searchr1 import SearchR1System
        FakeLLM.script = script
        system = SearchR1System("faux", FakeRetriever(), max_turns=max_turns)
        return system.run([{"id": "q", "question": "Q?", "golds": ["R"]}])[0]


def test_search_then_answer():
    t = _run([("<search>x</search>", "</search>"), ("<answer>R</answer>", None)])
    assert t["queries"] == ["x"]
    assert t["trajectory"].endswith("<answer>R</answer>")


def test_closing_tag_without_opening_tag_ends_without_crash():
    t = _run([("oups</search>", "</search>")])
    assert t["n_search"] == 0
    assert t["trajectory"] == "oups</search>"


def test_last_generation_never_searches():
    s = ("<search>x</search>", "</search>")
    t = _run([s, s, s], max_turns=2)
    assert t["n_search"] == 2
    assert t["trajectory"].endswith("<search>x</search>")
