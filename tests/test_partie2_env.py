"""Partie 2 : environnement fermé (HotpotQA distracteurs, BM25 par question) et boucle à n tirages."""
import sys
import types
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from retrieval.closed_bm25 import BM25Okapi, ClosedBM25Retriever  # noqa: E402

PARAS = ["Ed Wood\nEdward Davis Wood Jr. was an American filmmaker.",
         "Scott Derrickson\nScott Derrickson is an American director.",
         "Tyler Bates\nTyler Bates is a composer.",
         "Woodson, Arkansas\nWoodson is a town in Arkansas."]


def test_bm25_classe_le_bon_paragraphe():
    r = ClosedBM25Retriever([{"id": "q1", "paragraphs": PARAS}])
    assert r.search_in("q1", "Who is Scott Derrickson?", k=1)[0].startswith("Scott Derrickson")
    assert r.search_in("q1", "composer Tyler Bates", k=2)[0].startswith("Tyler Bates")


def test_idf_negatif_remplace_par_le_plancher():
    """« american » est dans 2 paragraphes sur 4 : idf = log(2,5/2,5) = 0 ; « is » dans 4 sur 4 : négatif."""
    bm = BM25Okapi(PARAS)
    assert all(v >= 0 for v in bm.idf.values())
    assert bm.idf["is"] == pytest.approx(0.25 * np.mean([np.log((4 - n + 0.5) / (n + 0.5))
                                                        for n in [len([p for p in bm.toks if m in p]) for m in bm.idf]]))


def test_ex_aequo_departages_par_l_ordre_du_fichier():
    r = ClosedBM25Retriever([{"id": "q", "paragraphs": ["A\nx", "B\nx", "C\nx"]}])
    assert [p[0] for p in r.search_in("q", "requête sans rapport", k=3)] == ["A", "B", "C"]


def test_format_identique_a_search_r1():
    out = ClosedBM25Retriever.format_searchr1(["Titre\nTexte\nsuite"])
    assert out == "Doc 1(Title: Titre) Texte\nsuite\n"


def _df(n_bridge=6, n_comp=4):
    import pandas as pd
    ctx = {"title": [f"T{i}" for i in range(10)], "sentences": [[f"s{i}."] for i in range(10)]}
    lignes = [{"id": f"b{i}", "question": f"qb{i}", "answer": f"a{i}", "type": "bridge", "level": "hard",
               "supporting_facts": {"title": ["T0", "T1"], "sent_id": [0, 0]}, "context": ctx} for i in range(n_bridge)]
    lignes += [{"id": f"c{i}", "question": f"qc{i}", "answer": "yes" if i < 2 else f"x{i}", "type": "comparison",
                "level": "hard", "supporting_facts": {"title": ["T2"], "sent_id": [0]}, "context": ctx} for i in range(n_comp)]
    lignes.append({"id": "court", "question": "q", "answer": "a", "type": "bridge", "level": "hard",
                   "supporting_facts": {"title": ["T0"], "sent_id": [0]},
                   "context": {"title": ["T0"], "sentences": [["s."]]}})
    return pd.DataFrame(lignes)


def test_preparer_exclut_oui_non_et_contextes_incomplets():
    from data.hotpot_distractor import preparer
    qs = preparer(_df())
    ids = {q["id"] for q in qs}
    assert "court" not in ids and "c0" not in ids and "c1" not in ids
    assert len(qs) == 6 + 2 and qs[0]["paragraphs"][0] == "T0\ns0."


def test_tirage_reproductible_et_exclusion(tmp_path):
    from data.hotpot_distractor import charger
    f = tmp_path / "v.parquet"
    _df(n_bridge=30).to_parquet(f)
    a = charger(f, "bridge", n=10, seed=42)
    assert [q["id"] for q in a] == [q["id"] for q in charger(f, "bridge", n=10, seed=42)]
    b = charger(f, "bridge", n=10, seed=7, exclure=[q["id"] for q in a])
    assert not {q["id"] for q in a} & {q["id"] for q in b}


class _Out:
    def __init__(self, text, stop, finish="stop"):
        self.outputs = [types.SimpleNamespace(text=text, stop_reason=stop, token_ids=[0], finish_reason=finish)]


class _LLM:
    script = []

    def __init__(self, *a, **k):
        self.i = 0

    def get_tokenizer(self):
        return types.SimpleNamespace(apply_chat_template=lambda m, **k: "CHAT:" + m[0]["content"],
                                     encode=lambda t: t.split())  # un « jeton » par mot

    def generate(self, prompts, params, use_tqdm=False):
        """Un tour = un appel : toutes les questions actives reçoivent la sortie prévue pour ce tour."""
        sortie = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return [_Out(*sortie) for _ in prompts]


def _systeme(script, **kw):
    fake_vllm = types.SimpleNamespace(LLM=_LLM, SamplingParams=lambda **k: None)
    with mock.patch.dict(sys.modules, {"vllm": fake_vllm}):
        sys.modules.pop("systems.searchr1", None)
        from systems.searchr1 import SearchR1System
        _LLM.script = script
        return SearchR1System("faux", ClosedBM25Retriever([{"id": "q1", "paragraphs": PARAS}]), **kw)


def test_boucle_recherche_fermee_et_tirages_multiples():
    s = _systeme([("<search>Scott Derrickson</search>", "</search>"), ("<answer>American</answer>", None)])
    t = s.run_samples([{"id": "q1", "question": "Q?", "golds": ["American"]}], n=3)
    assert len(t) == 3 and [x["sample"] for x in t] == [0, 1, 2]
    assert all(x["docs"][0][0].startswith("Scott Derrickson") for x in t)


def test_troncature_signalee_et_prompt_brut():
    s = _systeme([("<answer>coupé", None, "length")], use_chat_template=False)
    assert not s._initial_prompt("Q?").startswith("CHAT:")
    t = s.run([{"id": "q1", "question": "Q?", "golds": ["x"]}])[0]
    assert t["truncated"] is True


def test_contexte_depasse_arrete_proprement_la_trajectoire():
    """Pilote du 12 novembre : un contexte trop long faisait rejeter tout le lot par vLLM."""
    s = _systeme([("<search>Scott Derrickson</search>", "</search>"), ("<answer>x</answer>", None)],
                 max_tokens=5, max_model_len=110)  # le prompt (environ 90 « mots ») passe ; après une recherche, non
    t = s.run([{"id": "q1", "question": "Q?", "golds": ["x"]}])[0]
    assert t["context_exceeded"] is True and t["truncated"] is True and t["n_search"] == 1


def test_lecture_jsonl_robuste_au_separateur_unicode(tmp_path):
    """splitlines() coupe sur U+2028 (présent dans certains passages) ; split("\\n") non."""
    import json
    f = tmp_path / "t.jsonl"
    f.write_text(json.dumps({"id": "a", "texte": "avant\u2028après"}, ensure_ascii=False) + "\n")
    assert len(f.read_text().splitlines()) == 2  # le piège
    lignes = [json.loads(l) for l in f.read_text().split("\n") if l.strip()]
    assert lignes[0]["texte"] == "avant\u2028après"


def test_lecture_croisee_prefixe_identique_a_la_boucle_et_transmis_au_modele():
    """H3 : le début imposé contient les mêmes passages que ceux que la boucle insère, et arrive bien au modèle."""
    from scripts.run_lecture import construire_prefixe
    s = _systeme([("<search> Scott Derrickson </search>", "</search>"), ("<answer>American</answer>", None)])
    t = s.run([{"id": "q1", "question": "Q?", "golds": ["American"]}])[0]
    pref = construire_prefixe("q1", t["queries"], s.retriever, topk=s.topk)
    bloc_boucle = t["trajectory"][t["trajectory"].index("<information>"):t["trajectory"].index("</information>")]
    assert bloc_boucle in pref and pref.startswith("<search> Scott Derrickson </search>\n\n<information>")
    vus = []
    lecteur = _systeme([("<answer>American</answer>", None)], max_turns=0)
    gen = lecteur.llm.generate
    lecteur.llm.generate = lambda prompts, *a, **k: (vus.extend(prompts), gen(prompts, *a, **k))[1]
    r = lecteur.run([{"id": "q1", "question": "Q?", "golds": ["American"], "prefixe": pref}])[0]
    assert vus[0].endswith(pref) and r["trajectory"] == "<answer>American</answer>" and r["n_search"] == 0


def test_lecture_croisee_effets(tmp_path):
    import json
    from analysis.lecture_croisee import main
    rng = np.random.default_rng(0)
    # L'agent lit mieux (+0,2) ; ses plans ne valent pas mieux (0)
    p = {("A", "A"): 0.6, ("Qwen2.5-3B-Instruct", "A"): 0.4, ("A", "Qwen2.5-3B-Instruct"): 0.6,
         ("Qwen2.5-3B-Instruct", "Qwen2.5-3B-Instruct"): 0.4}
    for (lec, che), pr in p.items():
        lignes = [{"id": f"q{i}", "sample": s, "em": int(rng.random() < pr)} for i in range(300) for s in range(4)]
        (tmp_path / f"lecteur-{lec}_chercheur-{che}_bridge_m4.jsonl").write_text(
            "".join(json.dumps(l) + "\n" for l in lignes))
    r = main(["--indir", str(tmp_path)])["A_bridge"]
    assert r["H3"] and r["questions"] == 300
    assert abs(r["lecture"][0] - 0.2) < 0.05 and r["recherche"][1] < 0 < r["recherche"][2]
    assert abs(r["total"][0] - (r["lecture"][0] + r["recherche"][0])) < 1e-9
