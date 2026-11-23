"""H4 : sélection des exemples et masquage de la perte (scripts/sft_donnees.py, scripts/train_sft.py)."""
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.sft_donnees import construire, garder  # noqa: E402
from scripts.train_sft import masque_caracteres  # noqa: E402


def traj(i, texte, em=1, **kw):
    return dict({"id": i, "question": "q" + i, "trajectory": texte, "em": em, "truncated": False,
                 "context_exceeded": False, "n_search": 1, "sample": 0}, **kw)


def test_garder_filtre_et_dedoublonne():
    L = [traj("a", "x"), traj("a", "x"), traj("a", "y", em=0), traj("a", "z", truncated=True), traj("b", "w")]
    g = garder(L)
    assert {k: [t["trajectory"] for t in v] for k, v in g.items()} == {"a": ["x"], "b": ["w"]}


def test_construire_plafond_et_egalisation():
    a = [traj(f"q{i}", f"t{j}") for i in range(10) for j in range(6)]   # 10 q. × 6 réussites
    b = [traj(f"q{i}", f"u{j}") for i in range(5) for j in range(2)]    # 5 q. × 2
    jeux = construire({"a": a, "b": b}, par_question=4, seed=0)
    assert len(jeux["b"]) == 10 and len(jeux["a"]) == 10
    par_q = {}
    for t in jeux["a"]:
        par_q[t["id"]] = par_q.get(t["id"], 0) + 1
    assert max(par_q.values()) <= 4


def test_masque_prompt_et_information():
    p = "PROMPT"
    t = "<search>q</search>\n\n<information>DOC</information>\n\n<answer>X</answer>"
    g = masque_caracteres(p, t)
    texte = p + t
    entraine = "".join(c for c, k in zip(texte, g) if k)
    assert "PROMPT" not in entraine and "DOC" not in entraine and "<information>" not in entraine
    assert "<search>q</search>" in entraine and "<answer>X</answer>" in entraine


def test_logits_partiels_egaux_aux_complets():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from scripts.train_sft import perte_somme
    cfg = transformers.AutoConfig.for_model("qwen2", hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                            num_attention_heads=2, num_key_value_heads=1, vocab_size=100)
    torch.manual_seed(0)
    m = transformers.AutoModelForCausalLM.from_config(cfg).eval()
    rng = random.Random(0)
    ids = [rng.randrange(100) for _ in range(20)]
    labels = [-100] * 8 + ids[8:14] + [-100] * 3 + ids[17:]
    with torch.no_grad():
        p, n = perte_somme(m, ids, labels, torch)
        logits = m(input_ids=torch.tensor([ids])).logits[0]
        y = torch.tensor(labels)
        pos = torch.nonzero(y[1:] != -100).squeeze(1)
        ref = torch.nn.functional.cross_entropy(logits[pos].float(), y[pos + 1], reduction="sum")
    assert n == 9 and torch.allclose(p, ref, atol=1e-4)
