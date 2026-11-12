"""Analyse confirmatoire de la partie 2 sur des runs simulés : fusion des moitiés, appariement, Holm, verdicts."""
import json

import numpy as np

from analysis.confirmatoire_partie2 import NOM, charger, holm, main, reussites


def ecrire(dossier, nom, p_juste, n=8, N=40, seed=0):
    rng = np.random.default_rng(seed)
    lignes = []
    for q in range(N):
        for s in range(n):
            em = int(rng.random() < p_juste[q])
            lignes.append({"id": f"q{q}", "sample": s, "em": em, "status": "answer", "truncated": False,
                           "n_search": 1, "gen_tokens": 100})
    (dossier / f"{nom}.jsonl").write_text("".join(json.dumps(l) + "\n" for l in lignes))
    return lignes


def test_nom_de_fichier_reconnu_avec_moitie_et_graine():
    m = NOM.match("Qwen2.5-3B-Instruct_bridge_q500_n32_T4_temp0.7_chat_shard1of2")
    assert m and m["modele"] == "Qwen2.5-3B-Instruct" and m["sq"] is None
    m = NOM.match("qwen2.5-3b-it-em-ppo-v0.3_comparison_q50_n32_T0_temp0.7_brut_seed7")
    assert m and m["type"] == "comparison" and m["T"] == "0" and m["sq"] == "7"


def test_moities_fusionnees_et_questions_incompletes_ecartees(tmp_path):
    p = np.full(40, 0.5)
    L = ecrire(tmp_path, "Qwen2.5-3B-Instruct_bridge_q40_n8_T4_temp0.7_chat", p)
    (tmp_path / "Qwen2.5-3B-Instruct_bridge_q40_n8_T4_temp0.7_chat.jsonl").unlink()
    moitie = len(L) // 2
    for i, part in enumerate((L[:moitie], L[moitie:])):
        (tmp_path / f"Qwen2.5-3B-Instruct_bridge_q40_n8_T4_temp0.7_chat_shard{i}of2.jsonl").write_text(
            "".join(json.dumps(l) + "\n" for l in part))
    runs = charger(tmp_path)
    assert len(runs) == 1
    lignes = next(iter(runs.values()))
    assert len(lignes) == 320
    assert len(reussites(lignes[:-1], 8)) == 39  # la dernière question n'a plus que 7 tirages


def test_holm():
    assert holm([0.01, 0.04]) == [True, True]
    assert holm([0.03, 0.04]) == [False, False]  # 0,03 > 0,05 / 2
    assert holm([0.2, 0.01]) == [False, True]


def test_verdicts_elargissement_et_fiabilisation(tmp_path):
    rng = np.random.default_rng(1)
    # Questions accessibles avec p >= 0,5 : le départ les résout presque toujours en 8 essais. Avec p petit mais non
    # nul, un agent qui ne fait que fiabiliser « élargirait » en apparence à n fini (le départ les rate par hasard).
    base = rng.uniform(0.5, 0.8, 200)
    base[:60] = 0.0  # hors de portée du départ
    ecrire(tmp_path, "Qwen2.5-3B-Instruct_bridge_q200_n8_T4_temp0.7_chat", base, N=200, seed=2)
    # Agent qui élargit : résout une partie des questions hors de portée, et fait mieux partout
    elargit = np.clip(base + 0.15, 0, 1)
    elargit[:60] = 0.3
    ecrire(tmp_path, "qwen2.5-3b-it-em-ppo-v0.3_bridge_q200_n8_T4_temp0.7_chat", elargit, N=200, seed=3)
    # Agent qui fiabilise : mieux au premier essai sur les questions accessibles, rien sur les autres
    fiab = np.where(base > 0, np.clip(base + 0.15, 0, 0.95), 0.0)
    ecrire(tmp_path, "qwen2.5-3b-it-em-grpo-v0.3_bridge_q200_n8_T4_temp0.7_chat", fiab, N=200, seed=4)
    paires = main(["--indir", str(tmp_path), "--outdir", str(tmp_path / "out"), "--n-boot", "500", "--sans-figures"])
    v = {p["agent"]: p for p in paires}
    assert all(p["principal"] for p in paires)
    assert v["qwen2.5-3b-it-em-ppo-v0.3"]["res"]["H2"] == "élargissement"
    assert v["qwen2.5-3b-it-em-ppo-v0.3"]["elargissement_holm"]
    assert v["qwen2.5-3b-it-em-grpo-v0.3"]["res"]["H1"]
    assert v["qwen2.5-3b-it-em-grpo-v0.3"]["res"]["H2"] == "fiabilisation seule"
    assert (tmp_path / "out" / "tableau_verdicts.tex").exists()


def test_nom_sous_ensemble_difficile_et_graine_de_generation():
    m = NOM.match("Qwen2.5-3B-Instruct_bridge_q1000_n16_T4_temp0.7_chat_seed43_g1")
    assert m and m["sq"] == "43" and m["g"] == "1"
    m = NOM.match("qwen2.5-3b-it-em-ppo-v0.3_bridge_q180_n32_T4_temp0.7_chat_idsdifficiles-ponts_shard0of2")
    assert m and m["ids"] == "difficiles-ponts" and m["g"] is None
