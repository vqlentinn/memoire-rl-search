"""Chaîne d'analyse de la mini-expérience, testée sur les données fictives (sans GPU)."""
import numpy as np

from analysis.analyse_mini import analyser, charger, ecrire_tex, holm
from analysis.donnees_fictives import generer, main as generer_main  # noqa: F401


def test_holm():
    # p triées 0,01 < 0,02 < 0,04 : ajustées 0,03 ; max(0,03, 0,04) ; max(0,04, 0,04)
    assert np.allclose(holm([0.04, 0.01, 0.02]), [0.04, 0.03, 0.04])
    assert holm([0.9, 0.8])[0] == 1.0


def test_chaine_fictive(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["x", "--outdir", str(tmp_path / "in"), "--n", "30"])
    generer_main()
    donnees, fictif = charger(tmp_path / "in")
    assert fictif and len(donnees) == 12
    r = analyser(donnees)
    assert r["desc"][("A0", "nq")]["rech"][0] == 0
    assert r["desc"][("A1", "triviaqa")]["rech"][0] == 1
    assert len(r["tests"]["HA1"]["paires"]) == 6
    ecrire_tex(r, tmp_path, fictif)
    assert "\\fictiftrue" in (tmp_path / "valeurs.tex").read_text()
    assert (tmp_path / "resultats.tex").read_text().count("\\\\") == 8  # 2 lignes d'en-tête + 6 systèmes
