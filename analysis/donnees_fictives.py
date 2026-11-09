"""Données FICTIVES pour tester la chaîne d'analyse et la mise en page de l'article avant les vrais runs.

Ces lignes ne décrivent aucun modèle réel. Elles imitent le format de scripts/run_volet_a.py
(une ligne JSONL par question et par système) pour que analysis/analyse_mini.py tourne de bout en bout.
Chaque ligne porte "fictif": true, et les fichiers vont dans results/fictif/ (jamais dans results/ directement).

Les paramètres ci-dessous sont choisis à la main, arbitrairement. Ils ne sont PAS une prédiction :
la trame (étape 7.6) rappelle qu'analyser des données fabriquées ne fait que retrouver ce qu'on y a mis.

Modèle de génération (pour comprendre ce qu'on teste) :
- chaque question a une « facilité » latente z ~ N(0, 1), commune à tous les systèmes, ce qui rend les
  réponses corrélées d'un système à l'autre (comme dans la réalité : une question facile l'est pour tous) ;
- un système répond juste si z + 0,7 * bruit propre < seuil, le seuil étant fixé pour viser P_JUSTE ;
- le nombre de recherches suit une règle simple par système (voir n_recherches).

Usage : python analysis/donnees_fictives.py [--outdir results/fictif] [--n 100] [--seed 2026]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

SYSTEMES = ["A0", "A1", "A2", "A3", "A4", "A5"]
JEUX = ["nq", "triviaqa"]

# Probabilité visée de réponse juste (EM), choisie arbitrairement. Valeurs basses : couverture du corpus 1m.
P_JUSTE = {
    "nq":       {"A0": 0.12, "A1": 0.22, "A2": 0.19, "A3": 0.23, "A4": 0.31, "A5": 0.28},
    "triviaqa": {"A0": 0.36, "A1": 0.42, "A2": 0.40, "A3": 0.43, "A4": 0.45, "A5": 0.43},
}
BRUIT = 0.7
BUDGET = 4


def n_recherches(systeme, connue, rng):
    """Nombre de recherches fictif. `connue` : A0 répond juste à cette question."""
    if systeme == "A0":
        return 0
    if systeme == "A1":
        return 1
    if systeme == "A2":  # le prompt demande de répondre dès que possible
        if connue and rng.random() < 0.4:
            return 0
        return int(min(BUDGET, 1 + rng.binomial(3, 0.2)))
    if systeme == "A3":  # une recherche par phrase de raisonnement
        return int(min(BUDGET, 1 + rng.binomial(3, 0.75)))
    p = 0.45 if systeme == "A4" else 0.35  # A4, A5 : ne dépend pas de `connue`
    return int(1 + rng.binomial(3, p))


def generer(n, seed):
    rng = np.random.default_rng(seed)
    lignes = []
    for jeu in JEUX:
        z = rng.standard_normal(n)
        juste = {}
        for s in SYSTEMES:
            # seuil tel que P(z + BRUIT * e < seuil) = p, avec z + BRUIT * e ~ N(0, 1 + BRUIT^2)
            seuil = norm.ppf(P_JUSTE[jeu][s]) * np.sqrt(1 + BRUIT ** 2)
            juste[s] = (z + BRUIT * rng.standard_normal(n)) < seuil
        for s in SYSTEMES:
            for i in range(n):
                k = n_recherches(s, bool(juste["A0"][i]), rng)
                epuise = s in ("A3", "A4", "A5") and k == BUDGET and rng.random() < 0.15
                em = 0 if epuise else int(juste[s][i])
                lignes.append({
                    "id": f"{jeu}-fictif_{i:03d}",
                    "question": f"[question fictive {i} de {jeu}]",
                    "golds": ["[réponse fictive]"],
                    "trajectory": "[trajectoire fictive]",
                    "queries": [f"[requête fictive {j}]" for j in range(k)],
                    "docs": [],
                    "n_search": k,
                    "gen_tokens": int(rng.normal(60 + 90 * k, 20)),
                    "prediction": None if epuise else ("[réponse fictive]" if em else "[réponse fausse]"),
                    "status": "budget_exhausted" if epuise else "answer",
                    "system": s,
                    "dataset": jeu,
                    "em": em,
                    "f1": float(em) if em else float(np.clip(rng.normal(0.15, 0.15), 0, 0.9)),
                    "fictif": True,
                })
    return lignes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results/fictif")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    lignes = generer(args.n, args.seed)
    for s in SYSTEMES:
        for jeu in JEUX:
            f = out / f"{s}_{jeu}_e5_1m_n{args.n}.jsonl"
            with open(f, "w") as h:
                for l in lignes:
                    if l["system"] == s and l["dataset"] == jeu:
                        h.write(json.dumps(l, ensure_ascii=False) + "\n")
    print(f"{len(lignes)} lignes FICTIVES écrites dans {out}/")


if __name__ == "__main__":
    main()
