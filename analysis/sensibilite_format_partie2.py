"""Partie 2 : sensibilité au format de réponse (EXPLORATOIRE, 13 novembre 2026, après lecture des sorties).

Constat : beaucoup de trajectoires « sans réponse » de l'agent GRPO déclarent une réponse hors des balises attendues
(« &lt;answer&gt;X », tableau HTML « <td>Answer:</td><td>X », « 〈answer〉X〉 »…). La notation stricte de Search-R1 les
compte fausses. Deux notations de sensibilité, appliquées par la même règle aux deux modèles de chaque paire :
1. « réparée » (sensibilité principale) : si aucune réponse n'est balisée correctement, on cherche dans le texte écrit
   après le dernier bloc de passages une DÉCLARATION de réponse mal balisée (motif PAT ci-dessous), on garde la
   dernière, et on lui applique l'EM habituelle. Une simple mention n'est pas récupérée.
2. « mention » (borne haute, NON ÉQUITABLE) : une réponse de référence apparaît n'importe où dans ce texte. Elle
   avantage le modèle qui produit le plus de trajectoires sans réponse et de longs raisonnements (les agents) ; elle
   n'est rapportée que pour montrer jusqu'où va l'indulgence, pas pour conclure.

    python analysis/sensibilite_format_partie2.py --indir results/partie2
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.confirmatoire_partie2 import DEPART, analyser_paire, charger, court, reussites  # noqa: E402
import re  # noqa: E402

from eval.metrics import exact_match, normalize_answer  # noqa: E402


PAT = re.compile(r"(?:&lt;|<|〈|\[)\s*answer\s*(?:&gt;|>|〉|\])\s*([^<\n〉&\]]{1,120})"
                 r"|<td>\s*answer\s*:?\s*</td>\s*<td>\s*([^<\n]{1,120})"
                 r"|(?:^|\n)\s*answer\s*[:\n]\s*([^<\n]{1,120})", re.I)


def reparee(t):
    """EM après réparation d'une déclaration de réponse mal balisée (sinon EM stricte)."""
    if t["status"] == "answer" or not t.get("trajectory"):
        return int(t["em"])
    m = list(PAT.finditer(t["trajectory"].rsplit("</information>", 1)[-1]))
    if not m:
        return 0
    rep = [g for g in m[-1].groups() if g][0]
    rep = re.split(r"〈/|</|&lt;/|//", rep)[0].strip()
    return exact_match(rep, t["golds"]) if rep else 0


def mention(t):
    if t["status"] == "answer" or not t.get("trajectory"):
        return False
    fin = " " + normalize_answer(t["trajectory"].rsplit("</information>", 1)[-1]) + " "
    return any(normalize_answer(g) and f" {normalize_answer(g)} " in fin for g in t["golds"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/partie2")
    ap.add_argument("--out", default="docs/partie2/genere/sensibilite_format.json")
    args = ap.parse_args(argv)
    runs = charger(args.indir)
    res = {}
    souples = {}
    for (modele, cond), L in runs.items():
        if cond[6] != "42":
            continue
        sans = [t for t in L if t["status"] != "answer"]
        m = [mention(t) for t in sans]
        res[f"{court(modele)}_{cond[0]}_T{cond[3]}"] = {
            "sans_reponse": len(sans) / len(L), "dont_reponse_mentionnee": float(np.mean(m)) if m else 0.0,
            "em_stricte": float(np.mean([t["em"] for t in L])),
            "em_reparee": float(np.mean([reparee(t) for t in L])),
            "em_mention": float(np.mean([t["em"] or mention(t) for t in L])),
        }
        souples[("reparee", modele, cond)] = [dict(t, em=reparee(t)) for t in L]
        souples[("mention", modele, cond)] = [dict(t, em=int(t["em"] or mention(t))) for t in L]
    for k, v in sorted(res.items()):
        print(f"  {k:32} sans réponse {v['sans_reponse']:.1%}, dont réponse juste mentionnée {v['dont_reponse_mentionnee']:.1%}"
              f" | EM stricte {v['em_stricte']:.3f}, réparée {v['em_reparee']:.3f}, mention {v['em_mention']:.3f}")
    print("\n== Courbes appariées par notation (même règle pour les deux modèles)")
    for (notation, modele, cond), L in sorted(souples.items()):
        if modele in DEPART.values():
            continue
        dep = (notation, DEPART["it"], cond)
        if dep not in souples:
            continue
        n = cond[2]
        a, b = reussites(L, n), reussites(souples[dep], n)
        ids = sorted(set(a) & set(b))
        r = analyser_paire(np.array([a[i] for i in ids]), np.array([b[i] for i in ids]), n)
        res[f"paire_{notation}_{court(modele)}_{cond[0]}_T{cond[3]}"] = {k: r[k] for k in
                                                                    ("D", "simul_bas", "simul_haut", "H1", "H2", "frontiere")}
        print(f"  [{notation}] {court(modele)} {cond[0]} T={cond[3]} : D(k) " + " ".join(f"{x:+.3f}" for x in r["D"])
              + f" | bande à k=n [{r['simul_bas'][-1]:+.3f}, {r['simul_haut'][-1]:+.3f}] | H1 {r['H1']} | H2 {r['H2']}"
              f" | frontière {r['frontiere']['rl']} contre {r['frontiere']['base']}")
    Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
    return res


if __name__ == "__main__":
    main()
