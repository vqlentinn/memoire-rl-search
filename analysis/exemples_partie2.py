"""Partie 2 : exemples de trajectoires pour l'annexe (choix mécanique et reproductible, 14 novembre 2026).

Ponts, T = 4, questions graine 42. Pour chaque cas, on retient la trajectoire la plus courte (en caractères) qui
remplit le critère, pour que l'exemple tienne en annexe ; les blocs <information> sont abrégés.
1. question gagnée : l'agent PPO la résout au moins une fois, le modèle de départ jamais en 32 tirages ; on montre
   une réussite de l'agent qui a ramené les deux paragraphes utiles (cas de 93 % de ces réussites), puis la
   trajectoire la plus courte du départ sur la même question ;
2. presque juste : agent PPO, au moins une recherche, deux paragraphes utiles ramenés, réponse balisée, EM 0 et
   F1 >= 0,5 ;
3. balise mal formée : agent GRPO, statut « format », déclaration de réponse mal balisée récupérée par la notation
   « réparée » et juste après réparation ;
4. budget épuisé : agent PPO, statut budget_exhausted.

    python analysis/exemples_partie2.py --indir results/partie2
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.confirmatoire_partie2 import DEPART, charger, court  # noqa: E402
from analysis.sensibilite_format_partie2 import reparee  # noqa: E402

MAX_INFO = 110


def abreger(texte):
    def f(m):
        docs = re.split(r"\s*(?=Doc \d+\(Title:)", m.group(1).strip())
        docs = [d[:MAX_INFO] + " [...]" if len(d) > MAX_INFO else d for d in docs if d]
        return "<information>\n" + "\n".join(docs) + "\n</information>"
    texte = re.sub(r"<information>(.*?)</information>", f, texte, flags=re.S)
    texte = texte.replace("\u3008", "{U+3008}").replace("\u3009", "{U+3009}")  # chevrons absents de la police
    return re.sub(r"\n{2,}", "\n", texte).strip()


def bloc(titre, t, commentaire):
    gold = t["golds"] if isinstance(t["golds"], str) else " | ".join(t["golds"])
    return (f"\\subsection*{{{titre}}}\n{commentaire}\n\n\\noindent\\textbf{{Question :}} {echapper(t['question'])}"
            f" \\quad \\textbf{{Référence :}} {echapper(str(gold))}\n"
            "\\begin{Verbatim}[breaklines=true, breakanywhere=true, fontsize=\\scriptsize, frame=single]\n"
            f"{abreger(t['trajectory'])}\n\\end{{Verbatim}}\n")


def echapper(s):
    for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"),
                 ("_", "\\_"), ("{", "\\{"), ("}", "\\}"), ("~", "\\textasciitilde{}"), ("^", "\\^{}")):
        s = s.replace(a, b)
    return s


def plus_courte(L):
    return min(L, key=lambda t: (len(t["trajectory"]), t["id"], int(t["sample"])))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/partie2")
    ap.add_argument("--data", default="data/raw/hotpot_distractor/validation.parquet")
    ap.add_argument("--out", default="docs/partie2/genere/exemples_trajectoires.tex")
    args = ap.parse_args(argv)
    runs = {court(m): L for (m, c), L in charger(args.indir).items()
            if c[0] == "bridge" and c[3] == 4 and c[6] == "42"}
    dep, ppo, grpo = runs[court(DEPART["it"])], runs["it-em-ppo"], runs["it-em-grpo"]
    from analysis.recherche_partie2 import mesurer
    from data.hotpot_distractor import charger as charger_q
    from retrieval.closed_bm25 import ClosedBM25Retriever
    qs = {q["id"]: q for q in charger_q(args.data, "bridge", n=500, seed=42)}
    trouve = {(m["id"], m["sample"]): m["trouve_2"] for m in mesurer(ppo, qs, ClosedBM25Retriever(list(qs.values())))}
    succes_dep = defaultdict(int)
    for t in dep:
        succes_dep[t["id"]] += int(t["em"])
    gagnees = [t for t in ppo if t["em"] and succes_dep[t["id"]] == 0 and trouve[(t["id"], t["sample"])]]
    g = plus_courte(gagnees)
    echec_dep = plus_courte([t for t in dep if t["id"] == g["id"]])
    presque = plus_courte([t for t in ppo if t["status"] == "answer" and not t["em"] and float(t["f1"]) >= 0.5
                           and t["n_search"] > 0 and trouve[(t["id"], t["sample"])]])
    format_ = plus_courte([t for t in grpo if t["status"] == "format" and reparee(t)])
    budget = plus_courte([t for t in ppo if t["status"] == "budget_exhausted"])
    parties = [
        "% Fichier généré par analysis/exemples_partie2.py : ne pas modifier à la main.\n",
        "Les exemples sont choisis mécaniquement : pour chaque cas, la trajectoire la plus courte qui remplit le critère "
        "(\\texttt{analysis/exemples\\_partie2.py}). Les passages ramenés sont abrégés.\n",
        bloc("Une question gagnée : réussite de l'agent PPO", g,
             f"Le modèle de départ ne résout jamais cette question en 32 tirages ; l'agent PPO la résout "
             f"{sum(int(t['em']) for t in ppo if t['id'] == g['id'])} fois sur 32."),
        bloc("La même question : échec du modèle de départ", echec_dep, "Trajectoire la plus courte du modèle de départ."),
        bloc("Une réponse presque juste (agent PPO)", presque,
             f"Réponse « {echapper(presque['prediction'])} » : EM 0, F1 {float(presque['f1']):.2f}."),
        bloc("Une réponse mal balisée (agent GRPO)", format_,
             "Statut « format » : la réponse est déclarée hors des balises attendues. La notation stricte la compte "
             "fausse ; la notation « réparée » (section des analyses exploratoires) la récupère et la trouve juste. "
             "Le chevron typographique U+3008, que l'agent écrit à la place de « < », est noté \\texttt{\\{U+3008\\}}."),
        bloc("Un budget épuisé (agent PPO)", budget,
             "Quatre recherches faites, l'agent en demande une cinquième au lieu de répondre."),
    ]
    Path(args.out).write_text("\n".join(parties))
    print(f"écrit : {args.out} ({sum(len(p) for p in parties)} caractères)")
    for nom, t in (("gagnée", g), ("départ", echec_dep), ("presque", presque), ("format", format_), ("budget", budget)):
        print(f"  {nom:8} {t['id']} tirage {t['sample']} : {len(t['trajectory'])} caractères")


if __name__ == "__main__":
    main()
