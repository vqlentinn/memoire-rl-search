"""Assemble le mémoire complet à partir de l'article de la partie 1 et du chapitre de la partie 2.

Les deux documents sources restent la référence : on corrige là-bas, puis on relance ce script. Il écrit dans
docs/article/genere/ :
- memoire_preambule.tex : préambule de l'article (macros des résultats, styles), sans classe ni titre ;
- memoire_partie1.tex   : corps de l'article (de l'introduction à la conclusion) ;
- memoire_annexes1.tex  : annexes de l'article ;
- memoire_partie2.tex   : corps du chapitre de la partie 2 (de la question aux menaces à la validité).
docs/article/memoire.tex (écrit à la main) les assemble avec l'introduction, la discussion et la conclusion générales
(docs/article/memoire/*.tex). Compilation : `tectonic -X compile memoire.tex` dans docs/article/.

    python scripts/assemble_memoire.py
"""
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
ARTICLE = RACINE / "docs" / "article"
CHAPITRE2 = RACINE / "docs" / "partie2" / "chapitre" / "partie2.tex"


def entre(texte, debut, fin, inclure_debut=True):
    i = texte.index(debut)
    j = texte.index(fin, i)
    return texte[i if inclure_debut else i + len(debut):j]


def main():
    art = (ARTICLE / "main.tex").read_text()
    preambule = entre(art, "\n", r"\title{", inclure_debut=False)  # tout après \documentclass, avant le titre
    partie1 = entre(art, r"\section{Introduction}", r"\section*{Déclaration d'usage")
    annexes1 = entre(art, r"\appendix", r"\end{document}", inclure_debut=False)
    ch2 = CHAPITRE2.read_text()
    partie2 = entre(ch2, r"\section{Question et lien avec la partie 1}", r"\bibliographystyle")
    sortie = ARTICLE / "genere"
    entete = "% Fichier généré par scripts/assemble_memoire.py : ne pas modifier à la main.\n"
    (sortie / "memoire_preambule.tex").write_text(entete + preambule)
    (sortie / "memoire_partie1.tex").write_text(entete + partie1)
    (sortie / "memoire_annexes1.tex").write_text(entete + annexes1)
    (sortie / "memoire_partie2.tex").write_text(entete + partie2)
    for f in ("memoire_preambule", "memoire_partie1", "memoire_annexes1", "memoire_partie2"):
        print(f"{f}.tex : {len((sortie / (f + '.tex')).read_text().splitlines())} lignes")


if __name__ == "__main__":
    main()
