"""Sélection finale du corpus de la revue de littérature (étape 5.2, règle 9 de la trame).

Décision déléguée par l'étudiant à Claude le 9 novembre 2026 (« tu choisis, entre 30 et 70 articles »).
Les règles sont écrites ici pour être relues, contestées et rejouées :

R1  Corpus de départ (corpus.csv) : thèmes au cœur de l'article (T1 agents RL, T3 harnais fixes,
    T4 quand chercher, T7 retriever, T8 bruit) en priorité « essentiel » ; thème T2 (coût) en « essentiel »
    ou « utile », parce qu'il porte HA3 et H3.
R2  Tout article cité dans la revue de l'article (refs.bib, hors références de théorie et de méthode).
R3  Nouveaux candidats (OpenAlex, Consensus, équation 3) : inclus s'ils portent sur la décision de chercher,
    le coût des recherches, la comparaison à un harnais fixe, le retriever ou l'usage de l'information trouvée.
    Exclus (E1) : méthodes d'entraînement pures (récompense, crédit, données, distillation) sans l'un de ces
    objets ; (E3) population différente (domaines verticaux, questions ouvertes).
Hors corpus de revue, gardés à part : algorithmes et modèles (T5), jeux de données (T6), évaluation (T9),
références de théorie et de statistique.

    python scripts/selection_corpus.py --meta <json OpenAlex des retenus de l'équation 3>
Écrit docs/litterature/corpus_final.csv et met à jour les colonnes de décision des fichiers de tri.
"""
import argparse
import csv
import json
import re
from pathlib import Path

LIT = Path("docs/litterature")
BIB = Path("docs/article/refs.bib")
THEORIE_METHODE = {"pirolli1999information", "stigler1961economics", "azzopardi2011economics", "efron1979bootstrap",
                   "mcnemar1947note", "holm1979simple", "kwiatkowski2019natural", "joshi2017triviaqa",
                   "jin2024flashrag", "qwen2024qwen2", "schulman2017proximal", "shao2024deepseekmath",
                   "deepseekai2025deepseek"}
DECIDE_PAR = "Claude, par délégation de l'étudiant (9 nov. 2026)"

# R3 appliquée aux candidats OpenAlex / Consensus (décision sur titre, remarque et, quand il existe, résumé).
CANDIDATS_INCLUS = {
    "arXiv:2505.07903": "Réduire les recherches inutiles par RL : cible directe de H3",
    "arXiv:2510.04695": "Découple chercher et répondre : la décision de chercher comme objet",
    "10.18653/v1/2025.emnlp-main.998": "Recherches sous-optimales (trop ou trop peu) : H2, H3",
    "10.18653/v1/2025.emnlp-main.1095": "Chercheur entraîné contre RAG naïf : comparaison à un harnais",
    "arXiv:2604.15148": "Récompense par gain d'information : lien avec le cadre et avec H3",
}
EQ3_DOUTES = {  # les trois doutes restants de l'équation 3
    "W7221758497": ("Inclus", "", "Contrôle stratégique de la recherche (titre ; résumé absent d'OpenAlex)"),
    "W7168272857": ("Inclus", "", "Critique les stratégies de recherche statiques ou heuristiques"),
    "W7155506905": ("Exclu", "E1", "Architecture multi-agents de recherche web ; résumé absent"),
}


def norm(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())[:60]


def bib_entries():
    s = BIB.read_text()
    out = {}
    for e in re.split(r"\n@", s):
        k = re.search(r"\{([^,]+),", e)
        if not k:
            continue
        t = re.search(r"title\s*=\s*\{+(.+?)\}+,", e, re.S)
        ep = re.search(r"eprint\s*=\s*\{([\d.]+)\}", e)
        y = re.search(r"year\s*=\s*\{(\d+)", e)
        out[k.group(1)] = {"titre": " ".join(t.group(1).split()) if t else "", "arxiv": ep.group(1) if ep else "",
                           "annee": y.group(1) if y else ""}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    args = ap.parse_args()
    meta = {w["id"].split("/")[-1]: w for w in json.load(open(args.meta))["results"]}

    final, vus = [], {}

    def ajouter(titre, annee, source, theme, regle, raison, arxiv=""):
        cle = arxiv or norm(titre)
        for k in (arxiv, norm(titre)):
            if k and k in vus:
                vus[k]["sources"] += f" ; {source}"
                return
        row = {"titre": titre, "annee": annee, "theme": theme, "regle": regle, "raison": raison,
               "sources": source, "cite_dans_article": "", "arxiv": arxiv}
        final.append(row)
        vus[cle] = row
        vus[norm(titre)] = row

    # R1
    corpus = list(csv.DictReader(open(LIT / "corpus.csv", encoding="utf-8-sig")))
    for x in corpus:
        t, p = x["theme"][:2], x["priorite"]
        if (t in {"T1", "T3", "T4", "T7", "T8"} and p == "essentiel") or (t == "T2" and p in {"essentiel", "utile"}):
            ajouter(x["titre"], x["annee"], "corpus de départ", x["theme"], "R1", x["role_dans_le_memoire"], x["arxiv_id"])
    # R2
    bib = bib_entries()
    cites = {k: v for k, v in bib.items() if k not in THEORIE_METHODE}
    for k, v in cites.items():
        theme = next((x["theme"] for x in corpus if (v["arxiv"] and x["arxiv_id"] == v["arxiv"]) or norm(x["titre"]) == norm(v["titre"])), "")
        ajouter(v["titre"], v["annee"], f"cité dans l'article ({k})", theme, "R2", "Cité dans la revue de l'article", v["arxiv"])
    for r in final:
        for k, v in cites.items():
            if (v["arxiv"] and v["arxiv"] == r["arxiv"]) or norm(v["titre"]) == norm(r["titre"]):
                r["cite_dans_article"] = k
    # R3 : candidats OpenAlex / Consensus
    p = LIT / "candidats_openalex_consensus.csv"
    cand = list(csv.DictReader(open(p, encoding="utf-8-sig")))
    for x in cand:
        if x["id_ou_doi"] in CANDIDATS_INCLUS:
            x["decision_finale"], x["motif_final"] = "Inclus", ""
            ajouter(x["titre"], x["annee"], "candidat OpenAlex/Consensus", x["theme_probable"], "R3",
                    CANDIDATS_INCLUS[x["id_ou_doi"]], x["id_ou_doi"].replace("arXiv:", "") if x["id_ou_doi"].startswith("arXiv:") else "")
        else:
            x["decision_finale"] = "Exclu"
            x["motif_final"] = x["motif_exclusion"] or "E1"
        x["decide_par"] = DECIDE_PAR
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cand[0].keys())); w.writeheader(); w.writerows(cand)
    # R3 : équation 3
    p = LIT / "tri_equation3_proposition.csv"
    tri = list(csv.DictReader(open(p, encoding="utf-8-sig")))
    kid = [c for c in tri[0] if "openalex_id" in c][0]
    for x in tri:
        i, prop = x[kid], x["proposition_v2"]
        if i in EQ3_DOUTES:
            dec, mot, com = EQ3_DOUTES[i]
            x["commentaire_v2"] = com
        elif prop == "Inclus (proposition)":
            dec, mot = "Inclus", ""
        elif prop in ("Déjà au corpus", "Déjà candidat"):
            dec, mot = prop, ""
        else:
            dec, mot = "Exclu", x["motif_v2"] or "E1"
        x["decision_etudiant"] = ""
        x["decision_finale"], x["motif_final"], x["decide_par"] = dec, mot, DECIDE_PAR
        if dec == "Inclus":
            w_ = meta[i]
            ax = (w_.get("doi") or "").split("arxiv.")[-1] if "arxiv" in (w_.get("doi") or "") else ""
            ajouter(w_["title"], str(w_["publication_year"]), "équation 3 (OpenAlex)", "", "R3",
                    x["commentaire_v2"] or "Inclus après lecture du résumé", ax)
    fields = [c for c in tri[0].keys() if c != "decision_etudiant"]
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(tri)

    # Effectifs du compte rendu de sélection, calculés (jamais saisis à la main) -> docs/article/genere/selection.tex
    from collections import Counter
    final_cles = {r["arxiv"] for r in final if r["arxiv"]} | {norm(r["titre"]) for r in final}
    dans_final = lambda ax, t: (ax and ax in final_cles) or norm(t) in final_cles  # noqa: E731
    c_r1 = sum(1 for x in corpus if x["theme"][:2] in {"T1", "T3", "T4", "T7", "T8"} and x["priorite"] == "essentiel"
               or x["theme"][:2] == "T2" and x["priorite"] in {"essentiel", "utile"})
    c_meth = [x for x in corpus if x["theme"][:2] in {"T5", "T6", "T9"}]
    c_reste = [x for x in corpus if x["theme"][:2] not in {"T5", "T6", "T9"}]
    c_repeche = sum(1 for x in c_reste if dans_final(x["arxiv_id"], x["titre"])) - c_r1
    c_prio = len(c_reste) - c_r1 - c_repeche
    cand_exc = Counter(x["motif_final"] for x in cand if x["decision_finale"] == "Exclu")
    tri_exc = Counter(x["motif_final"] for x in tri if x["decision_finale"] == "Exclu")
    deja = sum(1 for x in tri if x["decision_finale"] in ("Déjà au corpus", "Déjà candidat"))
    hors_listes = sum(1 for r in final if r["regle"] == "R2" and r["sources"].startswith("cité")
                      and not any(dans_final(x["arxiv_id"], x["titre"]) and (x["arxiv_id"] == r["arxiv"] or norm(x["titre"]) == norm(r["titre"])) for x in corpus))
    lignes = [
        ("Corpus de départ (collecte thématique, arXiv)", len(corpus)),
        ("\\quad retenus par R1", c_r1),
        ("\\quad repêchés par R2 (cités dans l'article)", c_repeche),
        ("\\quad hors revue : algorithmes, modèles, jeux de données, évaluation (références de méthode)", len(c_meth)),
        ("\\quad non retenus : priorité « contexte », ou « utile » hors thème du coût", c_prio),
        ("Candidats OpenAlex et Consensus", len(cand)),
        ("\\quad inclus (R3)", sum(1 for x in cand if x["decision_finale"] == "Inclus")),
        ("\\quad exclus " + ", ".join(f"{k} : {v}" for k, v in sorted(cand_exc.items())), sum(cand_exc.values())),
        ("Résultats de l'équation 3", len(tri)),
        ("\\quad déjà présents dans les listes précédentes", deja),
        ("\\quad inclus (R3)", sum(1 for x in tri if x["decision_finale"] == "Inclus")),
        ("\\quad exclus " + ", ".join(f"{k} : {v}" for k, v in sorted(tri_exc.items())), sum(tri_exc.values())),
        ("Articles cités venus d'autres requêtes (sémantique, texte intégral)", hors_listes),
    ]
    tex = ["\\begin{tabular}{@{}p{10.4cm}r@{}}", "\\toprule", "Étape & Articles\\\\", "\\midrule"]
    tex += [f"{a} & {b}\\\\" for a, b in lignes]
    inclus_parts = c_r1 + c_repeche + sum(1 for x in cand if x["decision_finale"] == "Inclus") \
        + sum(1 for x in tri if x["decision_finale"] == "Inclus") + hors_listes
    tex += [f"Inclus avant fusion des doublons entre sources & {inclus_parts}\\\\",
            f"\\quad doublons fusionnés (même article trouvé par plusieurs voies) & {inclus_parts - len(final)}\\\\"]
    tex += ["\\midrule", f"Corpus final de la revue & {len(final)}\\\\",
            f"\\quad dont cités dans l'article & {sum(1 for r in final if r['cite_dans_article'])}\\\\",
            "\\bottomrule", "\\end{tabular}"]
    Path("docs/article/genere/selection.tex").write_text("% Généré par scripts/selection_corpus.py\n" + "\n".join(tex) + "\n")
    for a_, b_ in lignes:
        print(f"  {b_:4}  {a_}")

    out = LIT / "corpus_final.csv"
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["titre", "annee", "theme", "regle", "raison", "sources", "cite_dans_article", "arxiv"])
        w.writeheader(); w.writerows(final)
    from collections import Counter
    print(f"Corpus final : {len(final)} articles ; par règle {dict(Counter(r['regle'] for r in final))} ;"
          f" cités dans l'article : {sum(1 for r in final if r['cite_dans_article'])}")
    print("Candidats OpenAlex/Consensus :", dict(Counter(x["decision_finale"] for x in cand)))
    print("Équation 3 :", dict(Counter(x["decision_finale"] for x in tri)))


if __name__ == "__main__":
    main()
