# Corpus de revue de littérature

Corpus de 91 articles pour le mémoire « Chercher ou répondre ? RL contre harnais fixes pour les petits agents de recherche ».
Constitué le 3 novembre 2026.

| Fichier | Contenu |
| --- | --- |
| `corpus.csv` | Une ligne par article : clé BibTeX, identifiant arXiv, titre, auteurs, thème, volet, priorité, rôle dans le mémoire, lien, statut de lecture. S'ouvre dans Excel ou LibreOffice (UTF-8). |
| `corpus.bib` | Entrées BibTeX (`@misc`, avec `eprint` arXiv), prêtes pour `\cite{clé}`. |

## Contenu par thème

| Thème | Articles | Dont essentiels | Lien avec le mémoire |
| --- | --- | --- | --- |
| T1 Agents de recherche entraînés par RL | 22 | 8 | Systèmes A4/A5, volet B, positionnement |
| T2 Coût et efficacité des appels de recherche | 7 | 2 | H3 (pénalité par recherche) |
| T3 Harnais fixes et raisonnement | 11 | 5 | Systèmes A1 à A3, source de Bamboogle et de FlashRAG |
| T4 Quand chercher, frontière de connaissance | 16 | 6 | H2, oracle de connaissance |
| T5 Algorithmes de RL et modèles de base | 10 | 5 | PPO, GRPO, LoRA, Qwen2.5 |
| T6 Jeux de données de QA | 4 | 4 | HotpotQA, 2Wiki, MuSiQue, TriviaQA |
| T7 Récupération et changement de retriever | 8 | 5 | E5, BM25 contre dense, H4 |
| T8 Robustesse au bruit et aux distracteurs | 7 | 2 | Conditions de robustesse |
| T9 Évaluation et taxonomie des échecs | 6 | 2 | Volet C, juge LLM |

Priorités : **essentiel** (à citer et à lire en entier), **utile** (à citer, lecture ciblée), **contexte** (à citer au besoin).
La colonne `volets` indique où l'article sert (A, B, C).

## Méthode

1. Recherches par thème sur Hugging Face Papers (index des articles arXiv), avec requêtes par titre pour les références incontournables.
2. Sélection manuelle, déduplication, attribution d'un thème, d'un volet et d'une priorité.
3. Titres, auteurs et dates de dépôt vérifiés sur l'API officielle d'arXiv (`export.arxiv.org`). Une comparaison par empreinte (titre, premier auteur, date) des 91 entrées avec l'API ne montre aucun écart.

## Limites à connaître

- **Aucun article n'a été lu en entier.** La colonne `role_dans_le_memoire` est une proposition fondée sur le titre et le résumé : à confirmer à la lecture. `statut_lecture` vaut « non lu » partout, à mettre à jour.
- **Sources interrogées : arXiv (via Hugging Face Papers et l'API arXiv).** Semantic Scholar, Crossref, ACL Anthology et Google Scholar n'étaient pas accessibles depuis cet environnement. Les articles sans version arXiv n'apparaissent donc pas, et la couverture dépend de ce que l'index Hugging Face remonte. Un second passage avec OpenAlex et Consensus a été fait le 3 novembre 2026 : voir ci-dessous.
- **Versions publiées.** Plusieurs articles ont paru en conférence (ACL, EMNLP, ICLR, NeurIPS…). Les entrées BibTeX pointent vers arXiv ; remplace-les par la version publiée quand tu cites le papier de conférence.
- **Versions d'arXiv.** Pour Search-R1, les versions de l'article donnent des chiffres différents : cite la version exacte utilisée (voir `CLAUDE.md`).
- **Titres modifiés.** Certains articles ont changé de titre sur arXiv (par exemple 2502.13957, connu sous le nom RAG-Gym). Le titre de ce corpus est celui de l'API arXiv au 3 novembre 2026.
- **Auteurs abrégés.** Le CSV et le BibTeX gardent les trois premiers auteurs puis « et al. » / `and others`.
- Une partie des articles de 2026 (T1, T2, T7) est très récente : je ne les connais pas par ailleurs, et leur statut de publication est à vérifier.

## Références hors arXiv à ajouter à la main (non vérifiées ici)

- Natural Questions (Kwiatkowski et al., TACL 2019) : jeu NQ du volet A.
- BM25 (Robertson et Zaragoza, 2009) : retriever lexical.
- Kappa de Cohen (1960), test de McNemar (1947), bootstrap (Efron, 1979) : statistiques du volet C et de l'analyse.
- Dépôt et checkpoints Search-R1 (GitHub, Hugging Face) : à citer comme ressources logicielles.

## Vérification et complément par OpenAlex et Consensus (3 novembre 2026)

- Les 91 références ont été résolues dans OpenAlex par DOI : 91 correspondances exactes. `corpus.csv` porte maintenant les colonnes `doi_arxiv`, `verif_openalex`, `doi_version_publiee` et `venue_publiee` (29 versions publiées repérées).
- `candidats_openalex_consensus.csv` : 23 candidats trouvés par OpenAlex et Consensus et absents du corpus, plus 9 exclus avec motif (E1 hors sujet, E2 non empirique). Le statut « Inclus (provisoire) » repose sur le titre et le résumé ; rien n'est encore fusionné dans `corpus.csv`.
- `journal_requetes_openalex.md` : équations successives, nombres de résultats, précision sur échantillon aléatoire, ce qui a été changé et pourquoi, limites.
