# Journal des requêtes OpenAlex et Consensus

Étape 5.2 de la trame. Date : 3 novembre 2026. Outils : OpenAlex (connecteur MCP), Consensus (recherche sémantique).

## Question en trois blocs

| Bloc | Contenu |
| --- | --- |
| Objet étudié | agents de recherche (LLM) entraînés par apprentissage par renforcement, face aux pipelines RAG fixes |
| Mesure | décisions de recherche (quand, combien), précision (EM/F1), coût en appels |
| Population | petits modèles de langage, questions-réponses ouvertes et multi-sauts |

Articles témoins : Search-R1 (2503.09516), R1-Searcher (2503.05592), ReAct (2210.03629), IRCoT (2212.10509), Self-RAG (2310.11511).

## Journal

| N° | Outil | Équation ou requête | Nombre | Précision (échantillon aléatoire de 20) | Ce que j'ai changé et pourquoi |
| --- | --- | --- | --- | --- | --- |
| 1 | OpenAlex `search_works` (aperçu) | (RL OU GRPO OU policy optimization) ET (search agent OU agentic search OU retrieval-augmented OU search engine OU adaptive retrieval) ET (LLM OU language model), depuis 2022 | 1 036 | 7/20 (35 %) | Trop de bruit : réseaux 6G, médecine, psychiatrie. Ajout d'un bloc « tâche » (question answering, multi-hop, knowledge-intensive) |
| 2 | idem | bloc 1 + ET (question answering OU multi-hop OU knowledge-intensive OU open-domain) ; ajout de RLVR, agentic RAG, tool use | 336 | 7/20 strict (35 %), 10/20 en comptant les cas limites | Le nombre baisse mais la précision stagne : bruit multimodal (VQA), télédétection, parole. Remplacement de « tool use » et « open-domain » (trop larges), exclusion des termes parasites |
| 3 | idem | (RL OU GRPO OU RLVR) ET (search agent OU agentic search OU search-augmented OU search engine OU adaptive retrieval OU retrieval-augmented reasoning) ET (question answering OU multi-hop OU HotpotQA OU Natural Questions) ET (LLM) SAUF (multimodal, visual question, remote sensing, network), depuis 2023 | 103 | environ 15/20 (75 %) | Équation retenue. Les 83 résultats non tirés n'ont pas été triés : à faire (voir ci-dessous) |
| 4 | OpenAlex `resolve_references` | 91 DOI arXiv du corpus, 4 lots | 91 | 91 correspondances exactes | Trois lots lancés en parallèle ont été refusés (limite de 30 requêtes par seconde) : relancés un par un. Aucune référence incertaine ni rétractée |
| 5 | Consensus `search` | « Does reinforcement learning training improve small language model search agents compared with fixed retrieval-augmented pipelines on question answering? » | 10 | 10/10 sur le sujet | 4 articles absents du corpus (SEM, DeSA, DynaSearcher, Search-GRT) ajoutés aux candidats |
| 6 | OpenAlex `list_citations` | articles citant Search-R1 | 14 | 4/14 pertinents | Boule de neige peu exploitable : OpenAlex rattache à la notice arXiv seulement 14 citations (Consensus en compte plus de 1 600). 2 candidats retenus (SmartSearch, Tool-Star) |
| 7 | OpenAlex `search_works` (titre) | SEM, DeSA, DynaSearcher, UR2 | 0 / 1 / 1 / 1 | | SEM introuvable sous son titre : OpenAlex n'a qu'une copie Zenodo douteuse (revue IJTLS, auteurs incohérents). Identifiant arXiv retrouvé par recherche web (2505.07903) |

## Constats

- **Vérification.** Les 91 références du corpus existent dans OpenAlex et se résolvent par DOI. 29 ont une version publiée (colonnes `doi_version_publiee` et `venue_publiee` de `corpus.csv`). Les autres, dont ReAct, Self-RAG, Search-R1 et le RAG de Lewis et al., n'ont que la notice arXiv dans OpenAlex : cela ne prouve pas qu'elles ne sont pas publiées ailleurs. Les DOI des conférences 2026 sont ceux renvoyés par OpenAlex ; leur format est inhabituel, à vérifier en cliquant avant la bibliographie finale.
- **Titre changé.** 2502.13957 (RAG-Gym) est publié dans Nature Communications sous le titre « Supervising the search process produces reliable and generalizable information-seeking agents ».
- **Consensus.** L'outil renvoie des articles classés avec leur résumé, pas la répartition « oui / peut-être / non » du Consensus Meter. Les 10 résultats rapportent tous un gain de l'entraînement par RL ou un gain d'efficacité. C'est un biais de sélection probable : les articles qui proposent une méthode publient des résultats positifs. Les études qui comparent honnêtement le RL à des harnais fixes bien réglés sont rares, ce qui justifie la question du mémoire.
- **Articles témoins.** Seul Search-R1 a été retrouvé dans un échantillon de l'équation 3. Le rappel sur les cinq témoins n'a pas été testé.
- **Tri.** `candidats_openalex_consensus.csv` contient 32 lignes : 23 candidats (8 « Inclus (provisoire) », 15 « Doute ») et 9 exclus avec motif E1/E2. Décisions fondées sur le titre et le résumé seulement.

## Reste à faire

1. Trier à la main les 83 résultats restants de l'équation 3 (export OpenAlex, colonne Décision : Inclus / Exclu / Doute, motifs E1 à E5).
2. Lire les résumés des lignes « Doute » et trancher.
3. Fusionner les « Inclus » dans `corpus.csv` après lecture, avec leur clé BibTeX.
4. Tester le rappel de l'équation sur les cinq témoins.
