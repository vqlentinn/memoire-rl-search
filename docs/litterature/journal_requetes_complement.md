# Journal des requêtes : complément du 3 novembre 2026 (après-midi)

Complète `journal_requetes_openalex.md` (même dossier).

| N° | Outil | Requête | Nombre | Résultat | Ce que j'ai changé et pourquoi |
| --- | --- | --- | --- | --- | --- |
| 8 | OpenAlex `resolve_references` | 35 références de `docs/article/refs.bib` (lots de 25 et 10) | 35 | 35 exactes, 0 rétractée | 9 entrées citaient un préprint alors qu'une version publiée existe : remplacées (règle 9 de l'étape 5.2) |
| 9 | OpenAlex `search_works` (OQL) | Équation 3 réécrite depuis sa description dans le journal | 116 | La ligne 3 du journal indiquait 103 | L'équation exacte de la ligne 3 n'avait pas été conservée. Je garde 116, avec l'OQL ci-dessous |
| 10 | Équation 3, liste complète | Rappel sur les 5 témoins | 116 | 1 sur 5 : Search-R1 | ReAct, IRCoT et Self-RAG ne relèvent pas du bloc RL. Le résumé de R1-Searcher parle de « external search systems » et ne cite pas la tâche de questions-réponses. Déjà au corpus : pas de nouvelle version de l'équation |
| 11 | Équation 3, liste complète | Tri des 116 sur les titres | 116 | 4 inclus proposés, 58 doutes, 38 exclus, 11 déjà candidats, 5 déjà au corpus | Proposition de Claude : la décision revient à l'étudiant (`tri_equation3_proposition.csv`) |
| 12 | OpenAlex `search_works` (sémantique) | Comparaison empirique d'agents RL et de harnais fixes, à retriever et corpus identiques | 50 (15 lus) | Zhao et al. 2026, Pezzuti et al. 2026 : absents de l'équation 3 | Ajoutés après lecture du résumé, car ils touchent à la lacune et à la surrecherche |
| 13 | OpenAlex `resolve_references` | 6 références ajoutées à `refs.bib` | 6 | 6 exactes | Raikote 2026 : apparié sur le titre, pas de DOI |
| 14 | OpenAlex `search_works` (texte intégral) | « information foraging » ET agents de recherche ET LLM, depuis 2023 | 58 | 12 titres lus, 5 PDF contrôlés | Dans 4 PDF sur 5, le terme vient d'une seule référence citée (InForage). InForage mobilise vraiment la théorie |
| 15 | OpenAlex `search_works` (texte intégral) | Modèles économiques, satisficing ou rationalité limitée ET agents de recherche ET RL | 57 | Aucune mention de Stigler, d'Azzopardi ni de satisficing dans les 10 PDF contrôlés | Simon retiré du cadre |
| 16 | OpenAlex `resolve_references` | Pirolli et Card 1999, Azzopardi 2011, Holm 1979, Qian et Liu 2025, Zhang et al. 2026, Poudel et al. 2026 | 6 | 6 exactes | Ajoutés à `refs.bib` |
| 17 | OpenAlex `resolve_references` (9 novembre) | Les 46 références de `refs.bib`, par DOI ou par titre (lots de 25 et 21) | 46 | 43 exactes ; 3 non trouvées par le titre (R1-Searcher, apparié à tort à R1-Searcher++ ; la synthèse de Lin et al. ; ParallelSearch), toutes exactes par leur DOI arXiv. 0 rétractée | Vérification finale de l'étape 12 : zéro rouge. DOI arXiv ajoutés aux 3 entrées. 6 entrées passées à leur version publiée signalée par OpenAlex (FlashRAG, Adaptive-RAG, Active RAG, Mallen et al., InForage, Zhang et al.) |

OQL de l'équation 3 :

```
works where title-abstract-keywords has (("reinforcement learning" or GRPO or RLVR) and ("search agent" or "agentic search" or "search-augmented" or "search engine" or "adaptive retrieval" or "retrieval-augmented reasoning") and ("question answering" or "multi-hop" or HotpotQA or "Natural Questions") and (LLM or "language model") and not (multimodal or "visual question" or "remote sensing" or network)) and published since 2023
```

## Limites

- Tri fait sur les titres, sauf les 4 lignes « Inclus », dont le résumé a été lu. Aucun texte lu en entier.
- Une requête sémantique a trouvé deux articles importants que l'équation avait manqués : le rappel réel de l'équation reste inconnu.
- Non cités dans l'article à ce stade : les 58 doutes. Les lire peut changer la lacune annoncée.
