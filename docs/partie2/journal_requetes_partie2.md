# Partie 2 : journal des requêtes

Étape 5.2 de la trame, appliquée à la question de la partie 2. Requêtes lancées par Claude (OpenAlex, Consensus,
index arXiv de Hugging Face, recherche web) les 11 et 12 novembre 2026.

## Question en trois blocs

| Bloc | Contenu |
| --- | --- |
| Ce qu'on étudie | l'entraînement par renforcement (RL, RLVR, GRPO, PPO) |
| Ce qu'on mesure | la frontière de capacité : pass@k à grand k, couverture des solutions, « au-delà du modèle de départ » |
| Chez qui | les modèles de langage, en particulier les agents qui utilisent un moteur de recherche ou des outils |

Articles témoins : Yue et al. (2504.13837), Zhai et al. (2604.14877), Yang (2609.22547), ProRL (2505.24864),
Search-R1 (2503.09516).

## Journal

| N° | Outil | Requête | Nombre | Précision (20 au hasard) | Décision |
| --- | --- | --- | --- | --- | --- |
| P0 | Vérification (arXiv, NeurIPS, HF, web) | Les 16 références de la proposition initiale | 16 | 2 introuvables, 2 affirmations corrigées | Note corrigée (`note_litterature.md`, dernière section) |
| P1 | OpenAlex (OQL) | (RL OU RLVR OU GRPO OU « verifiable reward ») ET (pass@k OU « capability boundary » OU « reasoning boundary » OU « solution coverage » OU « beyond the base model » OU « reasoning capacity ») ET (LLM OU « language model »), depuis 2024 | 302 | environ 10 à 12 sur 20 (50 à 60 %) | Équation retenue pour le débat général |
| P2 v1 | OpenAlex (OQL) | bloc 1 ET (pass@k OU frontière OU exploration) ET (agent* OU « tool use » OU « multi-turn ») ET LLM | 642 | environ 4 sur 20 | Bruit : « exploration » et agent* ramènent robotique, StarCraft, multi-agents |
| P2 v2 | OpenAlex (OQL) | bloc 1 ET (pass@k OU frontière OU « base model ») ET (« search agent* » OU « LLM agent* » OU « tool-integrated » OU agentic OU « tool use » OU « retrieval-augmented ») | 138 | environ 2 à 3 sur 20 | « base model » ne filtre rien ; la littérature agentique sur la frontière est très mince. Passage à la boule de neige |
| B1 | OpenAlex `list_citations` | Articles citant Zhai et al. (2604.14877) | 6 | 3 sur 6 très pertinents | Zhu et al. (2611.06483), Chen et al. (effondrement de l'exploration), Kim et al. (Elicitation or Expansion?) |
| B2 | OpenAlex `list_citations` | Articles citant Yue et al. ET (agent OU outil OU recherche OU multi-tour) dans le titre ou le résumé | 127 (50 lus) | environ 8 sur 50 | « Atomic Skills », « When RLVR Shrinks », « Expanding LLM Agent Boundaries », RAPO agents |
| S1 | OpenAlex sémantique | « Le RL d'un agent de recherche élargit-il l'ensemble des questions résolues (pass@k) ou seulement le premier essai ? » | 50 (25 lus) | — | Aucun article ne mesure pass@k sur Search-R1 face à son modèle de départ |
| S2 | Index arXiv (HF Papers) | 4 requêtes : Search-R1 + pass@k ; frontière des agents de recherche ; élicitation ou élargissement ; pièges de pass@k | 32 | — | ProRL, MATH-Beyond, « From Base Rollouts », articles de méthodologie statistique |
| C1 | Consensus | « Le RLVR élargit-il la frontière de raisonnement au-delà du modèle de départ à grand pass@k ? » | 10 | 10 sur 10 | 1 « non » (étude d'évaluation), 9 « oui » dont 7 proposent une nouvelle méthode |
| C2 | Consensus | « Le RL élargit-il les capacités des agents à outils ou à recherche face au modèle de départ ? » | 10 | 10 sur 10 | Tous « oui », mais 9 sur 10 mesurent le premier essai ou une moyenne ; seul Zhai et al. mesure la frontière |
| V1 | OpenAlex `resolve_references` | Les 43 articles du corpus (lots de 25 et 13, plus 4 pivots) | 43 | 42 exactes, 1 incertaine | Zéro introuvable. Incertaine : Cheng et al. (Guru, NeurIPS 2025), titre et auteurs concordants |

OQL de l'équation P1 :

```
works where title-abstract-keywords has (("reinforcement learning" or RLVR or GRPO or "verifiable reward*")
and ("pass@k" or "capability boundary" or "reasoning boundary" or "solution coverage" or "beyond the base model"
or "reasoning capacity") and (LLM or "language model*")) and published since 2024
```

## Constats

- **La littérature sur la frontière des agents est très mince.** Deux études seulement (Zhai et al., Zhu et al.),
  toutes deux sur des agents entraînés par leurs auteurs ; aucune sur des checkpoints publics d'agents de recherche,
  aucune avec une inférence appariée. Le faible rendement de l'équation P2 en est un symptôme, pas un défaut à corriger.
- **Biais de publication, encore.** Sur la question générale, presque tous les « oui » viennent d'articles qui
  proposent une nouvelle méthode et la comparent au RL standard. Leur message implicite est que le RL standard
  (celui de Search-R1) n'élargit pas, ou peu. Sur les agents, les « oui » mesurent le premier essai, pas la frontière.
- **Pas de recoupement direct.** Le plus proche (Zhu et al., novembre 2026) porte sur du code, des appels de fonctions,
  du SQL et FrozenLake, avec Qwen3 entraîné par les auteurs, sans recherche documentaire ni test apparié.

## Corpus

`corpus_partie2.csv` : 43 articles en 5 thèmes (débat sans outil 11, agents 9, statistique de pass@k 9, méthodes pour
élargir 7, supervisé contre RL 7), avec la position de chacun dans le débat. Tri fondé sur les résumés ; seuls Zhai
et al., Yang et Zhu et al. ont été lus dans le texte (sections de méthode et de résultats).
