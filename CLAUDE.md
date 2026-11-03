# CLAUDE.md : contexte du projet pour Claude Code

Lis ce fichier en entier avant d'agir. Réponds toujours en français à l'utilisateur.

## L'utilisateur et l'objectif

Étudiant en master d'IA (M2), niveau PyTorch intermédiaire, plutôt théorique, prêt à apprendre sur le tas.
Il écrit un mémoire de 40 à 50 pages. **Budget : 0 €.** Calcul uniquement sur les GPU gratuits de Kaggle
(2×T4 ou 1×P100, environ 30 h par semaine, sessions de 9 à 12 h, 20 Go de sortie). Pas de crédits API.
Explique brièvement ce que tu fais : l'utilisateur veut comprendre le code, pas seulement l'obtenir.

## Sujet du mémoire

> À corpus et questions identiques, un agent de recherche entraîné par RL (Search-R1) décide-t-il mieux
> *quand* et *quoi* chercher qu'un harnais fixe (RAG, ReAct, IRCoT), et à quel coût ?

Hypothèses : H1 les gains du RL sont surtout sur les jeux vus à l'entraînement (NQ, HotpotQA) ; H2 le modèle RL
cherche presque toujours ; H3 une pénalité par recherche réduit les appels pour une perte d'EM limitée ;
H4 les gains du RL sont plus sensibles au changement de retriever que ceux d'un harnais.
Un résultat négatif est acceptable. Ce n'est **pas** une reproduction de Search-R1 : les scores absolus ne
seront pas comparables à ceux de l'article (corpus réduit), seule la comparaison relative compte.

## Volets

- **A (cœur, inférence seule) :** 6 systèmes. A0 réponse directe, A1 RAG classique, A2 ReAct par prompt,
  A3 IRCoT, A4 Search-R1 Qwen2.5-3B PPO v0.3, A5 Search-R1 Qwen2.5-3B GRPO v0.3. Sept jeux : NQ, TriviaQA,
  PopQA, HotpotQA, 2WikiMultiHopQA, MuSiQue, Bamboogle. 500 questions par jeu, graine 42, décodage glouton.
  Mêmes questions, même corpus, même retriever, top-3 passages pour tous les systèmes.
- **B (exploratoire) :** mini-GRPO avec TRL (`GRPOTrainer` + `environment_factory`), Qwen 0,5B ou 1,5B, LoRA,
  récompenses EM, EM − 0,1 × nb de recherches. Décision go/no-go en semaine 10.
- **C :** taxonomie des échecs sur 300 trajectoires (annotation manuelle de 100 + juge LLM open source, kappa).
- Oracle de connaissance (pour H2) : 8 échantillons sans recherche ; « connue » si au moins 6 sur 8 justes.
- Conditions de robustesse : BM25 au lieu de E5, 50 % de distracteurs en plus, passages gold retirés.
- Statistiques : bootstrap 1000 rééchantillonnages, McNemar apparié, kappa de Cohen.

## État actuel (3 novembre 2026)

Fait : squelette du dépôt, prompt exact de Search-R1, métriques EM/F1 (même normalisation que Search-R1),
statistiques, retriever BM25 (bm25s), chargement des jeux (FlashRAG), boucle de recherche vLLM (A4/A5),
script `scripts/smoke_test.py`, tests (`python -m pytest tests/` passe).

**Jamais exécuté sur GPU.** vLLM sur T4 n'a pas été testé. `data/load_qa.py` n'a pas été exécuté (noms de
splits à vérifier dans le dataset `RUC-NLPIR/FlashRAG_datasets`).

Prochaine étape : smoke test (20 questions NQ, checkpoint
`PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-ppo-v0.3`, sous-corpus 100k passages). Réussi si au moins
18 réponses sur 20 se terminent par `<answer>`. Ensuite : systèmes A0 à A3, construction du corpus, runs du volet A.

## Faits vérifiés (ne pas redécouvrir)

- Checkpoints Search-R1 : collection `PeterJinGo/search-r1-v03` (Qwen2.5 3B/7B/14B/32B, PPO et GRPO). Pas de
  checkpoint à 0,5 ou 1,5B trouvé : le 3B est la plus petite taille. Les versions de l'article donnent des chiffres
  différents, citer la version exacte.
- Corpus complet wiki-18 : 5,1 Go compressé, environ 132 Go avec l'index E5 : impossible sur Kaggle.
  Sous-corpus du dataset `maknee/wiki-18-subsets` : 10k, 100k (24 Mo, index 307 Mo), 1m (244 Mo, index 3 Go),
  5m (1,2 Go, index 15 Go). Utiliser 100k pour les tests, 1m comme cible du volet A (à valider).
- Format du prompt Search-R1 : `prompts/searchr1.txt`, copié de `infer.py` du dépôt officiel (Apache 2.0).
  Balises `<think>`, `<search>`, `<information>`, `<answer>`. Les modèles TRL du volet B utiliseront un autre
  format (function calling) : les évaluer chacun dans leur propre harnais.
- Le dépôt GitHub est privé : `vqlentinn/memoire-rl-search`.

## Règles du projet

- Chaque trajectoire est écrite en JSONL dans `results/` (question, système, texte complet, requêtes, documents,
  réponse, EM, F1, tokens, durée). Les analyses partent de ces fichiers, pas de notebooks en mémoire.
- Résultats écrits par lots : un run coupé doit reprendre là où il s'est arrêté.
- Les T4 n'ont pas de bf16 : fp16 ou quantification 4 bits.
- Graines et versions figées. Le protocole sera gelé en semaine 5 : ne pas modifier prompts ni métriques ensuite
  sans le dire explicitement.
- Ne jamais committer de données, d'index, de modèles ni de secrets (voir `.gitignore`). Pas de jeton dans le code.
- Ne rien affirmer sur des résultats non mesurés. Si une commande échoue, montre l'erreur complète à l'utilisateur.

## Kaggle depuis ici

Un jeton Kaggle existe dans `~/.kaggle/` sur la machine de l'utilisateur. Le client officiel est `pip install kaggle`.
Format du nouveau jeton et chemin exact : non confirmés, à tester. Lancer un notebook sur GPU par l'API demande un
`kernel-metadata.json` (`enable_gpu`, `enable_internet`) : non testé. Repli : `notebooks/kaggle_smoke_test.ipynb`,
à importer à la main dans Kaggle. Ne demande jamais le mot de passe Kaggle ou le jeton dans le chat.
