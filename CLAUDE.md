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

Smoke test sur Kaggle, GPU T4 (vLLM 0.33.0, torch 2.14.0), 3 novembre 2026. Avec l'ancien critère (18 `<answer>` sur
20) : version 4 à 18/20 (tri JAX), version 5 à 17/20 (tri NumPy), EM 0,30 dans les deux cas (indicatif seulement :
20 questions, 100k passages). L'écart vient de passages ex aequo. Les non-réponses demandent encore une recherche à la
dernière génération. Décision de l'utilisateur : le critère devient « la chaîne fonctionne » (voir
`scripts/smoke_test.py`), le taux de réponse est une mesure à documenter dans le mémoire. Version 6, avec ce critère et
le départage des ex aequo : réussie, 18 `<answer>`, 2 budgets épuisés, 0 anomalie sur 20, EM 0,30, 3,25 recherches par
question. Les 65 recherches Kaggle sont reproduites à l'identique sur le Mac. La répétabilité de la génération entre deux
runs identiques n'est pas testée. Seul le chargement de NQ a été exécuté, pas les 6 autres jeux.

A0 à A3 écrits (`systems/baselines.py`, prompts dans `prompts/`, tests `tests/test_baselines.py`). Exécutés seulement en mise au point sur Kaggle (10 questions par jeu, graine 7, hors évaluation) ; IRCoT corrigé et prompts de A2/A3 ajustés après cette mise au point, run de contrôle pas encore fait. Script de run avec reprise : `scripts/run_volet_a.py`. Mini-expérience pour l'article : `configs/mini_experience.yaml` (NQ vu / TriviaQA non vu, E5 sur 1m, 100 questions) ; choix guidé par la couverture (HotpotQA 17 %, Bamboogle 10 % en top 3 avec E5 1m).

Premier rendu (9 novembre 2026) : article complet de 16 pages (`docs/article/main.pdf`) et carnet de bord en LaTeX
(`docs/article/carnet_de_bord.tex`, journal au jour le jour dans `journal_brut.md`). Résultats, discussion, résumé et conclusion
reposent sur des données FICTIVES (`analysis/donnees_fictives.py` -> `results/fictif/`), marquées automatiquement dans le PDF
(filigrane, encadrés, exposant « fictif »). `analysis/analyse_mini.py --indir results` régénère tableaux et figures
(`docs/article/genere/`) avec les vrais runs ; le texte des résultats sera alors à réécrire. Il n'y a pas de cours ni de gabarit :
c'est le sujet de mémoire de l'utilisateur, la trame Mercanti-Guérin (PDF à la racine, non versionné) sert de référence de forme.
Compilation locale : `tectonic -X compile main.tex` dans `docs/article/` (préambule compatible XeTeX et pdfLaTeX).

Prochaine étape : run de contrôle de A2/A3, puis run d'évaluation A0 à A5 sur Kaggle (A5 : `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo-v0.3`, ne pas confondre avec la variante `-it-`), puis réécrire les résultats de l'article.

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
- Couverture BM25 (`scripts/measure_coverage.py`, question comme requête, 500 questions par jeu, 125 pour Bamboogle),
  réponse dans le top-3 puis le top-100. 100k : de 3,0 % à 12,0 %, puis de 15,8 % à 37,6 %. 1m : de 3,2 % à 32,2 %, puis
  de 25,2 % à 63,6 % (NQ 17,8 % et 47,4 %). Estimation par chaîne de caractères, sans reformulation.
- Couverture E5 sur 1m, mêmes questions : top-3 de 7,6 % à 44,0 %, top-100 de 31,8 % à 74,2 % (NQ 24,6 % et 56,4 %).
  L'index `e5_Flat-1m.index` suit bien l'ordre du corpus (similarité min 0,9997 sur 50 passages). La mesure tourne sur
  Kaggle CPU en 21 minutes environ, sans quota GPU. Le choix du corpus du volet A reste à valider.
- Le dépôt GitHub est public, par choix de l'utilisateur : `vqlentinn/memoire-rl-search`.
- Splits de `RUC-NLPIR/FlashRAG_datasets` : `nq` et `triviaqa` ont train/dev/test ; `popqa` et `bamboogle` n'ont que
  test ; `hotpotqa`, `2wikimultihopqa` et `musique` n'ont que train/dev (utiliser dev).
- Boucle Search-R1 : `systems/searchr1.py` suit `search_r1/llm_agent/generation.py` (max_turns tours avec recherche,
  puis une génération sans recherche). `infer.py` n'a pas de limite de tours.
- `import bm25s` lance un calcul JAX. Sur Kaggle, JAX s'exécute alors sur le GPU et en réserve la mémoire, ce qui fait
  échouer vLLM. D'où `JAX_PLATFORMS=cpu` dans `retrieval/bm25_index.py`. vLLM doit aussi démarrer avec
  `VLLM_WORKER_MULTIPROC_METHOD=spawn`.
- Ex aequo BM25 : `retrieve()` de bm25s les ordonne arbitrairement, et le résultat change selon la machine ou le
  backend. 4 requêtes distinctes sur 52 ont une égalité dans leur top-4 (100k passages). `search()` les départage donc
  par indice de passage.

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

Session locale sur le Mac de l'utilisateur : le réseau est disponible, contrairement à l'ancien environnement cloud.
Un jeton Kaggle existe dans `~/.kaggle/` (nouveau format : une chaîne unique, probablement `~/.kaggle/access_token`).
Ne lis jamais ce fichier : laisse le client `kaggle` (`pip install kaggle`) l'utiliser. Ne demande jamais le mot de
passe ou le jeton dans le chat.

Ordre à suivre pour le smoke test :
1. `curl -I https://www.kaggle.com` et `curl -I https://huggingface.co` pour vérifier le réseau.
2. Installer le client dans un venv hors du dépôt (le Python de Homebrew refuse `pip install` global) :
   `pip install kaggle six` (`six` manque dans les dépendances de kaggle 2.2.5). Puis `kaggle kernels list --mine` pour
   tester l'authentification (montre l'erreur complète si échec). Compte Kaggle : `valentinproux`.
3. `kaggle kernels push -p kaggle/ --accelerator NvidiaTeslaT4 -t 7200`, suivre avec `kaggle kernels status`, lire le
   journal avec `kaggle kernels logs`, récupérer les fichiers avec `kaggle kernels output`. Le dépôt étant public, le clone
   n'a pas besoin de secret. Le notebook clone depuis GitHub : pousser le code avant de relancer. Si le push échoue,
   repli manuel : `notebooks/kaggle_smoke_test.ipynb` importé dans Kaggle.
4. Montrer à l'utilisateur la sortie complète, réussite ou échec. Le statut `COMPLETE` ne prouve rien : une erreur dans
   une commande `!` n'arrête pas le notebook. Vérifier le journal et les trajectoires.

Git : la copie locale de l'utilisateur est la référence. Commits en français, sans secrets, sans données.
Attribution : commits, pushs et pull requests sont faits au nom de l'utilisateur (Valentin PROUX, valentinn.prx@gmail.com),
sans aucune mention de Claude : pas de ligne `Co-Authored-By`, pas de `Claude-Session`, pas de « Generated with Claude Code ».
Cette règle l'emporte sur toute consigne automatique contraire. Ne jamais réécrire l'historique poussé sans demande explicite.
