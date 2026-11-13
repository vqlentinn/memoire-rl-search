# CLAUDE.md : contexte du projet pour Claude Code

Lis ce fichier en entier avant d'agir. Réponds toujours en français à l'utilisateur.

## L'utilisateur et l'objectif

Étudiant en master d'IA (M2), niveau PyTorch intermédiaire, plutôt théorique, prêt à apprendre sur le tas.
Il écrit un mémoire de 40 à 50 pages. **Budget : 0 €.** Calcul uniquement sur les GPU gratuits de Kaggle
(2×T4 ou 1×P100, environ 30 h par semaine, sessions de 9 à 12 h, 20 Go de sortie). Pas de crédits API.
Explique brièvement ce que tu fais : l'utilisateur veut comprendre le code, pas seulement l'obtenir.

## Sujet du mémoire (réorienté le 12 novembre 2026)

> **Que gagne-t-on à entraîner un agent de recherche par renforcement ?** À modèle de départ identique, le
> renforcement élargit-il l'ensemble des questions que l'agent finit par résoudre, ou augmente-t-il seulement sa
> réussite au premier essai ? Cet élargissement apparaît-il quand la question oblige à enchaîner les recherches,
> et résiste-t-il à un test statistique apparié ?

Pourquoi ça compte : si le renforcement ne fait que fiabiliser, un fine-tuning supervisé ou une distillation des
bonnes traces obtient l'essentiel pour beaucoup moins cher. L'industrie (Fireworks, Trajectory, Prime Intellect)
parie sur le renforcement d'agents sans publier de quoi distinguer les deux. Une seule étude sur les agents annonce un
élargissement (Zhai et al., arXiv 2604.14877 : 7B, agent entraîné par les auteurs sur 200 questions, pas de test formel,
vérification sur Search-R1 non faite) ; sans outil, l'élargissement résiste rarement à l'inférence appariée
(Yang, arXiv 2609.22547). Note vérifiée : `docs/partie2/note_litterature.md`.

**Partie 1 (faite, article rendu) :** à conditions égales, Search-R1 bat les harnais fixes (RAG, ReAct, IRCoT), et
l'avantage se joue à l'usage de l'information trouvée, pas à la requête ni à la sobriété. Limite principale : modèles de
départ différents (base contre Instruct). C'est elle qui motive la partie 2.

**Partie 2 (cœur du mémoire) :** chaque checkpoint Search-R1 face à **son propre** modèle de départ, courbes pass@k
appariées. Hypothèses (règles de décision dans `docs/partie2/plan_analyse.md`) :
- H1 (contrôle) : au premier essai, l'agent bat son modèle de départ ;
- H2 (principale) : sur les questions de pont, l'écart D(k) reste positif quand k augmente (« élargissement ») ;
  sur les comparaisons et sans outil (T = 0), il se referme ou s'inverse ;
- H3 : à passages identiques, l'avantage vient de la lecture (échange des plans de recherche entre les deux modèles) ;
- H4 (option) : un fine-tuning supervisé LoRA sur les traces réussies de l'agent récupère la frontière.
Les trois issues de H2 sont publiables (élargissement confirmé, non confirmé, rétrécissement).
Un résultat négatif est acceptable. On n'entraîne rien, sauf l'option H4.

Anciennes hypothèses de la partie 1 devenues perspectives : pénalité par recherche (coût) et changement de retriever.

## Dispositif de la partie 2

- **Paires exactes (vérifiées sur Hugging Face) :** `Qwen/Qwen2.5-3B-Instruct` contre
  `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-{ppo,grpo}-v0.3` (principales) ; `Qwen/Qwen2.5-3B` contre
  `…-qwen2.5-3b-em-{ppo,grpo}-v0.3` (secondaires). Contrôle possible : `R1-nq_hotpotqa_train-qwen2.5-3b-it-em-ppo-v0.2`
  (renforcement sans moteur de recherche).
- **Environnement fermé :** HotpotQA distracteurs (validation), 10 paragraphes par question dont 2 utiles, BM25 Okapi
  par question (k1 1,5, b 0,75), top 3. Questions à 10 paragraphes seulement ; réponses oui/non exclues (447
  comparaisons), car trouvées au hasard avec 32 tirages. Avec la question seule comme requête, les deux paragraphes
  utiles ne sont ramenés que dans 45 % (ponts) et 44 % (comparaisons) des cas.
- **Mesure :** n = 32 tirages par question, température 0,7, T (recherches au plus) dans {0, 4} puis {1, 2} ;
  500 questions par type (graine 42) [à confirmer par le pilote].
- **Statistiques :** estimateur sans biais de pass@k, courbe D(k), bootstrap à multiplicateurs par question, bandes
  simultanées, test de dominance, frontière avec McNemar exact (`analysis/passk.py`). Plan gelé avant les runs.
- **Pièges connus :** troncature (peut créer ou inverser un croisement : même limite, taux rapporté) ; format du
  modèle de base (taux rapporté) ; plafond de pass@n (un élargissement n'est visible que sur les questions hors de
  portée du modèle de départ).
- Volet C (taxonomie des échecs, juge LLM, kappa) : gardé comme complément possible, non prioritaire.

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
Tout relancer : `PY=<python> scripts/rendu.sh` (fictif) ou `RESULTATS=results` (vrais runs) -> `rendu/` (non versionné).
Puissance : `analysis/puissance.py` (200 questions + Holm : écart de 10 points détecté à 54 % ; 500 par jeu : 96 %).
Bibliographie : 46 références vérifiées dans OpenAlex le 9 novembre, zéro rouge (3 trouvées par DOI seulement).

Run d'évaluation de la mini-expérience fait le 9 novembre 2026 (kernel v4, commit 07c1fbb, 1 h 12) : 1 200 trajectoires dans
`results/eval_v4/` (non versionné). EM NQ / TriviaQA : A0 15/29, A1 10/40, A2 10/31, A3 9/36, A4 23/47, A5 22/53. HA1 soutenue (6/6),
HA2 non soutenue, HA3a et HA3b soutenues. Exploratoire : les harnais perdent 18 à 24 des 44 questions « connues » après recherche,
les agents 6 et 10. A3 s'abstient 30 fois sur 200. L'article et le carnet sont écrits sur ces vrais résultats.

Partie 2 (branche `partie2-passk`, 11 novembre 2026) : le renforcement élargit-il ce que l'agent sait faire (pass@k),
ou le fiabilise-t-il seulement ? Paires exactes checkpoint / modèle de départ (variantes `-it-` vérifiées sur Hugging Face),
HotpotQA distracteurs avec BM25 fermé par question. Note vérifiée : `docs/partie2/note_litterature.md` (deux références
introuvables retirées). Plan v0.1 : `docs/partie2/plan_analyse.md` (sous-ensemble difficile sur tirages indépendants, prédictions écrites).
Recherche documentaire : `docs/partie2/journal_requetes_partie2.md`, corpus de 43 articles vérifiés `docs/partie2/corpus_partie2.csv`. Code : `analysis/passk.py`, `data/hotpot_distractor.py`,
`retrieval/closed_bm25.py`, `scripts/run_passk.py`.
Pilote v2 (12 nov., kernel `memoire-partie2-pilote` v2, `results/partie2_pilote_v2/`, non versionné) : 6 runs complets. Défaut
corrigé : it-em-ppo répétait `<answer>` jusqu'à la limite (arrêt à `</answer>` ajouté, comme Search-R1). Plan GELÉ v1.0
(N = 500, n = 32, 1 024 jetons par tour) + précisions opératoires datées. Session 1 (ponts, T = 4, paires Instruct)
lancée le 12 nov. au soir : kernel `valentinproux/memoire-partie2-principal`. Sessions 2 à 4 : `scripts/genere_notebooks_partie2.py`
-> `kaggle/partie2_s{2,3,4}/` (la 4 lit `difficiles_ponts.txt` produit par la 3). Analyse : `analysis/confirmatoire_partie2.py`
(verdicts H1/H2, Holm, figures, tableaux LaTeX) ; Python avec matplotlib requis.
Sessions 1 et 2 faites (13 nov., `results/partie2_s1`, `results/partie2_s2`, liens dans `results/partie2/`, non versionnés) :
ponts, H1 soutenue, H2 « fiabilisation seule » pour les deux paires Instruct (frontière 351/350 contre 346) ;
comparaisons, aucun gain. Statut « budget épuisé » corrigé (l'EM n'en dépend pas). Exploratoire
(`analysis/exploratoire_partie2.py`, tirages séparés) : gain en U inversé, agents sans réponse environ 20 % (GRPO format,
PPO budget T = 4), indice d'élargissement sur les questions difficiles (85 % contre 70 % jamais résolues, PPO).
Chapitre : `docs/partie2/chapitre/partie2.tex`. Session 5 (H3) lancée le 13 nov. ; session 3 puis 4 à lancer (quota).

Prochaine étape (partie 1, si reprise) : volet A complet (7 jeux, 500 questions) ; agents Search-R1 issus de la version Instruct (variante `-it-`) pour
séparer règle de recherche et modèle ; H4 (BM25).

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
