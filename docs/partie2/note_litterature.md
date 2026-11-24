# Partie 2 : le renforcement élargit-il ce qu'un agent de recherche sait faire ?

Note de littérature, version vérifiée du 11 novembre 2026. Chaque référence a été retrouvée sur arXiv, NeurIPS ou
Hugging Face. Elle corrige une première version : deux références introuvables ont été retirées, et deux
affirmations sur l'industrie ont été corrigées (voir la dernière section).

## 0. Synthèse (12 novembre 2026, après la recherche documentaire)

Recherche complète : `journal_requetes_partie2.md` ; corpus de 43 articles vérifiés : `corpus_partie2.csv`.

| Position | Sans outil | Agents |
| --- | --- | --- |
| Le RL standard n'élargit pas, il fiabilise | Yue et al. (NeurIPS 2025), « No Free Lunch » (théorème), « From Base Rollouts » (EMNLP 2026) | — |
| Le RL rétrécit | « When RLVR Shrinks », « Locked at the Entrance » | Effondrement de l'exploration en environnement peu familier (EMNLP 2026) |
| Le RL élargit, sous conditions | ProRL (entraînement prolongé), Guru (domaines peu vus au pré-entraînement), « Atomic Skills » (si les compétences de base existent) | Zhai et al. (tâches compositionnelles), Zhu et al. (si le budget T est suffisant) |
| Des méthodes modifiées élargissent | RL-PLUS, PKPO, curriculums, KL avant, e3, Poly-EPO | Exploration guidée par stratégie, RAPO agents |
| Les croisements observés sont souvent du bruit | Yang (inférence appariée : 1 sur 9 avant correction) | — |

Trois enseignements pour la partie 2 :

1. **Le débat ne porte plus sur « oui ou non », mais sur « quand ».** Les deux lectures sont compatibles selon le type
   de tâche (Zhai et al.), la durée d'entraînement (ProRL, Yao et al.), la familiarité du domaine (Guru, effondrement
   de l'exploration) et le budget d'interaction (Zhu et al.). D'où une hypothèse conditionnelle (H2 : pont contre
   comparaison, T = 4 contre T = 0), plutôt qu'un verdict global.
2. **Le RL standard est la cible des méthodes qui élargissent.** Presque toutes les études favorables proposent un
   correctif au RL standard ; Search-R1 est un RL standard (récompense de résultat, pénalité KL inverse). La
   littérature prédit donc plutôt un élargissement faible ou nul pour nos checkpoints, sauf sur les tâches
   compositionnelles où Zhai et al. l'observent.
3. **La place est libre.** Aucune étude ne mesure la frontière de checkpoints publics d'agents de recherche face à
   leur propre modèle de départ, et aucune étude sur les agents n'utilise d'inférence appariée.

## 1. Ce que dit la littérature

### Sans outil : « plus souvent », plutôt que « davantage »

- **Yue et al., NeurIPS 2025** — *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond
  the Base Model?* ([2504.13837](https://arxiv.org/abs/2504.13837)). Ils mesurent pass@k, la probabilité qu'au moins
  une de k tentatives soit juste. Le modèle entraîné gagne au premier essai, mais le modèle de départ le rattrape puis
  le dépasse quand k devient grand.
- **There Is No Free Lunch in RLVR: Every pass@1 Gain Has a pass@k Price** ([2610.39295](https://arxiv.org/abs/2610.39295),
  29 octobre 2026). Un résultat théorique fait de ce croisement un prix inévitable.

### Ce constat est fragile statistiquement

- **Chen Yang**, *When Pass@k Crossovers Are Real — and Why: Paired Inference and the Markov Kernel of RLVR*
  ([2609.22547](https://arxiv.org/abs/2609.22547), v3 du 3 novembre 2026). Une inférence appariée sur la courbe des
  écarts, avec un bootstrap à multiplicateurs par question, des bandes de confiance simultanées et un test de
  dominance.
  - Sur neuf checkpoints publics, un seul croisement passe le test, et seulement avant correction pour comparaisons
    multiples.
  - Une troncature asymétrique de la génération peut créer, masquer ou inverser un croisement.
  - Ajouter des questions augmente bien plus la puissance qu'ajouter des générations par question.
  - Nuance : dans ses expériences contrôlées (32 000 jetons), deux modèles de 1,5 milliard de paramètres montrent des
    croisements certifiés ; deux modèles de 7 milliards, aucune perte soutenue. Le phénomène existe, mais il est plus
    rare et plus petit qu'annoncé.

### Sur les agents, une seule étude annonce « davantage »

- **Zhai et al.**, *Does RL Expand the Capability Boundary of LLM Agents? A PASS@(k,T) Analysis*
  ([2604.14877](https://arxiv.org/abs/2604.14877), avril 2026, code public). Pass@(k,T) : k essais, au plus T tours
  d'interaction. Leur dispositif :
  - Qwen2.5-7B-Instruct ; **agent RL entraîné par les auteurs (GRPO) sur seulement 200 questions** HotpotQA ;
  - cadre « distracteurs » : 10 paragraphes par question, dont 2 utiles, recherche BM25 ;
  - 100 questions par catégorie : MATH-500 sans outil, comparaisons et ponts HotpotQA ; n = 64 tirages,
    température 0,7, T dans {0, 1, 2, 3, 5} ;
  - ponts : 81 questions résolues au moins une fois contre 77 pour le modèle de départ (5 gagnées, 1 perdue) ;
    au premier essai, l'agent fait moins bien (0,335 contre 0,363) ;
  - pas de test formel sur l'écart de frontière (une annexe de bootstrap existe ; elle n'a pas pu être lue) ;
  - un fine-tuning supervisé (LoRA) sur les mêmes données rétrécit la frontière : 3 questions gagnées, 7 perdues ;
  - la vérification sur les checkpoints publics de Search-R1 est renvoyée par les auteurs à « une version révisée ».
    Elle n'est pas faite.
  - Mécanisme : le gain vient de l'usage de l'information trouvée plutôt que de la requête, comme dans nos trajectoires
    de la partie 1.

### Fine-tuning supervisé ou renforcement : le fil commun est l'imitation

- **Chu et al.**, *SFT Memorizes, RL Generalizes* (ICML 2025, [2501.17161](https://arxiv.org/abs/2501.17161)) : le
  supervisé mémorise, le renforcement généralise ; le supervisé reste utile pour stabiliser le format.
- *Retaining by Doing: The Role of On-Policy Data in Mitigating Forgetting* ([2510.18874](https://arxiv.org/abs/2510.18874)) :
  le renforcement fait moins oublier, parce qu'il apprend sur ses propres générations.
- **Jin et al.**, *RL Is Neither a Panacea Nor a Mirage* ([2508.16546](https://arxiv.org/abs/2508.16546)) : le
  renforcement répare surtout la dérive causée par le supervisé (jeu de cartes « 24 points », hors distribution).
- **Zhang et al.**, *When Does Online Imitation Learning Help in LLM Post-Training?* (NeurIPS 2026,
  [2606.30445](https://arxiv.org/abs/2606.30445)) : quand l'expert est représentable par l'élève, l'imitation hors
  ligne égale déjà l'expert. Article surtout théorique (Countdown, raisonnement) : portée limitée pour les agents.

### Ajouts de la recherche du 12 novembre (résumés lus ; tous vérifiés dans OpenAlex)

**Agents :**
- **Zhu et al.**, *What Does Multi-Turn Reinforcement Learning Really Improve in LLM Agents?* ([2611.06483](https://arxiv.org/abs/2611.06483),
  4 novembre 2026 ; texte lu). Qwen3-4B et 8B, RL par les auteurs (GRPO, GiGPO, PPO), sur du code, des appels de
  fonctions, du SQL et FrozenLake ; K jusqu'à 128. L'agent reste au-dessus de la base quand k augmente, mais son
  avantage **dépend du budget d'interaction T et peut s'inverser quand T est petit**. Intervalles à 95 % sur les
  figures, pas de test apparié. Pas de recherche documentaire ni de Search-R1 : pas de recoupement avec la partie 2.
- **Chen et al.**, *Why Do LLM Agents Fail in Exploring New Environments?* (Findings EMNLP 2026) : en environnement
  peu familier, le RL fait monter pass@1 mais chuter pass@k (« effondrement de l'exploration »).
- **Kim et al.**, *Elicitation or Expansion? Demystifying Gains from Agent Harness Optimization*
  ([2610.27889](https://arxiv.org/abs/2610.27889)) : la même question, posée aux **harnais** ; les gains au premier
  essai d'un harnais optimisé peuvent s'inverser à grand nombre d'essais. Pont direct avec la partie 1.
- **Szot et al.**, *Expanding LLM Agent Boundaries with Strategy-Guided Exploration* ([2603.02045](https://arxiv.org/abs/2603.02045)).

**Débat sans outil :**
- **ProRL** (NeurIPS 2025, [2505.24864](https://arxiv.org/abs/2505.24864)) : un RL prolongé fait résoudre des
  problèmes que la base ne résout jamais, quel que soit le nombre d'essais.
- **Wen et al.** ([2506.14245](https://arxiv.org/abs/2506.14245)) : avec une mesure qui vérifie aussi le raisonnement
  (CoT-Pass@k), le RL élargit. La conclusion dépend de ce qu'on compte comme réussite.
- **Yao et al.** ([2510.04028](https://arxiv.org/abs/2510.04028)) : le RL rétrécit d'abord, puis élargit avec un
  entraînement plus long.
- **Yuan et al.** ([2606.15455](https://arxiv.org/abs/2606.15455)) : le RL standard est structurellement biaisé
  contre pass@k élevé ; une baisse moyenne n'exclut pas que des questions insolubles deviennent solubles.
- **Cheng et al.**, *Atomic Skills are the Prerequisite* ([2512.01970](https://arxiv.org/abs/2512.01970)) : le RL
  compose des compétences seulement si les compétences élémentaires existent déjà.

**Statistique et validité de la mesure :**
- **Chen et al. (2021)**, Codex ([2107.03374](https://arxiv.org/abs/2107.03374)) : l'estimateur sans biais de pass@k.
- **MATH-Beyond** ([2510.11653](https://arxiv.org/abs/2510.11653)) : un banc d'essai fait de problèmes hors de portée
  du modèle de départ, pour que l'élargissement soit mesurable. Réponse directe au risque de plafond.
- **On Randomness in Agentic Evals** ([2602.07150](https://arxiv.org/abs/2602.07150)), **When +1% Is Not Enough**
  ([2511.19794](https://arxiv.org/abs/2511.19794)), **Don't Pass@k** ([2510.04265](https://arxiv.org/abs/2510.04265)),
  **What pass@k Cannot Measure** ([2610.07405](https://arxiv.org/abs/2610.07405)).

### À lire (trouvés pendant la vérification, non lus)

- *What pass@k Cannot Measure* (NeurIPS 2026) : validité de construit de pass@k après renforcement.
- *The Debate on RLVR Reasoning Capability Boundary: Shrinkage, Expansion, or Both?* ([2510.04028](https://arxiv.org/abs/2510.04028)).
- *RL-PLUS: Countering Capability Boundary Collapse* ([2508.00222](https://arxiv.org/abs/2508.00222)).

## 2bis. Ce que la littérature change au dispositif

| Constat de la littérature | Conséquence pour le dispositif |
| --- | --- |
| Un élargissement n'est visible que sur les questions que la base ne résout jamais (MATH-Beyond, Yuan et al.) | Ajouter un **sous-ensemble difficile**, présélectionné sur des tirages **indépendants** de la base (voir le plan d'analyse), en plus de l'échantillon complet |
| L'avantage dépend du budget d'interaction et peut s'inverser à petit T (Zhu et al.) | Garder T comme variable : T = 0, 1, 2, 4 si le budget le permet, au minimum 0 et 4 |
| Le RL compose des compétences seulement si elles existent (« Atomic Skills ») | Justifie la distinction pont (dépendance entre recherches) contre comparaison (recherches indépendantes) |
| Les gains d'un harnais peuvent s'inverser à grand k (Kim et al.) | Bras optionnel : le même modèle Instruct avec un harnais ReAct, pour comparer RL et harnais **à pass@k**, et pas seulement au premier essai comme dans la partie 1 |
| La conclusion dépend de la définition de la réussite (Wen et al.) | Correspondance exacte en mesure principale ; correspondance souple et F1 en sensibilité, comme dans la partie 1 |
| Les évaluations d'agents sont très variables (Bjarnason et al.) | Inférence appariée par question (déjà prévue) ; graines rapportées |

## 2. Ce que fait l'industrie

- **Prime Intellect** fournit des briques ouvertes : la bibliothèque `verifiers`, l'Environments Hub et l'entraîneur
  `prime-rl` ([documentation](https://docs.primeintellect.ai/verifiers/v0/overview)).
- **Fireworks AI** vend du *reinforcement fine-tuning*. Sa vitrine est Genspark, un agent de recherche approfondie :
  qualité +12 %, appels d'outils +33 %, coût divisé par deux face à un modèle fermé ([étude de cas, octobre 2025](https://fireworks.ai/blog/genspark),
  sur Kimi K2). Ce sont leurs propres chiffres. Leur plateforme d'entraînement est annoncée en « preview » ; sa
  disponibilité générale n'a pas été confirmée.
- **Trajectory** propose de réentraîner par GRPO des agents à partir de traces de production (note de recherche du
  2 novembre 2026). Le gain annoncé de 4 % à 18 % n'a pas de méthodologie publiée
  ([couverture](https://www.startuphub.ai/ai-news/artificial-intelligence/2026/trajectory-demos-one-click-training-for-agents)).

Tous parient que le renforcement sur des agents vaut son coût. Aucun ne publie de quoi distinguer s'il fiabilise ou
s'il élargit. Or la différence a un prix : si le renforcement ne fait que fiabiliser, un fine-tuning supervisé ou une
distillation des bonnes traces obtient l'essentiel pour beaucoup moins cher.

## 3. Corrections apportées à la première version

| Première version | Correction |
| --- | --- |
| *RLVR Sharpens the Pencil* (novembre 2026), couverture qui augmente sur OLMo-2-1B | Introuvable : retiré |
| Baseten : la distillation avant le renforcement aide jusqu'à 8 milliards, puis l'avantage s'inverse | Introuvable : retiré |
| Fireworks : API d'entraînement disponible pour tous depuis août 2026 | Non confirmé : seule une « preview » est documentée |
| Genspark présenté comme récent | Étude de cas d'octobre 2025, sur Kimi K2 |
| Prime Intellect : environ 137 600 tâches de recherche (juillet 2026) | Chiffre introuvable : retiré |
| Article sur les croisements intitulé « When Pass@k Crossovers Are Real » | Titre exact confirmé (v3) ; nuance ajoutée sur les croisements certifiés à 1,5 milliard |
| Zhai et al. : dispositif résumé | Ajouté : l'agent RL est entraîné par les auteurs sur 200 questions ; le contrôle sans outil est MATH-500 |
| Variantes Instruct de Search-R1 « à vérifier » | Vérifié : `SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-{ppo,grpo}-v0.3` existent |

## Veille du 24 novembre 2026 (avant l'article)

Sources : recherche web, Consensus, index des articles Hugging Face, OpenAlex (aucun résultat sur la requête ciblée).
Vérifié sur la page arXiv (résumé) ; « texte » = texte complet consulté en partie.

- **Oh et al., « Sharpening Tax in Post-Training », arXiv 2610.01509 (1er oct. 2026)** (texte, 100 000 premiers
  caractères). 14 paires base / post-entraîné (Gemma-4, Ministral-3, Qwen2.5, Qwen3.5), 3 bancs agentiques (BFCL v4
  multi-tour, WebShop, ACEBench), pass@K jusqu'à 128. Le modèle de base, avec un harnais léger, dépasse souvent la
  version post-entraînée à grand K (36 cas sur 42). Taxe = différence d'aire sous la courbe pass@k. IC bootstrap, pas de
  test apparié ni de bande simultanée, pas de comparaison à coût égal, pas de vote majoritaire, pas d'équivalence en
  essais (seulement le budget de croisement k*). **Pas d'agent de recherche ni de HotpotQA.** Soutient notre résultat ;
  le titre « le renforcement fiabilise les agents » n'est donc plus neuf, nos apports doivent être ailleurs.
- **Zhai et al., arXiv 2604.14877** (déjà cité ; résumé relu). Annonce un élargissement par le renforcement sur les
  tâches compositionnelles de collecte d'information, et un **rétrécissement par le SFT à données égales** ; gain
  attribué à l'intégration de l'information trouvée. À discuter frontalement : notre H4 teste directement leur
  affirmation sur le SFT, et nos résultats (recherche plutôt que lecture, pas d'élargissement net) la contredisent.
- **Kim et al., « Elicitation or Expansion? Demystifying Gains from Agent Harness Optimization », arXiv 2610.27889
  (20 oct. 2026).** Même question (fiabiliser ou élargir) pour l'optimisation de harnais, pas pour le renforcement ;
  avantage à pass@1 qui peut s'inverser à grand budget.
- **Rath et al., « What pass@k Cannot Measure », arXiv 2610.07405 (atelier NeurIPS 2026, non archivé).** Maths,
  Qwen2.5-1.5B, GRPO contre RFT (fine-tuning par rejet, l'analogue de notre `it-sft-depart`) ; mesures de diversité
  (réponses uniques, entropie) ; aucune méthode n'améliore la couverture des problèmes difficiles. Proche de nos
  analyses de diversité et de H4, mais sans agent.
- **Zavidnyi et al., « Budget Internalization », arXiv 2611.26172 (19 nov. 2026).** Un plafond fixe pendant
  l'entraînement est intériorisé : gain près du budget, perte de l'adaptation au temps de calcul. Budget de jetons,
  maths ; analogue de notre observation T = 4 / T = 8 (budget de recherches).
- **Zhou, « When RLVR Shrinks the Reasoning Boundary », arXiv 2607.20543 (juil. 2026)** ; **Zhou et al., « Hard or
  Just Unreached? », ICML 2026** : pass@k = 0 n'est pas une difficulté intrinsèque (angle mort de l'échantillonnage).
  Proches de notre test miroir, mais aucun ne traite l'asymétrie d'un sous-ensemble choisi sur l'échec d'un seul des
  deux modèles comparés (d'après les résumés).
- **Essais équivalents :** aucune source trouvée qui exprime le modèle entraîné comme le modèle de départ avec g fois
  plus d'essais, ni qui prédit pass@k par pass_départ(g k). Les plus proches : « Sampling Efficiency Gap » de Yue et
  al. (pass@1 RL contre pass@256 base) et le budget de croisement k* d'Oh et al. Recherche non exhaustive.
- **Vote majoritaire base contre RL :** aucune comparaison directe trouvée.

### Lecture complète (24 novembre 2026)

Textes HTML d'arXiv lus en entier, sauf mention. Ce qui compte pour l'article :

- **Zhai et al. 2604.14877 (lu en entier).** Dispositif presque identique au nôtre : HotpotQA distracteurs, 10 paragraphes,
  BM25 fermé, ponts (C) et comparaisons (B). Mais : Qwen2.5-7B-Instruct, agent GRPO **entraîné par eux** en LoRA sur
  **200 questions**, 100 questions d'évaluation par catégorie, n = 64, T ≤ 5. Élargissement sur les ponts à
  PASS@(64,5) : RL 81, base 77, SFT 73 ; il repose sur **5 questions gagnées contre 1 perdue**. Bootstrap sur les
  tirages seulement (pas sur les questions) ; leurs IC des gains [2 ; 8] et pertes [1 ; 5] du RL se recouvrent ; aucun
  test formel. SFT sur des trajectoires **oracle** construites depuis les faits utiles annotés (pas des traces de
  l'agent). Échange de stratégies : RL stratégie + base raisonnement 0,148, base stratégie + RL raisonnement 0,143
  (léger avantage à la planification de recherche, comme chez nous), alors que le résumé attribue le gain à
  l'intégration de l'information. Vérification sur Search-R1 public : annoncée pour une version future, non faite.
  Diversité : stratégies uniques base 40,1, RL 45,5, SFT 14,7 sur 64.
  → Notre étude refait leur question avec des checkpoints publics, 500 questions, inférence appariée par question et un
  contrôle de sélection : leur « 5 contre 1 » est de l'ordre de ce qu'une fiabilisation ou une redistribution produit
  (notre simulation, notre test miroir).
- **Oh et al. 2610.01509, Sharpening Tax (lu en entier).** 14 paires, 3 bancs (BFCL, WebShop, ACEBench), N = 128,
  bootstrap apparié des tâches pour les IC, pas de test. Taxe positive dans 36 cas sur 42 ; **négative pour Qwen2.5-3B
  et 7B sur BFCL** (le post-entraînement répare le format des appels d'outils). Modèle théorique : une part λ des tâches
  poussée vers 0 ou 1 (bimodalisation) ; ce n'est pas notre modèle en puissance. Harnais spécial pour les modèles de
  base, harnais par défaut pour les post-entraînés (biais reconnu). Ne compare pas RL, SFT et distillation (limite
  déclarée) : H4 le fait.
- **Kim et al. 2610.27889, Elicitation or Expansion (lu en entier).** HotpotQA avec BM25 fixe et Qwen3-8B, optimisation
  de harnais. Exemple : le harnais optimisé interroge l'entité pont résolue et atteint le paragraphe dans 125 essais
  sur 128 (= notre chaînage). L'EM donne un classement faux sur HotpotQA (Qwen3-32B sous 8B), corrigé par un juge : à
  citer avec notre sensibilité à la notation. Pas de test formel ; pas d'équivalence en essais.
- **Rath et al. 2610.07405 (lu en entier).** Qwen2.5-1.5B, GSM8K. GRPO **réduit** la diversité (réponses uniques
  −1,30), RFT l'**augmente** (+0,56). RFT réduit la couverture des problèmes difficiles par rapport au départ (−3 à −3,5
  points à k ≤ 4) ; GRPO n'en diffère pas. Calcul de puissance (effet minimal détectable ≈ 0,02–0,03). Pas de vote
  majoritaire. → Prédiction plausible pour H4 : `it-sft-depart` (analogue du RFT) moins concentré que l'agent.
- **Zavidnyi et al. 2611.26172, Budget Internalization (lu en entier).** Budget de jetons, maths : au-delà du plafond
  d'entraînement, le modèle RL plafonne sous la base. Chez nous, le budget de **recherches** va dans l'autre sens :
  PPO à T = 8 gagne (EM 0,315 → 0,351). Contraste à signaler, sans conclure : budgets de nature différente.
- **Zhou 2607.20543, Pass@k Inversion (lu en entier).** Pertes concentrées sur les prompts « frontière » (succès rare
  mais récupérable chez la base) ; modèle à deux modes : la politique s'engage sur le mode faux. → Explique nos
  questions perdues : l'agent s'y « effondre » sur une même réponse fausse dans 57 % des cas.
- **Zhou et al. 2606.19636, Hard or Just Unreached (lu sauf les 6 derniers % ).** pass@6 = 0 n'est pas une difficulté
  intrinsèque. Ne traite ni la sélection sur un seul modèle ni la régression vers la moyenne : notre test miroir reste
  nouveau sur ce point.
- **Matrenok et al. 2611.03146, Controlling Sharpening (moitié principale lue ; annexes non lues).** Optimum fermé du RL
  KL-régularisé : inclinaison exponentielle de la politique de référence. Pour pass@1, cela multiplie les **cotes** de
  réussite de chaque question par un même facteur (p'/(1−p') = e^{1/β} p/(1−p)) : un autre modèle de fiabilisation que
  notre puissance 1 − p' = (1 − p)^g. Aucune équivalence en essais. → Test possible sur nos données : quel modèle
  (cotes, puissance, bimodalisation d'Oh et al.) prédit le mieux la courbe de l'agent ?
- **Qu 2611.11260, Primitives and Paths (lu en entier).** Régime « amplification » : gros gain à pass@1, peu de
  changement à grand k = notre signature. Tâches synthétiques.
- **Tomihari et Sato 2605.04542, Power Distribution (moitié principale lue).** π^α optimum du RL à auto-récompense ;
  pas d'équivalence en essais. Cite He et al. (2025) : à budget égal, échantillonner plus le modèle de base peut faire
  mieux (rejoint notre analyse à coût égal ; référence à retrouver).
- **Search-RSI 2611.27780 (moitié principale lue).** Pas de pass@k ni de comparaison à la base ; peu pertinent.

Bilan : aucune des sources lues ne définit d'équivalence en essais ni ne prédit la frontière par pass_départ(g k) ;
aucune ne traite l'asymétrie de sélection (test miroir) ; aucune ne compare le vote majoritaire base / RL ; aucune
n'étudie un agent de recherche public face à son modèle de départ avec inférence appariée par question.
