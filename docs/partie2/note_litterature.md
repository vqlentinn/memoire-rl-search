# Partie 2 : le renforcement élargit-il ce qu'un agent de recherche sait faire ?

Note de littérature, version vérifiée du 11 novembre 2026. Chaque référence a été retrouvée sur arXiv, NeurIPS ou
Hugging Face. Elle corrige une première version : deux références introuvables ont été retirées, et deux
affirmations sur l'industrie ont été corrigées (voir la dernière section).

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

### À lire (trouvés pendant la vérification, non lus)

- *What pass@k Cannot Measure* (NeurIPS 2026) : validité de construit de pass@k après renforcement.
- *The Debate on RLVR Reasoning Capability Boundary: Shrinkage, Expansion, or Both?* ([2510.04028](https://arxiv.org/abs/2510.04028)).
- *RL-PLUS: Countering Capability Boundary Collapse* ([2508.00222](https://arxiv.org/abs/2508.00222)).

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
