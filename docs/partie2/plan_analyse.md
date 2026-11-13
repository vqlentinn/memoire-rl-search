# Partie 2 : plan d'analyse

**Statut : version 1.0, GELÉE le 12 novembre 2026, après le pilote v2 et avant tout run principal.** Les paramètres
autrefois marqués « [pilote] » sont fixés (section « Pilote : résultats et choix », en fin de document). Toute
modification ultérieure sera datée, justifiée ici et signalée dans l'article.

## Question

À modèle de départ identique, l'entraînement par renforcement d'un agent de recherche élargit-il l'ensemble des
questions qu'il finit par résoudre, ou augmente-t-il seulement sa réussite au premier essai ? Cet élargissement
apparaît-il quand la question oblige à enchaîner les recherches, et résiste-t-il à un test statistique apparié ?

Lien avec la partie 1 : la partie 1 compare un agent entraîné à des harnais fixes, mais à partir de modèles de départ
différents (base contre Instruct). La partie 2 compare chaque agent à **son propre** modèle de départ.

## Modèles (paires exactes, aucun entraînement de notre part)

| Paire | Modèle de départ | Agent entraîné | Rôle |
| --- | --- | --- | --- |
| Instruct, PPO | `Qwen/Qwen2.5-3B-Instruct` | `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-ppo-v0.3` | principale |
| Instruct, GRPO | `Qwen/Qwen2.5-3B-Instruct` | `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo-v0.3` | principale |
| Base, PPO | `Qwen/Qwen2.5-3B` | `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-ppo-v0.3` | secondaire |
| Base, GRPO | `Qwen/Qwen2.5-3B` | `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo-v0.3` | secondaire |

Les paires Instruct sont principales : le modèle de base suit mal les balises de Search-R1, et ses échecs de format
gonfleraient l'écart. Contrôle possible, si le budget le permet : `R1-nq_hotpotqa_train-qwen2.5-3b-it-em-ppo-v0.2`,
entraîné par renforcement **sans** moteur de recherche.

Prompt : celui de Search-R1 (`prompts/searchr1.txt`), identique pour les deux modèles d'une paire. Gabarit de chat
pour les quatre paires (même entrée que celle que voient les agents) ; pour les paires base, le prompt brut est rapporté
en sensibilité (le pilote donne 62 % de bon format avec gabarit, 77 % sans).

## Environnement

- HotpotQA, cadre « distracteurs », partie validation (`hotpotqa/hotpot_qa`) : 10 paragraphes par question, dont 2
  utiles et 8 distracteurs.
- Recherche fermée : un index BM25 Okapi par question (k1 = 1,5, b = 0,75), comme Zhai et al. ; 3 paragraphes par
  recherche ; ex aequo départagés par l'ordre du fichier.
- Questions retenues : exactement 10 paragraphes (7 345 sur 7 405) ; réponses « yes » / « no » exclues (447
  comparaisons), car une réponse binaire est trouvée au moins une fois au hasard avec 32 tirages.
- **Mesure préalable (sans modèle)** : en cherchant avec la question telle quelle, BM25 ramène au moins un paragraphe
  utile dans 96 % (ponts) et 94 % (comparaisons) des cas, mais les deux seulement dans 45 % et 44 %. Une seconde
  recherche bien choisie est donc nécessaire dans plus de la moitié des cas, pour les deux types.

## Échantillon

- Ponts et comparaisons : N = 500 questions de chaque type, tirées avec la graine 42
  (`data/hotpot_distractor.charger`).
- Pilote : 50 questions de ponts tirées avec la graine 7, **disjointes** des questions d'évaluation.
- HotpotQA fait partie de l'entraînement de Search-R1 (partie train) : les questions de validation sont « vues »
  en distribution, jamais en contenu. À dire dans l'article.
- **Sous-ensemble difficile (ajouté le 12 novembre).** Un élargissement n'est mesurable que sur les questions que le
  modèle de départ ne résout jamais (MATH-Beyond ; Yuan et al.). Procédure en deux temps, sur des tirages
  **indépendants** :
  1. *Présélection* : le modèle de départ tire n_pré = 16 générations (graine de génération 1) sur un vivier de
     questions de ponts **distinct** des 500 de l'échantillon principal ; on garde celles qui n'ont aucune réussite.
  2. *Évaluation* : sur ces questions, les deux modèles de la paire tirent n = 32 générations **nouvelles** (graine
     de génération 0, comme l'échantillon principal).
  Réutiliser les tirages de la présélection donnerait au modèle de départ un pass@k nul par construction, et donc un
  élargissement fabriqué. Le test `test_selection_sur_les_memes_tirages_fabrique_un_elargissement` le montre sur
  simulation. Vivier : 1 000 questions de ponts tirées avec la graine 43, disjointes des 500 d'évaluation et des 50 du pilote.
  Le pilote donne 16 % de questions sans réussite en 32 essais : on attend environ 150 à 200 questions difficiles.

## Mesures

- n = 32 générations par question et par modèle, température 0,7 (comme Zhai et al.).
- T (recherches au plus) : 4 et 0 d'abord ; 1 et 2 ensuite si le budget le permet. Zhu et al. (2611.06483)
  montrent que l'avantage du RL dépend du budget d'interaction et peut s'inverser à petit T. T = 0 : aucune recherche permise ;
  une trajectoire qui tente de chercher se termine sans réponse.
- 1 024 jetons au plus par tour, contexte de 8 192 jetons, identiques pour les deux modèles d'une paire. La
  génération s'arrête à `</search>` (recherche) ou à `</answer>` (réponse), comme le post-traitement officiel de
  Search-R1. Une trajectoire dont le contexte ne laisse plus la place d'un tour est arrêtée et comptée fausse
  (statut `context_exceeded`, taux rapporté).
- Juste : correspondance exacte (normalisation de Search-R1). F1 et correspondance souple en sensibilité.
- Rapportés pour chaque modèle : taux de sorties sans réponse exploitable (format), taux de troncature, nombre moyen
  de recherches.

## Estimateurs et tests (`analysis/passk.py`)

- pass@k par question : estimateur sans biais 1 − C(n − c, k) / C(n, k), pour k dans {1, 2, 4, 8, 16, 32}.
- Courbe appariée : D(k) = moyenne sur les questions de pass@k(agent) − pass@k(départ).
- Incertitude : bootstrap à multiplicateurs gaussiens par question, 2 000 tirages, graine 0.
- Bandes de confiance simultanées à 95 % sur tous les k.
- Test de dominance (intersection-union) : « D(k) > 0 pour tout k », chaque t(k) au-dessus du quantile unilatéral
  à 95 %.
- Perte certifiée : au moins un k où la bande simultanée est entièrement sous 0.
- Frontière : questions résolues au moins une fois sur n (gagnées, perdues), test exact de McNemar.

## Hypothèses et règles de décision

Tests principaux : H2 sur les ponts à T = 4, pour les deux paires Instruct, sur l'échantillon complet ; correction de
Holm sur ces 2 tests. Le sous-ensemble difficile est secondaire, rapporté avec la même règle et sans correction.
Tout le reste est secondaire et rapporté comme tel.

| Hypothèse | Énoncé | Règle de décision |
| --- | --- | --- |
| H1 (contrôle) | Au premier essai, l'agent bat son modèle de départ | Borne basse de l'IC ponctuel de D(1) > 0 |
| H2 (principale) | Sur les ponts, l'écart reste positif quand k augmente | « Élargissement » : test de dominance rejeté ET borne basse de la bande simultanée à k = 32 > 0. « Fiabilisation seule » : H1 soutenue mais bande à k = 32 contenant 0. « Rétrécissement » : perte certifiée |
| H2' (secondaire) | Sur les comparaisons et à T = 0, l'écart se referme ou s'inverse | Mêmes règles ; aucune correction, rapporté comme exploratoire |
| H3 (secondaire) | À passages identiques, l'avantage vient de la lecture | Échange des plans de recherche entre les deux modèles ; protocole précis à écrire après le pilote |
| H4 (option) | Un fine-tuning supervisé LoRA sur les traces réussies de l'agent récupère la frontière | Ouvert seulement après les résultats principaux |

### Prédictions écrites avant le run (d'après la littérature)

- **H1 : soutenue** (gain au premier essai attendu, déjà observé dans la partie 1 à température nulle).
- **H2 sur l'échantillon complet : « fiabilisation seule » plus probable qu'« élargissement ».** Search-R1 est un RL
  standard (récompense de résultat, KL inverse), la cible de presque toutes les méthodes qui prétendent élargir ; et
  pass@32 risque de saturer.
- **H2 sur le sous-ensemble difficile : élargissement faible possible** sur les ponts (Zhai et al., tâches
  compositionnelles ; Yuan et al., des questions insolubles deviennent solubles).
- **H2' : pas d'élargissement à T = 0**, et un avantage plus faible à petit T (Zhu et al.).
Ces prédictions ne sont pas des règles de décision ; elles servent à montrer, après coup, ce qui était attendu.

### Bras optionnel : harnais contre RL à pass@k

Si le budget le permet, le modèle Instruct avec le harnais ReAct de la partie 1 (`prompts/react.txt`), dans le même
environnement fermé. Kim et al. (2610.27889) montrent que les gains au premier essai d'un harnais peuvent s'inverser à
grand nombre d'essais : la partie 1 comparait RL et harnais au premier essai seulement.

Les trois issues de H2 sont publiables : un élargissement qui résiste serait une réplication indépendante de Zhai
et al. sur des modèles publics ; un élargissement qui ne résiste pas serait le symétrique, pour les agents, du
résultat de Yang ; un rétrécissement montrerait que le renforcement fiabilise et rétrécit aussi pour les agents.

## Session pilote (étape 2) : ce qu'elle doit mesurer

Sur les 50 questions de mise au point, n = 32, T = 4, pour les deux paires Instruct et une paire base :

1. Le débit (trajectoires par seconde) sur une T4, pour fixer N et n dans le budget Kaggle (environ 30 h par semaine).
2. Le taux de réponses au bon format, avec et sans gabarit de chat pour le modèle de base.
3. Le taux de troncature à 512 jetons par tour.
4. La distribution des réussites par question, qui calibre une simulation de puissance avant de fixer N.
5. La part des questions **hors de portée** du modèle de départ (aucune réussite en n essais). C'est la marge
   d'élargissement : si elle est proche de 0, pass@n sature à 1 et un élargissement est indétectable quelle que
   soit la taille de l'étude. Dans ce cas, il faudra des questions plus difficiles (ou un n plus petit), à décider
   avant le gel du plan.

Ordre de grandeur à vérifier : 4 modèles × 2 valeurs de T × 1 000 questions × 32 tirages = 256 000 trajectoires,
contre 1 200 en 1 h 12 pour la partie 1 en décodage glouton. Repli si c'est trop lent : n = 16, paires Instruct
seules, puis T = 4 seul.

## Pilote : résultats et choix (12 novembre 2026)

Pilote v1 : échec (contexte de 4 096 jetons dépassé, lecture JSONL coupée sur U+2028), corrigé. Pilote v2 : 6 runs
complets de 1 600 trajectoires (50 questions de ponts, graine 7, n = 32, T = 4). Chiffres indicatifs, aucun test.

| Modèle | EM par tirage | Bon format | Tronquées | Débit (T4) |
| --- | --- | --- | --- | --- |
| Qwen2.5-3B-Instruct | 0,358 | 99,5 % | 0,1 % | 1,34 traj./s |
| it-em-grpo | 0,371 | 95,1 % | 15,1 % | 0,48 traj./s |
| it-em-ppo | 0,431 | 97,2 % | 42,0 % | 0,24 traj./s |
| Qwen2.5-3B (chat / brut) | 0,139 / 0,200 | 62 % / 77 % | 10 % / 4 % | 1,1 / 1,0 traj./s |
| em-ppo | 0,611 | 99,9 % | 0,0 % | 1,82 traj./s |

- **Défaut corrigé :** it-em-ppo répète « <answer>X</answer> » après sa réponse jusqu'à la limite de jetons. Arrêt à
  `</answer>` ajouté (voir Mesures). Rescoré au premier `</answer>`, le pilote donne pour les paires Instruct
  D(1) ≈ +0,03 à +0,05 et D(32) ≈ −0,04, frontière 40 contre 42 : rien ne change dans les tendances.
- **Marge d'élargissement :** 16 % des questions sans réussite du modèle Instruct en 32 essais (30 % pour Qwen2.5-3B).
  pass@32 ne sature donc pas ; le sous-ensemble difficile reste utile.
- **Taille :** écart-type par question de d_i(k) entre 0,19 et 0,30, quantile simultané 2,4. Avec N = 500, l'écart
  minimal détectable (puissance 80 %, α = 5 %) est d'environ 0,041 à k = 32 (bande simultanée) et de 0,023 à 0,034 à
  k = 1 (IC ponctuel). N = 1 000 ne le ramènerait qu'à 0,029 pour deux fois le coût : N = 500 et n = 32 sont retenus.
- **Budget** (débit après correction estimé à partir des jetons générés, à vérifier en session 1) : environ 14 h de
  GPU par type de question à T = 4 pour les trois modèles Instruct, moins à T = 0.

### Ordre des sessions Kaggle

1. **Session 1 (confirmatoire) :** ponts, T = 4, Qwen2.5-3B-Instruct, it-em-ppo, it-em-grpo (`notebooks/kaggle_partie2_principal.ipynb`).
2. Session 2 : comparaisons, T = 4, mêmes modèles (H2').
3. Session 3 : T = 0, ponts et comparaisons (H2') ; présélection du sous-ensemble difficile (Instruct, n_pré = 16).
4. Session 4 : sous-ensemble difficile, n = 32, tirages nouveaux.
5. Ensuite seulement : paires base, T ∈ {1, 2}, H3, bras harnais, H4.

## Précisions opératoires (12 novembre 2026, avant toute donnée de la session 1)

Écrites pendant que la session 1 tourne, sans avoir vu ses résultats ; elles précisent la mise en œuvre sans changer
les règles de décision.

- **Holm :** la p-valeur d'« élargissement » d'une paire est max(p_dominance, p_simultanée(k = 32)), puisque les deux
  conditions doivent être réunies (intersection-union) ; p_dominance = max_k p_unilatérale(k) (Berger, 1982) ;
  p_simultanée(k) = part des tirages bootstrap où max_k |t*(k)| ≥ |t(k)|. Holm porte sur ces deux p-valeurs (paires
  it-em-ppo et it-em-grpo, ponts, T = 4). Un « élargissement » non rejeté après Holm est requalifié « fiabilisation
  seule » si H1 est soutenue, « indéterminé » sinon. Code : `analysis/confirmatoire_partie2.py`, testé sur données
  simulées (`tests/test_confirmatoire_partie2.py`).
- **Questions incomplètes :** seules les questions avec n tirages pour les deux modèles entrent dans l'analyse ; leur
  nombre est rapporté.
- **Lecture d'un gain de frontière à n fini (piège noté en écrivant les tests) :** un agent qui ne fait que
  fiabiliser peut « gagner » des questions où le modèle de départ a une probabilité de succès faible mais non nulle
  et les rate par hasard en n essais. Un gain à k = n sur l'échantillon complet ne distingue donc pas à lui seul
  « élargir » de « fiabiliser les questions rares ». D'où le poids du sous-ensemble difficile (présélection sur
  tirages indépendants : la probabilité de succès du départ y est petite, et on mesure si l'agent la dépasse
  nettement) et la règle : le mot « élargissement » sans réserve n'est employé dans l'article que si le verdict tient
  aussi sur le sous-ensemble difficile.
- **Sessions 2 à 4 :** notebooks générés par `scripts/genere_notebooks_partie2.py`. Vivier de présélection :
  1 000 ponts, graine 43, hors évaluation et hors pilote, 16 tirages, graine de génération 1 ; évaluation du
  sous-ensemble difficile : graine de génération 0 (tirages nouveaux). Vérifié en local : évaluation, pilote et vivier
  sont disjoints (ponts : 500, 50, 1 000 ; il ne reste que 449 comparaisons hors évaluation et pilote).
- **H3, protocole (écrit avant toute donnée de la session 1) :** lecture croisée en plan 2 × 2 (lecteur ∈ {départ,
  agent} × chercheur ∈ {départ, agent}), ponts, 8 trajectoires de chercheur par question (tirages 0 à 7 de la
  session 1). Le lecteur reçoit, après le prompt, les requêtes du chercheur et les passages qu'elles ramènent
  (raisonnement retiré, dans les 4 cases), puis répond sans recherche supplémentaire (T = 0 ; une tentative compte
  sans réponse). Les trajectoires de chercheur sans recherche sont écartées et comptées. Effets par question :
  avantage de lecture (moyenne sur les chercheurs de agent − départ comme lecteur), avantage de recherche, interaction ;
  IC à 95 % par bootstrap à multiplicateurs. H3 soutenue si l'IC de l'avantage de lecture est au-dessus de 0
  (secondaire, sans correction). Code : `scripts/run_lecture.py`, `analysis/lecture_croisee.py` ; session 5.

## Amendement du 13 novembre 2026 (APRÈS la session 5) : H3, variante « réponse forcée »

Constat : sans le raisonnement du chercheur, les agents lecteurs tentent presque toujours une nouvelle recherche
(PPO ne répond que dans 3 à 5 % des lectures, GRPO dans 45 à 63 %, le modèle de départ dans 82 à 96 %). Le protocole
prévu compte ces tentatives comme des échecs : il mesure alors la propension à chercher encore, pas la lecture.
Le résultat prévu est rapporté tel quel (H3 non soutenue ; avantage de recherche positif). Variante ajoutée, déclarée
exploratoire : chaque lecture sans réponse est complétée en imposant « <answer> » après ce que le lecteur a écrit
(`scripts/run_lecture_forcee.py`, session 6), puis la même analyse (`analysis/lecture_croisee.py --force`).
Seul résultat de la session 5 non touché par ce défaut : avec le modèle de départ comme lecteur (qui répond), les
plans de recherche des agents valent plus que les siens.
