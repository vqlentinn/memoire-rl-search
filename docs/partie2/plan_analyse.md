# Partie 2 : plan d'analyse

**Statut : version 0.1, avant le pilote (12 novembre 2026), après la recherche documentaire.** Les paramètres marqués « [pilote] » seront fixés
d'après la session pilote, puis le plan sera gelé et commité **avant** tout run principal. Toute modification
ultérieure sera datée et signalée dans l'article.

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
pour les paires Instruct ; pour les paires base, gabarit de chat ou prompt brut selon le pilote [pilote].

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

- Ponts et comparaisons : N questions de chaque type [pilote ; cible 500], tirées avec la graine 42
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
  simulation. Taille du vivier [pilote], cible : 150 à 200 questions difficiles.

## Mesures

- n = 32 générations par question et par modèle, température 0,7 (comme Zhai et al.) [pilote : n peut descendre à 16].
- T (recherches au plus) : 4 et 0 d'abord ; 1 et 2 ensuite si le budget le permet. Zhu et al. (2611.06483)
  montrent que l'avantage du RL dépend du budget d'interaction et peut s'inverser à petit T. T = 0 : aucune recherche permise ;
  une trajectoire qui tente de chercher se termine sans réponse.
- 512 jetons par tour, identiques pour les deux modèles d'une paire.
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
