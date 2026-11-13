# Journal brut du travail avec les IA

Notes prises au jour le jour. La version mise en forme, à rendre, est `carnet_de_bord.tex` (PDF : `carnet_de_bord.pdf`).

Tenu au fil de l'eau par Claude Code, à la demande de l'étudiant, qui lui a aussi demandé de remplir les rubriques personnelles
le 3 novembre. Il ne contient que ce qui s'est réellement passé, d'après les échanges et les fichiers. À relire par l'étudiant avant
le rendu.

## Journal

### 3 novembre 2026

**Étapes 1 et 2 (préparer).** Projet Claude « memoire-rl-search » déjà ouvert, avec la description du mémoire.
Connecteur OpenAlex : absent au début de la journée, utilisé à partir de la reprise de l'étape 5 ci-dessous.

**Étape 5, première passe (hors OpenAlex).** Revue de littérature faite avec Claude Code sur le sujet du mémoire
(agents de recherche entraînés par RL contre harnais fixes). Résultat : 91 articles classés en 9 thèmes, avec
volet, priorité et rôle dans le mémoire (`docs/litterature/`).
- Sources essayées et bloquées : API arXiv, Semantic Scholar, Crossref, Hugging Face API depuis le shell (erreur 403 du
  proxy ou nom d'hôte introuvable) ; WebFetch refusait `export.arxiv.org`.
- Contournement : recherche par thème sur Hugging Face Papers (index arXiv), puis requête de l'API arXiv depuis le
  navigateur intégré, après autorisation de l'étudiant.
- Vérification : titre, premier auteur et date de chacune des 91 entrées comparés à l'API arXiv par empreinte, zéro écart.
- Limite reconnue : la trame demande des références vérifiées dans OpenAlex, ce qui n'a pas été fait. Aucun article n'a
  été lu en entier ; les rôles sont des propositions tirées du titre et du résumé.
- Gardé tel quel par l'étudiant : le corpus de 91 articles et ses thèmes. Refait par l'étudiant : rien à ce stade.

**Travail annexe sur git.** L'étudiant a demandé que les commits soient à son nom, sans mention de Claude. Historique
local réécrit (filter-branch), sauvegarde par bundle, règle ajoutée dans `CLAUDE.md`. Le push a dû être lancé par
l'étudiant lui-même (GitHub inaccessible depuis l'environnement de Claude). Incident : un commit contenant encore la
mention est apparu pendant la réécriture, créé par une autre session ouverte sur le dépôt.

**Étape 5 (reprise) : vérification et complément par OpenAlex et Consensus.**
Le premier corpus venait de Hugging Face Papers et de l'API arXiv, sans équation OpenAlex. Je l'ai repris avec la
méthode de la trame.
- Vérification des 91 références avec `resolve_references` (4 lots de 23 au plus) : 91 correspondances exactes par DOI,
  aucune incertaine, aucune rétractée. 29 ont une version publiée en conférence ou revue (DOI ajoutés à `corpus.csv`).
  Un titre a changé : RAG-Gym, publié dans Nature Communications sous un autre titre.
- Équation OpenAlex en trois blocs (objet, mesure, population), trois versions. Version 1 : 1 036 résultats, précision
  7/20 (bruit : réseaux 6G, médecine). Version 2 : 336 résultats, précision inchangée (35 %) malgré un bloc « tâche »
  ajouté, car de nouveaux bruits sont apparus (VQA multimodale, télédétection). Version 3 avec exclusions : 103 résultats,
  environ 15/20 (75 %). Chaque version est dans `docs/litterature/journal_requetes_openalex.md`.
- Consensus, question en version fermée : 10 articles, tous favorables à l'entraînement par RL. Pas de Consensus Meter
  fourni par l'outil. J'y vois un biais de publication, que je garde pour la discussion.
- Boule de neige depuis Search-R1 : seulement 14 citations dans OpenAlex (Consensus en compte plus de 1 600), donc peu
  utile. Cause probable : OpenAlex rattache mal les citations aux notices arXiv.
- Résultat : 23 candidats absents du corpus (dont SEM, directement lié à H3) et 9 exclus avec motif E1 ou E2, dans
  `candidats_openalex_consensus.csv`. Tri fondé sur les titres et résumés seulement.
- Pas encore fait : trier les 83 résultats restants de l'équation 3, tester le rappel sur les cinq articles témoins,
  lire les résumés des cas « Doute ».
- Piège repéré : SEM n'existe dans OpenAlex que sous la forme d'une copie Zenodo douteuse (revue sans lien, auteurs
  incohérents). J'ai retrouvé la vraie notice arXiv par recherche web avant de la retenir.

**Étape 6 : cadre théorique, modèle, hypothèses (première version).**
- Cadre proposé : économie de l'information (Stigler 1961, Simon 1955), références vérifiées dans OpenAlex. Point faible
  relevé : les articles du corpus ne mobilisent pas ces théories (la trame demande qu'elle le soit). À valider avec
  l'enseignante.
- Modèle à quatre variables (mode de décision, précision, familiarité du jeu, nombre de recherches), trois hypothèses
  « une flèche, une hypothèse » (HA1 effet, HA2 modération, HA3 mécanisme). Les hypothèses H3 (pénalité) et H4
  (retriever) du mémoire sont renvoyées en perspectives, car la mini-expérience ne peut pas les tester.

**Étapes 7 et 8 : préparer le terrain.**
- Systèmes A0 à A3 écrits (`systems/baselines.py`) avec leurs prompts, 9 tests sans GPU réussis. Pas encore exécutés
  sur Kaggle : aucun résultat de modèle réel à ce stade.
- Les fichiers de couverture déjà mesurés ont changé le plan. Part des questions dont la réponse est dans le top 3 :
  avec BM25 sur 100 k passages, HotpotQA 7 % et Bamboogle 3 % ; avec E5 sur 1 M passages, NQ 25 %, TriviaQA 44 %,
  HotpotQA 17 %, Bamboogle 10 %. J'ai donc remplacé HotpotQA/Bamboogle par NQ (vu) / TriviaQA (non vu), avec E5 sur 1 M
  (`configs/mini_experience.yaml`). Je garde ce choix comme décision de méthode à justifier dans l'article.

**Étape 12 (avance) : premier jet LaTeX.**
- `docs/article/main.tex`, `refs.bib`, `main.pdf` : 10 pages, introduction, revue, cadre, modèle (TikZ), hypothèses,
  méthode, limites écrites. Résultats, discussion (sauf limites), résumé et conclusion restent à écrire après les runs
  (champs rouges « À remplir »). Bibliographie : 35 références, toutes vérifiées dans OpenAlex. Le gabarit du cours
  (`main.tex` du .zip) n'est pas encore là : le préambule est provisoire.

**Étapes 7 et 8 (suite) : préparation du run Kaggle. Aucun système n'a encore tourné.**
- Tests : 22 réussis dans un venv hors du dépôt (20 existants, 2 ajoutés pour le tirage des questions).
- A5 : `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo-v0.3`, trouvé dans la collection Search-R1-v0.3 de Hugging
  Face. Il existe aussi une variante `-it-em-` : à ne pas confondre.
- Écart repéré, non tranché : d'après leur `config.json`, A4 et A5 partent de `Qwen/Qwen2.5-3B` alors que A0 à A3 utilisent
  `Qwen/Qwen2.5-3B-Instruct`. La comparaison ne change donc pas seulement le harnais. Décision : garder cette
  configuration, sur recommandation de Claude et sans objection de l'étudiant, et le signaler comme limite dans l'article.
- Tirage de mise au point : la graine 7 seule reprenait 3 questions d'évaluation sur NQ et 2 sur TriviaQA (sur 110 tirées).
  Ces questions sont maintenant exclues ; vérifié sur les vrais jeux, 0 recouvrement sur les 10 questions de chaque jeu.
  TriviaQA a été chargé pour la première fois à cette occasion.
- Git : le worktree `.litterature-wt` pointe vers un chemin de VM absent du Mac, et ses verrous sont ouverts par une VM
  active. Les fichiers de littérature ont été commités sans passer par ce worktree. Son retrait attend la réponse de
  l'étudiant.

**Lecture de la trame et contrôle de ce qui manque (après-midi).** Trame lue en entier. Écarts trouvés : carnet incomplet
(tableau des outils limité à 3 lignes), fiche données absente, Open Knowledge Maps non fait, type de revue non choisi
explicitement, tri de l'équation non fait, rappel sur les témoins non testé.

**Étape 5.2 : fin de la vérification.**
- Bibliographie : 35 références sur 35 exactes dans OpenAlex. Mais 9 citaient le préprint alors qu'une version publiée existe,
  et DeepSeek-R1 a changé de titre dans Nature. Corrigées. Pour Moskvoretskii et al., le deuxième auteur diffère entre arXiv
  (Lysyuk) et OpenAlex (Marina) : j'ai repris OpenAlex, à contrôler sur le PDF.
- Équation 3 : 103 résultats le matin, 116 l'après-midi. L'équation exacte n'avait pas été gardée, seulement sa description :
  je ne peux pas dire si l'écart vient de ma réécriture ou de la base. L'OQL est maintenant noté.
- Rappel : 1 témoin sur 5. Trois témoins sont des harnais fixes, que le bloc RL exclut par construction. R1-Searcher manque
  parce que son résumé n'emploie pas les mots de l'équation.
- Tri des 116 titres par Claude : 4 inclus proposés, 58 doutes, 38 exclus, 11 déjà candidats, 5 déjà au corpus. Non relu par
  l'étudiant à ce jour : ce tri reste une proposition.
- Une requête sémantique a trouvé ce que l'équation avait manqué, dont Zhao et al. (2026), une étude contrôlée des méthodes
  d'entraînement. Elle m'oblige à préciser la lacune de l'article : la comparaison contrôlée existe entre méthodes de RL, pas
  à notre connaissance entre RL et harnais fixes. La formulation reste prudente : résumés lus, textes non lus.
- Zhao et al. signalent aussi un défaut de couverture du corpus Wikipédia 2018, celui de notre expérience.

**Article.** Ajout d'une section sur la démarche de la revue, de 6 références et de la précision sur les modèles de A4 et A5.
Non recompilé sur le moment : Claude a affirmé qu'il n'y avait pas de LaTeX sur cette machine, ce qui était faux (`tectonic` était
installé, découvert plus tard).

**Étape 7 : run Kaggle.** Essai A1 réussi du premier coup (version 1 du kernel, NQ, 10 questions, graine 7, E5 sur 1m) :
10 réponses `answer` sur 10, 0 anomalie, code 0, 297 s. E5 sur le second T4 ne gêne pas vLLM (14,41 Gio libres sur 14,56 au
démarrage). EM 0,20 sur 10 questions de mise au point : ne vaut pas comme résultat. Trajectoires lues : le modèle répond parfois
à côté de la réponse attendue. Exemple : « Group 17, Period 5 » pour l'iode, contre « heaviest of the stable halogens » attendu.
Une question de NQ peut donc être difficile à noter en EM. Mode `dev` lancé ensuite (A0 à A3).

**Étape 7 : mise au point de A0 à A3 (version 2 du kernel, 10 questions par jeu, graine 7, aucune question d'évaluation).**
Run terminé, 8 fichiers, code 0 partout. La lecture des trajectoires a montré des défauts que le statut ne montrait pas :
- **A3 n'était pas IRCoT.** 20 questions sur 20 avec une seule recherche : le modèle écrivait tout son raisonnement d'un bloc et le
  code gardait le bloc entier dès qu'il contenait « the answer is ». A3 revenait donc à A1 avec un raisonnement. Bug du code écrit par Claude,
  que les tests n'avaient pas vu : le faux modèle des tests n'écrivait jamais toute sa chaîne d'un coup. Un ancien test figeait même ce
  comportement ; il a été changé. Corrigé : une seule phrase par étape, test ajouté.
- **A3 s'abstenait** (« Not provided in the given documents ») et ce texte comptait comme une réponse. Prompt modifié : si les
  documents n'aident pas, utiliser ses connaissances, comme le prompt de A1.
- **A2 répétait ses requêtes** et finissait sans réponse sur 7 questions sur 20, plus 1 anomalie. Prompt modifié : ne jamais répéter
  une requête, et répondre au mieux si les passages n'aident pas. L'anomalie : le modèle n'a rien généré après une observation.
Ces deux changements de prompt améliorent les harnais fixes : c'est voulu, pour ne pas comparer le RL à un harnais mal réglé. Ils
sont faits avant toute question d'évaluation. Les EM de ce run (10 questions) ne valent pas comme résultat et n'ont pas guidé ces
choix. Trajectoires gardées dans `results/dev/prompts_v1/`. Nouveau run de contrôle : [à faire].

**Étape 6 : reprise du cadre, du modèle et des hypothèses.** Demandée par l'étudiant après lecture de la trame.
- La trame demande si la théorie est mobilisée dans les articles retenus. Stigler et Simon ne l'étaient pas : point faible déjà noté.
- Recherche en texte intégral dans OpenAlex : « information foraging » avec agents de recherche, 58 résultats. PDF contrôlés : dans 4 sur
  5, le terme n'apparaît que dans une même référence, InForage (Qian et Liu, 2025). C'est lui qui mobilise vraiment la théorie : lu dans
  son texte. Sans ce contrôle, on pouvait croire à un usage répandu.
- Décision : cadre du fourragement de l'information (Pirolli et Card, 1999), avec Stigler pour la logique du coût. Simon est retiré :
  la trame conseille une théorie, deux au plus, et rien dans les textes contrôlés ne le mobilisait. Pirolli et Card n'est pas en accès
  libre : présenté d'après InForage, ce que l'article dit clairement.
- Modèle : cinq variables au lieu de quatre. HA3 mêlait deux flèches, contre la règle « une flèche, une hypothèse » : elle devient HA3a
  (mode et nombre de recherches) et HA3b (modération par la connaissance préalable).
- Règles de décision écrites avant tout run complet, avec correction de Holm (ajoutée car HA1 compte 6 comparaisons).
- `refs.bib` : 46 références. Les 6 ajouts sont vérifiés dans OpenAlex.
- Article compilé : 13 pages. Le code d'analyse de ces tests (Holm, écart d'écarts) reste à écrire.

### 9 novembre 2026

**Premier rendu complet de l'article, sur données fictives.** Demandé par l'étudiant : « un premier rendu, même si la
donnée est fausse, mais en suivant la forme du PDF » (la trame), avec « le schéma complet ».
- Données fictives : `analysis/donnees_fictives.py`, graine 2026, 1 200 lignes au format exact de `run_volet_a.py`, chacune
  marquée `"fictif": true`, écrites dans `results/fictif/`. Usage autorisé par la trame (7.6 : tester le programme).
- Analyse : `analysis/analyse_mini.py` applique les règles de décision du tableau 3 (McNemar + Holm, bootstrap) et écrit
  tableaux, valeurs citées et figures dans `docs/article/genere/`. Sur ces données : HA1 et HA2 non soutenues, HA3a et
  HA3b soutenues. Paramètres non réglés pour obtenir un verdict.
- Constat : j'avais donné à A4 9 points d'avance sur A1 pour NQ ; l'échantillon de 100 questions n'en montre que 7, et
  aucune paire n'est significative après Holm. La puissance de l'expérience est faible.
- Article : modèle conceptuel complet (H3 et H4 du mémoire en pointillé), schéma du protocole, résultats, discussion en
  cinq mouvements, résumé et conclusion. Filigrane, encadrés et exposant « fictif » activés automatiquement par le marqueur.
- Compilation avec `tectonic` : deux erreurs corrigées (style TikZ `out`, `\text` sans `amsmath`). 16 pages relues en image.
- Tests : `tests/test_analyse.py` ajouté ; 19 tests réussis dans un venv temporaire (les tests qui demandent `datasets`
  et `bm25s` n'ont pas été lancés ce jour-là).
- Pas de cours ni de gabarit : la mention « à valider avec l'enseignante » devient « avec la direction du mémoire ».
- Carnet de bord mis en forme en LaTeX (`carnet_de_bord.tex`), avec un registre des décisions justifiées ; ce journal
  devient une annexe.

**Deuxième passe du 9 novembre : conformité à la trame.** Demandée par l'étudiant : « relance tout, on continue d'améliorer
l'article […], recheck si on suit bien la trame ».
- Pages du site lues (introduction en entonnoir, abstract en 5 mouvements, types de revue). Écarts trouvés : introduction
  terminée par le plan (doit finir sur la contribution) ; résumé en rubriques sans chiffres (doit être un paragraphe de ~200 mots,
  chiffré, sans citation) ; sous-section « coût » en catalogue d'auteurs. Les trois réécrits. Abstract anglais ajouté.
- Bibliographie vérifiée d'un seul tenant (`resolve_references`, 46 entrées) : 43 exactes par titre ; 3 non trouvées par le
  titre (dont R1-Searcher apparié à tort à R1-Searcher++), exactes par DOI arXiv. 6 entrées passées à leur version publiée.
- Analyse de puissance par simulation (`analysis/puissance.py`) : 200 questions + Holm, écart réel de 10 points détecté
  dans 54 % des cas ; 500 questions par jeu, 96 %. Ajoutée à la méthode et aux limites.
- Annexes de l'article : méthode de la recherche documentaire (journal + OQL), consignes des systèmes, fiche données.
- Fichiers de littérature récupérés de la branche `litterature-corpus` dans `docs/litterature/`.
- `scripts/rendu.sh` relance tout et produit `rendu/PROUX_Valentin_{preprojet.pdf,source.zip,carnet.pdf}`. Le zip a été
  compilé seul dans un dossier vide : il ne lui manque rien.
- Carnet : tableau de conformité à la trame, décisions D17 à D19, chronologie, demandes de l'étudiant citées. L'étudiant
  avait autorisé Claude à inventer ; Claude n'a rien inventé de factuel et l'a écrit.

**Run d'évaluation et article sur les vrais résultats (9 novembre, fin de matinée).**
- Contrôle de A2/A3 déjà fait (kernel v3, commit fcef699) : A2 20/20 réponses sans requête répétée ; A3 3,4 recherches par
  question mais 3 abstentions sur 20. Prompt de A3 gelé (D20). Notebook passé en mode `complet`, kernel v4 lancé à 9 h 18.
- Kernel v4 terminé à 10 h 39 : 1 200 trajectoires, 12 runs à code 0, aucune erreur, mêmes questions pour les 6 systèmes,
  4 324 s de GPU. Trajectoires lues : format Search-R1 correct pour A4 ; anomalies de A2 = vrais ratés de format.
- Verdicts du programme commité avant le run : HA1 soutenue (6/6, écarts de 10 à 17 points, p Holm < 0,001) ; HA2 non
  soutenue (12,8 points sur NQ, 14,3 sur TriviaQA) ; HA3a soutenue (+0,92 recherche) ; HA3b soutenue (+0,74 sur 44 questions).
- Constat non prévu : sur NQ, les trois harnais font moins bien que le modèle seul. Analyse exploratoire (D22) : sur les 44
  questions « connues », les harnais en perdent 18 à 24 après recherche, les agents 6 et 10. Exemples vérifiés dans les passages.
- Résumé, résultats, discussion et conclusion réécrits ; relecture « rapporteur » avec le prompt de la trame : 6 corrections.

**Analyses approfondies (9 novembre, après-midi).** Demandées par l'étudiant (« article pas assez consistant, plus de schémas,
analyse poussée »). `analysis/analyse_approfondie.py`, exploratoire, sur les 1 200 trajectoires :
- Trouver ou lire : réponse présente dans les passages pour 38 % des trajectoires des harnais et 42 % de celles des agents ;
  EM si présente 50 % contre 69 % ; EM si absente 6 % contre 12 %.
- Décomposition contrefactuelle de l'écart (13,6 points [9,2 ; 18,2]) : trouver 2,1 [0,9 ; 3,4], lire 8,0 [4,9 ; 11,4],
  répondre sans appui 3,5 [0,7 ; 6,4]. Bootstrap sur les questions.
- Requêtes : la 1re requête trouve la réponse pour 31 à 37 % des questions selon le système ; A5 recopie la question à 90 %,
  A4 à 70 % ; requête répétée pour 29 % (A4) et 26 % (A5) des questions. Jetons : environ 150 à 166 pour les systèmes itératifs.
- Gains / pertes face à A0, typologie automatique des erreurs, matrice des 15 comparaisons (Bonferroni), arbre TikZ.

**Analyses avancées (9 novembre, fin d'après-midi).** `analysis/analyse_avancee.py`, exploratoire :
- Régression logistique (1 000 trajectoires, SE groupées par question) : RL OR 2,62 [1,44 ; 4,78] ; réponse dans les passages
  OR 12,96 ; RL × réponse dans les passages OR 1,03 (p = 0,94) ; connue OR 5,61 ; TriviaQA OR 3,90. Interprétation « lisent
  mieux » corrigée en « répondent mieux à information égale ».
- Gain marginal des recherches : 1re ≈ 31 à 37 % ; 2e +1 à +3 points (A4 +10) ; 3e et 4e 0 à 1,5. IRCoT fait ≥ 3 recherches
  sur 80 % des questions, A4 sur 66 %.
- Sensibilité HA1 : souple +12,5 [8,1 ; 17,2], 6/6 ; F1 +15,5 [11,3 ; 19,9]. Odds ratios McNemar 3,8 à 6,0.
- Par jeu : trouver 1,2 (NQ) et 2,7 (TQA) points ; lire 6,5 et 9,7 ; sans appui 5,1 et 2,0.
- Complémentarité : aucun système ne répond juste à 52 % des questions ; Jaccard A4/A5 0,69, harnais 0,48 en moyenne.
- Schéma TikZ de la boucle d'entraînement de Search-R1 ajouté à la méthode.

**Formalisation (9 novembre, soir).** Sur proposition de Claude, décidée par l'étudiant : récompense de Search-R1 (r = EM) et
récompense avec coût (r = EM − c · n_rech) dans le cadre théorique, avec lecture en clair ; annexe D sur le gradient de
politique, l'avantage, PPO, GRPO, la pénalité KL et le masquage des passages ; une phrase dans l'état de l'art.

### 11 et 12 novembre 2026

**Réorientation du mémoire.** L'étudiant a apporté une piste rédigée ailleurs : le renforcement élargit-il ce qu'un agent
sait faire (pass@k), ou le fiabilise-t-il seulement ? Vérification de chaque référence (arXiv, NeurIPS, Hugging Face) :
deux introuvables (« RLVR Sharpens the Pencil », Baseten) retirées, Fireworks corrigé (étude Genspark d'octobre 2025,
plateforme en preview), nuances ajoutées (Zhai et al. entraînent leur agent sur 200 questions ; Yang trouve des
croisements certifiés à 1,5B). Variantes Instruct de Search-R1 vérifiées. Décision de l'étudiant (12 novembre) : la
partie 2 devient le cœur du mémoire, l'article rendu en devient la partie 1.
- Étape 1 (sans GPU) : `analysis/passk.py` (12 tests), environnement fermé HotpotQA, boucle à n tirages, note vérifiée,
  plan v0. Découvertes : 447 réponses oui/non à exclure ; deux paragraphes utiles ramenés par la question seule dans
  45 % des cas seulement ; risque de plafond de pass@n.
- Pilote Kaggle lancé le 12 novembre (version 1 de `memoire-partie2-pilote`).
- **Pilote v1 (kernel v1, 12 novembre, 15 h 21 à 16 h 19) : en erreur.** Trois runs sur six plantent : contexte de
  4 096 jetons dépassé (paragraphes HotpotQA longs ; un seul dépassement fait rejeter tout le lot par vLLM). Un
  quatrième finit ses 1 600 trajectoires mais plante à la relecture : `splitlines()` coupe sur U+2028, présent dans
  certains passages. Les deux bugs venaient du code écrit par Claude ; corrigés partout (y compris dans les scripts de la
  partie 1), avec un test chacun. Résultats indicatifs des runs complets : paire base, agent 0,617 contre 0,154 à k = 1,
  mais 0,740 contre 0,760 à k = 32 (fiabilisation sans élargissement, comme Yue et al.) ; 24 % des questions hors de
  portée du modèle de départ (pas de plafond) ; agent Instruct GRPO tronqué à 28,5 % avec 512 jetons par tour ; modèle
  de base au bon format 61 % du temps avec gabarit de chat ; débit environ 2 trajectoires par seconde et par T4.
  Pilote v2 : 1 024 jetons par tour, contexte de 8 192, les six runs avec les mêmes réglages.
- Recherche documentaire de la partie 2 (12 novembre) : équation OpenAlex P1 (302 résultats, environ 55 % de précision),
  P2 agents en deux versions (642 puis 138, précision faible : littérature mince), boule de neige sur Zhai et al. et
  Yue et al., deux questions Consensus, index arXiv. Corpus de 43 articles, tous vérifiés (42 exacts, 1 incertain).
  Pas de recoupement : Zhu et al. (2611.06483, Pass@(K,T) sur des agents) ne touche ni la recherche documentaire ni
  Search-R1. Biais de publication : presque tous les « oui » proposent une nouvelle méthode. Conséquences : sous-ensemble
  difficile présélectionné sur des tirages indépendants (le test montre qu'une réutilisation fabrique un élargissement),
  T comme variable, bras harnais optionnel, prédictions écrites avant le run.

## Les outils utilisés

| Étape | Outil | Ce que l'outil a fait | Ce que j'ai gardé, refait ou vérifié |
| --- | --- | --- | --- |
| 5 | Claude Code, Hugging Face Papers, API arXiv | recherche, tri, corpus de 91 articles | Corpus gardé tel quel après recoupement automatique avec l'API arXiv (0 écart). Aucun article lu en entier : les rôles dans le mémoire sont des propositions à confirmer |
| 5 | OpenAlex (équations, `resolve_references`, `list_citations`) | vérification des 91 références, trois équations, candidats | Résultats gardés. Équation 1 abandonnée après mesure de sa précision (35 %). Tri des 83 résultats restants et test sur les articles témoins non faits |
| 3 | Consensus | test de la question en version fermée, 10 articles | Résultats comparés au corpus (4 absents). Pas de Consensus Meter fourni par l'outil, donc pas de répartition à reporter |
| 3 | Open Knowledge Maps | non utilisé | Étape de la trame non faite à ce jour |
| 5.2 | OpenAlex via Claude Code | vérification de `refs.bib`, liste de l'équation 3, tri proposé | Références corrigées vers les versions publiées. Tri : proposition de Claude, non relue par l'étudiant à ce jour |
| 6 | Claude Code, OpenAlex, PDF d'articles | cadre théorique, modèle, hypothèses, schémas TikZ | Première version (Stigler, Simon) refaite après vérification dans les textes : Simon retiré, fourragement de l'information ajouté. L'étudiant a demandé cette étape et n'a pas réécrit le texte. Cadre pas encore soumis à l'enseignante. NotebookLM non utilisé |
| 7 | Claude Code, Kaggle (GPU T4) | systèmes A0 à A3, script de run, notebook | Trame prévue avec Colab : Kaggle choisi pour le GPU. Bug de A3 trouvé à la lecture des trajectoires. L'étudiant a fixé le protocole et validé les lancements ; le code n'a pas été refait à la main |
| 12 | Claude Code | premier jet LaTeX, modèle TikZ | Gabarit du cours absent, préambule provisoire. Compilé avec `tectonic`, pages relues en image par Claude. Pas encore importé dans Overleaf |

## Mes difficultés

- Aucune API bibliographique joignable depuis l'environnement de Claude, sauf arXiv via le navigateur : plusieurs
  échecs avant de trouver ce contournement.
- La première sélection d'articles venait de Claude, pas d'une équation OpenAlex ; reprise avec trois équations, dont la
  première avait 35 % de précision.
- L'outil Consensus ne donne pas le Consensus Meter attendu par la trame.
- OpenAlex compte très peu de citations pour les préprints arXiv récents : la boule de neige y est peu fiable.
- Limite de 30 requêtes par seconde chez OpenAlex : trois lots lancés ensemble ont été refusés, relancés un par un.
- Un commit contenant encore la mention de Claude est apparu pendant la réécriture de l'historique (autre session
  ouverte) : il a fallu adapter le script de push.
- Gabarit LaTeX : il n'y a pas de cours, c'est mon sujet de mémoire. La trame sert de référence ; le préambule de l'article est le mien (précisé le 9 novembre).

- La bibliographie vérifiée (35 sur 35 exactes) citait quand même 9 préprints à la place des versions publiées : « la référence
  existe » ne veut pas dire « c'est la bonne version ». Résolu.
- L'équation exacte du matin n'avait pas été conservée : impossible d'expliquer l'écart entre 103 et 116. Non résolu.
- Sans filtre, le tirage de mise au point (graine 7) reprenait 3 questions d'évaluation sur NQ et 2 sur TriviaQA. Résolu.
- Le worktree de la littérature pointait vers une machine virtuelle : commit par un index temporaire, puis retrait avec
  l'accord de l'étudiant.

## Mon rapport d'étonnement

Écarts réels entre l'attendu et l'observé (à confirmer ou corriger par moi) :
- Une équation large paraissait suffisante ; mesurée sur 20 résultats au hasard, elle n'avait que 35 % de précision.
- OpenAlex ne compte que 14 citations pour Search-R1, alors que Consensus en compte plus de 1 600 : les chiffres de
  citation dépendent de l'outil.
- Observé : une équation précise à environ 75 % ne retrouve qu'un témoin sur cinq, et une requête en langage naturel a trouvé
  une étude qu'elle avait manquée. L'attendu : le journal l'avait « retenue » au vu de sa précision.
- Attendu : la consigne supposait qu'une autre graine donnerait d'autres questions (« graine 7, PAS les 100 questions d'évaluation »).
  Observé : 3 questions sur 110 en commun pour NQ, 2 pour TriviaQA.
- Attendu : les tests sans GPU, tous réussis, laissaient penser que A3 fonctionnait. Observé : A3 ne faisait qu'une recherche par
  question, sans aucune erreur ; seule la lecture des trajectoires l'a montré.
- Observé : le résumé de Wu et al. parle d'« un modèle » pour les 27,7 % ; le texte précise qu'il s'agit de Search-R1, un de nos
  systèmes. Lire le texte change ce qu'on peut affirmer.
- Un article utile (SEM) n'existe dans OpenAlex que sous une copie douteuse : une référence trouvée par un outil doit
  être recoupée avec la source.

## Ce que j'aimerais approfondir

- Lire les 58 « doutes » du tri de l'équation 3, puis améliorer son rappel.
- Lire en entier les articles « essentiels » du corpus (statut « non lu » partout aujourd'hui).
- Comprendre pourquoi les articles de méthode ne publient presque que des gains (biais de publication) et comment
  l'expérience du mémoire peut le contourner.

## Annexes à joindre

- Carnet des requêtes : `docs/litterature/journal_requetes_openalex.md` et `journal_requetes_complement.md`.
- Fiche données : `docs/article/fiche_donnees.md`.

## 12 novembre 2026 (soir) : pilote v2

Kernel v2 terminé (COMPLETE, journaux lus) : 6 runs × 1 600 trajectoires, 50 questions de pont (graine 7, hors évaluation),
n = 32, T = 4, température 0,7. Seule « erreur » des journaux : avertissement FlashAttention 2 non disponible sur T4 (sans effet).
- EM par trajectoire : Instruct 0,358 ; it-em-grpo 0,371 ; it-em-ppo 0,431 ; Qwen2.5-3B 0,139 (chat) / 0,200 (brut) ; em-ppo 0,611.
- Format correct du modèle de base : 62 % avec gabarit de chat, 77 % en prompt brut.
- Défaut trouvé : it-em-ppo continue après sa réponse et répète « <answer>X</answer> » jusqu'à 1 024 jetons
  (42 % des trajectoires, 0,24 trajectoire/s contre 1,3 à 1,8 pour les autres). Correction : arrêt aussi à `</answer>`,
  comme le post-traitement officiel de Search-R1. Rescoré hors ligne au premier `</answer>` : conclusions indicatives inchangées.
- Indicatif (50 questions, aucune conclusion) : paires Instruct, D(1) ≈ +0,03 à +0,05, D(32) ≈ −0,04 ; frontière 40 contre 42.
  Paire de base : D(1) +0,47, D(32) +0,02. Questions hors de portée du modèle Instruct : 16 %.

## 13 novembre 2026 : sessions 1 et 2 de l'étude principale (partie 2)

Session 1 (ponts) et session 2 (comparaisons) terminées dans la nuit (COMPLETE, journaux lus, aucune erreur hors
l'avertissement FlashAttention 2) : 4 × 24 000 trajectoires, 500 questions complètes par run. Durées : Instruct
environ 1 h 15 par moitié (1,8 traj./s), GRPO 5 h 50 (0,8), PPO 7 h à 8 h (0,6).
- Confirmatoire (plan gelé, sans modification) : ponts, H1 soutenue (PPO +0,065, GRPO +0,038 à k = 1), H2
  « fiabilisation seule » pour les deux paires (D(32) +0,010 et +0,008, Holm p = 1 ; frontière 351 et 350 contre 346).
  Comparaisons : pas de gain au premier essai, frontière inchangée.
- Erreur de code trouvée : le statut « budget épuisé » se déclenchait sur n'importe quel `<search>` de la trajectoire.
  Corrigé (seule une recherche demandée après le dernier bloc de passages compte) ; l'EM n'en dépend pas.
- Exploratoire (`analysis/exploratoire_partie2.py`, tirages séparés 0-15 / 16-31) : gain en U inversé selon la
  difficulté ; pertes sur les questions faciles dues aux sorties sans réponse (GRPO : format ; PPO : budget T = 4).
  Questions jamais résolues par le départ sur 0-15 : 85 % le restent sur 16-31 pour le départ, 70 % pour PPO.
- Vérifié pour H3 : les passages recalculés depuis les requêtes sont identiques à ceux de la trajectoire dans 99,4 %
  des cas ; les écarts viennent de faux blocs `<information>` écrits par le modèle lui-même.
- Session 5 (lecture croisée) lancée le 13 au matin ; session 3 en attente (quota).

## 13 novembre 2026 (après-midi) : sessions 5 et 6 (H3), qualité de recherche

- Session 5 (lecture croisée prévue, 7 cases, environ 23 000 lectures, COMPLETE, journaux propres) : H3 non soutenue
  par la règle prévue, mais le protocole mesure la propension à chercher encore : sans son raisonnement, PPO lecteur ne
  répond que dans 3 à 5 % des cas (GRPO 45 à 63 %, départ 82 à 96 %). Avec le départ comme lecteur, les plans des
  agents valent 8 à 9 points de plus.
- Amendement déclaré (plan, section dédiée) : variante « réponse forcée », session 6 (COMPLETE, environ 10 000
  lectures complétées, 1 h). Lecture +0,048 et +0,049 (IC au-dessus de 0), recherche +0,077 et +0,061.
- Sans GPU (`analysis/recherche_partie2.py`) : les agents trouvent les deux paragraphes utiles dans 54 % des
  trajectoires de pont, le départ dans 32 %. Décomposition exacte : tout le gain d'EM naturel vient de « trouver »
  (+0,064) ; « lire » nul (PPO) ou négatif (GRPO), car l'agent ne répond pas dans 14 à 20 % des cas où il a trouvé.
- État de l'art de la partie 2 rédigé dans le chapitre. OpenAlex en panne (erreur serveur) : 7 articles cités par
  identifiant arXiv ou DOI, à convertir en entrées bibliographiques.
- Sensibilité au format (`analysis/sensibilite_format_partie2.py`) : GRPO déclare souvent sa réponse hors balises
  (`&lt;answer&gt;`, tableau HTML, chevrons). Réparation stricte (déclaration seule, puis EM) : verdict H2 inchangé ;
  GRPO +0,063 au premier essai sur les ponts, +0,064 sur les comparaisons. La notation « mention » donnerait un
  élargissement partout mais avantage les agents (plus de sorties sans réponse) : rapportée comme borne haute seulement.
- Mémoire complet assemblé (`docs/article/memoire.tex`, 55 pages, `scripts/assemble_memoire.py`). Bibliographie de la
  partie 2 complétée : 8 références vérifiées sur l'API arXiv et Crossref (OpenAlex en panne).
