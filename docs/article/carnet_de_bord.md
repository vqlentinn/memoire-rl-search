# Carnet de bord : travail avec les IA

Brouillon tenu au fil de l'eau par Claude Code, à relire et à reprendre par l'étudiant avant le rendu. Il ne contient que ce
qui s'est réellement passé. Les passages entre crochets `[à compléter]` sont ceux que seul l'étudiant peut écrire (son
ressenti, ses choix, ce qu'il a refait lui-même).

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
- Gardé tel quel par l'étudiant : [à compléter]. Refait par l'étudiant : [à compléter].

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
  `Qwen/Qwen2.5-3B-Instruct`. La comparaison ne change donc pas seulement le harnais. [à compléter : décision]
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
- Tri des 116 titres par Claude : 4 inclus proposés, 58 doutes, 38 exclus, 11 déjà candidats, 5 déjà au corpus. Décision de
  l'étudiant : [à compléter].
- Une requête sémantique a trouvé ce que l'équation avait manqué, dont Zhao et al. (2026), une étude contrôlée des méthodes
  d'entraînement. Elle m'oblige à préciser la lacune de l'article : la comparaison contrôlée existe entre méthodes de RL, pas
  à notre connaissance entre RL et harnais fixes. La formulation reste prudente : résumés lus, textes non lus.
- Zhao et al. signalent aussi un défaut de couverture du corpus Wikipédia 2018, celui de notre expérience.

**Article.** Ajout d'une section sur la démarche de la revue, de 6 références et de la précision sur les modèles de A4 et A5.
Non recompilé : pas de LaTeX sur cette machine, `main.pdf` est donc en retard sur `main.tex`.

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

## Les outils utilisés

| Étape | Outil | Ce que l'outil a fait | Ce que j'ai gardé, refait ou vérifié |
| --- | --- | --- | --- |
| 5 | Claude Code, Hugging Face Papers, API arXiv | recherche, tri, corpus de 91 articles | Corpus gardé tel quel après recoupement automatique avec l'API arXiv (0 écart). Aucun article lu en entier : les rôles dans le mémoire sont des propositions à confirmer |
| 5 | OpenAlex (équations, `resolve_references`, `list_citations`) | vérification des 91 références, trois équations, candidats | Résultats gardés. Équation 1 abandonnée après mesure de sa précision (35 %). Tri des 83 résultats restants et test sur les articles témoins non faits |
| 3 | Consensus | test de la question en version fermée, 10 articles | Résultats comparés au corpus (4 absents). Pas de Consensus Meter fourni par l'outil, donc pas de répartition à reporter |
| 3 | Open Knowledge Maps | non utilisé à ce stade | [à compléter : à faire par l'étudiant] |
| 5.2 | OpenAlex via Claude Code | vérification de `refs.bib`, liste de l'équation 3, tri proposé | Références corrigées vers les versions publiées. Tri : proposition seulement, [à compléter : décisions de l'étudiant] |
| 6 | Claude Code, OpenAlex | cadre théorique, modèle, hypothèses | Stigler et Simon vérifiés dans OpenAlex. [à compléter : ce que l'étudiant a gardé ou refait, avis de l'enseignante] |
| 7 | Claude Code, Kaggle (GPU T4) | systèmes A0 à A3, script de run, notebook | Trame prévue avec Colab : Kaggle choisi pour le GPU. Tests relancés (22 réussis). [à compléter] |
| 12 | Claude Code | premier jet LaTeX, modèle TikZ | Gabarit du cours absent, préambule provisoire. [à compléter] |

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
- Le gabarit LaTeX du cours (`main.tex`) n'est pas encore en ma possession : à demander à l'enseignante.

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
  une étude qu'elle avait manquée. [à compléter : ce que j'attendais]
- Observé : deux tirages de graines différentes se recoupent. [à compléter : ce que j'attendais]
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
