# Carnet de bord : travail avec les IA

Brouillon tenu au fil de l'eau. Il ne contient que ce qui s'est réellement passé. Les passages entre crochets
`[à compléter]` sont ceux que seul l'étudiant peut écrire (son ressenti, ses choix, ce qu'il a refait lui-même).

## Journal

### 3 novembre 2026

**Étapes 1 et 2 (préparer).** Projet Claude « memoire-rl-search » déjà ouvert, avec la description du mémoire.
Connecteur OpenAlex : pas encore branché. [à compléter : date où il le sera]

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

## Les outils utilisés

| Étape | Outil | Ce que l'outil a fait | Ce que j'ai gardé, refait ou vérifié |
| --- | --- | --- | --- |
| 5 | Claude Code, Hugging Face Papers, API arXiv | recherche, tri, corpus de 91 articles | Corpus gardé tel quel après recoupement automatique avec l'API arXiv (0 écart). Aucun article lu en entier : les rôles dans le mémoire sont des propositions à confirmer |
| 5 | OpenAlex (équations, `resolve_references`, `list_citations`) | vérification des 91 références, trois équations, candidats | Résultats gardés. Équation 1 abandonnée après mesure de sa précision (35 %). Tri des 83 résultats restants et test sur les articles témoins non faits |
| 3 | Consensus | test de la question en version fermée, 10 articles | Résultats comparés au corpus (4 absents). Pas de Consensus Meter fourni par l'outil, donc pas de répartition à reporter |

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

## Mon rapport d'étonnement

Écarts réels entre l'attendu et l'observé (à confirmer ou corriger par moi) :
- Une équation large paraissait suffisante ; mesurée sur 20 résultats au hasard, elle n'avait que 35 % de précision.
- OpenAlex ne compte que 14 citations pour Search-R1, alors que Consensus en compte plus de 1 600 : les chiffres de
  citation dépendent de l'outil.
- Un article utile (SEM) n'existe dans OpenAlex que sous une copie douteuse : une référence trouvée par un outil doit
  être recoupée avec la source.

## Ce que j'aimerais approfondir

- Trier les 83 résultats restants de l'équation 3 et mesurer le rappel sur les cinq articles témoins.
- Lire en entier les articles « essentiels » du corpus (statut « non lu » partout aujourd'hui).
- Comprendre pourquoi les articles de méthode ne publient presque que des gains (biais de publication) et comment
  l'expérience du mémoire peut le contourner.
