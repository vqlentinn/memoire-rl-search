# Fiche données (étape 7 de la trame)

Version du 3 novembre 2026, avant le run complet. Les champs vides seront remplis depuis les fichiers `results/`.

| Rubrique | Contenu |
| --- | --- |
| Source : questions | Existante. `RUC-NLPIR/FlashRAG_datasets` (Hugging Face), configurations `nq` et `triviaqa`, split `test` : 3 610 et 11 313 questions |
| Source : corpus | Existante. `maknee/wiki-18-subsets` (Hugging Face), `wiki-18-1m.jsonl.gz` et `e5_Flat-1m.index`. Jeu tiers sans fiche descriptive : l'alignement index/corpus a été vérifié sur 50 passages (similarité min 0,9997) |
| Source : modèles | `Qwen/Qwen2.5-3B-Instruct` (A0 à A3) ; `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-ppo-v0.3` (A4) ; `...-em-grpo-v0.3` (A5) |
| Source : trajectoires | Produites par ce projet (`scripts/run_volet_a.py`, Kaggle GPU T4). Ce sont des sorties réelles de modèles, pas des données fabriquées |
| Dates | Collecte des questions et du corpus : à la date de chaque run. Période couverte par le corpus : instantané Wikipédia 2018. Date du run complet : [à remplir] |
| Une ligne | Une question traitée par un système. 100 questions × 2 jeux × 6 systèmes, soit 1 200 lignes attendues. Lignes obtenues : [à remplir] |
| Échantillon | 100 questions tirées avec la graine 42. Mise au point sur un autre tirage (graine 7), sans aucune question d'évaluation |
| Droit d'utilisation | [à vérifier avant le rendu : licences de FlashRAG_datasets, wiki-18-subsets, Qwen2.5 et des checkpoints Search-R1] |
| Qui manque | Corpus réduit à 1 million de passages : réponse dans le top 3 pour 25 % des questions NQ et 44 % des TriviaQA (estimation haute, 500 questions). Corpus Wikipédia de 2018 : les faits récents manquent. Questions en anglais seulement |
| Ce que j'ai modifié | Aucun nettoyage ni retrait de question. Seul ajout : « ? » en fin de question s'il manque, comme Search-R1. Réponses normalisées pour le calcul de l'EM et du F1, sans toucher au texte brut des trajectoires |
| Test final | Un camarade relance le notebook et obtient les mêmes chiffres : [non fait] |
