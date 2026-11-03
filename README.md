# Chercher ou répondre ? RL contre harnais fixes pour les petits agents de recherche

Mémoire de Master 2 (IA). Ce dépôt contient le code des expériences.

**Question :** à corpus et questions identiques, un agent de recherche entraîné par RL
(Search-R1) décide-t-il mieux *quand* et *quoi* chercher qu'un harnais fixe
(RAG, ReAct, IRCoT), et à quel coût ?

## Volets

| Volet | Contenu | Calcul |
| --- | --- | --- |
| A | Comparaison contrôlée de 6 systèmes (A0 à A5) sur 7 jeux de QA | Inférence seule |
| B | Mini-entraînement GRPO (TRL) avec plusieurs récompenses | Exploratoire |
| C | Taxonomie des échecs sur 300 trajectoires | Annotation |

| Code | Système |
| --- | --- |
| A0 | Réponse directe, sans recherche |
| A1 | RAG classique (1 recherche) |
| A2 | ReAct par prompt |
| A3 | IRCoT |
| A4 | Search-R1 Qwen2.5-3B PPO v0.3 |
| A5 | Search-R1 Qwen2.5-3B GRPO v0.3 |

## Structure

```
configs/     un YAML par expérience
prompts/     prompts des systèmes (searchr1.txt = texte exact du dépôt Search-R1)
data/        chargement des jeux de QA, construction du corpus
retrieval/   index BM25 (bm25s) et dense (E5 + FAISS)
systems/     un module par système
eval/        métriques (EM, F1), oracle, statistiques
train/       volet B (GRPO)
scripts/     points d'entrée (smoke test, runs)
notebooks/   notebook Kaggle prêt à lancer (smoke test)
analysis/    notebooks de figures
results/     JSONL des trajectoires (non versionné)
```

## Démarrage

```bash
pip install -r requirements.txt
python -m pytest tests/                 # vérifie les métriques
python scripts/smoke_test.py --n 20     # sur Kaggle, GPU T4
```

## Règles du projet

- Chaque trajectoire est écrite en JSONL dans `results/` : question, système, texte complet,
  requêtes, documents, réponse, EM, F1, tokens, durée. Les analyses partent de ces fichiers.
- Résultats écrits par lots : un run coupé reprend là où il s'est arrêté.
- Graines et versions de bibliothèques figées. Protocole gelé en semaine 5.
- Données et index lourds : jamais dans git. Dataset Kaggle privé ou Hub Hugging Face.

## Ressources

- Search-R1 : https://github.com/PeterGriffinJin/Search-R1 (Apache 2.0)
- Checkpoints : https://huggingface.co/collections/PeterJinGo/search-r1-v03
- Jeux de QA : `RUC-NLPIR/FlashRAG_datasets` (format utilisé par Search-R1)
- Note de projet complète : document Claude « Mémoire M2 — Agents de recherche entraînés par RL »
