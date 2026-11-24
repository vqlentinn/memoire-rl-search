#!/usr/bin/env bash
# Après une session Kaggle de la partie 2 : télécharge les sorties, vérifie le journal et les trajectoires,
# relie les fichiers dans results/partie2/ et relance les analyses.
#   PY=<python avec matplotlib> scripts/apres_session.sh 3
# Session faite sur Colab : copier le dossier Drive MyDrive/memoire-rl-search/sN/ dans results/partie2_sN/, puis
#   LOCAL=1 PY=<python> scripts/apres_session.sh N   (pas de téléchargement Kaggle)
# Le statut COMPLETE ne prouve rien : ce script s'arrête si un run a échoué ou si une question est incomplète.
set -euo pipefail
S="$1"
PY="${PY:-python3}"
K="${KAGGLE:-$HOME/.venvs/memoire-rl-search/bin/kaggle}"
cd "$(dirname "$0")/.."
DEST="results/partie2_s${S}"
mkdir -p "$DEST"
if [ -z "${LOCAL:-}" ]; then
  "$K" kernels output "valentinproux/memoire-partie2-s${S}" -p "$DEST" -o >/dev/null
fi

echo "== Journal : erreurs Python"
grep -h -E "Traceback|Error|RuntimeError" "$DEST"/*.log "$DEST"/logs/*.log 2>/dev/null | sort | uniq -c | head -20 || true

echo "== Contrôle des fichiers (questions complètes, statuts)"
"$PY" - "$DEST" <<'EOF'
import json, sys
from collections import Counter
from pathlib import Path
d = Path(sys.argv[1])
res = json.loads(next(d.glob("resume_s*.json")).read_text())
echecs = [s for s in res["status"] if s[2] != 0]
for nom, v in res["resume"].items():
    print(f"  {nom[:90]:90} traj {v['trajectoires']:6} complètes {v['questions_completes']:4} EM {v['em']:.3f} {v['statuts']}")
print("  runs en échec :", echecs or "aucun")
sys.exit(1 if echecs else 0)
EOF

echo "== Liens dans results/partie2/"
for f in "$DEST"/results/partie2/*.jsonl; do
  ln -sf "../partie2_s${S}/results/partie2/$(basename "$f")" "results/partie2/$(basename "$f")"
done
[ -f "$DEST/difficiles_ponts.txt" ] && echo "  difficiles_ponts.txt : $(wc -l < "$DEST/difficiles_ponts.txt") questions"

echo "== Analyses"
"$PY" analysis/confirmatoire_partie2.py --indir results/partie2
"$PY" analysis/sensibilite_format_partie2.py --indir results/partie2 | tail -12
"$PY" analysis/complements_partie2.py --indir results/partie2 | tail -10
"$PY" analysis/miroir_partie2.py --indir results/partie2
"$PY" analysis/replication_partie2.py --indir results/partie2
"$PY" analysis/synthese_partie2.py
"$PY" analysis/cout_partie2.py --indir results/partie2
if ls results/partie2/*it-sft-* >/dev/null 2>&1; then "$PY" analysis/sft_partie2.py --indir results/partie2; fi
"$PY" analysis/budget_partie2.py --indir results/partie2
"$PY" analysis/equivalents_partie2.py --indir results/partie2
"$PY" analysis/diversite_partie2.py --indir results/partie2
"$PY" analysis/modeles_fiabilisation_partie2.py --indir results/partie2

# 2WikiMultiHopQA (sessions 12 et 13) : seulement quand les agents et le départ (2 graines × 2 moitiés) sont là
n2w() { ls results/partie2/ 2>/dev/null | grep "_compositional_" | grep -c "$1" || true; }
if [ "$(n2w '^Qwen2.5-3B-Instruct_')" -ge 4 ] && [ "$(n2w 'it-em-')" -ge 2 ]; then
  echo "== 2WikiMultiHopQA"
  "$PY" analysis/equivalents_partie2.py --indir results/partie2 --type compositional
  "$PY" analysis/diversite_partie2.py --indir results/partie2 --type compositional
  "$PY" analysis/modeles_fiabilisation_partie2.py --indir results/partie2 --type compositional
fi

# 7B (sessions 14 à 17) : seulement quand les deux moitiés de l'agent et du départ sont là
n7b() { ls results/partie2/ 2>/dev/null | grep "_bridge_" | grep -c "$1" || true; }
if [ "$(n7b '^qwen2.5-7b-it-em-grpo')" -ge 2 ] && [ "$(n7b '^Qwen2.5-7B-Instruct_')" -ge 2 ]; then
  echo "== 7B"
  "$PY" analysis/equivalents_partie2.py --indir results/partie2 --taille 7
  "$PY" analysis/diversite_partie2.py --indir results/partie2 --taille 7
  "$PY" analysis/modeles_fiabilisation_partie2.py --indir results/partie2 --taille 7
fi
