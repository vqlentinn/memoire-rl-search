#!/usr/bin/env bash
# Relance toute la chaîne du rendu : données -> analyse -> puissance -> PDF -> dossier rendu/.
#
#   PY=/chemin/vers/python scripts/rendu.sh           # données fictives (tant qu'il n'y a pas de vrais runs)
#   PY=... RESULTATS=results scripts/rendu.sh          # vrais fichiers JSONL du run d'évaluation
#
# Le dossier rendu/ (non versionné) contient les trois fichiers demandés par la trame (étape 12) :
# PROUX_Valentin_preprojet.pdf, PROUX_Valentin_source.zip (à importer dans Overleaf), PROUX_Valentin_carnet.pdf.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
RESULTATS="${RESULTATS:-results/fictif}"

if [ "$RESULTATS" = "results/fictif" ]; then
  echo "== Données FICTIVES (analysis/donnees_fictives.py)"
  "$PY" analysis/donnees_fictives.py --outdir results/fictif
fi
echo "== Analyse de $RESULTATS"
"$PY" analysis/analyse_mini.py --indir "$RESULTATS"
echo "== Puissance (simulation)"
"$PY" analysis/puissance.py

echo "== Compilation"
( cd docs/article
  for doc in main carnet_de_bord; do
    tectonic -X compile "$doc.tex" 2>&1 | grep -E "^error|Writing \`$doc.pdf" || true
    [ -s "$doc.pdf" ] || { echo "échec de la compilation de $doc.tex"; exit 1; }
  done
  rm -f ./*.blg ./*.log )

echo "== Dossier rendu/"
rm -rf rendu && mkdir -p rendu
cp docs/article/main.pdf rendu/PROUX_Valentin_preprojet.pdf
cp docs/article/carnet_de_bord.pdf rendu/PROUX_Valentin_carnet.pdf
( cd docs/article && zip -qr ../../rendu/PROUX_Valentin_source.zip main.tex refs.bib genere -x '*.png' )
ls -la rendu
