#!/usr/bin/env bash
# Relance toute la chaîne du rendu : données -> analyse -> puissance -> PDF -> dossier rendu/.
#
#   PY=/chemin/vers/python scripts/rendu.sh           # vrais résultats s'ils existent, sinon données fictives
#   PY=... RESULTATS=results/eval_v4 scripts/rendu.sh  # vrais fichiers JSONL (choix par défaut s'ils existent)
#
# Le dossier rendu/ (non versionné) contient les trois fichiers demandés par la trame (étape 12) :
# PROUX_Valentin_preprojet.pdf, PROUX_Valentin_source.zip (à importer dans Overleaf), PROUX_Valentin_carnet.pdf.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
# Par défaut : les vrais résultats du run d'évaluation s'ils existent, sinon les données fictives.
if [ -z "${RESULTATS:-}" ]; then
  if [ -d results/eval_v4 ]; then RESULTATS=results/eval_v4; else RESULTATS=results/fictif; fi
fi

if [ "$RESULTATS" = "results/fictif" ]; then
  echo "== Données FICTIVES (analysis/donnees_fictives.py)"
  "$PY" analysis/donnees_fictives.py --outdir results/fictif
fi
echo "== Analyse de $RESULTATS"
"$PY" analysis/analyse_mini.py --indir "$RESULTATS"
echo "== Analyses exploratoires"
"$PY" analysis/analyse_approfondie.py --indir "$RESULTATS" | tail -2
"$PY" analysis/analyse_avancee.py --indir "$RESULTATS" | tail -1
echo "== Sélection du corpus de la revue (règles R1 à R3)"
"$PY" scripts/selection_corpus.py --meta docs/litterature/openalex_equation3_retenus.json | tail -3
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
