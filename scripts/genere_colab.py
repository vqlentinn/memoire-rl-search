"""Génère des notebooks Google Colab (GPU T4 gratuit) pour les sessions de run_passk définies pour Kaggle.

Mêmes runs, mêmes options que scripts/genere_notebooks_partie2.py, avec trois différences :
- un seul GPU : les runs s'enchaînent, et les moitiés « --shard i/2 » prévues pour deux GPU sont fusionnées en un run
  entier (le tirage des questions est fait avant le découpage : mêmes questions, fichier sans suffixe _shard) ;
- les résultats sont écrits au fil de l'eau sur Google Drive (MyDrive/memoire-rl-search/sN/) : une déconnexion ne
  perd au plus que le lot en cours ;
- reprise : run_passk.py reprend là où il s'est arrêté ; relancer le notebook suffit.
Les sessions d'entraînement (H4) et de lecture croisée ne sont pas concernées.

    python scripts/genere_colab.py 13          # notebooks/colab_partie2_s13.ipynb
Ouvrir : https://colab.research.google.com/github/vqlentinn/memoire-rl-search/blob/partie2-passk/notebooks/colab_partie2_s13.ipynb
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from genere_notebooks_partie2 import SESSIONS  # noqa: E402

RACINE = Path(__file__).resolve().parent.parent


def sans_shard(opts):
    out, i = [], 0
    while i < len(opts):
        if opts[i] == "--shard":
            i += 2
            continue
        out.append(opts[i])
        i += 1
    return out


def runs_un_gpu(session):
    vus, runs = set(), []
    for file in session["files"]:
        for model, opts in file:
            r = (model, tuple(sans_shard(opts)))
            if r not in vus:
                vus.add(r)
                runs.append((model, list(r[1])))
    return runs


def code(texte):
    return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
            "source": texte.splitlines(keepends=True)}


def generer(num):
    s = SESSIONS[num]
    if "files" not in s or s.get("fusion") or s.get("sources"):
        raise SystemExit(f"session {num} : seules les sessions run_passk sans entrée Kaggle sont prises en charge")
    runs = runs_un_gpu(s)
    lignes_runs = "".join(f"    ({m!r}, {o!r}),\n" for m, o in runs)
    cellules = [
        {"cell_type": "markdown", "metadata": {}, "source": [
            f"# Partie 2, {s['titre']} (version Colab, un GPU)\n\n", f"{s['texte']}\n\n",
            "Avant de lancer : **Exécution > Modifier le type d'exécution > GPU T4**. Garder l'onglet ouvert (la version "
            "gratuite coupe une session inactive). Si la session est coupée : relancer toutes les cellules, les runs "
            "reprennent là où ils se sont arrêtés.\n\n",
            "Résultats : `MyDrive/memoire-rl-search/" f"s{num}/`. Notebook généré par `scripts/genere_colab.py`.\n"]},
        code('BRANCH = "partie2-passk"\n'
             "N_QUESTIONS, N_SAMPLES, T, TEMP = 500, 32, 4, 0.7   # valeurs par défaut ; les options d'un run priment\n"
             "MAX_TOKENS, MAX_MODEL_LEN = 1024, 8192   # figés au pilote v2\n"
             f"SESSION = {num}\n"
             f"RUNS = [\n{lignes_runs}]\n"),
        code("from google.colab import drive\n"
             "drive.mount('/content/drive')\n"
             "from pathlib import Path\n"
             "WORK = Path(f'/content/drive/MyDrive/memoire-rl-search/s{SESSION}')\n"
             "(WORK / 'logs').mkdir(parents=True, exist_ok=True)\n"
             "print(WORK)\n"),
        code("!nvidia-smi --query-gpu=name,memory.total --format=csv\n"
             "import subprocess\n"
             'REPO = "vqlentinn/memoire-rl-search"\n'
             'CLONE = Path("/content/memoire-rl-search")\n'
             "if not CLONE.exists():\n"
             '    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, f"https://github.com/{REPO}.git", '
             "str(CLONE)], check=True)\n"
             'print(subprocess.run(["git", "-C", str(CLONE), "log", "-1", "--format=%H %s"], capture_output=True, '
             "text=True).stdout)\n"),
        code("!pip install -q vllm==0.33.0 pandas pyarrow huggingface_hub 2>&1 | tail -3\n"
             "!pip freeze > {WORK}/pip_freeze.txt\n"
             "!python -c \"import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, "
             "'| gpus', torch.cuda.device_count())\"\n"
             "import sys\n"
             "sys.path.insert(0, str(CLONE))\n"
             "from data.hotpot_distractor import telecharger\n"
             'PARQUET = telecharger("/content/hotpot")\n'),
        code("import os, time, json\n"
             "status = []\n"
             "env = dict(os.environ, VLLM_WORKER_MULTIPROC_METHOD='spawn')\n"
             "for model, opts in RUNS:\n"
             "    nom = model.split('/')[-1] + ''.join('_' + o.strip('-').replace('/', 'of') for o in opts if o)\n"
             "    cmd = ['python', '-u', 'scripts/run_passk.py', '--model', model, '--type', 'bridge',\n"
             "           '--n-questions', str(N_QUESTIONS), '--n-samples', str(N_SAMPLES), '--max-turns', str(T),\n"
             "           '--temperature', str(TEMP), '--data', PARQUET, '--max-tokens', str(MAX_TOKENS),\n"
             "           '--max-model-len', str(MAX_MODEL_LEN), '--outdir', str(WORK / 'results' / 'partie2')] + opts\n"
             "    t0 = time.time()\n"
             "    with open(WORK / 'logs' / f'{nom}.log', 'a') as f:\n"
             "        code = subprocess.run(cmd, cwd=CLONE, env=env, stdout=f, stderr=subprocess.STDOUT).returncode\n"
             "    status.append((nom, 0, code, round(time.time() - t0)))\n"
             "    print(f'{nom} : code {code}, {time.time() - t0:.0f} s', flush=True)\n"
             "    if code != 0:\n"
             "        print(open(WORK / 'logs' / f'{nom}.log').read()[-3000:])\n"),
        code("from collections import Counter\n"
             "resume = {}\n"
             "for fichier in sorted((WORK / 'results' / 'partie2').glob('*.jsonl')):\n"
             "    L = [json.loads(l) for l in fichier.read_text().split('\\n') if l.strip()]\n"
             "    if not L:\n"
             "        continue\n"
             "    par_q = Counter(l['id'] for l in L)\n"
             "    resume[fichier.stem] = {'trajectoires': len(L),\n"
             "        'questions_completes': sum(v >= N_SAMPLES for v in par_q.values()),\n"
             "        'em': round(sum(l['em'] for l in L) / len(L), 3), 'statuts': dict(Counter(l['status'] for l in L)),\n"
             "        'tronquees': round(sum(l['truncated'] for l in L) / len(L), 3),\n"
             "        'recherches': round(sum(l['n_search'] for l in L) / len(L), 2),\n"
             "        'jetons': round(sum(l['gen_tokens'] for l in L) / len(L))}\n"
             "(WORK / f'resume_s{SESSION}.json').write_text(json.dumps({'status': status, 'resume': resume}, indent=1, "
             "ensure_ascii=False))\n"
             "print(json.dumps({'status': status, 'resume': resume}, indent=1, ensure_ascii=False))\n"
             "echecs = [s for s in status if s[2] != 0]\n"
             "if echecs:\n"
             "    raise RuntimeError(f'Runs en échec : {echecs}')\n"),
    ]
    nb = {"cells": cellules, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                                          "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                          "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    chemin = RACINE / "notebooks" / f"colab_partie2_s{num}.ipynb"
    chemin.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    return chemin, runs


if __name__ == "__main__":
    for a in sys.argv[1:] or ["13"]:
        chemin, runs = generer(int(a))
        print(chemin)
        for r in runs:
            print("  ", r)
