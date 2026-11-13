"""Génère les notebooks Kaggle des sessions 2 à 4 de l'étude principale (plan gelé v1.0, « Ordre des sessions »).

Chaque session est une liste de deux files (une par T4) de runs (modèle, options de scripts/run_passk.py). Les options
d'un run s'ajoutent à la fin de la commande commune : argparse garde la dernière valeur, elles priment donc.
Les cellules d'installation et de résumé sont reprises de notebooks/kaggle_partie2_principal.ipynb (session 1).

    python scripts/genere_notebooks_partie2.py
"""
import json
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
MODELE = RACINE / "notebooks" / "kaggle_partie2_principal.ipynb"
QWEN_IT = "Qwen/Qwen2.5-3B-Instruct"
R = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-"
PPO, GRPO = R + "it-em-ppo-v0.3", R + "it-em-grpo-v0.3"

COMP = ["--type", "comparison"]
T0 = ["--max-turns", "0"]
# Vivier de présélection : 1 000 ponts, graine 43, hors évaluation et hors pilote (graine 7, 50), 16 tirages, graine
# de génération 1 (les tirages d'évaluation du sous-ensemble difficile utiliseront la graine 0 : tirages nouveaux).
VIVIER = ["--n-questions", "1000", "--seed-questions", "43", "--exclure-tirage", "7:50", "--n-samples", "16",
          "--seed", "1"]
IDS = ["--ids", "/kaggle/input/memoire-partie2-s3/difficiles_ponts.txt"]

SESSIONS = {
    2: {
        "titre": "session 2 : comparaisons, T = 4 (H2')",
        "texte": "500 questions de comparaison (graine 42), mêmes modèles et réglages que la session 1.",
        "files": [[(PPO, COMP), (QWEN_IT, COMP + ["--shard", "0/2"])],
                  [(GRPO, COMP), (QWEN_IT, COMP + ["--shard", "1/2"])]],
    },
    3: {
        "titre": "session 3 : sans recherche (T = 0) et présélection du sous-ensemble difficile",
        "texte": ("T = 0 sur les ponts et les comparaisons pour les trois modèles (H2'). Présélection : le modèle de "
                  "départ tire 16 générations sur un vivier de 1 000 ponts (graine 43), disjoint de l'évaluation et du "
                  "pilote ; les questions sans aucune réussite forment `difficiles_ponts.txt`, utilisé en session 4."),
        "files": [[(QWEN_IT, VIVIER + ["--shard", "0/2"]), (PPO, T0), (PPO, T0 + COMP), (QWEN_IT, T0)],
                  [(QWEN_IT, VIVIER + ["--shard", "1/2"]), (GRPO, T0), (GRPO, T0 + COMP), (QWEN_IT, T0 + COMP)]],
        "difficiles": True,
    },
    4: {
        "titre": "session 4 : sous-ensemble difficile (ponts, T = 4, tirages nouveaux)",
        "texte": ("Questions sans réussite du modèle de départ à la présélection (sortie de la session 3, attachée au "
                  "kernel). 32 tirages **nouveaux** (graine de génération 0) pour les trois modèles."),
        "files": [[(PPO, IDS), (QWEN_IT, IDS + ["--shard", "0/2"])],
                  [(GRPO, IDS), (QWEN_IT, IDS + ["--shard", "1/2"])]],
        "sources": ["valentinproux/memoire-partie2-s3"],
    },
    5: {
        "titre": "session 5 : lecture croisée (H3)",
        "texte": ("Chaque lecteur reçoit les requêtes et passages d'un chercheur (8 trajectoires par question de la "
                  "session 1, raisonnement retiré) et répond sans chercher davantage. Plan 2 × 2 par paire ; la case "
                  "« départ lit départ » est commune aux deux paires. Sortie de la session 1 attachée au kernel."),
        # (lecteur, préfixe du nom des fichiers du chercheur)
        "lecture": [[(PPO, "qwen2.5-3b-it-em-ppo-v0.3_bridge"), (PPO, "Qwen2.5-3B-Instruct_bridge"),
                     (QWEN_IT, "qwen2.5-3b-it-em-ppo-v0.3_bridge"), (QWEN_IT, "Qwen2.5-3B-Instruct_bridge")],
                    [(GRPO, "qwen2.5-3b-it-em-grpo-v0.3_bridge"), (GRPO, "Qwen2.5-3B-Instruct_bridge"),
                     (QWEN_IT, "qwen2.5-3b-it-em-grpo-v0.3_bridge")]],
        "sources": ["valentinproux/memoire-partie2-principal"],
    },
    7: {
        "titre": "session 7 : sensibilité au budget de recherche (ponts, T = 8, paire PPO)",
        "texte": ("Option (discussion générale, limites) : l'agent PPO bute sur T = 4 dans 14 à 24 % de ses trajectoires. "
                  "Même protocole que la session 1, T = 8, pour l'agent PPO et le modèle de départ (moitiés sur deux GPU)."),
        "files": [[(PPO, ["--max-turns", "8", "--shard", "0/2"]), (QWEN_IT, ["--max-turns", "8", "--shard", "0/2"])],
                  [(PPO, ["--max-turns", "8", "--shard", "1/2"]), (QWEN_IT, ["--max-turns", "8", "--shard", "1/2"])]],
    },
    6: {
        "titre": "session 6 : lecture croisée, variante « réponse forcée » (exploratoire)",
        "texte": ("Amendement du 13 novembre, après la session 5 : les lectures sans réponse (le lecteur voulait chercher "
                  "encore) sont complétées en imposant « <answer> » après ce que le lecteur a écrit. Sorties des sessions "
                  "1 et 5 attachées au kernel."),
        # (lecteur, préfixe du nom du chercheur) : mêmes 7 cases que la session 5
        "forcee": [[(PPO, "qwen2.5-3b-it-em-ppo-v0.3_bridge"), (PPO, "Qwen2.5-3B-Instruct_bridge"),
                    (QWEN_IT, "qwen2.5-3b-it-em-ppo-v0.3_bridge"), (QWEN_IT, "Qwen2.5-3B-Instruct_bridge")],
                   [(GRPO, "qwen2.5-3b-it-em-grpo-v0.3_bridge"), (GRPO, "Qwen2.5-3B-Instruct_bridge"),
                    (QWEN_IT, "qwen2.5-3b-it-em-grpo-v0.3_bridge")]],
        "sources": ["valentinproux/memoire-partie2-principal", "valentinproux/memoire-partie2-s5"],
    },
}

CELLULE_FORCEE = '''import os, threading, time, glob
WORK = Path("/kaggle/working")
(WORK / "logs").mkdir(exist_ok=True)
status, verrou = [], threading.Lock()
COURT = {"PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-ppo-v0.3": "qwen2.5-3b-it-em-ppo-v0.3",
         "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo-v0.3": "qwen2.5-3b-it-em-grpo-v0.3",
         "Qwen/Qwen2.5-3B-Instruct": "Qwen2.5-3B-Instruct"}

def trouver(motif):
    f = sorted(glob.glob(f"/kaggle/input/**/{motif}", recursive=True))
    assert f, f"aucun fichier pour {motif}"
    return f

def file_forcee(gpu, runs):
    env = dict(os.environ, VLLM_WORKER_MULTIPROC_METHOD="spawn", CUDA_VISIBLE_DEVICES=str(gpu))
    for lecteur, prefixe in runs:
        chercheur = prefixe.rsplit("_", 1)[0]
        lecture = trouver(f"lecteur-{COURT[lecteur]}_chercheur-{chercheur}_bridge_m8.jsonl")[0]
        nom = f"{COURT[lecteur]}_lit_{chercheur}_force"
        cmd = ["python", "-u", "scripts/run_lecture_forcee.py", "--lecteur", lecteur, "--lecture", lecture,
               "--chercheur", *trouver(f"{prefixe}_q500_n32_T4_temp0.7_chat*.jsonl"), "--data", PARQUET,
               "--max-model-len", str(MAX_MODEL_LEN), "--outdir", str(WORK / "results" / "partie2_lecture")]
        t0 = time.time()
        with open(WORK / "logs" / f"{nom}.log", "a") as f:
            code = subprocess.run(cmd, cwd=CLONE, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        with verrou:
            status.append((nom, gpu, code, round(time.time() - t0)))
            print(f"GPU {gpu} : {nom} code {code}, {time.time() - t0:.0f} s", flush=True)

fils = [threading.Thread(target=file_forcee, args=(g, runs)) for g, runs in enumerate(FILES)]
for f in fils: f.start()
for f in fils: f.join()
'''

CELLULE_LECTURE = '''import os, threading, time, glob
WORK = Path("/kaggle/working")
(WORK / "logs").mkdir(exist_ok=True)
status, verrou = [], threading.Lock()

def fichiers_chercheur(prefixe):
    # Trajectoires de la session 1 (ponts, T = 4, n = 32), moitiés « shard » comprises
    f = sorted(glob.glob(f"/kaggle/input/**/{prefixe}_q500_n32_T4_temp0.7_chat*.jsonl", recursive=True))
    assert f, f"aucun fichier pour {prefixe}"
    return f

def file_de_lectures(gpu, runs):
    env = dict(os.environ, VLLM_WORKER_MULTIPROC_METHOD="spawn", CUDA_VISIBLE_DEVICES=str(gpu))
    for lecteur, prefixe in runs:
        nom = f"{lecteur.split('/')[-1]}_lit_{prefixe}"
        cmd = ["python", "-u", "scripts/run_lecture.py", "--lecteur", lecteur, "--chercheur", *fichiers_chercheur(prefixe),
               "--m", str(M), "--data", PARQUET, "--max-tokens", str(MAX_TOKENS), "--max-model-len", str(MAX_MODEL_LEN),
               "--outdir", str(WORK / "results" / "partie2_lecture")]
        t0 = time.time()
        with open(WORK / "logs" / f"{nom}.log", "a") as f:
            code = subprocess.run(cmd, cwd=CLONE, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        with verrou:
            status.append((nom, gpu, code, round(time.time() - t0)))
            print(f"GPU {gpu} : {nom} code {code}, {time.time() - t0:.0f} s", flush=True)

fils = [threading.Thread(target=file_de_lectures, args=(g, runs)) for g, runs in enumerate(FILES)]
for f in fils: f.start()
for f in fils: f.join()
'''

CELLULE_RUNS = '''import os, threading, time
WORK = Path("/kaggle/working")
(WORK / "logs").mkdir(exist_ok=True)
status, verrou = [], threading.Lock()

def file_de_runs(gpu, runs):
    env = dict(os.environ, VLLM_WORKER_MULTIPROC_METHOD="spawn", CUDA_VISIBLE_DEVICES=str(gpu))
    for model, opts in runs:
        nom = model.split("/")[-1] + "".join("_" + o.strip("-").replace("/", "of").replace(":", "-")
                                             for o in opts if not o.startswith("/kaggle"))
        cmd = ["python", "-u", "scripts/run_passk.py", "--model", model, "--type", "bridge",
               "--n-questions", str(N_QUESTIONS), "--n-samples", str(N_SAMPLES), "--max-turns", str(T),
               "--temperature", str(TEMP), "--data", PARQUET,
               "--max-tokens", str(MAX_TOKENS), "--max-model-len", str(MAX_MODEL_LEN),
               "--outdir", str(WORK / "results" / "partie2")] + opts
        t0 = time.time()
        with open(WORK / "logs" / f"{nom}.log", "a") as f:
            code = subprocess.run(cmd, cwd=CLONE, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        with verrou:
            status.append((nom, gpu, code, round(time.time() - t0)))
            print(f"GPU {gpu} : {nom} code {code}, {time.time() - t0:.0f} s", flush=True)

fils = [threading.Thread(target=file_de_runs, args=(g, runs)) for g, runs in enumerate(FILES)]
for f in fils: f.start()
for f in fils: f.join()
'''

CELLULE_DIFFICILES = '''# Sous-ensemble difficile : questions du vivier sans aucune réussite du modèle de départ (16 tirages, graine 1).
from collections import defaultdict
c = defaultdict(int); vus = defaultdict(set)
for f in sorted((WORK / "results" / "partie2").glob("Qwen2.5-3B-Instruct_bridge_q1000_n16_*_seed43_g1_shard*.jsonl")):
    for l in f.read_text().split("\\n"):
        if l.strip():
            x = json.loads(l); c[x["id"]] += x["em"]; vus[x["id"]].add(x["sample"])
completes = [i for i in c if len(vus[i]) >= 16]
difficiles = sorted(i for i in completes if c[i] == 0)
(WORK / "difficiles_ponts.txt").write_text("".join(i + "\\n" for i in difficiles))
print(f"{len(completes)} questions complètes du vivier, {len(difficiles)} sans réussite ({len(difficiles) / max(len(completes), 1):.1%})")
'''


def source(texte):
    return texte.splitlines(keepends=True)


def generer(num, s, modele):
    cellules = json.loads(json.dumps(modele["cells"]))
    cellules[0]["source"] = source(
        f"# Partie 2 : étude principale pass@k, {s['titre']}\n\nPlan gelé : `docs/partie2/plan_analyse.md` (v1.0). "
        f"{s['texte']}\n\nNotebook généré par `scripts/genere_notebooks_partie2.py` : le modifier là, pas ici.\n")
    files = ",\n".join("    [" + ", ".join(f"({m!r}, {o!r})" for m, o in file) + "]"
                        for file in s.get("files") or s.get("lecture") or s["forcee"])
    cellules[1]["source"] = source(
        'BRANCH = "partie2-passk"\n'
        "N_QUESTIONS, N_SAMPLES, T, TEMP = 500, 32, 4, 0.7   # valeurs par défaut ; les options d'un run priment\n"
        "MAX_TOKENS, MAX_MODEL_LEN = 1024, 8192   # figés au pilote v2 ; arrêt à </search> et </answer>\n"
        + ("M = 8   # trajectoires de chercheur par question\n" if "lecture" in s else "")
        + f"FILES = [\n{files},\n]\n")
    cellules[4]["source"] = source(CELLULE_FORCEE if "forcee" in s else CELLULE_LECTURE if "lecture" in s
                                   else CELLULE_RUNS)
    resume = "".join(cellules[5]["source"]).replace("resume_principal.json", f"resume_s{num}.json")
    if "lecture" in s or "forcee" in s:
        resume = resume.replace('"partie2").glob', '"partie2_lecture").glob')
        resume = resume.replace('"questions_completes": sum(v >= N_SAMPLES for v in par_q.values()),', '"questions": len(par_q),')
        resume = resume.replace('                "recherches": round(sum(l["n_search"] for l in L) / len(L), 2),\n', "")
    cellules[5]["source"] = source(resume)
    if s.get("difficiles"):
        cellules.insert(5, {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                            "source": source("import json\n" + CELLULE_DIFFICILES)})
    for c in cellules:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    nb = dict(modele, cells=cellules)
    chemin = RACINE / "notebooks" / f"kaggle_partie2_s{num}.ipynb"
    chemin.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    meta_dir = RACINE / "kaggle" / f"partie2_s{num}"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta = {"id": f"valentinproux/memoire-partie2-s{num}", "title": f"memoire-partie2-s{num}",
            "code_file": f"../../notebooks/kaggle_partie2_s{num}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "dataset_sources": [], "competition_sources": [], "kernel_sources": s.get("sources", [])}
    (meta_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return chemin


if __name__ == "__main__":
    modele = json.loads(MODELE.read_text())
    for num, s in SESSIONS.items():
        print(generer(num, s, modele))
