"""Partie 2 : analyse confirmatoire, appliquée telle qu'écrite dans le plan gelé (docs/partie2/plan_analyse.md, v1.0).

Entrée : les JSONL de scripts/run_passk.py (un fichier par modèle et condition ; les moitiés « _shard0of2 » et
« _shard1of2 » d'un même run sont fusionnées). Sorties dans --outdir :
- resultats.json : description de chaque run, puis pour chaque paire et condition D(k), bandes, tests, verdicts ;
- tableau_runs.tex, tableau_verdicts.tex : tableaux LaTeX prêts à \\input ;
- passk_<paire>_<condition>.pdf : courbes pass@k des deux modèles, et D(k) avec bandes ponctuelle et simultanée.

Règles de décision (plan, section « Hypothèses et règles de décision ») :
- H1 : borne basse de l'IC ponctuel à 95 % de D(1) > 0.
- H2, « élargissement » : test de dominance rejeté (t(k) > z_0,95 pour tout k) ET borne basse de la bande simultanée
  à k = n > 0. « Rétrécissement » : perte certifiée (bande simultanée entièrement sous 0 en au moins un k).
  « Fiabilisation seule » : H1 soutenue, pas d'élargissement ni de rétrécissement. Sinon « indéterminé ».
- Holm sur les 2 tests principaux (ponts, T = 4, paires Instruct). Précision opératoire écrite avant de voir les
  données de la session 1 : la p-valeur d'« élargissement » d'une paire est max(p_dominance, p_simul(k = n)), car les
  deux conditions doivent être réunies (intersection-union) ; Holm est appliqué à ces deux p-valeurs au seuil 5 %.

    python analysis/confirmatoire_partie2.py --indir results/partie2 --outdir docs/partie2/genere
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.passk import bootstrap_multiplicateur, courbes, difference_appariee, frontiere, grille_k  # noqa: E402

NOM = re.compile(r"^(?P<modele>.+?)_(?P<type>bridge|comparison)_q(?P<N>\d+)_n(?P<n>\d+)_T(?P<T>\d+)"
                 r"_temp(?P<temp>[\d.]+)_(?P<tag>chat|brut)(?:_seed(?P<sq>\d+))?(?:_ids(?P<ids>[^_]+))?(?:_g(?P<g>\d+))?"
                 r"(?:_shard\d+of\d+)?$")
DEPART = {"it": "Qwen2.5-3B-Instruct", "base": "Qwen2.5-3B"}
PRINCIPAUX = ("qwen2.5-3b-it-em-ppo-v0.3", "qwen2.5-3b-it-em-grpo-v0.3")
ALPHA = 0.05


def charger(indir):
    """{(modele, condition): [lignes]} ; condition = (type, N, n, T, temp, tag, questions, graine de génération)."""
    runs = defaultdict(list)
    for f in sorted(Path(indir).glob("*.jsonl")):
        m = NOM.match(f.stem)
        if not m:
            print(f"  ignoré (nom inattendu) : {f.name}")
            continue
        g = m.groupdict()
        # graine des questions, ou nom de la liste d'ids (sous-ensemble difficile) ; graine de génération à part
        cond = (g["type"], int(g["N"]), int(g["n"]), int(g["T"]), g["temp"], g["tag"],
                ("ids-" + g["ids"]) if g["ids"] else (g["sq"] or "42"), int(g["g"] or 0))
        runs[(g["modele"], cond)] += [json.loads(l) for l in f.read_text().split("\n") if l.strip()]
    return dict(runs)


def reussites(lignes, n):
    """{id: nombre de tirages justes}, questions complètes seulement (n tirages distincts)."""
    par_q = defaultdict(dict)
    for l in lignes:
        par_q[l["id"]][l.get("sample", len(par_q[l["id"]]))] = l["em"]
    return {i: int(sum(v.values())) for i, v in par_q.items() if len(v) >= n}


def decrire(lignes):
    st = Counter(l["status"] for l in lignes)
    m = len(lignes)
    return {
        "trajectoires": m, "questions": len({l["id"] for l in lignes}),
        "em": float(np.mean([l["em"] for l in lignes])),
        "sans_reponse": 1 - st["answer"] / m, "format": st["format"] / m,
        "budget_epuise": st["budget_exhausted"] / m, "contexte_depasse": st["context_exceeded"] / m,
        "sans_recherche_permise": st["no_search_allowed"] / m,
        "tronquees": float(np.mean([l["truncated"] for l in lignes])),
        "recherches": float(np.mean([l["n_search"] for l in lignes])),
        "jetons": float(np.mean([l["gen_tokens"] for l in lignes])),
    }


def famille(modele):
    return "it" if "-it-" in modele else "base"


def verdict_h2(r, ks):
    elargit = r["dominance"] and r["bande_simul"][0][-1] > 0
    if r["perte_certifiee"]:
        return "rétrécissement"
    if elargit:
        return "élargissement"
    if r["bande_ponct"][0][0] > 0:
        return "fiabilisation seule"
    return "indéterminé"


def analyser_paire(c_rl, c_b, n, n_boot=2000):
    ks = grille_k(n)
    d, D = difference_appariee(c_rl, c_b, n, ks)
    r = bootstrap_multiplicateur(d, n_boot=n_boot, alpha=ALPHA, seed=0)
    f = frontiere(c_rl, c_b)
    return {
        "ks": ks, "questions": len(c_rl),
        "passk_agent": courbes(c_rl, n, ks).mean(0).tolist(), "passk_depart": courbes(c_b, n, ks).mean(0).tolist(),
        "D": D.tolist(), "sigma": r["sigma"].tolist(),
        "ponct_bas": r["bande_ponct"][0].tolist(), "ponct_haut": r["bande_ponct"][1].tolist(),
        "simul_bas": r["bande_simul"][0].tolist(), "simul_haut": r["bande_simul"][1].tolist(),
        "q_simul": float(r["q_simul"]), "t_min": r["t_min"], "dominance": r["dominance"],
        "p_dominance": r["p_dominance"], "p_simul_kmax": float(r["p_simul"][-1]),
        "p_elargissement": max(r["p_dominance"], float(r["p_simul"][-1])),
        "perte_certifiee": r["perte_certifiee"], "gain_certifie": r["gain_certifie"],
        "H1": bool(r["bande_ponct"][0][0] > 0), "H2": verdict_h2(r, ks),
        "hors_portee_depart": float(np.mean(c_b == 0)), "hors_portee_agent": float(np.mean(c_rl == 0)),
        "frontiere": f,
    }


def holm(pvals, alpha=ALPHA):
    """Rejets de Holm (ordre des entrées conservé)."""
    ordre = np.argsort(pvals)
    rejet = [False] * len(pvals)
    for rang, i in enumerate(ordre):
        if pvals[i] <= alpha / (len(pvals) - rang):
            rejet[i] = True
        else:
            break
    return rejet


def figure(res, titre, chemin):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ks = res["ks"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    a1.plot(ks, res["passk_depart"], "-o", color="#6b7280", lw=2, ms=4, label="modèle de départ")
    a1.plot(ks, res["passk_agent"], "-o", color="#2563eb", lw=2, ms=4, label="agent (RL)")
    a1.set(xscale="log", xticks=ks, xticklabels=ks, xlabel="k (essais)", ylabel="pass@k", ylim=(0, 1))
    a1.legend(frameon=False, loc="lower right")
    a2.fill_between(ks, res["simul_bas"], res["simul_haut"], color="#2563eb", alpha=0.15, lw=0,
                    label="bande simultanée 95 %")
    a2.fill_between(ks, res["ponct_bas"], res["ponct_haut"], color="#2563eb", alpha=0.3, lw=0,
                    label="IC ponctuel 95 %")
    a2.plot(ks, res["D"], "-o", color="#2563eb", lw=2, ms=4, label="D(k)")
    a2.axhline(0, color="#111827", lw=0.8)
    a2.set(xscale="log", xticks=ks, xticklabels=ks, xlabel="k (essais)", ylabel="D(k) = agent − départ")
    a2.legend(frameon=False, fontsize=8)
    for a in (a1, a2):
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", color="#e5e7eb", lw=0.6)
    fig.suptitle(titre, fontsize=10)
    fig.tight_layout()
    fig.savefig(chemin)
    plt.close(fig)


def court(modele):
    return modele.replace("qwen2.5-3b-", "").replace("-v0.3", "")


def latex_runs(desc):
    L = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
         r"Modèle & Condition & Traj. & EM & Sans rép. & Tronq. & Rech. & Jetons \\", r"\midrule"]
    for (modele, cond), d in sorted(desc.items()):
        L.append(f"{court(modele)} & {cond[0]}, T={cond[3]}, {cond[5]} & {d['trajectoires']} & {d['em']:.3f} & "
                 f"{d['sans_reponse']:.1%} & {d['tronquees']:.1%} & {d['recherches']:.2f} & {d['jetons']:.0f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L).replace("%", r"\%")


def latex_verdicts(paires):
    L = [r"\begin{tabular}{llrrrrrl}", r"\toprule",
         r"Agent & Condition & $N$ & $D(1)$ & $D(n)$ & Gagnées / perdues & $p$ (Holm) & Verdict H2 \\", r"\midrule"]
    for p in paires:
        r, f = p["res"], p["res"]["frontiere"]
        ph = f"{p['p_holm']:.3f}" if p.get("p_holm") is not None else "--"
        L.append(f"{court(p['agent'])} & {p['cond'][0]}, T={p['cond'][3]} & {r['questions']} & {r['D'][0]:+.3f} & "
                 f"{r['D'][-1]:+.3f} & {f['gagnees']} / {f['perdues']} & {ph} & {r['H2']} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/partie2")
    ap.add_argument("--outdir", default="docs/partie2/genere")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--sans-figures", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    runs = charger(args.indir)
    desc = {k: decrire(v) for k, v in runs.items()}

    paires = []
    for (modele, cond), lignes in runs.items():
        if modele in DEPART.values():
            continue
        cle_dep = (DEPART[famille(modele)], cond)
        if cle_dep not in runs:
            continue
        n = cond[2]
        a, b = reussites(lignes, n), reussites(runs[cle_dep], n)
        ids = sorted(set(a) & set(b))
        if not ids:
            continue
        c_rl, c_b = np.array([a[i] for i in ids]), np.array([b[i] for i in ids])
        res = analyser_paire(c_rl, c_b, n, args.n_boot)
        res["questions_incompletes"] = cond[1] - len(ids)  # questions sans n tirages complets pour les deux
        paires.append({"agent": modele, "depart": cle_dep[0], "cond": cond, "res": res,
                       "principal": modele in PRINCIPAUX and cond[0] == "bridge" and cond[3] == 4
                       and cond[5] == "chat" and cond[6] == "42"})

    # Holm sur les tests principaux seulement
    princ = [p for p in paires if p["principal"]]
    if princ:
        pv = [p["res"]["p_elargissement"] for p in princ]
        rej = holm(pv)
        m = len(pv)
        ordre = np.argsort(pv)
        ajust, courant = [0.0] * m, 0.0
        for rang, i in enumerate(ordre):
            courant = max(courant, min(1.0, (m - rang) * pv[i]))
            ajust[i] = courant
        for p, r, pa in zip(princ, rej, ajust):
            p["p_holm"], p["elargissement_holm"] = pa, r
            # Le verdict principal exige le rejet après Holm ; sinon l'élargissement n'est pas retenu.
            if p["res"]["H2"] == "élargissement" and not r:
                p["res"]["H2"] = "fiabilisation seule" if p["res"]["H1"] else "indéterminé"

    paires.sort(key=lambda p: (not p["principal"], p["cond"], p["agent"]))
    print("== Runs")
    for (modele, cond), d in sorted(desc.items()):
        print(f"  {court(modele):28} {cond} traj {d['trajectoires']:6} EM {d['em']:.3f} sans réponse "
              f"{d['sans_reponse']:.1%} tronquées {d['tronquees']:.1%} rech {d['recherches']:.2f}")
    print("\n== Paires")
    for p in paires:
        r = p["res"]
        print(f"  {'[principal] ' if p['principal'] else ''}{court(p['agent'])} contre {p['depart']} {p['cond']}, "
              f"{r['questions']} questions")
        print("    k      " + " ".join(f"{k:>7}" for k in r["ks"]))
        print("    D(k)   " + " ".join(f"{x:+7.3f}" for x in r["D"]))
        print("    simul. " + " ".join(f"[{a:+.2f},{b:+.2f}]" for a, b in zip(r["simul_bas"], r["simul_haut"])))
        print(f"    H1 {'soutenue' if r['H1'] else 'non soutenue'} | H2 : {r['H2']} | p élargissement "
              f"{r['p_elargissement']:.3g}" + (f" (Holm {p['p_holm']:.3g})" if "p_holm" in p else "")
              + f" | frontière {r['frontiere']}")
        if not args.sans_figures:
            nom = f"passk_{court(p['agent'])}_{p['cond'][0]}_T{p['cond'][3]}_{p['cond'][5]}_q{p['cond'][6]}.pdf"
            figure(r, f"{court(p['agent'])} contre {p['depart']} ({p['cond'][0]}, T = {p['cond'][3]}, "
                      f"{r['questions']} questions)", out / nom)

    (out / "resultats.json").write_text(json.dumps({
        "runs": [{"modele": k[0], "condition": k[1], **v} for k, v in desc.items()],
        "paires": [{k: v for k, v in p.items()} for p in paires],
    }, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    (out / "tableau_runs.tex").write_text(latex_runs(desc) + "\n")
    (out / "tableau_verdicts.tex").write_text(latex_verdicts(paires) + "\n")
    return paires


if __name__ == "__main__":
    main()
