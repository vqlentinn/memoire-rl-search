"""Partie 2 : diversité des réponses, vote majoritaire et chaînage des requêtes (EXPLORATOIRE, 24 novembre 2026).

Trois mesures par question ou par trajectoire, sur l'échantillon principal (ponts, 500 questions, 32 tirages) :
1. Diversité : nombre de réponses distinctes (après normalisation EM) parmi les 32 tirages, entropie (nats) de leur
   distribution, part de la réponse modale. Une fiabilisation concentre la probabilité : moins de réponses distinctes.
2. Vote majoritaire maj@k : on tire k des 32 tirages (sans remise), la réponse la plus fréquente parmi celles déclarées
   est jugée (ex aequo départagés au hasard ; aucune réponse déclarée = échec). Moyenne sur 200 sous-ensembles par
   question et par k. maj@k profite d'une distribution concentrée sur la bonne réponse, pass@k de sa diversité.
3. Chaînage (questions de pont) : une requête de rang >= 2 nomme le titre d'un paragraphe utile absent de la question
   (l'entité « pont »), alors qu'une requête antérieure a déjà ramené l'autre paragraphe utile. C'est le geste propre
   à une question de pont : lire le premier paragraphe pour savoir quoi chercher ensuite. Mesure par chaîne de
   caractères (titre normalisé, sans parenthèse), donc minorée.
Plus : sur les questions perdues à la frontière (départ >= 1 réussite, agent 0 sur 32), part où l'agent s'est
« effondré » sur une même réponse fausse (réponse modale fausse déclarée dans au moins la moitié des tirages).
IC à 95 % par bootstrap apparié des questions.

    python analysis/diversite_partie2.py --indir results/partie2
"""
import argparse
import json
import re
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.confirmatoire_partie2 import charger, court, taille_de  # noqa: E402
from analysis.recherche_partie2 import norm, titres  # noqa: E402
from eval.metrics import exact_match, normalize_answer  # noqa: E402

N = 32
KS = [1, 2, 4, 8, 16, 32]
COULEURS = {"Qwen2.5-3B-Instruct": "#6b7280", "it-em-ppo": "#2563eb", "it-em-grpo": "#d97706"}
NOMS = {"Qwen2.5-3B-Instruct": "départ", "it-em-ppo": "agent PPO", "it-em-grpo": "agent GRPO"}


def diversite(trajs):
    rep = [normalize_answer(t["prediction"]) if t.get("prediction") else None for t in trajs]
    c = Counter(x for x in rep if x is not None)
    if not c:
        return 0, 0.0, 0.0, None
    p = np.array(list(c.values())) / len(rep)
    mode, n_mode = c.most_common(1)[0]
    return len(c), float(-(p * np.log(p)).sum()), n_mode / len(rep), mode


def maj_k(reps, justes, k, rng, n_sous=200):
    """reps : réponses normalisées (None si aucune) ; justes : {réponse: 0/1}. Espérance de maj@k par sous-échantillons."""
    tot = 0.0
    for _ in range(n_sous):
        sous = [reps[j] for j in rng.choice(len(reps), size=k, replace=False)]
        c = Counter(x for x in sous if x is not None)
        if not c:
            continue
        m = max(c.values())
        tete = [x for x, v in c.items() if v == m]
        tot += np.mean([justes[x] for x in tete])  # ex aequo : choix au hasard, en espérance
    return tot / n_sous


def chainage(t, q, retr):
    caches = [g for g in q["gold_titles"] if norm(re.sub(r"\(.*?\)", "", g)) not in norm(q["question"])]
    nommes = set(q["gold_titles"]) - set(caches)
    vus = set()
    for j, r in enumerate(t["queries"]):
        if j >= 1 and (nommes & vus) and any(norm(re.sub(r"\(.*?\)", "", g)) in norm(r) for g in caches):
            return True
        vus |= titres(retr.search_in(t["id"], r, k=3))
    return False


def boot_diff(x, y, n_boot=5000, seed=0):
    d = np.asarray(x, float) - np.asarray(y, float)
    rng = np.random.default_rng(seed)
    b = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return float(d.mean()), [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def mesurer(L, qs, retr, rng):
    par = defaultdict(list)
    for t in L:
        par[t["id"]].append(t)
    out = {}
    for i, T in par.items():
        q = qs[i]
        nd, ent, part, mode = diversite(T)
        reps = [normalize_answer(t["prediction"]) if t.get("prediction") else None for t in T]
        justes = {r: exact_match(r, q["golds"]) for r in set(reps) if r is not None}
        ch = [chainage(t, q, retr) for t in T]
        out[i] = {"distinctes": nd, "entropie": ent, "part_modale": part,
                  "mode_juste": int(mode is not None and justes[mode] == 1), "reussites": int(sum(t["em"] for t in T)),
                  "maj": [maj_k(reps, justes, k, rng) for k in KS],
                  "chainage": float(np.mean(ch)), "multi": float(np.mean([len(t["queries"]) >= 2 for t in T])),
                  "em_ch": [t["em"] for t, c in zip(T, ch) if c], "em_nch": [t["em"] for t, c in zip(T, ch) if not c]}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/partie2")
    ap.add_argument("--genere", default="docs/partie2/genere")
    ap.add_argument("--data", default=None, help="défaut : HotpotQA local, ou 2Wiki téléchargé")
    ap.add_argument("--type", default="bridge", choices=["bridge", "compositional"])
    ap.add_argument("--taille", type=int, default=3, choices=[3, 7], help="7 : paire 7B GRPO (amendement du 24 nov.)")
    args = ap.parse_args(argv)
    from retrieval.closed_bm25 import ClosedBM25Retriever
    if args.type == "bridge":
        from data.hotpot_distractor import charger as charger_q
        args.data = args.data or "data/raw/hotpot_distractor/validation.parquet"
    else:
        from data.wiki2_distractor import charger as charger_q
    qs = {q["id"]: q for q in charger_q(args.data, args.type, n=500, seed=42)}
    retr = ClosedBM25Retriever(list(qs.values()))
    runs = {(court(m), c[3]): L for (m, c), L in charger(args.indir).items()
            if c[0] == args.type and c[6] == "42" and c[2] == N and c[3] in (4, 8) and c[7] == 0 and taille_de(m) == args.taille}
    dep = f"Qwen2.5-{args.taille}B-Instruct"
    # un générateur par run, graine tirée de son nom : ses maj@k ne dépendent pas des autres runs présents
    M = {k: mesurer(L, qs, retr, np.random.default_rng(zlib.crc32(f"{k[0]}_T{k[1]}".encode())))
         for k, L in sorted(runs.items())}
    ids = sorted(set.intersection(*[set(v) for v in M.values()]))
    res = {"questions": len(ids), "ks": KS, "modeles": {}, "ecarts": {}}
    for (m, T), x in M.items():
        v = lambda champ: np.array([x[i][champ] for i in ids], float)  # noqa: E731
        em_ch = [e for i in ids for e in x[i]["em_ch"]]
        em_nch = [e for i in ids for e in x[i]["em_nch"]]
        maj = np.array([x[i]["maj"] for i in ids]).mean(0)
        res["modeles"][f"{m}_T{T}"] = {
            "distinctes": float(v("distinctes").mean()), "entropie": float(v("entropie").mean()),
            "part_modale": float(v("part_modale").mean()), "mode_juste": float(v("mode_juste").mean()),
            "maj": maj.tolist(), "chainage": float(v("chainage").mean()), "deux_requetes_ou_plus": float(v("multi").mean()),
            "em_si_chainage": float(np.mean(em_ch)) if em_ch else None, "em_sinon": float(np.mean(em_nch)),
            "n_chainage": len(em_ch), "hist_distinctes": np.bincount(v("distinctes").astype(int), minlength=N + 1).tolist()}
        r = res["modeles"][f"{m}_T{T}"]
        print(f"== {m} T={T} : réponses distinctes {r['distinctes']:.2f}, entropie {r['entropie']:.2f}, part modale "
              f"{r['part_modale']:.2f}, mode juste {r['mode_juste']:.3f} | maj@k " +
              " ".join(f"{k}:{a:.3f}" for k, a in zip(KS, r["maj"])) +
              f" | chaînage {r['chainage']:.3f} (≥ 2 requêtes {r['deux_requetes_ou_plus']:.2f}), EM si chaînage "
              f"{r['em_si_chainage']:.3f} ({r['n_chainage']} traj.), sinon {r['em_sinon']:.3f}")
    agents = (("it-em-ppo", 4), ("it-em-grpo", 4), ("it-em-ppo", 8)) if args.taille == 3 else (("7b-it-em-grpo", 4),)
    for ag, T in agents:
        if (ag, T) not in M or (dep, T) not in M:
            continue
        a, b = M[(ag, T)], M[(dep, T)]
        e = {}
        for champ in ("distinctes", "entropie", "part_modale", "mode_juste", "chainage"):
            e[champ] = boot_diff([a[i][champ] for i in ids], [b[i][champ] for i in ids])
        e["maj32"] = boot_diff([a[i]["maj"][-1] for i in ids], [b[i]["maj"][-1] for i in ids])
        # maj@32 du départ contre premier essai de l'agent
        e["maj32_depart_moins_pass1_agent"] = boot_diff([b[i]["maj"][-1] for i in ids], [a[i]["reussites"] / N for i in ids])
        # questions perdues à la frontière : effondrement sur une réponse fausse
        perdues = [i for i in ids if b[i]["reussites"] > 0 and a[i]["reussites"] == 0]
        gagnees = [i for i in ids if a[i]["reussites"] > 0 and b[i]["reussites"] == 0]
        eff = lambda x, I: float(np.mean([x[i]["part_modale"] >= 0.5 and not x[i]["mode_juste"] for i in I])) if I else None  # noqa: E731
        e["perdues"] = {"n": len(perdues), "agent_effondre": eff(a, perdues),
                        "distinctes_agent": float(np.mean([a[i]["distinctes"] for i in perdues])) if perdues else None,
                        "distinctes_depart": float(np.mean([b[i]["distinctes"] for i in perdues])) if perdues else None}
        e["gagnees"] = {"n": len(gagnees), "depart_effondre": eff(b, gagnees)}
        e["effondrement_global"] = {"agent": eff(a, ids), "depart": eff(b, ids)}
        # corrélation, par question, entre baisse de diversité et gain au premier essai
        dd = np.array([a[i]["distinctes"] - b[i]["distinctes"] for i in ids], float)
        g1 = np.array([(a[i]["reussites"] - b[i]["reussites"]) / N for i in ids], float)
        e["corr_baisse_diversite_gain1"] = float(np.corrcoef(-dd, g1)[0, 1])
        res["ecarts"][f"{ag}_T{T}"] = e
        print(f"-- {ag} T={T} contre départ : " + ", ".join(f"{k} {v[0]:+.3f} [{v[1][0]:+.3f}, {v[1][1]:+.3f}]"
                                                         for k, v in e.items() if isinstance(v, tuple)))
        print(f"   perdues {e['perdues']} | gagnées {e['gagnees']} | effondrement {e['effondrement_global']} | "
              f"corr(baisse de diversité, gain à k = 1) {e['corr_baisse_diversite_gain1']:+.2f}")
    suffixe = ("" if args.type == "bridge" else "_2wiki") + ("" if args.taille == 3 else f"_{args.taille}b")
    Path(args.genere, f"diversite{suffixe}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    if suffixe == "":
        figure(res, Path(args.genere, "diversite_ponts.pdf"))
    return res


def figure(res, chemin):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.5))
    for m in ("Qwen2.5-3B-Instruct", "it-em-ppo", "it-em-grpo"):
        r = res["modeles"][f"{m}_T4"]
        h = np.array(r["hist_distinctes"], float)
        cdf = np.cumsum(h) / h.sum()
        a1.step(np.arange(len(cdf)), cdf, where="post", color=COULEURS[m], lw=2,
                label=f"{NOMS[m]} (moyenne {r['distinctes']:.1f})")
        ls = "-" if m != "Qwen2.5-3B-Instruct" else "--"
        a2.plot(KS, r["maj"], ls, marker="o", ms=4.5, color=COULEURS[m], lw=2, label=f"{NOMS[m]}")
    a1.set(xlabel="réponses distinctes parmi 32 essais", ylabel="part cumulée des questions", xlim=(0, 32), ylim=(0, 1.01))
    a1.set_title("Diversité des réponses (ponts, $T = 4$)", fontsize=10)
    a2.set(xscale="log", xlabel="k (essais soumis au vote)", ylabel="maj@k")
    a2.set_xticks(KS, [str(k) for k in KS])
    a2.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    a2.set_title("Vote majoritaire", fontsize=10)
    for ax in (a1, a2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e7eb", lw=0.6)
        ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(chemin)
    plt.close(fig)


if __name__ == "__main__":
    main()
