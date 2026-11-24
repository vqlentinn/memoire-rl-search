"""Partie 2 : quel type de fiabilisation ? Comparaison de modèles de transformation p_départ -> p_agent
(EXPLORATOIRE, 24 novembre 2026, sans GPU).

Chaque question i a une probabilité de réussite p_i pour le modèle de départ et f(p_i) pour l'agent. On compare
plusieurs formes de f, proposées dans la littérature ou par nous :
- identité : f(p) = p (aucun effet) ;
- puissance (essais équivalents, chapitre) : 1 − f(p) = (1 − p)^g, soit pass_agent(k) = pass_départ(g k) ;
- cotes (optimum du RL KL-régularisé pour pass@1, Matrenok et al. 2026, annexe B) : f(p) = p e^b / (1 − p + p e^b),
  les cotes de chaque question multipliées par e^b ;
- logit-linéaire (2 paramètres, contient « cotes » pour c = 1) : logit f(p) = b + c logit p ;
- bimodalisation (Oh et al. 2026, théorème 2) : avec probabilité l1 la question est poussée à 1, avec l0 à 0, sinon
  p inchangée ;
- puissance + bimodalisation partielle n'est pas testée (trop de paramètres pour 500 questions).
Aucune de ces formes ne rend résoluble une question à p = 0, sauf la bimodalisation (poussée vers 1).

Vraisemblance. La loi des p_i est estimée sur les 64 essais du départ (bêta-binomiale à zéros, comme
analysis/extrapolation_partie2.py), puis fixée. Pour chaque question, on intègre sur p :
L_i(θ) = ∫ Bin(c_départ,i | 64, p) Bin(c_agent,i | 32, f_θ(p)) dπ(p) / ∫ Bin(c_départ,i | 64, p) dπ(p),
soit la vraisemblance conditionnelle des essais de l'agent sachant ceux du départ. Quadrature : 400 quantiles (1 000 : mêmes résultats, vérifié) de la
loi bêta, plus la masse en 0. Comparaison : log-vraisemblance, AIC, et prédiction de la courbe pass@k de l'agent
(espérance a posteriori, comparée à l'estimateur sans biais observé). IC de l'écart de log-vraisemblance entre les
deux meilleurs modèles par bootstrap des questions (paramètres réajustés à chaque tirage).

    python analysis/modeles_fiabilisation_partie2.py --indir results/partie2
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, gammaln, logit
from scipy.stats import beta as loi_beta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.confirmatoire_partie2 import charger, court, reussites, taille_de  # noqa: E402
from analysis.extrapolation_partie2 import ajuster  # noqa: E402
from analysis.passk import pass_at_k  # noqa: E402

KS = [1, 2, 4, 8, 16, 32]
EPS = 1e-12


def log_binom(c, n, p):
    p = np.clip(p, EPS, 1 - EPS)
    return gammaln(n + 1) - gammaln(c + 1) - gammaln(n - c + 1) + c * np.log(p) + (n - c) * np.log1p(-p)


def grille(a, b, pi0, J=400):
    """Points de quadrature (p, poids) : masse pi0 en 0 et J quantiles de Bêta(a, b)."""
    p = loi_beta.ppf((np.arange(J) + 0.5) / J, a, b)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.concatenate([[0.0], p]), np.concatenate([[pi0], np.full(J, (1 - pi0) / J)])


GH_X, GH_W = np.polynomial.hermite_e.hermegauss(21)  # quadrature de Gauss-Hermite (loi normale centrée réduite)
GH_W = GH_W / GH_W.sum()


def _bruit(p, centre, sigma):
    """Espérance de f quand logit f = centre(p) + sigma ε, ε ~ N(0, 1) : matrice (points de p × nœuds), f(0) = 0."""
    z = centre[:, None] + sigma * GH_X[None, :]
    f = expit(z)
    f[p == 0, :] = 0.0
    return f


TRANSFOS = {
    "identité": (0, lambda p, t: p),
    "puissance": (1, lambda p, t: 1 - (1 - p) ** np.exp(t[0])),
    "cotes": (1, lambda p, t: np.where(p > 0, expit(logit(np.clip(p, 1e-9, 1 - 1e-9)) + t[0]), 0.0)),
    "logit-linéaire": (2, lambda p, t: np.where(p > 0, expit(t[0] + np.exp(t[1]) * logit(np.clip(p, 1e-9, 1 - 1e-9))), 0.0)),
}


def posterieur(cb, p, w, nb=64):
    """Loi a posteriori (questions × points) de p sachant les essais du départ ; calculée une fois."""
    lb = log_binom(cb[:, None], nb, p[None, :]) + np.log(np.maximum(w, 1e-300))[None, :]
    lb[:, 0] = np.where(cb == 0, np.log(max(w[0], 1e-300)), -np.inf)  # p = 0 : seul c = 0 est possible
    return np.exp(lb - np.logaddexp.reduce(lb, axis=1)[:, None])


def binom_pmf(c, n, q):
    q = np.clip(q, 0.0, 1.0)
    out = np.exp(log_binom(c[:, None], n, q[None, :]))
    out[:, q == 0] = (c == 0)[:, None]
    out[:, q == 1] = (c == n)[:, None]
    return out


def centre(nom, p, t):
    lp = logit(np.clip(p, 1e-9, 1 - 1e-9))
    if nom == "cotes+dispersion":
        return lp + t[0]
    # puissance+dispersion : centre = logit de 1 − (1 − p)^g
    return logit(np.clip(1 - (1 - p) ** np.exp(t[0]), 1e-12, 1 - 1e-12))


def log_vrais(nom, theta, post, ca, p, na=32):
    """Somme sur les questions de log L_i (vraisemblance des essais de l'agent sachant ceux du départ)."""
    if nom in DISPERSION:
        f = _bruit(p, centre(nom, p, theta), np.exp(theta[1]))  # (points, nœuds)
        pm = np.zeros((len(ca), len(p)))
        for j, wj in enumerate(GH_W):
            pm += wj * binom_pmf(ca, na, f[:, j])
        li = (post * pm).sum(1)
        return float(np.log(np.maximum(li, 1e-300)).sum())
    if nom == "bimodalisation":
        l1, l0 = expit(theta[0]), expit(theta[1]) * (1 - expit(theta[0]))
        li = l1 * (ca == na) + l0 * (ca == 0) + (1 - l1 - l0) * (post * binom_pmf(ca, na, p)).sum(1)
    else:
        li = (post * binom_pmf(ca, na, TRANSFOS[nom][1](p, theta))).sum(1)
    return float(np.log(np.maximum(li, 1e-300)).sum())


DISPERSION = ("cotes+dispersion", "puissance+dispersion")


def nb_param(nom):
    return 2 if nom in DISPERSION or nom == "bimodalisation" else TRANSFOS[nom][0]


def ajuster_transfo(nom, post, ca, p):
    k = nb_param(nom)
    if k == 0:
        return np.array([]), log_vrais(nom, [], post, ca, p)
    departs = {1: [[0.0], [0.7], [-0.5]], 2: [[0.0, 0.0], [0.5, 0.3], [-2.0, -2.0]]}[k]
    meilleur = None
    for x0 in departs:
        r = minimize(lambda t: -log_vrais(nom, t, post, ca, p), x0, method="Nelder-Mead",
                     options={"xatol": 1e-5, "fatol": 1e-6, "maxiter": 2000})
        if meilleur is None or r.fun < meilleur.fun:
            meilleur = r
    return meilleur.x, -meilleur.fun


def passk_predit(nom, theta, post, p):
    """Espérance a posteriori de pass@k de l'agent, sachant les essais du départ, moyennée sur les questions."""
    if nom in DISPERSION:
        f = _bruit(p, centre(nom, p, theta), np.exp(theta[1]))
        return np.array([(post * ((1 - (1 - f) ** k) @ GH_W)[None, :]).sum(1).mean() for k in KS])
    if nom == "bimodalisation":
        l1, l0 = expit(theta[0]), expit(theta[1]) * (1 - expit(theta[0]))
        return np.array([l1 + (1 - l1 - l0) * (post * (1 - (1 - p) ** k)).sum(1).mean() for k in KS])
    fp = TRANSFOS[nom][1](p, theta)
    return np.array([(post * (1 - (1 - fp) ** k)).sum(1).mean() for k in KS])


def parametres_lisibles(nom, t):
    if nom == "puissance":
        return {"g": float(np.exp(t[0]))}
    if nom == "cotes":
        return {"facteur_cotes": float(np.exp(t[0]))}
    if nom == "logit-linéaire":
        return {"b": float(t[0]), "c": float(np.exp(t[1]))}
    if nom == "cotes+dispersion":
        return {"facteur_cotes": float(np.exp(t[0])), "sigma_logit": float(np.exp(t[1]))}
    if nom == "puissance+dispersion":
        return {"g": float(np.exp(t[0])), "sigma_logit": float(np.exp(t[1]))}
    if nom == "bimodalisation":
        l1 = float(expit(t[0]))
        return {"vers_1": l1, "vers_0": float(expit(t[1]) * (1 - l1))}
    return {}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/partie2")
    ap.add_argument("--genere", default="docs/partie2/genere")
    ap.add_argument("--type", default="bridge", choices=["bridge", "compositional"])
    ap.add_argument("--n-boot", type=int, default=100)
    ap.add_argument("--grille", type=int, default=400)
    ap.add_argument("--taille", type=int, default=3, choices=[3, 7],
                    help="7 : paire 7B GRPO ; loi du départ sur 64 essais si le run de graine 1 existe, sinon 32")
    args = ap.parse_args(argv)
    dep = f"Qwen2.5-{args.taille}B-Instruct"
    # runs de la taille demandée seulement : l'intersection des questions ne dépend pas des runs d'une autre taille
    R = {(court(m), c[3], c[7]): reussites(L, 32) for (m, c), L in charger(args.indir).items()
         if c[0] == args.type and c[6] == "42" and c[2] == 32 and c[3] in (4, 8) and taille_de(m) == args.taille}
    ids = sorted(set.intersection(*[set(v) for v in R.values()]))
    C = {k: np.array([v[i] for i in ids]) for k, v in R.items()}
    # deux runs du départ (contrôle départ / départ possible) ou un seul (32 essais, pas de contrôle)
    second = (dep, 4, 1) if (dep, 4, 1) in C else (dep, 8, 0) if args.type == "bridge" and (dep, 8, 0) in C else None
    cb = C[(dep, 4, 0)] + (C[second] if second else 0)
    nb = 64 if second else 32
    candidats = [("it-em-ppo", 4), ("it-em-grpo", 4), ("it-em-ppo", 8), ("it-sft-ppo", 4), ("it-sft-depart", 4)] if args.taille == 3 else [("7b-it-em-grpo", 4)]
    agents = [ag for ag in candidats if (ag[0], ag[1], 0) in C]
    a, b, pi0, _ = ajuster(cb, n=nb)
    p, w = grille(a, b, pi0, args.grille)
    post = posterieur(cb, p, w, nb=nb)
    print(f"loi du départ ({nb} essais) : Bêta({a:.3f}, {b:.3f}), part à 0 : {pi0:.3f}")
    noms = ["identité", "puissance", "cotes", "logit-linéaire", "bimodalisation", "puissance+dispersion", "cotes+dispersion"]
    res = {"questions": len(ids), "essais_depart": nb, "loi_depart": [a, b, pi0], "ks": KS, "agents": {}}
    rng = np.random.default_rng(0)
    for ag in agents:
        ca = C[(ag[0], ag[1], 0)]
        obs = np.array([pass_at_k(32, ca, k).mean() for k in KS])
        r = {"observe": obs.tolist(), "modeles": {}}
        for nom in noms:
            t, ll = ajuster_transfo(nom, post, ca, p)
            k = nb_param(nom)
            pred = passk_predit(nom, t, post, p)
            r["modeles"][nom] = {"theta": t.tolist(), "parametres": parametres_lisibles(nom, t), "logvrais": ll,
                                 "aic": 2 * k - 2 * ll, "passk_predit": pred.tolist(),
                                 "erreur_max": float(np.abs(pred - obs).max()), "erreur_k32": float(pred[-1] - obs[-1])}
        best = min(r["modeles"], key=lambda n: r["modeles"][n]["aic"])
        for nom in noms:
            r["modeles"][nom]["delta_aic"] = r["modeles"][nom]["aic"] - r["modeles"][best]["aic"]
        # bootstrap de l'écart de log-vraisemblance puissance − cotes (1 paramètre chacun : comparable directement)
        diffs = []
        for _ in range(args.n_boot):
            j = rng.integers(0, len(ids), len(ids))
            diffs.append(ajuster_transfo("puissance", post[j], ca[j], p)[1] - ajuster_transfo("cotes", post[j], ca[j], p)[1])
        r["puissance_moins_cotes"] = {"ecart": r["modeles"]["puissance"]["logvrais"] - r["modeles"]["cotes"]["logvrais"],
                                      "ic": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}
        res["agents"][f"{ag[0]}_T{ag[1]}"] = r
        print(f"== {ag[0]} T={ag[1]} : pass@k observé " + " ".join(f"{x:.3f}" for x in obs))
        for nom in noms:
            m = r["modeles"][nom]
            print(f"   {nom:15} ΔAIC {m['delta_aic']:8.1f} | {m['parametres']} | erreur max sur pass@k {m['erreur_max']:.3f}, "
                  f"à k = 32 {m['erreur_k32']:+.3f}")
        x = r["puissance_moins_cotes"]
        print(f"   log-vraisemblance puissance − cotes : {x['ecart']:+.1f} [{x['ic'][0]:+.1f} ; {x['ic'][1]:+.1f}]")
    # Contrôle : le départ contre lui-même (deux runs de 32 essais, même comportement) ; attendu : facteur 1, sigma 0
    res["controle_depart_depart"] = {}
    if second:
        c1, c2 = C[(dep, 4, 0)], C[second]
        a1, b1, z1, _ = ajuster(c1, n=32)
        p1, w1 = grille(a1, b1, z1, args.grille)
        post1 = posterieur(c1, p1, w1, nb=32)
        for nom in ("identité", "cotes", "cotes+dispersion"):
            t1, l1 = ajuster_transfo(nom, post1, c2, p1)
            res["controle_depart_depart"][nom] = {"logvrais": l1, "parametres": parametres_lisibles(nom, t1)}
            print(f"contrôle départ / départ, {nom:17} log-vraisemblance {l1:.1f} {parametres_lisibles(nom, t1)}")
    else:
        print("contrôle départ / départ : un seul run du départ, non calculé")
    suffixe = ("" if args.type == "bridge" else "_2wiki") + ("" if args.taille == 3 else f"_{args.taille}b")
    Path(args.genere, f"modeles_fiabilisation{suffixe}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    if suffixe == "":
        figure(res, Path(args.genere, "modeles_fiabilisation_ponts.pdf"))
    return res


def figure(res, chemin):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    styles = {"identité": ("#9ca3af", ":"), "puissance": ("#2563eb", "-"), "cotes": ("#d97706", "--"),
              "bimodalisation": ("#0d9488", (0, (1, 1))), "cotes+dispersion": ("#111827", "-")}
    titres = {"it-em-ppo_T4": "agent PPO, $T = 4$", "it-em-grpo_T4": "agent GRPO, $T = 4$",
              "it-em-ppo_T8": "agent PPO, $T = 8$"}
    ag_fig = {c: r for c, r in res["agents"].items() if c in titres}  # modèles SFT (H4) : hors figure
    fig, axes = plt.subplots(1, len(ag_fig), figsize=(9.6, 3.3), sharey=True)
    for ax, (cle, r) in zip(np.atleast_1d(axes), ag_fig.items()):
        obs = np.array(r["observe"])
        for nom, (coul, ls) in styles.items():
            ax.plot(KS, np.array(r["modeles"][nom]["passk_predit"]) - obs, color=coul, ls=ls, lw=1.8, label=nom)
        ax.axhline(0, color="#111827", lw=0.8)
        ax.set(xscale="log", xlabel="k", title=titres.get(cle, cle))
        ax.title.set_fontsize(10)
        ax.set_xticks(KS, [str(k) for k in KS])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e7eb", lw=0.6)
    np.atleast_1d(axes)[0].set_ylabel("pass@k prédit − observé")
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(chemin)
    plt.close(fig)


if __name__ == "__main__":
    main()
