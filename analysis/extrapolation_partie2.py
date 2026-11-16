"""Partie 2 : au-delà de 32 essais, extrapolation par un modèle de la difficulté (EXPLORATOIRE, 16 novembre 2026).

Les estimateurs sans biais de pass@k s'arrêtent à k = n = 32. Pour dire où vont les courbes au-delà, il faut un
modèle de la probabilité de réussite p_i de chaque question. Deux modèles, ajustés par maximum de vraisemblance aux
nombres de réussites c_i sur 32 tirages, pour chaque modèle de langage séparément :
- BB : p_i ~ Bêta(a, b), c_i | p_i ~ Binomiale(32, p_i) (bêta-binomiale). pass@k = 1 − B(a, b + k) / B(a, b) ;
  pass@∞ = 1 : toute question finit par être résolue ;
- ZIBB : une part pi0 de questions hors de portée (p_i = 0), les autres comme BB. pass@k = (1 − pi0)(1 − B(a, b+k)/B(a, b)),
  pass@∞ = 1 − pi0.
Le choix entre les deux n'est pas identifiable à partir de 32 tirages (une masse en 0 et une bêta très concentrée près
de 0 donnent presque les mêmes c_i) : l'écart entre leurs extrapolations mesure la part de l'extrapolation qui vient
du modèle et non des données. Adéquation : pass@k prédit contre estimé à k <= 32, et test du rapport de vraisemblance.
IC à 95 % des écarts par bootstrap apparié des questions (réajustement des deux modèles à chaque tirage).

    python analysis/extrapolation_partie2.py --indir results/partie2
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln, gammaln

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.confirmatoire_partie2 import DEPART, charger, court, reussites  # noqa: E402
from analysis.passk import courbes  # noqa: E402

N = 32
KS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024]


def log_bb(c, n, a, b):
    return gammaln(n + 1) - gammaln(c + 1) - gammaln(n - c + 1) + betaln(c + a, n - c + b) - betaln(a, b)


def ajuster(c, n=N, zero=True):
    """Maximum de vraisemblance ; paramètres en log (a, b) et logit (pi0). Renvoie (a, b, pi0, logvrais)."""
    c = np.asarray(c)
    compte = np.bincount(c, minlength=n + 1)  # la vraisemblance ne dépend que de l'histogramme
    vals = np.arange(n + 1)

    def nll(th):
        a, b = np.exp(th[0]), np.exp(th[1])
        pi0 = 1 / (1 + np.exp(-th[2])) if zero else 0.0
        lp = log_bb(vals, n, a, b) + np.log1p(-pi0)
        lp[0] = np.logaddexp(np.log(pi0), lp[0]) if zero else lp[0]
        return -(compte * lp).sum()
    meilleurs = None
    for x0 in ([np.log(0.5), np.log(1.5), -1.0], [np.log(0.2), np.log(0.5), -2.0], [np.log(1.0), np.log(3.0), 0.0]):
        r = minimize(nll, x0 if zero else x0[:2] + [0.0], method="Nelder-Mead",
                     options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 4000})
        if meilleurs is None or r.fun < meilleurs.fun:
            meilleurs = r
    th = meilleurs.x
    pi0 = float(1 / (1 + np.exp(-th[2]))) if zero else 0.0
    return float(np.exp(th[0])), float(np.exp(th[1])), pi0, float(-meilleurs.fun)


def pass_modele(a, b, pi0, k):
    return (1 - pi0) * (1 - np.exp(betaln(a, b + k) - betaln(a, b)))


def courbe(c, zero):
    a, b, pi0, ll = ajuster(c, zero=zero)
    return np.array([pass_modele(a, b, pi0, k) for k in KS] + [1 - pi0]), (a, b, pi0, ll)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/partie2")
    ap.add_argument("--genere", default="docs/partie2/genere")
    ap.add_argument("--n-boot", type=int, default=400)
    args = ap.parse_args(argv)
    runs = charger(args.indir)
    dep = court(DEPART["it"])
    res = {"ks": KS + ["inf"]}
    rng = np.random.default_rng(0)
    for typ in ("bridge", "comparison"):
        sel = {court(m): L for (m, c), L in runs.items() if c[0] == typ and c[3] == 4 and c[6] == "42"}
        for ag in ("it-em-ppo", "it-em-grpo"):
            a, b = reussites(sel[ag], N), reussites(sel[dep], N)
            ids = sorted(set(a) & set(b))
            ca, cb = np.array([a[i] for i in ids]), np.array([b[i] for i in ids])
            emp_a = courbes(ca, N, [1, 2, 4, 8, 16, 32]).mean(0)
            emp_b = courbes(cb, N, [1, 2, 4, 8, 16, 32]).mean(0)
            r = {"questions": len(ids)}
            for zero, nom in ((True, "ZIBB"), (False, "BB")):
                pa, pa_par = courbe(ca, zero)
                pb, pb_par = courbe(cb, zero)
                boot = []
                for _ in range(args.n_boot):
                    j = rng.integers(0, len(ids), len(ids))
                    boot.append(courbe(ca[j], zero)[0] - courbe(cb[j], zero)[0])
                boot = np.array(boot)
                r[nom] = {"agent": pa.tolist(), "depart": pb.tolist(), "D": (pa - pb).tolist(),
                          "ic_bas": np.percentile(boot, 2.5, 0).tolist(), "ic_haut": np.percentile(boot, 97.5, 0).tolist(),
                          "parametres_agent": pa_par, "parametres_depart": pb_par,
                          "erreur_max_k32_agent": float(np.abs(pa[:6] - emp_a).max()),
                          "erreur_max_k32_depart": float(np.abs(pb[:6] - emp_b).max())}
            # rapport de vraisemblance ZIBB contre BB (pi0 au bord : loi mélange 1/2 chi2_0 + 1/2 chi2_1)
            for m in ("agent", "depart"):
                lr = 2 * (r["ZIBB"][f"parametres_{m}"][3] - r["BB"][f"parametres_{m}"][3])
                r[f"lr_zero_{m}"] = {"statistique": float(lr),
                                     "p": float(0.5 * (1 - __import__("scipy.stats").stats.chi2.cdf(max(lr, 0), 1)))}
            r["empirique"] = {"agent": emp_a.tolist(), "depart": emp_b.tolist()}
            res[f"{ag}_{typ}"] = r
            z, bb = r["ZIBB"], r["BB"]
            print(f"== {ag} {typ} ({len(ids)} q.) | erreur max à k<=32 : ZIBB {z['erreur_max_k32_agent']:.3f}/"
                  f"{z['erreur_max_k32_depart']:.3f}, BB {bb['erreur_max_k32_agent']:.3f}/{bb['erreur_max_k32_depart']:.3f}")
            print(f"   pi0 agent {z['parametres_agent'][2]:.3f}, départ {z['parametres_depart'][2]:.3f} | "
                  f"LR zéro : agent {r['lr_zero_agent']['statistique']:.1f} (p {r['lr_zero_agent']['p']:.3g}), "
                  f"départ {r['lr_zero_depart']['statistique']:.1f} (p {r['lr_zero_depart']['p']:.3g})")
            for nom in ("ZIBB", "BB"):
                x = r[nom]
                print(f"   {nom:4} D(k) " + " ".join(f"{k}:{d:+.3f}[{lo:+.3f},{hi:+.3f}]" for k, d, lo, hi in
                                          zip(res["ks"], x["D"], x["ic_bas"], x["ic_haut"]) if k in (32, 128, 1024, "inf")))
    Path(args.genere, "extrapolation.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    figure(res, Path(args.genere, "extrapolation_ponts.pdf"))
    return res


def figure(res, chemin):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ks = KS
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), sharey=True)
    for ax, ag, titre in zip(axes, ("it-em-ppo", "it-em-grpo"), ("agent PPO", "agent GRPO")):
        r = res[f"{ag}_bridge"]
        for nom, coul, ls in (("ZIBB", "#2563eb", "-"), ("BB", "#d97706", "--")):
            x = r[nom]
            ax.fill_between(ks, x["ic_bas"][:-1], x["ic_haut"][:-1], color=coul, alpha=0.12, lw=0)
            lab = "avec questions hors de portée (ZIBB)" if nom == "ZIBB" else "sans (bêta-binomiale)"
            ax.plot(ks, x["D"][:-1], ls, color=coul, lw=2, label=f"modèle {lab}")
        emp = np.array(r["empirique"]["agent"]) - np.array(r["empirique"]["depart"])
        ax.plot([1, 2, 4, 8, 16, 32], emp, "o", color="#111827", ms=4.5, label="estimé sans modèle (k ≤ 32)")
        ax.axvline(32, color="#9ca3af", lw=0.8, ls=":")
        ax.axhline(0, color="#111827", lw=0.8)
        ax.set(xscale="log", xlabel="k (essais)", title=titre)
        ax.set_xticks([1, 4, 32, 128, 1024], ["1", "4", "32", "128", "1024"])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.title.set_fontsize(10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e7eb", lw=0.6)
    axes[0].set_ylabel("D(k) = agent − départ")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(chemin)
    plt.close(fig)


if __name__ == "__main__":
    main()
