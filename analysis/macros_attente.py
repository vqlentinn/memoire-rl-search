"""Valeurs des sections en attente (H4, 2WikiMultiHopQA, 7B) pour l'article, lues dans les JSON des analyses.

Écrit `macros_attente.tex` : une définition par valeur disponible, sous une clé (ex. `w2.ppo.g`). Dans l'article,
`\\val{w2.ppo.g}` affiche la valeur, ou `[PENDING: w2.ppo.g]` en rouge tant qu'elle n'est pas mesurée : aucun chiffre
n'est recopié à la main, et le document compile à tout moment. Les prédictions écrites avant les données
(amendements des 23 et 24 novembre, `docs/partie2/plan_analyse.md`) sont évaluées ici, pas dans le texte.

    python analysis/macros_attente.py                      # après apres_session.sh
    python analysis/macros_attente.py --verifier docs/papier/main.tex   # clés utilisées mais non définies
"""
import argparse
import json
import math
import re
from pathlib import Path

VERDICT_H2 = {"fiabilisation seule": "sharpening only", "élargissement": "expansion",
              "rétrécissement": "narrowing", "indéterminé": "indeterminate"}
VERDICT_H4 = {"soutenue": "supported", "partiellement": "partially supported", "non soutenue": "not supported"}


def lire(genere, nom):
    p = Path(genere, nom)
    return json.loads(p.read_text()) if p.exists() else None


def fin(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def f3(x):
    return f"{x:.3f}" if fin(x) else None


def f2(x):
    return f"{x:.2f}" if fin(x) else None


def s3(x):
    return f"${x:+.3f}$" if fin(x) else None


def ic3(a, b):
    return f"$[{a:+.3f}, {b:+.3f}]$" if fin(a) and fin(b) else None


def ic2(a, b):
    return f"[{a:.2f}, {b:.2f}]" if fin(a) and fin(b) else None


def tenue(b):
    return "held" if b else "not held"


def paire(res):
    """Verdicts H1/H2 et courbes d'une paire (résultat d'analyser_paire)."""
    fr = res["frontiere"]
    return {"H1": "yes" if res["H1"] else "no", "H2": VERDICT_H2[res["H2"]],
            "pass1": f3(res["passk_agent"][0]), "pass32": f3(res["passk_agent"][-1]),
            "base.pass1": f3(res["passk_depart"][0]), "base.pass32": f3(res["passk_depart"][-1]),
            "D1": s3(res["D"][0]), "D32": s3(res["D"][-1]), "band32": ic3(res["simul_bas"][-1], res["simul_haut"][-1]),
            "front": str(fr["rl"]), "base.front": str(fr["base"]), "gagnees": str(fr["gagnees"]),
            "perdues": str(fr["perdues"]), "pmcnemar": f3(fr["p_mcnemar"]), "questions": str(res["questions"])}


def equivalents(e):
    m, lo, hi = e["m"], e["m_bas"], e["m_haut"]
    return {"g": f2(e["g"]), "gic": ic2(*e["g_ic"]), "m32": f2(m[-1]), "m32ic": ic2(lo[-1], hi[-1]),
            "obs32": f3(e["pass32_observe"]), "pred32": f3(e["pass32_predit"]),
            "ecart": s3(e["ecart_observe_moins_predit"]), "ecartic": ic3(*e["ecart_ic"])}


def diversite(mod, ec=None):
    v = {"distinctes": f"{mod['distinctes']:.1f}", "chainage": f"{100 * mod['chainage']:.1f}\\%",
         "maj32": f3(mod["maj"][-1])}
    if ec:
        v.update({"ddistinctes": f"${ec['distinctes'][0]:+.1f}$", "dchainage": s3(ec["chainage"][0]),
                  "dmaj32": s3(ec["maj32"][0]), "dmaj32ic": ic3(*ec["maj32"][1])})
    return v


def fiabilisation(a):
    mo = a["modeles"]
    return {"lambda": f2(mo["cotes"]["parametres"]["facteur_cotes"]),
            "sigma": f"{mo['cotes+dispersion']['parametres']['sigma_logit']:.1f}",
            "llgap": f"{-a['puissance_moins_cotes']['ecart']:.0f}",
            "llgapic": "[{:.0f}, {:.0f}]".format(*sorted(-x for x in a["puissance_moins_cotes"]["ic"]))}


def bloc_agent(pref, res_paire, eq, div, mf, cle_ag):
    """Toutes les valeurs d'un agent face à son départ, sous le préfixe `pref.` ; sources absentes ignorées."""
    v = {}
    if res_paire:
        v.update(paire(res_paire))
    if eq and cle_ag in eq.get("agents", {}):
        v.update(equivalents(eq["agents"][cle_ag]))
    if div and cle_ag in div.get("modeles", {}):
        v.update(diversite(div["modeles"][cle_ag], div.get("ecarts", {}).get(cle_ag)))
    if mf and cle_ag in mf.get("agents", {}):
        v.update(fiabilisation(mf["agents"][cle_ag]))
    return {f"{pref}.{k}": x for k, x in v.items() if x is not None}


def paire_de(resultats, type_q, motif):
    """Paire principale (500 questions, graine 42, T = 4, n = 32, graine de génération 0) dont l'agent contient `motif`."""
    for p in (resultats or {}).get("paires", []):
        c = p["cond"]
        if c[0] == type_q and c[2] == 32 and c[3] == 4 and c[6] == "42" and c[7] == 0 and motif in p["agent"]:
            return p["res"]
    return None


def predictions_echange(res_paire, eq_ag, div_ec):
    """Prédictions 2 à 4 de l'amendement 2Wiki (aussi appliquées au 7B pour la 2)."""
    out = {}
    if eq_ag:
        e, lo, hi = eq_ag["ecart_observe_moins_predit"], *eq_ag["ecart_ic"]
        if fin(e):
            out["p.loi"] = tenue(lo <= 0 <= hi and abs(e) < 0.03)
        m, mb, mh = eq_ag["m"], eq_ag["m_bas"], eq_ag["m_haut"]
        out["p.tendance"] = tenue(mb[0] <= m[-1] <= mh[0]) if fin(m[-1]) else "not computable"
    if div_ec and res_paire:
        moins = div_ec["distinctes"][1][1] < 0
        plus = div_ec["chainage"][1][0] > 0
        out["p.diversite"] = tenue(moins and plus and div_ec["maj32"][0] > res_paire["D"][-1])
    return out


def macros(genere):
    R = lire(genere, "resultats.json")
    v = {}

    # --- H4 (session 11) : sft.json, et les analyses communes pour les modèles SFT
    sft = lire(genere, "sft.json")
    eq, div, mf = lire(genere, "equivalents.json"), lire(genere, "diversite.json"), lire(genere, "modeles_fiabilisation.json")
    for court, pref in (("it-sft-ppo", "h4.ppo"), ("it-sft-depart", "h4.dep")):
        v.update(bloc_agent(pref, None, eq, div, mf, f"{court}_T4"))
        if sft and court in sft:
            x = sft[court]
            res = {"H1": x["H1"], "H2": x["H2"], "passk_agent": x["passk"], "passk_depart": x["passk_depart"],
                   "D": x["D_depart"], "simul_bas": x["simul_bas"], "simul_haut": x["simul_haut"],
                   "frontiere": x["frontiere"], "questions": x["questions"]}
            v.update({f"{pref}.{k}": y for k, y in paire(res).items() if y is not None})
            v.update({f"{pref}.H4": VERDICT_H4[x["H4"]], f"{pref}.rho": f2(x["rho"]), f"{pref}.rhoic": ic2(*x["rho_ic"]),
                      f"{pref}.E1": s3(x["E_ppo"][0]), f"{pref}.E1ic": ic3(x["E_bas"][0], x["E_haut"][0]),
                      f"{pref}.E32": s3(x["E_ppo"][-1]), f"{pref}.E32ic": ic3(x["E_bas"][-1], x["E_haut"][-1]),
                      f"{pref}.recherches": f2(x["recherches"]), f"{pref}.jetons": f"{x['jetons']:.0f}",
                      f"{pref}.sansrep": f"{100 * (1 - x['statuts'].get('answer', 0)):.1f}\\%"})
    if sft:
        v.update({"h4.ppo3b.pass1": f3(sft["ppo"]["em"]), "h4.ppo3b.recherches": f2(sft["ppo"]["recherches"]),
                  "h4.base.recherches": f2(sft["depart"]["recherches"])})
        if "it-sft-ppo" in sft and "it-sft-depart" in sft:
            a, b = sft["it-sft-ppo"], sft["it-sft-depart"]
            v["h4.p.rhoppo"] = tenue(0.3 <= a["rho"] <= 0.8 and a["H4"] == "partiellement")
            v["h4.p.rhodep"] = tenue(b["rho"] < 0.5)
            v["h4.p.frontiere"] = tenue(all(x["H2"] in ("fiabilisation seule", "indéterminé") for x in (a, b)))

    # --- 2WikiMultiHopQA (sessions 12 et 13)
    eq, div, mf = (lire(genere, f"{n}_2wiki.json") for n in ("equivalents", "diversite", "modeles_fiabilisation"))
    if eq:
        v["w2.base.pass64"] = f3(eq["depart_64"][2])
    if div and "Qwen2.5-3B-Instruct_T4" in div["modeles"]:
        v.update({f"w2.base.{k}": x for k, x in diversite(div["modeles"]["Qwen2.5-3B-Instruct_T4"]).items()})
    verdicts = []
    for court, pref in (("it-em-ppo", "w2.ppo"), ("it-em-grpo", "w2.grpo")):
        rp = paire_de(R, "compositional", f"3b-{court}-")
        cle = f"{court}_T4"
        v.update(bloc_agent(pref, rp, eq, div, mf, cle))
        v.update({f"{pref}.{k}": x for k, x in predictions_echange(
            rp, (eq or {}).get("agents", {}).get(cle), (div or {}).get("ecarts", {}).get(cle)).items()})
        if rp:
            verdicts.append(rp["H1"] and rp["H2"] == "fiabilisation seule")
    if len(verdicts) == 2:
        v["w2.p.verdicts"] = tenue(all(verdicts))

    # --- 7B (sessions 14 à 17), comparé au GRPO 3B sur HotpotQA
    eq, div, mf = (lire(genere, f"{n}_7b.json") for n in ("equivalents", "diversite", "modeles_fiabilisation"))
    rp = paire_de(R, "bridge", "7b-it-em-grpo")
    cle = "7b-it-em-grpo_T4"
    v.update(bloc_agent("b7", rp, eq, div, mf, cle))
    if div and "Qwen2.5-7B-Instruct_T4" in div["modeles"]:
        v.update({f"b7.base.{k}": x for k, x in diversite(div["modeles"]["Qwen2.5-7B-Instruct_T4"]).items()})
    if eq:
        v["b7.essaisbase"] = str(eq.get("essais_depart", 64))
    if rp:
        v["b7.p.H1"] = tenue(rp["H1"])
        v["b7.p.H2"] = tenue(rp["H2"] == "fiabilisation seule")
    eq3, mf3 = lire(genere, "equivalents.json"), lire(genere, "modeles_fiabilisation.json")
    if eq and cle in eq["agents"] and eq3:
        v["b7.p.loi"] = predictions_echange(None, eq["agents"][cle], None).get("p.loi", "not computable")
        if mf and cle in mf["agents"] and mf3:
            l7 = mf["agents"][cle]["modeles"]["cotes"]["parametres"]["facteur_cotes"]
            l3 = mf3["agents"]["it-em-grpo_T4"]["modeles"]["cotes"]["parametres"]["facteur_cotes"]
            v["b7.p.plusfort"] = tenue(l7 > l3 and eq["agents"][cle]["g"] > eq3["agents"]["it-em-grpo_T4"]["g"])
    return v


def ecrire(v, chemin):
    L = ["% Généré par analysis/macros_attente.py : ne pas modifier à la main.",
         f"% {len(v)} valeurs disponibles ; les autres s'affichent [PENDING: clé]."]
    L += [f"\\expandafter\\def\\csname v@{k}\\endcsname{{{x}}}" for k, x in sorted(v.items())]
    Path(chemin).write_text("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--genere", default="docs/partie2/genere")
    ap.add_argument("--out", default=None, help="défaut : <genere>/macros_attente.tex")
    ap.add_argument("--verifier", default=None, help="fichier .tex : liste les clés \\val{} utilisées non définies")
    args = ap.parse_args(argv)
    v = macros(args.genere)
    ecrire(v, args.out or Path(args.genere, "macros_attente.tex"))
    print(f"{len(v)} valeurs écrites")
    if args.verifier:
        cles = sorted(set(re.findall(r"\\val\{([^}]+)\}", Path(args.verifier).read_text())))
        manque = [c for c in cles if c not in v]
        print(f"{len(cles)} clés utilisées dans {args.verifier}, {len(manque)} en attente :")
        for c in manque:
            print("  ", c)
    return v


if __name__ == "__main__":
    main()
