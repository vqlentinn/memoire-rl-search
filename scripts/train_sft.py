"""H4 : fine-tuning supervisé LoRA de Qwen2.5-3B-Instruct sur des trajectoires réussies (amendement du 23 nov. 2026).

Chaque exemple = prompt Search-R1 dans le gabarit de dialogue (identique à l'évaluation, systems/searchr1.py) suivi de
la trajectoire générée. La perte ne porte que sur les jetons générés par le modèle : le prompt et les blocs
« <information>…</information> » (passages renvoyés par le moteur) sont masqués, comme dans Search-R1.
Réglages figés par le plan : LoRA r 16, alpha 32, dropout 0,05 sur toutes les projections ; 2 époques ; taux 1e-4,
cosinus, échauffement 3 % ; lot effectif de 16 trajectoires (micro-lot 1) ; perte moyenne par jeton sur le lot effectif ;
fp16 avec poids LoRA en fp32 et mise à l'échelle du gradient ; graine 0. Les logits ne sont calculés qu'aux positions
entraînées (logits_to_keep) : sur un T4, les logits complets d'une trajectoire de 4 000 jetons pèseraient plus de 2 Go.

    python scripts/train_sft.py --data data/sft/sft_ppo.jsonl --out /kaggle/working/lora_sft_ppo
    python scripts/train_sft.py --fusion /kaggle/working/lora_sft_ppo --out /tmp/qwen2.5-3b-it-sft-ppo
"""
import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE = "Qwen/Qwen2.5-3B-Instruct"
PROMPT = (Path(__file__).resolve().parent.parent / "prompts" / "searchr1.txt").read_text()
INFO_RE = re.compile(r"<information>.*?</information>", re.S)
CIBLES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def prompt_initial(tok, question):
    msgs = [{"role": "user", "content": PROMPT.format(question=question)}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def masque_caracteres(prompt, trajectoire):
    """Liste de booléens, un par caractère de prompt + trajectoire : True si le caractère est entraîné."""
    garde = [False] * len(prompt) + [True] * len(trajectoire)
    for m in INFO_RE.finditer(trajectoire):
        for j in range(len(prompt) + m.start(), len(prompt) + m.end()):
            garde[j] = False
    return garde


def encoder(tok, question, trajectoire):
    """(input_ids, labels) ; un jeton est entraîné si son premier caractère l'est (-100 sinon)."""
    p = prompt_initial(tok, question)
    texte = p + trajectoire
    enc = tok(texte, add_special_tokens=False, return_offsets_mapping=True)
    garde = masque_caracteres(p, trajectoire)
    labels = [i if (a < b and garde[a]) else -100 for i, (a, b) in zip(enc["input_ids"], enc["offset_mapping"])]
    return enc["input_ids"], labels


def perte_somme(model, ids, labels, torch):
    """Somme des pertes de prédiction du jeton suivant aux positions entraînées, et leur nombre."""
    x = torch.tensor([ids], device=model.device)
    y = torch.tensor(labels, device=model.device)
    cible = torch.nonzero(y[1:] != -100).squeeze(1)  # la position t prédit le jeton t + 1
    if cible.numel() == 0:
        return None, 0
    try:
        logits = model(input_ids=x, logits_to_keep=cible).logits[0]
    except TypeError:  # transformers sans logits_to_keep : logits complets
        logits = model(input_ids=x).logits[0][cible]
    perte = torch.nn.functional.cross_entropy(logits.float(), y[cible + 1], reduction="sum")
    return perte, int(cible.numel())


def entrainer(args):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    gpu = torch.cuda.is_available() and not args.cpu
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.base)
    exemples = [json.loads(l) for l in Path(args.data).read_text().split("\n") if l.strip()]
    if args.limite:
        exemples = exemples[:args.limite]
    donnees, ecartes = [], 0
    for e in exemples:
        ids, lab = encoder(tok, e["question"], e["trajectory"])
        if len(ids) > args.max_len:
            ecartes += 1
            continue
        donnees.append((ids, lab))
    print(f"{len(donnees)} exemples ({ecartes} écartés, plus de {args.max_len} jetons) ; "
          f"{sum(len(i) for i, _ in donnees)} jetons, {sum(sum(l != -100 for l in lab) for _, lab in donnees)} entraînés",
          flush=True)

    dtype = torch.float16 if gpu else torch.float32
    if args.config_test:  # petit modèle aléatoire pour les tests sur CPU
        from transformers import AutoConfig
        cfg = AutoConfig.for_model("qwen2", **json.loads(args.config_test), vocab_size=len(tok))
        model = AutoModelForCausalLM.from_config(cfg)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype)
    model.to("cuda" if gpu else "cpu")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout,
                                             target_modules=CIBLES, task_type="CAUSAL_LM"))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()  # poids LoRA en fp32 : la mise à l'échelle fp16 refuse des gradients fp16
    model.print_trainable_parameters()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    pas_par_epoque = math.ceil(len(donnees) / args.lot)
    total = pas_par_epoque * args.epoques
    sched = get_cosine_schedule_with_warmup(opt, int(round(args.echauffement * total)), total)
    scaler = torch.amp.GradScaler("cuda", enabled=gpu)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    journal, t0, pas = [], time.time(), 0
    model.train()
    for ep in range(args.epoques):
        ordre = list(range(len(donnees)))
        random.Random(args.seed + ep).shuffle(ordre)
        for debut in range(0, len(ordre), args.lot):
            groupe = [donnees[i] for i in ordre[debut:debut + args.lot]]
            n_jetons = sum(sum(l != -100 for l in lab[1:]) for _, lab in groupe)
            somme = 0.0
            for ids, lab in groupe:
                with torch.autocast("cuda", dtype=torch.float16, enabled=gpu):
                    perte, n = perte_somme(model, ids, lab, torch)
                if perte is None:
                    continue
                scaler.scale(perte / n_jetons).backward()
                somme += float(perte)
            scaler.unscale_(opt)
            norme = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            pas += 1
            journal.append({"pas": pas, "epoque": ep + 1, "perte": somme / max(n_jetons, 1), "norme": norme,
                            "lr": sched.get_last_lr()[0], "secondes": round(time.time() - t0)})
            if pas % args.log_every == 0 or pas == total:
                print(f"  pas {pas}/{total} (époque {ep + 1}) perte {journal[-1]['perte']:.4f} norme {norme:.2f} "
                      f"lr {journal[-1]['lr']:.2e} {time.time() - t0:.0f} s", flush=True)
        model.save_pretrained(out)  # adaptateur à chaque fin d'époque (reprise manuelle possible)
        (out / "journal_entrainement.json").write_text(json.dumps(journal))
    meta = {"base": args.base, "data": str(args.data), "exemples": len(donnees), "ecartes": ecartes,
            "pas": pas, "epoques": args.epoques, "r": args.r, "alpha": args.alpha, "dropout": args.dropout,
            "lr": args.lr, "lot": args.lot, "max_len": args.max_len, "seed": args.seed,
            "secondes": round(time.time() - t0), "perte_finale": journal[-1]["perte"] if journal else None}
    (out / "meta_entrainement.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1), flush=True)
    return meta


def fusionner(args):
    """Fusionne l'adaptateur dans le modèle de base (fp16) pour l'évaluation par vLLM."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float16)
    model = PeftModel.from_pretrained(base, args.fusion).merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.out)
    print(f"modèle fusionné : {args.out}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fusion", default=None, help="dossier d'un adaptateur : fusion au lieu d'entraînement")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--epoques", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--echauffement", type=float, default=0.03)
    ap.add_argument("--lot", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=6144)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--limite", type=int, default=0, help="tests : seulement les premiers exemples")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--config-test", default=None, help="tests : config JSON d'un petit Qwen2 aléatoire")
    args = ap.parse_args(argv)
    return fusionner(args) if args.fusion else entrainer(args)


if __name__ == "__main__":
    main()
