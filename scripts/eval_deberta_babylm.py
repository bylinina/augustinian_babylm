#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_deberta_babylm.py
======================

Evaluate a trained BabyLM DeBERTa masked-LM:
  1. Qualitative: fill-the-[MASK] predictions on a few sample sentences, so you
     can eyeball whether the model learned sensible English.
  2. Quantitative: pseudo-perplexity (PPPL) over the validation set — the
     standard MLM eval. Lower is better. We also report plain masked-token
     accuracy at 15% masking.

The model + tokenizer are loaded from a local path OR an HF repo id (so this
works on the just-trained output dir on Snellius, or on a pushed repo).

Usage (CLI / Snellius):
    python eval_deberta_babylm.py --model ./deberta-babylm-75k-base
    python eval_deberta_babylm.py --model augustinian-babylm/deberta-base-75k
Usage (Colab cell):
    main(["--model", "augustinian-babylm/deberta-base-75k"])
"""

import os
import sys
import math
import argparse
import subprocess


def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)


def _ensure_deps():
    try:
        import torch, transformers, datasets  # noqa
    except Exception:
        _pip("-U", "torch", "transformers", "datasets", "huggingface_hub", "tqdm")


def build_parser():
    p = argparse.ArgumentParser(description="Evaluate a BabyLM DeBERTa MLM")
    p.add_argument("--model", type=str, required=True,
                   help="Local path or HF repo id of the trained model.")
    p.add_argument("--tokenizer", type=str, default=None,
                   help="Defaults to --model (tokenizer saved alongside it).")
    p.add_argument("--valid_url", type=str,
                   default="https://raw.githubusercontent.com/Leukas/babylm25/main/data/bb25_small.dev",
                   help="URL or local path to the validation text (one seg/line).")
    p.add_argument("--max_seq_len", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--mlm_prob", type=float, default=0.15)
    p.add_argument("--max_eval_lines", type=int, default=2000,
                   help="Cap dev lines for a quick PPPL estimate. 0 = all.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--samples_only", action="store_true",
                   help="Skip the PPPL pass; just show mask-fill examples.")
    return p


SAMPLE_SENTENCES = [
    "the dog ran across the [MASK] to fetch the ball",
    "she opened the [MASK] and started to read quietly",
    "he was very [MASK] when he heard the good news",
    "they went to the [MASK] to buy some milk and bread",
    "the little boy [MASK] his mother for a cookie",
]


def fetch(path_or_url, dest):
    if os.path.exists(path_or_url):
        return path_or_url
    if path_or_url.startswith(("http://", "https://")):
        if not os.path.exists(dest):
            subprocess.run(["wget", "-q", "-O", dest, path_or_url], check=True)
        return dest
    raise FileNotFoundError(path_or_url)


def show_mask_fills(model, tokenizer, device, topk=5):
    import torch
    mask_id = tokenizer.mask_token_id
    print("\n" + "=" * 70 + "\nMASKED-TOKEN PREDICTIONS\n" + "=" * 70)
    model.eval()
    for sent in SAMPLE_SENTENCES:
        enc = tokenizer(sent, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        # find the [MASK] position(s)
        positions = (enc["input_ids"][0] == mask_id).nonzero(as_tuple=True)[0]
        if len(positions) == 0:
            print(f"  (no mask found in: {sent!r})")
            continue
        pos = positions[0].item()
        probs = logits[0, pos].softmax(-1)
        top = probs.topk(topk)
        preds = [(tokenizer.decode([tid]).strip(), f"{p:.2f}")
                 for tid, p in zip(top.indices.tolist(), top.values.tolist())]
        print(f"\n  {sent}")
        print("    -> " + ", ".join(f"{w!r} ({p})" for w, p in preds))


def pseudo_perplexity(model, tokenizer, lines, device, args):
    """Approximate MLM eval: mask 15% of tokens per sequence (like training),
    average cross-entropy over masked positions -> exp() = pseudo-perplexity.
    This is the cheap batched estimate (not the exact one-mask-at-a-time PPPL),
    which is the standard quick proxy and matches the training objective."""
    import torch
    from transformers import set_seed
    set_seed(args.seed)
    model.eval()

    mask_id = tokenizer.mask_token_id
    vocab = tokenizer.vocab_size
    special_ids = set(tokenizer.all_special_ids)

    total_loss, total_masked, total_correct = 0.0, 0, 0
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

    bs = args.batch_size
    from tqdm.auto import tqdm
    for i in tqdm(range(0, len(lines), bs), desc="PPPL"):
        batch_lines = lines[i:i + bs]
        enc = tokenizer(batch_lines, return_tensors="pt", padding=True,
                        truncation=True, max_length=args.max_seq_len).to(device)
        input_ids = enc["input_ids"].clone()
        labels = torch.full_like(input_ids, -100)

        # choose maskable positions: not padding, not special tokens
        prob = torch.full(input_ids.shape, args.mlm_prob, device=device)
        for sid in special_ids:
            prob[input_ids == sid] = 0.0
        prob[enc["attention_mask"] == 0] = 0.0
        masked = torch.bernoulli(prob).bool()
        if masked.sum() == 0:
            continue
        labels[masked] = input_ids[masked]

        # 80/10/10 like training
        rand = torch.rand(input_ids.shape, device=device)
        do_mask = masked & (rand < 0.8)
        do_rand = masked & (rand >= 0.8) & (rand < 0.9)
        input_ids[do_mask] = mask_id
        input_ids[do_rand] = torch.randint(0, vocab, (int(do_rand.sum()),),
                                            device=device)

        with torch.no_grad():
            logits = model(input_ids=input_ids,
                           attention_mask=enc["attention_mask"]).logits
        sel = labels != -100
        sel_logits = logits[sel]
        sel_labels = labels[sel]
        total_loss += loss_fn(sel_logits, sel_labels).item()
        total_correct += (sel_logits.argmax(-1) == sel_labels).sum().item()
        total_masked += int(sel.sum().item())

    avg_loss = total_loss / max(1, total_masked)
    return {
        "masked_tokens": total_masked,
        "avg_mlm_loss": avg_loss,
        "pseudo_perplexity": math.exp(avg_loss),
        "masked_accuracy": total_correct / max(1, total_masked),
    }


def main(argv=None):
    _ensure_deps()
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    args = build_parser().parse_args(argv)
    tok_src = args.tokenizer or args.model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading model:     {args.model}")
    print(f"Loading tokenizer: {tok_src}")

    tokenizer = AutoTokenizer.from_pretrained(tok_src)
    model = AutoModelForMaskedLM.from_pretrained(args.model, trust_remote_code=True).to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"Loaded: {n/1e6:.1f}M params | vocab {tokenizer.vocab_size:,}")

    show_mask_fills(model, tokenizer, device)

    if args.samples_only:
        return

    # validation set
    vfile = fetch(args.valid_url, "/tmp/_dev.txt")
    with open(vfile, encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if args.max_eval_lines and len(lines) > args.max_eval_lines:
        lines = lines[:args.max_eval_lines]
    print(f"\nValidation lines used: {len(lines):,}")

    metrics = pseudo_perplexity(model, tokenizer, lines, device, args)
    print("\n" + "=" * 70 + "\nQUANTITATIVE\n" + "=" * 70)
    print(f"  masked tokens     : {metrics['masked_tokens']:,}")
    print(f"  avg MLM loss      : {metrics['avg_mlm_loss']:.4f}")
    print(f"  pseudo-perplexity : {metrics['pseudo_perplexity']:.2f}")
    print(f"  masked accuracy   : {metrics['masked_accuracy']*100:.1f}%")
    return metrics


if __name__ == "__main__":
    main()
