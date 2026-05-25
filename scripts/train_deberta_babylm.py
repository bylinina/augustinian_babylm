#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_deberta_babylm.py
=======================

Pretrain a (small) DeBERTa-v3 masked-LM on the BabyLM corpus, following the
same recipe as Lukas's 2025 repo (github.com/Leukas/babylm25, train_mask.py),
using one of our byte-level BPE tokenizers from the augustinian-babylm org.

Runs both on Colab (just call main() / run as a cell) and on Surf Snellius
(submit the SLURM script at the bottom of this file, or run the python directly
inside an srun/sbatch GPU allocation).

------------------------------------------------------------------------------
Hyperparameters — from the paper's Table 3 (with context warmup 64 -> 128):
  optimizer        AdamW, betas=(0.9, 0.999), eps=1e-8  (standard AdamW defaults;
                   the paper says only "AdamW")
  lr               2e-4
  weight_decay     0.01  ("Decay" in the table)
  scheduler        cosine, fixed warmup = 4000 steps
  epochs           50
  batch_size       256 effective, grad_acc 4 (=> 64 per device). The paper's
                   "64, 256" scales batch with context for memory only; we hold
                   effective batch at 256 and use grad_acc, so the optimizer math
                   is unchanged (HF Trainer can't vary batch mid-run anyway).
  mlm_prob         0.15   (mask 0.8 / random 0.1 / keep 0.1)
  dropout          0.1
  seed             0
  context warmup   "0:64,5:128"  -> ctx 64 for epochs 0-4, ctx 128 from epoch 5
  vocab size       set by --tokenizer (paper used 40k; we sweep 50k/75k/100k)
  architecture     from microsoft/deberta-v3-base config, then
                   hidden_size=768, intermediate_size=3072,
                   max_position_embeddings=1024

Faithful deviations:
  * Tokenizer is loaded with AutoTokenizer, NOT DebertaV2Tokenizer. On current
    transformers (5.x) the DebertaV2Tokenizer path is broken and cannot read our
    byte-level BPE; AutoTokenizer loads tokenizer.json directly. Everything
    downstream (config.vocab_size = tok.vocab_size, special-token ids) is identical.
  * Defaults to Lukas's BASE size (12 layers / 768 hidden). Pass --preset small
    for a faster 6-layer model that's lighter on a Colab GPU.

Usage (Colab cell):           main([])              # all defaults
Usage (CLI / Snellius):       python train_deberta_babylm.py --epochs 10 ...
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


# ==========================================================================
# Argument parsing — defaults mirror Lukas's train_mask.py
# ==========================================================================
def build_parser():
    p = argparse.ArgumentParser(description="DeBERTa MLM pretraining on BabyLM")
    # data + tokenizer
    p.add_argument("--train_url", type=str,
                   default="https://raw.githubusercontent.com/Leukas/babylm25/main/data/bb24.train",
                   help="URL or local path to the training text (one segment/line).")
    p.add_argument("--valid_url", type=str,
                   default="https://raw.githubusercontent.com/Leukas/babylm25/main/data/bb25_small.dev",
                   help="URL or local path to the validation text.")
    p.add_argument("--tokenizer", type=str, required=True,
                   help="HF repo id of the BPE tokenizer to use, e.g. "
                        "augustinian-babylm/babylm-bpe-75k (required).")
    p.add_argument("--base_config", type=str, default="microsoft/deberta-v3-base",
                   help="Config to inherit DeBERTa architecture defaults from.")
    # optimization (paper hyperparameters, Table 3)
    p.add_argument("--lr", type=float, default=2e-4,
                   help="Default 2e-4 (paper). Lukas's repo default was 0.007; "
                        "lower to ~1e-4 if you see early divergence.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=256,
                   help="Effective batch (paper: 256 at long context). The "
                        "paper's '64, 256' scales batch with context purely for "
                        "memory; we instead hold effective batch at 256 and use "
                        "--grad_acc for memory, so the optimizer math is unchanged.")
    p.add_argument("--grad_acc", type=int, default=4,
                   help="Mini-batches per step. Default 4 -> 64/device at batch "
                        "256, fits base size on one GPU. Effective batch unchanged.")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=4000,
                   help="Fixed warmup steps (paper: 4000). Set <0 to fall back to "
                        "--warmup_ratio instead.")
    p.add_argument("--warmup_ratio", type=float, default=0.01,
                   help="Used only if --warmup_steps < 0.")
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    p.add_argument("--mlm_prob", type=float, default=0.15)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    # context-size warmup: "0:64,5:128" == ctx 64 from epoch 0, ctx 128 from epoch 5
    p.add_argument("--max_seq_len", type=str, default="0:64,5:128")
    # architecture
    p.add_argument("--preset", choices=["small", "base"], default="base",
                   help="base = Lukas's 12L/768H (default); small = 6L/768H (fast).")
    p.add_argument("--hidden_size", type=int, default=768)
    p.add_argument("--intermediate_size", type=int, default=3072)
    p.add_argument("--num_hidden_layers", type=int, default=None,
                   help="Override preset layer count.")
    p.add_argument("--num_attention_heads", type=int, default=None)
    p.add_argument("--max_position_embeddings", type=int, default=1024)
    # io / logging
    p.add_argument("--output_path", type=str, default="./deberta-babylm-out")
    p.add_argument("--logging_steps", type=int, default=100)
    p.add_argument("--eval_steps", type=int, default=1000)
    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--fp16", action="store_true", help="Force fp16 (else bf16 if available).")
    p.add_argument("--push_to_hub", type=str, default=None,
                   help="Optional repo id to push the trained model to, e.g. "
                        "augustinian-babylm/deberta-small-50k")
    p.add_argument("--hub_private", action="store_true", default=True)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--debug", action="store_true", help="Tiny subset + 1 epoch.")
    return p


def parse_max_seq_len(spec):
    """'0:64,5:256' -> [(0,64),(5,256)] ; bare '128' -> [(0,128)]."""
    if ":" not in spec:
        return [(0, int(spec))]
    out = []
    for part in spec.split(","):
        ep, sl = part.split(":")
        out.append((int(ep), int(sl)))
    return sorted(out)


def seq_len_for_epoch(epoch, schedule):
    sl = schedule[0][1]
    for start_ep, length in schedule:
        if epoch >= start_ep:
            sl = length
    return sl


# ==========================================================================
# Data
# ==========================================================================
def _fetch(path_or_url, dest):
    if os.path.exists(path_or_url):
        return path_or_url
    if path_or_url.startswith(("http://", "https://")):
        if not os.path.exists(dest):
            print(f"Downloading {path_or_url}")
            subprocess.run(["wget", "-q", "--show-progress", path_or_url, "-O", dest],
                           check=True)
        return dest
    raise FileNotFoundError(path_or_url)


def load_text_dataset(train_url, valid_url, tokenizer, workdir, debug=False):
    from datasets import load_dataset
    os.makedirs(workdir, exist_ok=True)
    train_file = _fetch(train_url, os.path.join(workdir, "train.txt"))
    valid_file = _fetch(valid_url, os.path.join(workdir, "valid.txt"))

    ds = load_dataset("text", data_files={"train": train_file, "validation": valid_file})
    if debug:
        ds["train"] = ds["train"].select(range(min(2000, len(ds["train"]))))
        ds["validation"] = ds["validation"].select(range(min(500, len(ds["validation"]))))

    # Tokenize WITHOUT padding/truncation here; the collator pads per-batch and
    # we re-truncate per seq-len phase. Keep non-empty lines only.
    def tok_fn(batch):
        return tokenizer(batch["text"], add_special_tokens=True,
                         truncation=True, max_length=512)

    ds = ds.filter(lambda ex: len((ex["text"] or "").strip()) > 0)
    ds = ds.map(tok_fn, batched=True, remove_columns=["text"],
                desc="tokenizing")
    return ds


# ==========================================================================
# Model
# ==========================================================================
def build_model(args, tokenizer):
    from transformers import AutoConfig, AutoModelForMaskedLM

    config = AutoConfig.from_pretrained(args.base_config, trust_remote_code=True)
    # ---- match Lukas: inherit base config, override these ----
    config.vocab_size = tokenizer.vocab_size
    config.max_position_embeddings = args.max_position_embeddings
    config.hidden_size = args.hidden_size
    config.intermediate_size = args.intermediate_size
    config.dropout = args.dropout
    config.hidden_dropout_prob = args.dropout
    # special-token ids from our tokenizer
    config.pad_token_id = tokenizer.pad_token_id
    config.bos_token_id = tokenizer.cls_token_id
    config.cls_token_id = tokenizer.cls_token_id
    config.eos_token_id = tokenizer.sep_token_id
    config.sep_token_id = tokenizer.sep_token_id

    # ---- size: preset unless explicitly overridden ----
    layers = args.num_hidden_layers
    heads = args.num_attention_heads
    if layers is None:
        layers = 6 if args.preset == "small" else 12
    if heads is None:
        heads = 12  # 768 / 12 = 64 dim/head
    config.num_hidden_layers = layers
    config.num_attention_heads = heads

    model = AutoModelForMaskedLM.from_config(config, trust_remote_code=True)
    n = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.preset} | {layers}L/{config.hidden_size}H/{heads}heads "
          f"| vocab {config.vocab_size:,} | {n/1e6:.1f}M params")
    return model


# ==========================================================================
# Training (HF Trainer; reproduces Lukas's optimizer + cosine schedule + seqlen warmup)
# ==========================================================================
def train(args):
    import torch
    from transformers import (AutoTokenizer, Trainer, TrainingArguments,
                              DataCollatorForLanguageModeling, set_seed)
    from transformers.optimization import get_cosine_schedule_with_warmup

    set_seed(args.seed)
    schedule = parse_max_seq_len(args.max_seq_len)
    workdir = os.path.join(args.output_path, "_data")

    # NOTE: AutoTokenizer, not DebertaV2Tokenizer (see header).
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    ds = load_text_dataset(args.train_url, args.valid_url, tokenizer, workdir, args.debug)
    model = build_model(args, tokenizer)

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_prob)

    # effective steps for the cosine schedule (mirrors his total_steps math)
    per_device = max(1, args.batch_size // args.grad_acc)
    steps_per_epoch = math.ceil(len(ds["train"]) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    # paper uses a fixed warmup of 4000 steps; fall back to ratio if asked
    if args.warmup_steps is not None and args.warmup_steps >= 0:
        warmup_steps = args.warmup_steps
    else:
        warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    warmup_steps = min(warmup_steps, max(1, total_steps - 1))  # never exceed run
    if args.debug:
        args.epochs, total_steps, warmup_steps = 1, max(1, steps_per_epoch), 1

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_bf16 = bf16_ok and not args.fp16
    use_fp16 = args.fp16 or (torch.cuda.is_available() and not bf16_ok)

    # Build kwargs, then keep only those the INSTALLED transformers accepts.
    # transformers 5.x removed several args (overwrite_output_dir, ...) and
    # renamed evaluation_strategy -> eval_strategy; this survives both.
    import inspect
    ta_kwargs = dict(
        output_dir=args.output_path,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=per_device,
        per_device_eval_batch_size=per_device,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=use_fp16,
        dataloader_num_workers=args.num_workers,
        report_to=(["wandb"] if args.wandb else []),
        seed=args.seed,
        remove_unused_columns=False,
    )
    valid = set(inspect.signature(TrainingArguments.__init__).parameters)
    # handle the 4.x/5.x evaluation_strategy <-> eval_strategy rename
    if "eval_strategy" not in valid and "evaluation_strategy" in valid:
        ta_kwargs["evaluation_strategy"] = ta_kwargs.pop("eval_strategy")
    dropped = [k for k in ta_kwargs if k not in valid]
    for k in dropped:
        ta_kwargs.pop(k)
    if dropped:
        print(f"[compat] dropped TrainingArguments kwargs not in this "
              f"transformers version: {dropped}")
    targs = TrainingArguments(**ta_kwargs)

    # custom optimizer (betas 0.9/0.95) + cosine schedule with 1% warmup
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2), eps=1e-8,
        weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # ---- seq-len warmup: a callback that re-truncates input_ids per epoch ----
    class SeqLenWarmup:
        """Lukas trains short then long. We emulate it by capping sequence
        length per epoch via the collator's max length."""
        def __init__(self, sched):
            self.sched = sched
        def cap_for(self, epoch):
            return seq_len_for_epoch(int(epoch), self.sched)

    seqcap = SeqLenWarmup(schedule)

    class CappingCollator:
        def __init__(self, base, seqcap):
            self.base = base
            self.seqcap = seqcap
            self.epoch = 0
        def __call__(self, features):
            cap = self.seqcap.cap_for(self.epoch)
            for f in features:
                if len(f["input_ids"]) > cap:
                    f["input_ids"] = f["input_ids"][:cap]
                    if "attention_mask" in f:
                        f["attention_mask"] = f["attention_mask"][:cap]
            return self.base(features)

    capping = CappingCollator(collator, seqcap)

    from transformers import TrainerCallback

    class EpochTracker(TrainerCallback):
        def on_epoch_begin(self, a, state, control, **kw):
            capping.epoch = int(state.epoch or 0)
            print(f"[seqlen warmup] epoch {capping.epoch} -> max_len "
                  f"{seqcap.cap_for(capping.epoch)}")

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=capping,
        optimizers=(optimizer, scheduler),
        callbacks=[EpochTracker()],
    )

    print(f"\nTraining: {len(ds['train']):,} train / {len(ds['validation']):,} valid | "
          f"{total_steps:,} steps | warmup {warmup_steps} | "
          f"precision {'bf16' if use_bf16 else ('fp16' if use_fp16 else 'fp32')}")
    trainer.train()
    trainer.save_model(args.output_path)
    tokenizer.save_pretrained(args.output_path)

    # final eval -> perplexity
    metrics = trainer.evaluate()
    if "eval_loss" in metrics:
        metrics["perplexity"] = math.exp(metrics["eval_loss"])
        print(f"Final eval loss {metrics['eval_loss']:.4f} | "
              f"ppl {metrics['perplexity']:.2f}")

    if args.push_to_hub:
        from huggingface_hub import create_repo, HfApi
        create_repo(args.push_to_hub, repo_type="model",
                    private=args.hub_private, exist_ok=True)
        HfApi().upload_folder(folder_path=args.output_path,
                              repo_id=args.push_to_hub, repo_type="model")
        print(f"Pushed -> https://huggingface.co/{args.push_to_hub}")
    return metrics


# ==========================================================================
# Entry point (works as a Colab cell: main([]) , or CLI)
# ==========================================================================
def main(argv=None):
    _ensure_deps()
    args = build_parser().parse_args([] if argv == [] else argv)
    os.makedirs(args.output_path, exist_ok=True)
    return train(args)


if __name__ == "__main__":
    main()
