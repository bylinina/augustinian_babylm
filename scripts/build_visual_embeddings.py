#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_visual_embeddings.py
==========================

Build DeBERTa-ready visual embedding tables from the grounding dataset, in a
SINGLE compact pass (no big patch-grid cache).

Per your choices:
  * compact output — we pool regions during encoding and keep only the pooled
    768-d vectors, accumulated per token. Storage is ~MB, not hundreds of GB.
    Trade-off: changing the pooling strategy later requires re-encoding.
  * pluggable pooling — mean by default (POOLERS dict).
  * encoders — DINOv3 and iBOT share the patch-ViT path (encode full image,
    slice patches inside bbox, pool). SAM is a separate backend (mask-prompt)
    left as a clearly marked stub to fill in next, because uniform bbox-patch
    pooling would defeat SAM's purpose (rectangular background contamination).

Flow:
  load annotations (image_uid, bbox, text) + images
  for each unique image: encode once -> patch grid (CLS + register tokens dropped)
  for each annotation row on that image:
      bbox -> patch indices (null bbox = whole image), pool -> region vector
      attribute region vector to the token id(s) of `text`
  per token: average -> E_raw [V,768]
  postprocess (mean-center, L2-norm, x0.55 -> std~0.02) -> E_init [V,768]
  save E_raw / E_init / coverage  (per encoder x tokenizer)

The expensive ViT pass runs once per unique image. Because output is per-token,
the per-token aggregation depends on the tokenizer, so run once per tokenizer
(or pass several with --tokenizers a=repo,b=repo to share the encode pass).
"""

import os
import sys
import io
import json
import math
import argparse
import subprocess
from collections import defaultdict


def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)


def _ensure_deps():
    try:
        import torch, transformers, datasets, numpy, pandas, PIL, safetensors  # noqa
    except Exception:
        _pip("-U", "torch", "transformers", "datasets", "huggingface_hub",
             "numpy", "pandas", "Pillow", "safetensors", "timm", "tqdm")


# ==========================================================================
# Geometry: orig-pixel xywh bbox -> processed patch-grid flat indices
# ==========================================================================
def bbox_to_patch_indices(bbox_xywh, orig_hw, grid_hw, patch_size=16):
    orig_h, orig_w = orig_hw
    grid_h, grid_w = grid_hw
    proc_h, proc_w = grid_h * patch_size, grid_w * patch_size
    sx, sy = proc_w / float(orig_w), proc_h / float(orig_h)
    x, y, w, h = bbox_xywh
    px0, py0, px1, py1 = x * sx, y * sy, (x + w) * sx, (y + h) * sy
    c0 = int(math.floor(px0 / patch_size)); r0 = int(math.floor(py0 / patch_size))
    c1 = int(math.ceil(px1 / patch_size)) - 1; r1 = int(math.ceil(py1 / patch_size)) - 1
    c0 = max(0, min(c0, grid_w - 1)); c1 = max(0, min(c1, grid_w - 1))
    r0 = max(0, min(r0, grid_h - 1)); r1 = max(0, min(r1, grid_h - 1))
    if c1 < c0:
        c0 = c1 = max(0, min(int((px0 + px1) / 2 / patch_size), grid_w - 1))
    if r1 < r0:
        r0 = r1 = max(0, min(int((py0 + py1) / 2 / patch_size), grid_h - 1))
    idxs = []
    for r in range(r0, r1 + 1):
        base = r * grid_w
        for c in range(c0, c1 + 1):
            idxs.append(base + c)
    return idxs


# ==========================================================================
# Pooling (pluggable). Each takes [n_sel, hidden] -> [hidden].
# ==========================================================================
def mean_pool(vecs):
    return vecs.mean(axis=0)

POOLERS = {"mean": mean_pool}


# ==========================================================================
# Encoder backends
# ==========================================================================
class PatchViTBackend:
    """DINOv3 / iBOT-style: full-image encode -> patch tokens (CLS+registers
    stripped) -> slice patches inside bbox -> pool."""
    def __init__(self, model_id, device):
        from transformers import AutoImageProcessor, AutoModel
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        self.device = device
        self.patch_size = getattr(self.model.config, "patch_size", 16)
        self.num_register = int(getattr(self.model.config, "num_register_tokens", 0) or 0)
        self.n_prefix = 1 + self.num_register
        self.hidden = self.model.config.hidden_size
        print(f"  [{model_id}] patch={self.patch_size} registers={self.num_register} "
              f"prefix={self.n_prefix} hidden={self.hidden}")

    def encode(self, pil_image):
        import torch
        orig_w, orig_h = pil_image.size
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        _, _, proc_h, proc_w = inputs["pixel_values"].shape
        with torch.inference_mode():
            hs = self.model(**inputs).last_hidden_state[0]   # [n_tok, hidden]
        patches = hs[self.n_prefix:, :]
        gh, gw = proc_h // self.patch_size, proc_w // self.patch_size
        if patches.shape[0] != gh * gw:
            patches = patches[:gh * gw, :]
        return patches.float().cpu().numpy(), (gh, gw), (orig_h, orig_w)

    def region_vector(self, encoded, bbox, pooler):
        patches, grid_hw, orig_hw = encoded
        if bbox is None or (hasattr(bbox, "__len__") and len(bbox) == 0):
            sel = patches
        else:
            idxs = bbox_to_patch_indices(list(bbox), orig_hw, grid_hw, self.patch_size)
            sel = patches[idxs] if idxs else patches
        return pooler(sel) if sel.shape[0] else None


class SAMBackend:
    """SAM region strategy (your spec): bbox as prompt -> predicted object mask
    -> pool the vision-encoder PATCH features over the mask. Avoids the
    rectangular-background contamination of plain bbox pooling.

    Dimensionality note: SAM's ViT backbone is 768-dim (hidden_size) but its
    neck projects to 256 (output_channels); `outputs.image_embeddings` is the
    256-d post-neck map. To stay 768-d (comparable to DINOv3/iBOT) we pool the
    PRE-neck 768-d patch features from model.vision_encoder. We verify the 768
    shape at load and fail loudly if the attribute path differs in your
    transformers version (rather than silently pooling the wrong tensor).

    Whole-image rows (bbox is None): no prompt, pool all patch features.
    """
    def __init__(self, model_id, device):
        from transformers import SamModel, SamProcessor
        self.processor = SamProcessor.from_pretrained(model_id)
        self.model = SamModel.from_pretrained(model_id).to(device).eval()
        self.device = device
        self.hidden = self.model.config.vision_config.hidden_size  # expect 768
        if self.hidden != 768:
            print(f"  [WARN] SAM vision hidden_size={self.hidden}, not 768")
        # SAM resizes to a square (default 1024); patch grid is image/patch.
        self.img_size = self.model.config.vision_config.image_size
        self.patch_size = self.model.config.vision_config.patch_size
        self.grid = self.img_size // self.patch_size  # e.g. 1024/16 = 64
        print(f"  [{model_id}] SAM vision hidden={self.hidden} "
              f"grid={self.grid}x{self.grid} patch={self.patch_size}")

    def _vit_patch_features(self, pixel_values):
        """Run ONLY the SAM vision encoder and return pre-neck 768-d patch
        features as [grid*grid, 768]. Defensive about the output layout."""
        import torch
        with torch.inference_mode():
            ve_out = self.model.vision_encoder(
                pixel_values, output_hidden_states=True)
        # SAM's vision encoder output is [B, grid, grid, C] (post-neck=256) for
        # last_hidden_state; the pre-neck 768-d features are the last entry of
        # hidden_states. Find the one whose channel dim == self.hidden.
        cand = []
        if getattr(ve_out, "hidden_states", None) is not None:
            cand.extend(ve_out.hidden_states)
        if getattr(ve_out, "last_hidden_state", None) is not None:
            cand.append(ve_out.last_hidden_state)
        feat = None
        for t in reversed(cand):  # prefer the deepest layer at 768
            if t is not None and t.shape[-1] == self.hidden:
                feat = t
                break
        if feat is None:
            raise RuntimeError(
                f"Could not find a {self.hidden}-d SAM vision feature map. "
                f"Available shapes: {[tuple(t.shape) for t in cand if t is not None]}. "
                f"Inspect model.vision_encoder output for your transformers "
                f"version and adjust _vit_patch_features().")
        # feat is [B, grid, grid, hidden] (or [B, n, hidden]); flatten to [n, hidden]
        feat = feat[0]
        if feat.dim() == 3:           # [grid, grid, hidden]
            feat = feat.reshape(-1, self.hidden)
        return feat.float().cpu().numpy()

    def encode(self, pil_image):
        """Returns (patch_features[n,768], grid_hw, orig_hw, inputs) — inputs kept
        so region prompting can reuse the processed image."""
        import torch
        orig_w, orig_h = pil_image.size
        inputs = self.processor(pil_image, return_tensors="pt").to(self.device)
        feats = self._vit_patch_features(inputs["pixel_values"])
        return feats, (self.grid, self.grid), (orig_h, orig_w), inputs

    def region_vector(self, encoded, bbox, pooler):
        import numpy as np
        import torch
        feats, grid_hw, orig_hw, inputs = encoded
        if bbox is None or (hasattr(bbox, "__len__") and len(bbox) == 0):
            return pooler(feats)  # whole image
        # bbox xywh (orig px) -> xyxy for SAM prompt
        x, y, w, h = [float(v) for v in bbox]
        boxes = [[[x, y, x + w, y + h]]]  # [image, box_batch, 4]
        prompt = self.processor(
            images=None, input_boxes=boxes,
            original_sizes=inputs["original_sizes"],
            reshaped_input_sizes=inputs.get("reshaped_input_sizes"),
            return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model(
                pixel_values=inputs["pixel_values"],
                input_boxes=prompt["input_boxes"],
                multimask_output=False)
        # mask back to original size, then to the patch grid
        masks = self.processor.image_processor.post_process_masks(
            out.pred_masks.cpu(), inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu())
        m = masks[0][0, 0].numpy()  # [orig_h, orig_w] bool/float
        gh, gw = grid_hw
        # downsample mask to grid by average-pool, threshold -> patch selection
        import numpy as np
        oh, ow = m.shape
        ys = (np.arange(gh) * oh / gh).astype(int)
        xs = (np.arange(gw) * ow / gw).astype(int)
        grid_mask = m[np.ix_(ys, xs)] > 0.5
        idxs = np.where(grid_mask.reshape(-1))[0]
        if len(idxs) == 0:
            return None  # mask didn't cover any patch; caller skips
        return pooler(feats[idxs])


class IBOTTimmBackend:
    """iBOT ViT-B/16 via timm, loading the OFFICIAL ByteDance checkpoint
    (github.com/bytedance/ibot) — no third-party HF mirror.

    iBOT's released file is a raw .pth, not an AutoModel repo, so we:
      1. build a plain vit_base_patch16_224 skeleton (num_classes=0, no head)
      2. load the iBOT state dict, extracting the backbone (key 'teacher' or
         'student', stripping 'backbone.'/'module.' prefixes and head/proj keys)
      3. use forward_features -> [B, n_tok, 768], drop CLS (+ any register/dist
         tokens timm reports) to get patch tokens.

    `model_id` here is the iBOT checkpoint: a local path or an http(s) URL to the
    .pth (e.g. the ByteDance release URL for ViT-B/16 ImageNet-22k).
    """
    def __init__(self, model_id, device, ckpt_key="teacher",
                 arch="vit_base_patch16_224"):
        import timm, torch
        self.device = device
        self.model = timm.create_model(arch, pretrained=False, num_classes=0)
        self.patch_size = self.model.patch_embed.patch_size[0]
        self.hidden = self.model.embed_dim
        # number of non-patch prefix tokens timm prepends (cls [+ reg])
        self.n_prefix = getattr(self.model, "num_prefix_tokens", 1)

        sd = self._load_state_dict(model_id, ckpt_key)
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        loaded = len(sd) - len(unexpected)
        print(f"  [iBOT {arch}] loaded {loaded} tensors from {model_id} "
              f"(key={ckpt_key}); missing={len(missing)} unexpected={len(unexpected)}")
        if loaded < 50:
            raise RuntimeError(
                f"Only {loaded} tensors matched — checkpoint key/prefix likely "
                f"wrong. Inspect the .pth keys and adjust ckpt_key/_clean_key.")
        self.model = self.model.to(device).eval()

        cfg = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(**cfg, is_training=False)
        print(f"  [iBOT] hidden={self.hidden} patch={self.patch_size} "
              f"prefix_tokens={self.n_prefix} input={cfg.get('input_size')}")

    @staticmethod
    def _clean_key(k):
        for p in ("module.", "backbone.", "0.backbone."):
            if k.startswith(p):
                k = k[len(p):]
        return k

    def _load_state_dict(self, model_id, ckpt_key):
        import torch
        if model_id.startswith(("http://", "https://")):
            ckpt = torch.hub.load_state_dict_from_url(model_id, map_location="cpu")
        else:
            ckpt = torch.load(model_id, map_location="cpu")
        # iBOT wraps weights under 'teacher'/'student'; fall back to common keys
        if isinstance(ckpt, dict):
            for k in (ckpt_key, "state_dict", "model"):
                if k in ckpt and isinstance(ckpt[k], dict):
                    ckpt = ckpt[k]
                    break
        # keep only backbone tensors, drop head/projector
        out = {}
        for k, v in ckpt.items():
            ck = self._clean_key(k)
            if any(s in ck for s in ("head", "projection", "proj", "last_layer")):
                continue
            out[ck] = v
        return out

    def encode(self, pil_image):
        import torch
        orig_w, orig_h = pil_image.size
        x = self.transform(pil_image).unsqueeze(0).to(self.device)
        _, _, proc_h, proc_w = x.shape
        with torch.inference_mode():
            tokens = self.model.forward_features(x)[0]   # [n_tok, hidden]
        patches = tokens[self.n_prefix:, :]
        gh, gw = proc_h // self.patch_size, proc_w // self.patch_size
        if patches.shape[0] != gh * gw:
            patches = patches[:gh * gw, :]
        return patches.float().cpu().numpy(), (gh, gw), (orig_h, orig_w)

    # same region selection as the patch-ViT path
    region_vector = PatchViTBackend.region_vector


BACKENDS = {"dinov3": PatchViTBackend, "ibot": IBOTTimmBackend, "sam": SAMBackend}


# ==========================================================================
# Load unique images referenced by the grounding dataset
# ==========================================================================
def load_unique_images(dataset_repo, limit=0):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from datasets import load_dataset
    from PIL import Image
    ann_path = hf_hub_download(dataset_repo, "annotations.parquet", repo_type="dataset")
    ann = pd.read_parquet(ann_path, columns=["image_uid"])
    uids = list(dict.fromkeys(ann["image_uid"].tolist()))
    if limit:
        uids = uids[:limit]
    uid_set = set(uids)
    print(f"Unique images: {len(uids):,}")
    imgs = load_dataset("parquet",
                        data_files=f"hf://datasets/{dataset_repo}/images/images-*.parquet",
                        split="train", streaming=True)
    found = {}
    for row in imgs:
        u = row["image_uid"]
        if u in uid_set and u not in found:
            img = row["image"]
            if not isinstance(img, Image.Image):
                img = Image.open(io.BytesIO(img["bytes"]))
            found[u] = img.convert("RGB")
            if len(found) == len(uid_set):
                break
    print(f"Resolved {len(found):,}/{len(uids):,} images")
    return found


# ==========================================================================
# Build
# ==========================================================================
def build(args):
    import numpy as np
    import pandas as pd
    import torch
    from transformers import AutoTokenizer
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import save_file
    from tqdm.auto import tqdm

    pooler = POOLERS[args.pooling]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | encoder={args.encoder_name} ({args.encoder})")

    # tokenizers: name=repo pairs (one or several sharing the encode pass)
    tok_specs = dict(s.split("=", 1) for s in args.tokenizers.split(","))
    tokenizers = {tag: AutoTokenizer.from_pretrained(repo) for tag, repo in tok_specs.items()}
    for tag, t in tokenizers.items():
        print(f"  tokenizer {tag}: {tok_specs[tag]} (vocab {t.vocab_size:,})")

    # annotations grouped by image
    ann_path = hf_hub_download(args.dataset_repo, "annotations.parquet", repo_type="dataset")
    ann = pd.read_parquet(ann_path, columns=["image_uid", "bbox", "text"])
    if args.limit_rows:
        ann = ann.head(args.limit_rows)
    rows_by_img = defaultdict(list)
    for r in ann.itertuples(index=False):
        rows_by_img[r.image_uid].append((r.bbox, r.text))
    print(f"Annotation rows: {len(ann):,} over {len(rows_by_img):,} images")

    images = load_unique_images(args.dataset_repo, args.limit_images)
    backend_cls = BACKENDS[args.encoder_name]
    if args.encoder_name == "ibot":
        backend = backend_cls(args.encoder, device, ckpt_key=args.ckpt_key)
    else:
        backend = backend_cls(args.encoder, device)
    H = backend.hidden

    # per-tokenizer running sums
    sums = {tag: np.zeros((t.vocab_size, H), np.float64) for tag, t in tokenizers.items()}
    counts = {tag: np.zeros((t.vocab_size,), np.int64) for tag, t in tokenizers.items()}

    for uid, rows in tqdm(rows_by_img.items(), desc="encoding+pooling"):
        if uid not in images:
            continue
        encoded = backend.encode(images[uid])
        for bbox, text in rows:
            rv = backend.region_vector(encoded, bbox, pooler)
            if rv is None:
                continue
            for tag, t in tokenizers.items():
                ids = t(str(text), add_special_tokens=False)["input_ids"]
                if not ids:
                    continue
                targets = [ids[-1]] if args.seed_last_subword else ids
                for tid in targets:
                    sums[tag][tid] += rv
                    counts[tag][tid] += 1

    # finalize each tokenizer table
    for tag, t in tokenizers.items():
        V = t.vocab_size
        seeded = counts[tag] > 0
        n_seeded = int(seeded.sum())
        E_raw = np.zeros((V, H), np.float32)
        E_raw[seeded] = (sums[tag][seeded] / counts[tag][seeded, None]).astype(np.float32)
        X = E_raw[seeded].astype(np.float64)
        X = X - X.mean(0, keepdims=True)
        n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1.0
        X = args.scale * (X / n)
        E_init = np.zeros((V, H), np.float32)
        E_init[seeded] = X.astype(np.float32)
        std = float(E_init[seeded].std()) if n_seeded else 0.0
        print(f"[{tag}] seeded {n_seeded:,}/{V:,} ({100*n_seeded/V:.1f}%) | std {std:.4f}")

        out = os.path.join(args.out_dir, args.encoder_name, tag)
        os.makedirs(out, exist_ok=True)
        save_file({"embeddings": E_raw}, os.path.join(out, "E_raw.safetensors"))
        save_file({"embeddings": E_init, "seeded_mask": seeded.astype(np.int8)},
                  os.path.join(out, "E_init.safetensors"))
        pd.DataFrame({"token_id": np.arange(V), "n_regions": counts[tag],
                      "seeded": seeded}).to_parquet(os.path.join(out, "coverage.parquet"))
        with open(os.path.join(out, "config.json"), "w") as f:
            json.dump({"encoder": args.encoder_name, "encoder_id": args.encoder,
                       "tokenizer": tok_specs[tag], "vocab_size": V, "hidden": H,
                       "pooling": args.pooling, "scale": args.scale,
                       "seed_last_subword": args.seed_last_subword,
                       "n_seeded": n_seeded, "realised_std": std}, f, indent=2)
        if args.push_to_hub:
            from huggingface_hub import HfApi, create_repo
            create_repo(args.push_to_hub, repo_type="dataset", private=True, exist_ok=True)
            HfApi().upload_folder(folder_path=out, repo_id=args.push_to_hub,
                                  repo_type="dataset",
                                  path_in_repo=f"{args.encoder_name}/{tag}")
            print(f"  pushed [{tag}] -> {args.push_to_hub}/{args.encoder_name}/{tag}")


def build_parser():
    p = argparse.ArgumentParser(description="Build visual embedding tables (compact)")
    p.add_argument("--dataset_repo", type=str,
                   default="augustinian-babylm/multimodal-babylm-grounding")
    p.add_argument("--encoder_name", type=str, default="dinov3", choices=list(BACKENDS))
    p.add_argument("--encoder", type=str,
                   default="facebook/dinov3-vitb16-pretrain-lvd1689m",
                   help="HF model id. iBOT: ViT-B/16 repo with --encoder_name ibot. "
                        "SAM: facebook/sam-vit-base with --encoder_name sam.")
    p.add_argument("--tokenizers", type=str,
                   default="50k=augustinian-babylm/babylm-bpe-50k,"
                           "75k=augustinian-babylm/babylm-bpe-75k,"
                           "100k=augustinian-babylm/babylm-bpe-100k",
                   help="Comma-separated tag=repo pairs; all share one encode pass.")
    p.add_argument("--out_dir", type=str, default="./visual_embeddings")
    p.add_argument("--pooling", type=str, default="mean", choices=list(POOLERS))
    p.add_argument("--scale", type=float, default=0.55)
    p.add_argument("--seed_last_subword", action="store_true", default=True)
    p.add_argument("--ckpt_key", type=str, default="teacher",
                   help="For iBOT: which key in the .pth holds the backbone "
                        "(teacher or student).")
    p.add_argument("--limit_images", type=int, default=0)
    p.add_argument("--limit_rows", type=int, default=0)
    p.add_argument("--push_to_hub", type=str, default=None)
    return p


def main(argv=None):
    _ensure_deps()
    build(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
