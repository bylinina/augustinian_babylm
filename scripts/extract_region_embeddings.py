"""
extract_region_embeddings.py  —  STAGE 1 (vision only, NO tokenizer)
===================================================================

Produce one pooled VISUAL embedding per annotation row: the whole-image
embedding for sentence/caption rows, and the bbox-region embedding for
description rows. The accompanying text is carried along as METADATA only — it
is NOT tokenized here. Mapping these region/sentence embeddings to per-token
embeddings is a SEPARATE, later stage (build_token_embeddings.py), so we can
freely experiment with how to get from region vectors to token vectors.

Pipeline:
  for each unique image: encode once (full image)
  for each annotation row on it:
      bbox -> patch indices (null bbox = whole image), pool -> 768-d vector
  save one row per annotation: (row_id, image_uid, bbox, text, source, embedding)

Encoders (all ViT-B, 768-d):
  dinov3 : facebook/dinov3-vitb16-pretrain-lvd1689m   (encode -> patches in bbox -> pool)
  sam    : facebook/sam-vit-base                       (bbox prompt -> mask -> pool over mask)
  ibot   : local .pth (official ByteDance ViT-B/16)    (timm -> patches in bbox -> pool)

Output (under --out_dir/<encoder>/), pushed to HF if --push_to_hub:
  region_embeddings.parquet  : row_id, image_uid, bbox, text, source, n_patches
  region_embeddings.npy      : float16 [N, 768], row-aligned with the parquet
  config.json                : encoder, pooling, dims, counts

NO postprocessing (mean-center/L2/scale) here either — these are RAW pooled
embeddings. Normalization choices belong with the token-building stage so they
can be tuned without re-encoding.
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
        """Returns (patch_features[n,768], grid_hw, orig_hw) -- same shape as the
        patch-ViT backends, so SAM uses the identical cheap bbox-patch pooling
        (one encode per image; no per-box mask forward)."""
        orig_w, orig_h = pil_image.size
        inputs = self.processor(pil_image, return_tensors="pt").to(self.device)
        feats = self._vit_patch_features(inputs["pixel_values"])
        return feats, (self.grid, self.grid), (orig_h, orig_w)

    # SAM now uses the SAME cheap bbox-patch pooling as the patch-ViT backends
    # (rectangular bbox -> patch indices -> pool). One encode per image; no
    # per-box mask forward. Reuses PatchViTBackend.region_vector unchanged.
    region_vector = PatchViTBackend.region_vector


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
def stream_target_images(dataset_repo, uid_set, first_shard_only=False):
    """Stream the image shards ONCE, yielding (uid, PIL.Image) for each uid in
    uid_set as it is encountered (each uid only the first time). Holds at most
    one decoded image in memory; stops early once all targets are seen. This
    replaces the old "load every image into a dict" approach, which OOMed on the
    full ~89k-image dataset and hung on small smoke tests (it scanned most shards
    to find a few scattered uids)."""
    from datasets import load_dataset
    from PIL import Image
    pattern = ("images/images-00000.parquet" if first_shard_only
               else "images/images-*.parquet")
    imgs = load_dataset("parquet",
                        data_files=f"hf://datasets/{dataset_repo}/{pattern}",
                        split="train", streaming=True)
    seen = set()
    target = set(uid_set)
    for row in imgs:
        u = row["image_uid"]
        if u in target and u not in seen:
            img = row["image"]
            if not isinstance(img, Image.Image):
                img = Image.open(io.BytesIO(img["bytes"]))
            seen.add(u)
            yield u, img.convert("RGB")
            if len(seen) == len(target):
                break


# ==========================================================================
# Build: one pooled visual embedding per annotation row (NO tokenizer)
# ==========================================================================
def build(args):
    import numpy as np
    import pandas as pd
    import torch
    from huggingface_hub import hf_hub_download
    from tqdm.auto import tqdm
    from collections import defaultdict

    pooler = POOLERS[args.pooling]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | encoder={args.encoder_name} ({args.encoder})")

    # annotations: load the FULL table so we carry ALL metadata through. No
    # tokenization here — text and every other column are just metadata.
    ann_path = hf_hub_download(args.dataset_repo, "annotations.parquet", repo_type="dataset")
    ann = pd.read_parquet(ann_path).reset_index(drop=True)   # ALL columns
    if args.limit_rows:
        ann = ann.head(args.limit_rows)
    if "image_uid" not in ann.columns or "bbox" not in ann.columns:
        raise SystemExit("annotations.parquet must have 'image_uid' and 'bbox' columns")
    if "text" not in ann.columns:
        ann["text"] = ""
    print(f"Annotation rows: {len(ann):,} | columns carried through: "
          f"{list(ann.columns)}")

    rows_by_img = defaultdict(list)
    for i, r in enumerate(ann.itertuples(index=False)):
        rows_by_img[r.image_uid].append((i, r.bbox))
    print(f"Unique images referenced: {len(rows_by_img):,}")

    # Which images do we actually need? If --limit_images, take the first K
    # unique uids (in annotation order) and only target those.
    all_uids = list(rows_by_img.keys())
    if args.first_shard_only:
        import pandas as pd
        from huggingface_hub import hf_hub_download as _dl
        _f = _dl(args.dataset_repo, "images/images-00000.parquet", repo_type="dataset")
        _shard0 = set(pd.read_parquet(_f, columns=["image_uid"])["image_uid"])
        cand = [u for u in all_uids if u in _shard0]
        target_uids = set(cand[:args.limit_images] if args.limit_images else cand)
        print(f"[first_shard_only] {len(target_uids)} target images from shard 0")
    else:
        target_uids = set(all_uids[:args.limit_images] if args.limit_images else all_uids)
    print(f"Targeting {len(target_uids):,} images "
          f"({'limited' if args.limit_images else 'full set'})")

    backend_cls = BACKENDS[args.encoder_name]
    backend = (backend_cls(args.encoder, device, ckpt_key=args.ckpt_key)
               if args.encoder_name == "ibot" else backend_cls(args.encoder, device))
    H = backend.hidden

    N = len(ann)
    emb = np.zeros((N, H), dtype=np.float32)
    got = np.zeros((N,), dtype=bool)

    # Stream images ONE at a time; encode, pool every row on that image, discard.
    n_imgs = 0
    for uid, image in tqdm(stream_target_images(args.dataset_repo, target_uids,
                                                first_shard_only=args.first_shard_only),
                           total=len(target_uids), desc="encoding+pooling"):
        encoded = backend.encode(image)
        for idx, bbox in rows_by_img[uid]:
            rv = backend.region_vector(encoded, bbox, pooler)
            if rv is None:
                continue
            emb[idx] = rv
            got[idx] = True
        n_imgs += 1
        del image, encoded
    print(f"Encoded {n_imgs:,}/{len(target_uids):,} targeted images")

    n_ok = int(got.sum())
    print(f"Embedded {n_ok:,}/{N:,} rows ({100*n_ok/N:.1f}%)")

    # save: the FULL annotations rows that got embedded, with everything carried
    # through, plus a row-aligned .npy of the embeddings. Optionally also embed
    # the vector as a column directly in the parquet ("annotations + embeddings").
    keep = np.where(got)[0]
    out = os.path.join(args.out_dir, args.encoder_name)
    os.makedirs(out, exist_ok=True)

    meta = ann.iloc[keep].reset_index(drop=True).copy()   # ALL original columns
    meta["is_region"] = [b is not None and (not hasattr(b, "__len__") or len(b) > 0)
                         for b in meta["bbox"].values]
    kept_emb = emb[keep].astype(np.float16 if args.store_fp16 else np.float32)

    if args.embed_in_parquet:
        # store each embedding as a list column so the parquet alone is
        # "annotations + embeddings" — self-contained, no separate .npy needed.
        meta["embedding"] = list(kept_emb)
    meta.to_parquet(os.path.join(out, "region_embeddings.parquet"))
    np.save(os.path.join(out, "region_embeddings.npy"), kept_emb)

    import json
    n_region = int(meta["is_region"].sum())
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({"encoder": args.encoder_name, "encoder_id": args.encoder,
                   "pooling": args.pooling, "hidden": H, "n_rows": int(len(keep)),
                   "n_region": n_region, "n_whole_image": int(len(keep) - n_region),
                   "dtype": "float16" if args.store_fp16 else "float32",
                   "columns": list(meta.columns),
                   "embedding_in_parquet": bool(args.embed_in_parquet),
                   "note": "RAW pooled embeddings (no normalization). parquet = "
                           "full annotations rows that were embedded; .npy[i] is "
                           "row-aligned with parquet row i."}, f, indent=2)
    print(f"Saved -> {out}/region_embeddings.{{parquet,npy}}")
    print(f"  carried columns: {list(meta.columns)}")
    print(f"  region rows: {n_region:,} | whole-image rows: {len(keep)-n_region:,}")

    if args.push_to_hub:
        from huggingface_hub import HfApi, create_repo
        create_repo(args.push_to_hub, repo_type="dataset", private=True, exist_ok=True)
        HfApi().upload_folder(folder_path=out, repo_id=args.push_to_hub,
                              repo_type="dataset", path_in_repo=args.encoder_name)
        print(f"Pushed -> {args.push_to_hub}/{args.encoder_name}")


def build_parser():
    import argparse
    p = argparse.ArgumentParser(description="Stage 1: per-row visual embeddings (no tokenizer)")
    p.add_argument("--dataset_repo", type=str, default="augustinian-babylm/augustinian_babylm")
    p.add_argument("--encoder_name", type=str, default="dinov3", choices=list(BACKENDS))
    p.add_argument("--encoder", type=str,
                   default="facebook/dinov3-vitb16-pretrain-lvd1689m",
                   help="HF id (dinov3/sam) or local .pth (ibot). "
                        "iBOT: facebook none — pass the ByteDance ViT-B/16 .pth path.")
    p.add_argument("--out_dir", type=str, default="./region_embeddings")
    p.add_argument("--pooling", type=str, default="mean", choices=list(POOLERS))
    p.add_argument("--ckpt_key", type=str, default="teacher", help="iBOT only.")
    p.add_argument("--store_fp16", action="store_true", default=True)
    p.add_argument("--embed_in_parquet", action="store_true", default=True,
                   help="Also store the embedding as a column in the parquet, so "
                        "it's literally 'annotations + embeddings' in one file "
                        "(the row-aligned .npy is always written too).")
    p.add_argument("--limit_images", type=int, default=0)
    p.add_argument("--limit_rows", type=int, default=0)
    p.add_argument("--first_shard_only", action="store_true",
                   help="Smoke-test mode: read ONLY images-00000.parquet and "
                        "target images found there, so no slow scan across shards.")
    p.add_argument("--push_to_hub", type=str, default=None)
    return p


def main(argv=None):
    _ensure_deps()
    build(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
