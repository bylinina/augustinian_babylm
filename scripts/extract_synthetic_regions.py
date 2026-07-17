#!/usr/bin/env python
"""
Stage 3 (synthetic): OWLv2-detect target words in generated images,
pool SAM features in the detected boxes VIA THE ORIGINAL BACKEND
(imported from extract_region_embeddings.py -- identical feature space),
and emit Stage-1-format region rows (text = the word, source = synthetic).

Words a detector cannot find contribute no row: failed generations are
silently (and correctly) dropped rather than noise-seeded.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from extract_region_embeddings import SAMBackend, mean_pool  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--detector", default="google/owlv2-base-patch16-ensemble")
    ap.add_argument("--sam", default="facebook/sam-vit-base")
    ap.add_argument("--threshold", type=float, default=0.25)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    device = "cuda" if torch.cuda.is_available() else "cpu"

    det_proc = Owlv2Processor.from_pretrained(args.detector)
    det = Owlv2ForObjectDetection.from_pretrained(args.detector).to(device).eval()
    sam = SAMBackend(args.sam, device)

    images_dir = Path(args.images_dir)
    index = [json.loads(l) for l in (images_dir / "index.jsonl").open()]
    print(f"{len(index)} images to process")

    rows, vecs = [], []
    n_det = n_nodet = 0
    for i, rec in enumerate(index):
        img_path = images_dir / rec["file"]
        words = sorted(set(rec["matched"]))
        if not words or not img_path.exists():
            continue
        img = Image.open(img_path).convert("RGB")
        queries = [f"a photo of a {w}" for w in words]
        inputs = det_proc(text=[queries], images=img,
                          return_tensors="pt").to(device)
        with torch.inference_mode():
            out = det(**inputs)
        res = det_proc.post_process_grounded_object_detection(
            out, threshold=args.threshold,
            target_sizes=torch.tensor([img.size[::-1]]))[0]
        best = {}
        for score, label, box in zip(res["scores"], res["labels"], res["boxes"]):
            li = int(label)
            if li not in best or float(score) > best[li][0]:
                best[li] = (float(score), [float(x) for x in box])
        if not best:
            n_nodet += 1
            continue
        encoded = sam.encode(img)
        for li, (score, xyxy) in best.items():
            x0, y0, x1, y1 = xyxy
            bbox = [x0, y0, max(1.0, x1 - x0), max(1.0, y1 - y0)]  # xywh
            rv = sam.region_vector(encoded, bbox, mean_pool)
            if isinstance(rv, tuple):
                vec, n_patches = rv[0], rv[1]
            else:
                vec, n_patches = rv, -1
            rows.append({"row_id": len(rows), "image_uid": rec["file"],
                         "bbox": bbox, "text": words[li],
                         "source": "synthetic", "n_patches": int(n_patches),
                         "det_score": round(score, 3)})
            vecs.append(np.asarray(vec, dtype=np.float16))
            n_det += 1
        if i % 200 == 0:
            print(f"  {i}/{len(index)} images | {n_det} word-regions", flush=True)

    import pandas as pd
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out / "region_embeddings.parquet")
    np.save(out / "region_embeddings.npy", np.stack(vecs))
    (out / "config.json").write_text(json.dumps({
        "encoder": "sam", "source": "synthetic", "detector": args.detector,
        "threshold": args.threshold, "n_rows": len(rows),
        "n_images_no_detection": n_nodet}, indent=2))
    uniq = len({r["text"] for r in rows})
    print(f"DONE: {len(rows)} regions | {uniq} unique words | "
          f"{n_nodet} images w/o any detection -> {out}")


if __name__ == "__main__":
    main()
