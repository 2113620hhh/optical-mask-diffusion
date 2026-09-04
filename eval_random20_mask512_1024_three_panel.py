#!/usr/bin/env python3
"""Evaluate the fixed 20 validation samples at 512 and replicated 1024 masks.

The network is evaluated once at its native 512 mask resolution.  The 1024
case is a nearest-neighbour 2x replication of that hard mask; its optical
output and target are both 384x384 so that the compared field of view remains
matched to the native 512->192 experiment.  This script is read-only with
respect to the checkpoint and training output.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from 训练扩散专家掩码模型 import (
    ExpertMaskDataset,
    FFTPlanCache,
    load_metadata,
    normalize_target,
    parse_amp_dtype,
    predict_direct_logits,
)
from eval_random3_mask_resolutions import load_model
from 光学FFT前向传播_小分辨率 import cosine_score_image


SAMPLE_INDICES = (
    151, 303, 401, 556, 610, 847, 1215, 1297, 1307, 1331,
    1379, 1383, 1770, 2014, 2327, 2425, 2492, 2873, 2912, 2918,
)


def panel_image(value: torch.Tensor, *, binary: bool, size: int = 360):
    from PIL import Image

    array = value.detach().cpu().float().numpy()
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    if binary:
        array = (array > 0.5).astype(np.float32)
    else:
        lo, hi = float(array.min()), float(array.max())
        array = (array - lo) / (hi - lo) if hi > lo + 1e-8 else np.zeros_like(array)
    image = Image.fromarray(np.uint8(np.clip(array, 0.0, 1.0) * 255.0), mode="L")
    return image.resize((size, size), resample=Image.Resampling.NEAREST).convert("RGB")


def save_three_panel(path: Path, *, target: torch.Tensor, mask: torch.Tensor,
                     light: torch.Tensor, title: str) -> None:
    from PIL import Image, ImageDraw

    panel_size, gap, header = 360, 12, 44
    labels = ("target", "inferred mask", "FFT light field")
    images = (
        panel_image(target, binary=False, size=panel_size),
        panel_image(mask, binary=True, size=panel_size),
        panel_image(light, binary=False, size=panel_size),
    )
    canvas = Image.new("RGB", (3 * panel_size + 2 * gap, header + panel_size + 26), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill="black")
    for idx, (label, image) in enumerate(zip(labels, images)):
        x = idx * (panel_size + gap)
        draw.text((x + 8, header + 4), label, fill="black")
        canvas.paste(image, (x, header + 26))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {out_dir}")
    out_dir.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    metadata = load_metadata(args.dataset_dir)
    dataset = ExpertMaskDataset(args.dataset_dir, args.split, metadata=metadata, cache_size=1)
    if max(SAMPLE_INDICES) >= len(dataset):
        raise ValueError("Fixed sample indices exceed validation dataset")
    model, ckpt = load_model(checkpoint, device)
    train_args = dict(ckpt.get("train_args", {}))
    if int(train_args.get("mask_resolution", 512)) != 512 or int(train_args.get("target_resolution", 192)) != 192:
        raise ValueError("Expected native 512x512 mask and 192x192 target checkpoint")
    amp_dtype = parse_amp_dtype(args.amp_dtype)

    # Cache CPU tensors before releasing the large model, leaving memory for
    # the 1024-resolution optical FFT.
    selected = []
    try:
        with torch.inference_mode():
            for ordinal, index in enumerate(SAMPLE_INDICES):
                item = dataset[index]
                target = normalize_target(item["target"]).to(device)
                optics = item["optics"].to(device=device, dtype=torch.float32).reshape(1, -1)
                logits = predict_direct_logits(
                    model, target, optics, mask_resolution=512,
                    logit_clip=float(train_args.get("logit_clip", 12.0)),
                    prediction_type=str(train_args.get("prediction_type", "x0")),
                    amp=bool(args.amp), amp_dtype=amp_dtype,
                )
                native_mask = (torch.sigmoid(logits) >= float(args.mask_threshold)).float()
                selected.append({
                    "index": int(index),
                    "sample_id": int(item.get("sample_id", index)),
                    "pattern_type": str(item.get("pattern_type", "unknown")),
                    "target": target[0, 0].cpu(),
                    "mask": native_mask[0, 0].cpu(),
                    "optics": optics.cpu(),
                })
                print(f"MASK {ordinal + 1}/20 index={index}", flush=True)
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rows = []
    for mask_resolution, target_resolution in ((512, 192), (1024, 384)):
        print(f"FFT mask={mask_resolution} target={target_resolution} inter=10", flush=True)
        plans = {}
        for ordinal, sample in enumerate(selected):
            native = sample["mask"].to(device)[None, None]
            mask = F.interpolate(native, size=(mask_resolution, mask_resolution), mode="nearest")
            target = F.interpolate(
                sample["target"].to(device)[None, None],
                size=(target_resolution, target_resolution), mode="nearest",
            )
            optics = sample["optics"].to(device).clone()
            optics[..., 3] = 10.0
            key = tuple(float(x) for x in optics.reshape(-1).tolist())
            if key not in plans:
                plans[key] = FFTPlanCache(
                    mask_resolution=mask_resolution,
                    target_resolution=target_resolution,
                    device=device, max_size=1,
                    forward_model="asf_linear_convolution_fft",
                    inter_num_override=10.0,
                )
            started = time.perf_counter()
            light = plans[key].forward(mask, optics)
            score = float(cosine_score_image(light, target).mean().item())
            row = {
                "sample_index": sample["index"],
                "sample_id": sample["sample_id"],
                "pattern_type": sample["pattern_type"],
                "mask_resolution": mask_resolution,
                "target_resolution": target_resolution,
                "inter_num": 10.0,
                "score": score,
                "mask_mean": float(mask.mean().item()),
                "elapsed_s": time.perf_counter() - started,
            }
            rows.append(row)
            sample_dir = out_dir / "images" / f"mask_{mask_resolution:04d}" / f"sample_{ordinal:02d}_idx_{sample['index']:05d}"
            save_three_panel(
                sample_dir / "three_panel.png",
                target=target[0, 0].cpu(), mask=mask[0, 0].cpu(), light=light[0, 0].cpu(),
                title=f"sample {sample['index']}  mask={mask_resolution}  score={score:.5f}  inter=10",
            )
            print(f"[{mask_resolution} {ordinal + 1}/20] score={score:.6f} time={row['elapsed_s']:.1f}s", flush=True)
        del plans
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "split": args.split,
        "sample_indices": list(SAMPLE_INDICES),
        "mask_threshold": float(args.mask_threshold),
        "inter_num": 10.0,
        "physical_contract": "native 512->192; 1024 nearest-neighbor mask replication and matched 384x384 target/output FOV",
        "results": {},
    }
    for resolution in (512, 1024):
        current = [row for row in rows if row["mask_resolution"] == resolution]
        summary["results"][str(resolution)] = {
            "target_resolution": 192 if resolution == 512 else 384,
            "score_mean": float(np.mean([row["score"] for row in current])),
            "score_min": float(np.min([row["score"] for row in current])),
            "score_max": float(np.max([row["score"] for row in current])),
        }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
