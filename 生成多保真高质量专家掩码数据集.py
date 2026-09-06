#!/usr/bin/env python3
"""Generate the simple 120K dataset with the validated multi-fidelity optimizer.

The source dataset is read-only.  Only native ``shard_*.npz`` files are
optimized; their ``_rot90_cw`` partners are produced by exact rotation.  Each
worker keeps one proxy propagation plan and one final linear-ASF reference plan
on its GPU, processes one sample at a time, and atomically commits complete
shards.  This makes a stopped run safely resumable without a second mask cache.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_DIR = Path(__file__).resolve().parent
LAB_DIR = PROJECT_DIR / "expert_mask_quality_lab_20260829"
if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))

from direct_linear_gumbel_experiment import DirectLinearASFPlan, ncc, normalize_mean
from fast_multifidelity_hybrid import exponential_value, remember_candidate
from 光学FFT前向传播_小分辨率 import (
    SmallFFTForwardPlan,
    gradient_cosine_score_image,
    highpass_cosine_score_image,
)
from 生成多精度正确专家掩码数据集 import (
    REQUIRED_FIELDS,
    atomic_json,
    atomic_npz,
    is_valid_shard,
    load_source,
    make_output_arrays,
    native_source_shards,
    rotated_output,
)


GENERATOR_VERSION = "multifidelity-linear-asf-v1"
PARTIAL_SAVE_EVERY = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate higher-quality 512x512 expert masks with proxy optimization and final linear-ASF acceptance."
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--mask-resolution", type=int, default=512)
    parser.add_argument("--target-resolution", type=int, default=192)
    parser.add_argument("--proxy-inter-num", type=int, default=6)
    parser.add_argument("--reference-inter-num", type=int, default=10)
    parser.add_argument("--continuous-steps", type=int, default=250)
    parser.add_argument("--binary-steps", type=int, default=250)
    parser.add_argument("--continuous-lr", type=float, default=0.1)
    parser.add_argument("--binary-lr", type=float, default=0.1)
    parser.add_argument("--tau-start", type=float, default=1.0)
    parser.add_argument("--tau-end", type=float, default=0.2)
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--candidate-every", type=int, default=10)
    parser.add_argument("--deterministic-every", type=int, default=20)
    parser.add_argument("--quality-attempts", type=int, default=1)
    parser.add_argument("--direct-polish-steps", type=int, default=0)
    parser.add_argument("--direct-polish-lr", type=float, default=0.03)
    parser.add_argument(
        "--min-ncc",
        type=float,
        default=0.0,
        help="Optional quality threshold; zero disables quality-gate blocking.",
    )
    parser.add_argument("--field-threshold", type=float, default=0.57)
    parser.add_argument("--foreground-threshold", type=float, default=0.05)
    parser.add_argument("--min-free-gib", type=float, default=12.0)
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--max-samples-per-shard", type=int, default=0)
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--allow-incomplete-finalize", action="store_true")
    parser.add_argument(
        "--strict-quality-gate",
        action="store_true",
        help="Fail a rank when a fixed-mask NCC is below --min-ncc; default keeps the best mask and continues.",
    )
    return parser.parse_args()


def config_dict(args: argparse.Namespace) -> dict[str, Any]:
    keys = (
        "mask_resolution",
        "target_resolution",
        "proxy_inter_num",
        "reference_inter_num",
        "continuous_steps",
        "binary_steps",
        "continuous_lr",
        "binary_lr",
        "tau_start",
        "tau_end",
        "candidate_count",
        "candidate_every",
        "deterministic_every",
        "quality_attempts",
        "direct_polish_steps",
        "direct_polish_lr",
        "min_ncc",
        "field_threshold",
        "foreground_threshold",
        "seed",
    )
    return {
        "generator_version": GENERATOR_VERSION,
        "source_dir": str(Path(args.source_dir).resolve()),
        **{key: getattr(args, key) for key in keys},
    }


def ensure_config(out_dir: Path, args: argparse.Namespace) -> None:
    path = out_dir / "generation_config.json"
    expected = config_dict(args)
    if path.is_file():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != expected:
            changed = {key for key in set(actual) | set(expected) if actual.get(key) != expected.get(key)}
            resumable_tuning = {
                "continuous_steps",
                "binary_steps",
                "quality_attempts",
                "direct_polish_steps",
                "direct_polish_lr",
                "min_ncc",
                "strict_quality_gate",
            }
            if changed and changed.issubset(resumable_tuning):
                atomic_json(path, expected)
                print(
                    f"Updated resumable optimization settings: {sorted(changed)}",
                    flush=True,
                )
            else:
                raise RuntimeError(
                    f"Existing output uses a different generation configuration: {path}"
                )
    else:
        atomic_json(path, expected)


def ensure_free_space(path: Path, minimum_gib: float) -> None:
    free = shutil.disk_usage(path).free
    minimum = int(float(minimum_gib) * (1024**3))
    if free < minimum:
        raise RuntimeError(
            f"Free disk space is {free / 1024**3:.2f} GiB, below the protected reserve "
            f"of {float(minimum_gib):.2f} GiB"
        )


def partial_path(out_dir: Path, split: str, source_path: Path) -> Path:
    """Return the resumable state file for one source shard."""
    return out_dir / "partials" / split / f"{source_path.name}.partial.npz"


def empty_metric() -> dict[str, float]:
    return {
        "score": 0.0,
        "highpass_score": 0.0,
        "gradient_score": 0.0,
        "mask_mean": 0.0,
        "quality_pass": 0.0,
    }


def load_partial(
    path: Path,
    count: int,
    mask_resolution: int,
    target_resolution: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]], np.ndarray] | None:
    """Load a partial shard, returning None for missing or stale state."""
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as state:
            completed = state["completed"].astype(bool, copy=False)
            masks = state["masks"].astype(np.uint8, copy=False)
            lights = state["lights"].astype(np.float16, copy=False)
            scores = state["score"].astype(np.float32, copy=False)
            highpass = state["highpass_score"].astype(np.float32, copy=False)
            gradient = state["gradient_score"].astype(np.float32, copy=False)
            mask_mean = state["mask_mean"].astype(np.float32, copy=False)
        expected_mask = (count, 1, int(mask_resolution), int(mask_resolution))
        expected_light = (count, 1, int(target_resolution), int(target_resolution))
        if (
            completed.shape != (count,)
            or masks.shape != expected_mask
            or lights.shape != expected_light
            or any(values.shape != (count,) for values in (scores, highpass, gradient, mask_mean))
        ):
            raise ValueError("partial state shape does not match the source shard")
        metrics = [empty_metric() for _ in range(count)]
        for index in np.flatnonzero(completed):
            metrics[int(index)] = {
                "score": float(scores[index]),
                "highpass_score": float(highpass[index]),
                "gradient_score": float(gradient[index]),
                "mask_mean": float(mask_mean[index]),
                "quality_pass": 0.0,
            }
        return masks.copy(), lights.copy(), metrics, completed.copy()
    except Exception as error:
        print(f"[partial] ignoring invalid state {path}: {type(error).__name__}: {error}", flush=True)
        return None


def save_partial(
    path: Path,
    masks: np.ndarray,
    lights: np.ndarray,
    metrics: list[dict[str, float]],
    completed: np.ndarray,
) -> None:
    """Atomically persist enough state to resume inside a shard."""
    atomic_npz(
        path,
        {
            "completed": completed.astype(np.uint8, copy=False),
            "masks": masks.astype(np.uint8, copy=False),
            "lights": lights.astype(np.float16, copy=False),
            "score": np.asarray([row["score"] for row in metrics], dtype=np.float32),
            "highpass_score": np.asarray(
                [row["highpass_score"] for row in metrics], dtype=np.float32
            ),
            "gradient_score": np.asarray(
                [row["gradient_score"] for row in metrics], dtype=np.float32
            ),
            "mask_mean": np.asarray([row["mask_mean"] for row in metrics], dtype=np.float32),
        },
    )


def optimize_attempt(
    target: torch.Tensor,
    proxy: SmallFFTForwardPlan,
    reference: DirectLinearASFPlan,
    args: argparse.Namespace,
    seed: int,
) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    stage_times: dict[str, float] = {}

    started = time.perf_counter()
    latent = (
        torch.randn(
            (1, 1, int(args.mask_resolution), int(args.mask_resolution)),
            device=target.device,
        )
        * 0.1
    ).requires_grad_(True)
    optimizer = torch.optim.Adam([latent], lr=float(args.continuous_lr))
    best_soft_score = -1.0
    best_probability: torch.Tensor | None = None
    for _ in range(int(args.continuous_steps)):
        optimizer.zero_grad(set_to_none=True)
        probability = torch.sigmoid(latent)
        score = ncc(normalize_mean(proxy.forward(probability)), target)[0]
        (1.0 - score).backward()
        optimizer.step()
        score_value = float(score.detach())
        if score_value > best_soft_score:
            best_soft_score = score_value
            best_probability = probability.detach().clone()
    stage_times["continuous_seconds"] = time.perf_counter() - started
    if best_probability is None:
        raise RuntimeError("Continuous optimization produced no probability map")

    probability = best_probability.clamp(1e-4, 1.0 - 1e-4)
    signed = 0.5 * torch.log(probability / (1.0 - probability))[0, 0]
    logits = torch.stack((-signed, signed), dim=-1).detach().requires_grad_(True)
    del latent, probability, best_probability, optimizer

    started = time.perf_counter()
    optimizer = torch.optim.Adam([logits], lr=float(args.binary_lr))
    candidates: list[tuple[float, torch.Tensor]] = []
    for step in range(1, int(args.binary_steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        tau = exponential_value(args.tau_start, args.tau_end, step, int(args.binary_steps))
        sampled = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)[..., 1]
        sampled = sampled[None, None]
        score = ncc(normalize_mean(proxy.forward(sampled)), target)[0]
        (1.0 - score).backward()
        optimizer.step()

        if step % max(1, int(args.candidate_every)) == 0 or step == int(args.binary_steps):
            remember_candidate(
                candidates,
                float(score.detach()),
                sampled,
                int(args.candidate_count),
            )
        if (
            step % max(1, int(args.deterministic_every)) == 0
            or step == int(args.binary_steps)
        ):
            with torch.no_grad():
                hard = (logits[..., 1] >= logits[..., 0]).to(torch.float32)[None, None]
                hard_score = float(ncc(normalize_mean(proxy.forward(hard)), target)[0])
            remember_candidate(candidates, hard_score, hard, int(args.candidate_count))
    stage_times["binary_seconds"] = time.perf_counter() - started
    final_logits = logits.detach().clone()
    del logits, optimizer

    started = time.perf_counter()
    best_score = -1.0
    best_mask: torch.Tensor | None = None
    best_light: torch.Tensor | None = None
    with torch.no_grad():
        for _, mask_cpu in reversed(candidates):
            mask = mask_cpu.to(device=target.device, dtype=torch.float32)
            light = reference.forward_native(mask)
            score = float(ncc(light, target)[0])
            if score > best_score:
                best_score = score
                best_mask = mask.detach().clone()
                best_light = light.detach().clone()
    stage_times["reference_seconds"] = time.perf_counter() - started
    if best_mask is None or best_light is None:
        raise RuntimeError("No fixed binary candidate was produced")
    return best_score, best_mask, best_light, final_logits, stage_times


def direct_polish(
    logits: torch.Tensor,
    target: torch.Tensor,
    reference: DirectLinearASFPlan,
    best_score: float,
    best_mask: torch.Tensor,
    best_light: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[float, torch.Tensor, torch.Tensor, float]:
    started = time.perf_counter()
    logits = logits.detach().requires_grad_(True)
    optimizer = torch.optim.Adam([logits], lr=float(args.direct_polish_lr))
    steps = int(args.direct_polish_steps)
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        tau = exponential_value(0.5, 0.2, step, steps)
        sampled = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)[..., 1]
        sampled = sampled[None, None]
        light = reference.forward_native(sampled)
        score = ncc(light, target)[0]
        (1.0 - score).backward()
        optimizer.step()
        score_value = float(score.detach())
        if score_value > best_score:
            best_score = score_value
            best_mask = sampled.detach().clone()
            best_light = light.detach().clone()
        if step % 10 == 0 or step == steps:
            with torch.no_grad():
                hard = (logits[..., 1] >= logits[..., 0]).to(torch.float32)[None, None]
                hard_light = reference.forward_native(hard)
                hard_score = float(ncc(hard_light, target)[0])
            if hard_score > best_score:
                best_score = hard_score
                best_mask = hard.detach().clone()
                best_light = hard_light.detach().clone()
    return best_score, best_mask, best_light, time.perf_counter() - started


def optimize_sample(
    target: torch.Tensor,
    proxy: SmallFFTForwardPlan,
    reference: DirectLinearASFPlan,
    args: argparse.Namespace,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float], int, dict[str, float]]:
    target = normalize_mean(target)
    best_score = -1.0
    best_mask: torch.Tensor | None = None
    best_light: torch.Tensor | None = None
    best_logits: torch.Tensor | None = None
    best_times: dict[str, float] = {}
    attempts_used = 0

    for attempt in range(max(1, int(args.quality_attempts))):
        score, mask, light, logits, stage_times = optimize_attempt(
            target,
            proxy,
            reference,
            args,
            int(seed) + attempt * 1_000_003,
        )
        attempts_used = attempt + 1
        if score > best_score:
            best_score = score
            best_mask = mask
            best_light = light
            best_logits = logits
            best_times = stage_times
        if best_score >= float(args.min_ncc):
            break

    if (
        best_score < float(args.min_ncc)
        and int(args.direct_polish_steps) > 0
        and best_logits is not None
        and best_mask is not None
        and best_light is not None
    ):
        best_score, best_mask, best_light, polish_seconds = direct_polish(
            best_logits,
            target,
            reference,
            best_score,
            best_mask,
            best_light,
            args,
        )
        best_times["direct_polish_seconds"] = polish_seconds

    if best_mask is None or best_light is None:
        raise RuntimeError("Optimizer did not return a mask")
    quality_pass = best_score >= float(args.min_ncc)
    if not quality_pass and bool(args.strict_quality_gate):
        raise RuntimeError(
            f"Final fixed-mask NCC {best_score:.6f} is below required {float(args.min_ncc):.6f}"
        )

    with torch.no_grad():
        highpass = float(highpass_cosine_score_image(best_light, target)[0])
        gradient = float(gradient_cosine_score_image(best_light, target)[0])
    metric = {
        "score": float(best_score),
        "highpass_score": highpass,
        "gradient_score": gradient,
        "mask_mean": float(best_mask.mean()),
        "quality_pass": float(quality_pass),
    }
    return best_mask, best_light, metric, attempts_used, best_times


def write_failure(
    out_dir: Path,
    split: str,
    source_path: Path,
    sample_index: int,
    source: dict[str, np.ndarray],
    error: Exception,
) -> None:
    path = out_dir / "failures" / split / f"{source_path.stem}_sample{sample_index:03d}.json"
    atomic_json(
        path,
        {
            "source_shard": str(source_path.resolve()),
            "sample_index": int(sample_index),
            "sample_id": int(source["sample_id"][sample_index]),
            "pattern_type": str(source["pattern_type"][sample_index]),
            "error": f"{type(error).__name__}: {error}",
            "no_output_shard_committed": True,
        },
    )


def worker(args: argparse.Namespace, source_dir: Path, out_dir: Path, splits: list[str]) -> None:
    if int(args.world_size) < 1 or not 0 <= int(args.rank) < int(args.world_size):
        raise ValueError("rank must satisfy 0 <= rank < world_size")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if int(args.mask_resolution) != 512 or int(args.target_resolution) != 192:
        raise ValueError("This production generator is validated for 512x512 masks and 192x192 targets")

    all_native = native_source_shards(source_dir, splits)
    assigned = all_native[int(args.rank) :: int(args.world_size)]
    if int(args.max_shards) > 0:
        assigned = assigned[: int(args.max_shards)]
    if not assigned:
        print(f"[rank {args.rank}] no assigned native shards", flush=True)
        return

    first_source = load_source(assigned[0][1], 1)
    optics = first_source["optics"][0].astype(np.float64)
    proxy_optics = optics.copy()
    proxy_optics[3] = float(args.proxy_inter_num)
    proxy = SmallFFTForwardPlan(
        proxy_optics,
        mask_resolution=int(args.mask_resolution),
        target_resolution=int(args.target_resolution),
        device=device,
        forward_model="angular_spectrum_padded_fft",
    )
    reference = DirectLinearASFPlan(
        optics,
        mask_resolution=int(args.mask_resolution),
        target_resolution=int(args.target_resolution),
        inter_num=int(args.reference_inter_num),
        device=device,
    )

    quality_path = out_dir / "quality" / f"rank{int(args.rank):03d}.csv"
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not quality_path.exists() or quality_path.stat().st_size == 0
    handle = quality_path.open("a", newline="", encoding="utf-8")
    columns = (
        "split",
        "shard",
        "sample_index",
        "sample_id",
        "pattern_type",
        "ncc",
        "highpass_score",
        "gradient_score",
        "mask_mean",
        "attempts",
        "seconds",
        "peak_gpu_gib",
    )
    writer = csv.DictWriter(handle, fieldnames=columns)
    if write_header:
        writer.writeheader()
        handle.flush()

    rank_started = time.perf_counter()
    completed_shards = 0
    try:
        for split, source_path in assigned:
            source = load_source(source_path, int(args.max_samples_per_shard))
            count = int(source["target"].shape[0])
            out_path = out_dir / split / source_path.name
            rotated_source_path = source_path.with_name(source_path.stem + "_rot90_cw.npz")
            rotated_out_path = out_dir / split / rotated_source_path.name
            has_rotated = rotated_source_path.is_file()
            if is_valid_shard(out_path, count) and (
                not has_rotated or is_valid_shard(rotated_out_path, count)
            ):
                print(f"[rank {args.rank}] skip complete {split}/{source_path.name}", flush=True)
                completed_shards += 1
                continue

            if not np.allclose(source["optics"], optics[None], rtol=1e-6, atol=1e-12):
                raise ValueError(f"Optical parameters differ from the cached plans in {source_path}")
            partial_file = partial_path(out_dir, split, source_path)
            partial = load_partial(
                partial_file,
                count,
                int(args.mask_resolution),
                int(args.target_resolution),
            )
            if partial is None:
                masks = np.zeros(
                    (count, 1, int(args.mask_resolution), int(args.mask_resolution)),
                    dtype=np.uint8,
                )
                lights = np.zeros(
                    (count, 1, int(args.target_resolution), int(args.target_resolution)),
                    dtype=np.float16,
                )
                metrics = [empty_metric() for _ in range(count)]
                completed = np.zeros((count,), dtype=bool)
            else:
                masks, lights, metrics, completed = partial
                below_gate = completed & np.asarray(
                    [row["score"] < float(args.min_ncc) for row in metrics],
                    dtype=bool,
                )
                if np.any(below_gate):
                    completed[below_gate] = False
                    print(
                        f"[rank {args.rank}] retrying {int(below_gate.sum())} below-gate "
                        f"sample(s) in {split}/{source_path.name}",
                        flush=True,
                    )
                print(
                    f"[rank {args.rank}] resume partial {split}/{source_path.name} "
                    f"{int(completed.sum())}/{count} samples",
                    flush=True,
                )
            for sample_index in range(count):
                if completed[sample_index]:
                    continue
                ensure_free_space(out_dir, float(args.min_free_gib))
                sample_started = time.perf_counter()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                target = torch.from_numpy(
                    source["target"][sample_index : sample_index + 1].astype(np.float32)
                ).to(device)
                sample_seed = (
                    int(args.seed)
                    + int(source["sample_id"][sample_index]) * 104_729
                    + int(args.rank) * 1_000_000_007
                )
                try:
                    mask, light, metric, attempts, stage_times = optimize_sample(
                        target,
                        proxy,
                        reference,
                        args,
                        sample_seed,
                    )
                except Exception as error:
                    write_failure(out_dir, split, source_path, sample_index, source, error)
                    raise
                seconds = time.perf_counter() - sample_started
                peak_gpu_gib = (
                    torch.cuda.max_memory_allocated(device) / (1024**3)
                    if device.type == "cuda"
                    else 0.0
                )
                masks[sample_index] = mask.cpu().numpy().astype(np.uint8, copy=False)[0]
                lights[sample_index] = light.cpu().numpy().astype(np.float16, copy=False)[0]
                metrics[sample_index] = metric
                completed[sample_index] = True
                if (int(completed.sum()) % PARTIAL_SAVE_EVERY == 0) or sample_index == count - 1:
                    save_partial(partial_file, masks, lights, metrics, completed)
                writer.writerow(
                    {
                        "split": split,
                        "shard": source_path.name,
                        "sample_index": sample_index,
                        "sample_id": int(source["sample_id"][sample_index]),
                        "pattern_type": str(source["pattern_type"][sample_index]),
                        "ncc": f"{metric['score']:.9f}",
                        "highpass_score": f"{metric['highpass_score']:.9f}",
                        "gradient_score": f"{metric['gradient_score']:.9f}",
                        "mask_mean": f"{metric['mask_mean']:.9f}",
                        "attempts": attempts,
                        "seconds": f"{seconds:.3f}",
                        "peak_gpu_gib": f"{peak_gpu_gib:.3f}",
                    }
                )
                handle.flush()
                print(
                    f"[rank {args.rank} {split} {source_path.name} {sample_index + 1}/{count}] "
                    f"ncc={metric['score']:.6f} attempts={attempts} seconds={seconds:.2f} "
                    f"quality={'pass' if metric.get('quality_pass', 1.0) else 'below_gate'} "
                    f"peak_gpu={peak_gpu_gib:.2f}GiB stages={json.dumps(stage_times, separators=(',', ':'))}",
                    flush=True,
                )

            output = make_output_arrays(
                source,
                masks,
                lights,
                metrics,
            )
            ensure_free_space(out_dir, float(args.min_free_gib))
            atomic_npz(out_path, output)
            if has_rotated:
                rotated_source = load_source(rotated_source_path, int(args.max_samples_per_shard))
                atomic_npz(rotated_out_path, rotated_output(output, rotated_source))
            partial_file.unlink(missing_ok=True)
            completed_shards += 1
            elapsed = time.perf_counter() - rank_started
            print(
                f"[rank {args.rank}] committed {split}/{source_path.name} "
                f"native_shards={completed_shards}/{len(assigned)} elapsed={elapsed:.1f}s",
                flush=True,
            )
    finally:
        handle.close()


def finalize(args: argparse.Namespace, source_dir: Path, out_dir: Path, splits: list[str]) -> None:
    expected: list[tuple[str, str]] = []
    for split, native in native_source_shards(source_dir, splits):
        expected.append((split, native.name))
        rotated = native.with_name(native.stem + "_rot90_cw.npz")
        if rotated.is_file():
            expected.append((split, rotated.name))
    missing = [
        f"{split}/{name}"
        for split, name in expected
        if not is_valid_shard(out_dir / split / name)
    ]
    if missing and not bool(args.allow_incomplete_finalize):
        raise RuntimeError(f"Cannot finalize: {len(missing)} expected shards are missing or invalid")

    counts = {split: 0 for split in splits}
    score_sum = 0.0
    score_count = 0
    score_min = math.inf
    for split, name in expected:
        path = out_dir / split / name
        if not is_valid_shard(path):
            continue
        with np.load(path, allow_pickle=False) as shard:
            count = int(shard["target"].shape[0])
            values = shard["score"].astype(np.float64)
        counts[split] += count
        score_sum += float(values.sum())
        score_count += int(values.size)
        score_min = min(score_min, float(values.min()))

    source_metadata_path = source_dir / "metadata.json"
    source_metadata = (
        json.loads(source_metadata_path.read_text(encoding="utf-8"))
        if source_metadata_path.is_file()
        else {}
    )
    source_metadata.pop("complex_append_history", None)
    metadata = dict(source_metadata)
    metadata.update(
        {
            "dataset_type": "expert_mask_diffusion_dataset",
            "source_dataset_dir": str(source_dir.resolve()),
            "source_filter": "native shard_*.npz plus exact rot90 partners; complex8x excluded",
            "source_dataset_preserved": True,
            "mask_resolution": int(args.mask_resolution),
            "target_resolution": int(args.target_resolution),
            "split": counts,
            "attempted": counts,
            "target_types": [
                value
                for value in source_metadata.get("target_types", [])
                if str(value).startswith("real_")
            ],
            "expert_fields": list(REQUIRED_FIELDS[:4]),
            "forward_model": "linear_asf_reference_with_angular_spectrum_proxy",
            "expert_optimizer": {
                "generator_version": GENERATOR_VERSION,
                "method": "continuous proxy Adam then hard Gumbel-Softmax candidate search",
                "objective": "maximize continuous-light NCC",
                "acceptance": "fixed binary mask evaluated by final inter=10 finite-support linear ASF",
                "proxy_inter_num": int(args.proxy_inter_num),
                "reference_inter_num": int(args.reference_inter_num),
                "continuous_steps": int(args.continuous_steps),
                "binary_steps": int(args.binary_steps),
                "candidate_count": int(args.candidate_count),
                "quality_attempts": int(args.quality_attempts),
                "direct_polish_steps_on_failure": int(args.direct_polish_steps),
                "minimum_ncc": float(args.min_ncc),
                "binary_dice_used_for_acceptance": False,
            },
            "rotation_augmentation": {
                "suffix": "_rot90_cw",
                "method": "exact clockwise rotation of target, mask, and light field",
            },
            "distributed_generation": {
                "enabled": int(args.world_size) > 1,
                "world_size": int(args.world_size),
                "rank_assignment": "native_shard_index % world_size == rank",
                "one_worker_per_gpu_recommended": True,
            },
            "quality": {
                "complete": not missing,
                "samples": int(score_count),
                "ncc_mean": score_sum / max(1, score_count),
                "ncc_min": None if score_count == 0 else score_min,
                "minimum_required": float(args.min_ncc),
                "missing_shards": len(missing),
            },
            "created_by": Path(__file__).name,
        }
    )
    atomic_json(out_dir / "metadata.json", metadata)
    print(json.dumps(metadata["quality"], ensure_ascii=False), flush=True)


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = [value.strip() for value in str(args.splits).split(",") if value.strip()]
    if not splits:
        raise ValueError("No splits selected")
    ensure_config(out_dir, args)
    if bool(args.finalize_only):
        finalize(args, source_dir, out_dir, splits)
    else:
        ensure_free_space(out_dir, float(args.min_free_gib))
        worker(args, source_dir, out_dir, splits)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FATAL: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
