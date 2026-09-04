#!/usr/bin/env python3
"""Direct high-sampling linear-ASF binary mask optimization experiment.

This is an isolated research script.  It reads an existing validation shard and
writes only below the selected output directory.  The propagation and optimizer
follow ``asf_inverse_gpu/gradient_backward.py`` while adding reproducible fixed
binary-mask tracking and cross-model evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.fft import next_fast_len


LAB_DIR = Path(__file__).resolve().parent
PROJECT_DIR = LAB_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from 光学FFT前向传播_小分辨率 import (  # noqa: E402
    SmallFFTForwardPlan,
    cosine_score_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default=str(PROJECT_DIR / "real_circuit_special_expert_val/val/shard_00000.npz"),
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--inter-num", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--tau-start", type=float, default=1.0)
    parser.add_argument("--tau-end", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--deterministic-every", type=int, default=25)
    parser.add_argument("--target-ncc", type=float, default=0.95)
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument(
        "--objective-resolution",
        choices=("optical", "native"),
        default="optical",
        help=(
            "optical compares the high-sampling field to a repeated target, matching the "
            "expert demo; native first integrates the same field to the real 192x192 detector."
        ),
    )
    parser.add_argument(
        "--initialization",
        choices=("random", "target", "target_resized", "old_expert", "continuous"),
        default="random",
    )
    parser.add_argument(
        "--initial-probability",
        default="",
        help="NPZ containing a probability array, required for continuous initialization.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(LAB_DIR / "runs/direct_linear_gumbel_sample0"),
    )
    return parser.parse_args()


def normalize_mean(image: torch.Tensor) -> torch.Tensor:
    return image / image.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-12)


def ncc(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction_flat = prediction.reshape(prediction.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    numerator = (prediction_flat * target_flat).sum(dim=1)
    denominator = prediction_flat.square().sum(dim=1).sqrt()
    denominator = denominator * target_flat.square().sum(dim=1).sqrt()
    return numerator / denominator.clamp_min(1e-12)


class DirectLinearASFPlan:
    """Expert-compatible finite-support Rayleigh-Sommerfeld linear convolution."""

    def __init__(
        self,
        optics: np.ndarray,
        *,
        mask_resolution: int,
        target_resolution: int,
        inter_num: int,
        device: torch.device,
    ) -> None:
        self.device = device
        self.mask_resolution = int(mask_resolution)
        self.target_resolution = int(target_resolution)
        self.inter_num = int(inter_num)

        self.aperture = float(optics[0])
        self.wavelength = float(optics[1])
        self.distance = float(optics[2])
        self.pixel_pitch = self.aperture / self.mask_resolution
        self.dx = self.pixel_pitch / self.inter_num
        self.n_up = self.mask_resolution * self.inter_num
        self.roi = self.target_resolution * self.inter_num
        self.roi_start = (self.n_up - self.roi) // 2

        self.asf_n = (
            self.mask_resolution * 2 * self.inter_num
            - 1
            - ((self.mask_resolution - self.target_resolution) // 2 * self.inter_num)
        )
        self.asf_length = self.asf_n * self.dx
        fft_n = next_fast_len(self.n_up + self.asf_n - 1)
        self.fft_shape = (int(fft_n), int(fft_n))
        self.same_start = (self.asf_n - 1) // 2

        f_max = (self.asf_length / 2.0) / (
            self.wavelength
            * math.sqrt((self.asf_length / 2.0) ** 2 + self.distance**2)
        )
        nyquist_dx = 1.0 / (2.0 * f_max)
        if self.dx > nyquist_dx * (1.0 + 1e-6):
            raise RuntimeError(
                f"ASF phase is undersampled: dx={self.dx:.6e}, limit={nyquist_dx:.6e}"
            )

        started = time.perf_counter()
        x = torch.linspace(
            -self.asf_length / 2.0,
            self.asf_length / 2.0,
            self.asf_n,
            dtype=torch.float32,
            device=device,
        )
        y = torch.linspace(
            self.asf_length / 2.0,
            -self.asf_length / 2.0,
            self.asf_n,
            dtype=torch.float32,
            device=device,
        )
        radius_squared = y[:, None].square() + x[None, :].square()
        radius_squared.add_(self.distance**2)
        radius = radius_squared.sqrt()
        phase = radius.mul(2.0 * math.pi / self.wavelength).sub_(math.pi / 2.0)
        amplitude = radius_squared.reciprocal().mul_(self.distance / self.wavelength)
        asf = torch.polar(amplitude, phase)
        asf.div_(asf.abs().square().sum().sqrt().clamp_min(1e-12))
        self.asf_fft = torch.fft.fft2(asf, s=self.fft_shape)

        del x, y, radius_squared, radius, phase, amplitude, asf
        torch.cuda.empty_cache()
        self.metadata = {
            "mask_resolution": self.mask_resolution,
            "target_resolution": self.target_resolution,
            "inter_num": self.inter_num,
            "aperture_m": self.aperture,
            "wavelength_m": self.wavelength,
            "distance_m": self.distance,
            "pixel_pitch_m": self.pixel_pitch,
            "dx_m": self.dx,
            "n_up": self.n_up,
            "roi": self.roi,
            "asf_n": self.asf_n,
            "fft_shape": list(self.fft_shape),
            "kernel_build_seconds": time.perf_counter() - started,
        }

    def forward_optical(self, mask: torch.Tensor) -> torch.Tensor:
        if mask.ndim == 2:
            mask = mask[None, None]
        elif mask.ndim == 3:
            mask = mask[:, None]
        if tuple(mask.shape[-2:]) != (self.mask_resolution, self.mask_resolution):
            raise ValueError(f"Unexpected mask shape {tuple(mask.shape)}")

        mask_up = mask.repeat_interleave(self.inter_num, dim=-2)
        mask_up = mask_up.repeat_interleave(self.inter_num, dim=-1)
        mask_fft = torch.fft.fft2(mask_up.to(torch.complex64), s=self.fft_shape)
        field_full = torch.fft.ifft2(mask_fft * self.asf_fft, s=self.fft_shape)
        start = self.same_start + self.roi_start
        field_roi = field_full[..., start : start + self.roi, start : start + self.roi]
        intensity = field_roi.real.square() + field_roi.imag.square()
        return normalize_mean(intensity.to(torch.float32))

    def forward_native(self, mask: torch.Tensor) -> torch.Tensor:
        optical = self.forward_optical(mask)
        native = F.interpolate(
            optical,
            size=(self.target_resolution, self.target_resolution),
            mode="area",
        )
        return normalize_mean(native)


def centered_target_initialization(target: torch.Tensor, mask_resolution: int) -> torch.Tensor:
    binary = (target > 0.05).to(torch.float32)
    height, width = binary.shape[-2:]
    left = (mask_resolution - width) // 2
    right = mask_resolution - width - left
    top = (mask_resolution - height) // 2
    bottom = mask_resolution - height - top
    return F.pad(binary, (left, right, top, bottom), value=0.0)


def resized_target_initialization(target: torch.Tensor, mask_resolution: int) -> torch.Tensor:
    resized = F.interpolate(
        target,
        size=(mask_resolution, mask_resolution),
        mode="nearest",
    )
    return (resized > 0.05).to(torch.float32)


def logits_from_mask(mask: torch.Tensor, magnitude: float = 0.25) -> torch.Tensor:
    mask_hw = mask[0, 0]
    signed = (mask_hw * 2.0 - 1.0) * magnitude
    return torch.stack((-signed, signed), dim=-1)


@torch.no_grad()
def evaluate_fixed_mask(
    mask: torch.Tensor,
    target_native: torch.Tensor,
    target_optical: torch.Tensor,
    linear_plan: DirectLinearASFPlan,
    angular_plan: SmallFFTForwardPlan,
) -> dict[str, float]:
    linear_optical = linear_plan.forward_optical(mask)
    linear_native = F.interpolate(
        linear_optical,
        size=target_native.shape[-2:],
        mode="area",
    )
    linear_native = normalize_mean(linear_native)
    angular_native = normalize_mean(angular_plan.forward(mask))
    return {
        "linear_optical_ncc": float(ncc(linear_optical, target_optical)[0]),
        "linear_native_ncc": float(ncc(linear_native, target_native)[0]),
        "angular_native_ncc": float(cosine_score_image(angular_native, target_native)[0]),
        "mask_mean": float(mask.mean()),
    }


def atomic_save_npz(path: Path, **arrays: Any) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("This direct high-sampling experiment requires a CUDA GPU")

    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    with np.load(args.source, allow_pickle=False) as shard:
        index = int(args.sample_index)
        target_np = shard["target"][index : index + 1].astype(np.float32)
        optics = shard["optics"][index].astype(np.float64)
        old_expert_np = shard["expert_mask"][index : index + 1].astype(np.float32)
        sample_id = int(shard["sample_id"][index])
        pattern_type = str(shard["pattern_type"][index])

    target_native = normalize_mean(torch.from_numpy(target_np).to(device))
    target_optical = target_native.repeat_interleave(int(args.inter_num), dim=-2)
    target_optical = target_optical.repeat_interleave(int(args.inter_num), dim=-1)
    old_expert = torch.from_numpy(old_expert_np).to(device)

    print("building direct linear ASF plan", flush=True)
    linear_plan = DirectLinearASFPlan(
        optics,
        mask_resolution=512,
        target_resolution=192,
        inter_num=int(args.inter_num),
        device=device,
    )
    print(json.dumps(linear_plan.metadata, ensure_ascii=False), flush=True)
    angular_optics = optics.copy()
    angular_optics[3] = float(args.inter_num)
    angular_plan = SmallFFTForwardPlan(
        angular_optics,
        mask_resolution=512,
        target_resolution=192,
        device=device,
        forward_model="angular_spectrum_padded_fft",
    )

    if args.initialization == "random":
        logits = torch.randn(512, 512, 2, dtype=torch.float32, device=device)
    elif args.initialization == "target":
        initial_mask = centered_target_initialization(target_native, 512)
        logits = logits_from_mask(initial_mask)
    elif args.initialization == "target_resized":
        initial_mask = resized_target_initialization(target_native, 512)
        logits = logits_from_mask(initial_mask)
    elif args.initialization == "old_expert":
        logits = logits_from_mask(old_expert)
    else:
        if not args.initial_probability:
            raise ValueError("--initial-probability is required for continuous initialization")
        with np.load(args.initial_probability, allow_pickle=False) as packed:
            probability_np = packed["probability"].astype(np.float32)
        probability = torch.from_numpy(probability_np).to(device).clamp(1e-4, 1.0 - 1e-4)
        signed = 0.5 * torch.log(probability / (1.0 - probability))[0, 0]
        logits = torch.stack((-signed, signed), dim=-1)
    logits.requires_grad_(True)
    optimizer = torch.optim.Adam([logits], lr=float(args.lr))

    baseline_masks = {
        "centered_target": centered_target_initialization(target_native, 512),
        "resized_target": resized_target_initialization(target_native, 512),
        "old_expert": old_expert,
    }
    baselines: dict[str, dict[str, float]] = {}
    for label, mask in baseline_masks.items():
        baselines[label] = evaluate_fixed_mask(
            mask, target_native, target_optical, linear_plan, angular_plan
        )
        print(f"BASELINE {label} {baselines[label]}", flush=True)

    best_score = -math.inf
    best_step = -1
    best_mask_cpu: torch.Tensor | None = None
    best_kind = ""
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    decay = math.log(float(args.tau_start) / float(args.tau_end)) / max(1, int(args.steps))

    def objective_score(mask: torch.Tensor) -> torch.Tensor:
        optical_light = linear_plan.forward_optical(mask)
        if args.objective_resolution == "optical":
            return ncc(optical_light, target_optical)[0]
        native_light = F.interpolate(
            optical_light,
            size=target_native.shape[-2:],
            mode="area",
        )
        return ncc(normalize_mean(native_light), target_native)[0]

    for step in range(1, int(args.steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        tau = float(args.tau_start) * math.exp(-decay * (step - 1))
        sampled_mask = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)[..., 1]
        sampled_mask_bchw = sampled_mask[None, None]
        sampled_score = objective_score(sampled_mask_bchw)
        (1.0 - sampled_score).backward()
        optimizer.step()

        score_value = float(sampled_score.detach())
        if score_value > best_score:
            best_score = score_value
            best_step = step
            best_kind = "gumbel_argmax"
            best_mask_cpu = sampled_mask.detach().to("cpu", torch.uint8).clone()

        deterministic_value = math.nan
        should_check_deterministic = (
            step == 1
            or step == int(args.steps)
            or step % max(1, int(args.deterministic_every)) == 0
        )
        if should_check_deterministic:
            with torch.no_grad():
                deterministic_mask = (logits[..., 1] >= logits[..., 0]).to(torch.float32)
                deterministic_value = float(objective_score(deterministic_mask[None, None]))
            if deterministic_value > best_score:
                best_score = deterministic_value
                best_step = step
                best_kind = "deterministic_argmax"
                best_mask_cpu = deterministic_mask.to("cpu", torch.uint8).clone()

        elapsed = time.perf_counter() - started
        row = {
            "step": step,
            "tau": tau,
            "sampled_fixed_ncc": score_value,
            "deterministic_argmax_ncc": deterministic_value,
            "best_fixed_ncc": best_score,
            "best_step": best_step,
            "best_kind": best_kind,
            "elapsed_seconds": elapsed,
            "max_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        }
        rows.append(row)
        if step == 1 or step % max(1, int(args.log_every)) == 0 or step == int(args.steps):
            print(
                f"step={step:4d} tau={tau:.6f} sampled={score_value:.7f} "
                f"det={deterministic_value:.7f} best={best_score:.7f} "
                f"elapsed={elapsed:.1f}s mem={row['max_gpu_memory_gib']:.2f}GiB",
                flush=True,
            )
        if bool(args.early_stop) and best_score >= float(args.target_ncc):
            print(f"early stop: best fixed NCC reached {best_score:.7f}", flush=True)
            break

    if best_mask_cpu is None:
        raise RuntimeError("No binary mask candidate was recorded")
    best_mask = best_mask_cpu.to(device=device, dtype=torch.float32)[None, None]
    final_metrics = evaluate_fixed_mask(
        best_mask, target_native, target_optical, linear_plan, angular_plan
    )
    final_metrics.update(
        {
            "best_training_ncc": best_score,
            "best_step": best_step,
            "best_kind": best_kind,
            "total_seconds": time.perf_counter() - started,
            "steps_completed": len(rows),
            "objective_resolution": args.objective_resolution,
        }
    )

    with (out_dir / "convergence.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    atomic_save_npz(
        out_dir / "best_fixed_mask.npz",
        mask=best_mask_cpu.numpy()[None, None],
        target=target_np,
        optics=optics.astype(np.float32),
        sample_id=np.asarray(sample_id, dtype=np.int64),
        pattern_type=np.asarray(pattern_type),
    )
    report = {
        "source": str(Path(args.source).resolve()),
        "sample_index": int(args.sample_index),
        "sample_id": sample_id,
        "pattern_type": pattern_type,
        "arguments": vars(args),
        "plan": linear_plan.metadata,
        "baselines": baselines,
        "final": final_metrics,
    }
    with (out_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print("FINAL " + json.dumps(final_metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
