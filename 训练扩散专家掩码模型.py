#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
import time
from collections import OrderedDict
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

from 光学FFT前向传播_小分辨率 import (
    SmallFFTForwardPlan,
    centered_cosine_score_image,
    cosine_score_image,
    gradient_cosine_score_image,
    highpass_cosine_score_image,
    save_gray_png,
)
from 小分辨率轨迹数据集 import read_npz_array_header
from 扩散专家掩码模型 import ExpertMaskDiffusionUNet, build_model_from_config, model_config_from_args
from optical_visual_quality import (
    binary_printing_dice_score,
    binary_printing_topology_metrics,
    expert_optical_distillation_loss,
    optical_visual_quality_metrics,
    soft_binary_printing_loss,
    visual_quality_training_loss,
)


def init_distributed() -> tuple[bool, int, int, int]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    timeout_s = int(os.environ.get("DDP_TIMEOUT_SECONDS", "3600"))
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo", timeout=timedelta(seconds=timeout_s))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank


def cleanup_distributed(enabled: bool) -> None:
    if enabled and dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return int(rank) == 0


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def move_batch(batch: dict[str, Any], device: torch.device, *, non_blocking: bool = True) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=non_blocking) if torch.is_tensor(value) else value
    return out


def parse_amp_dtype(value: str) -> torch.dtype:
    text = str(value).lower()
    if text in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if text in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError(f"Unsupported amp dtype: {value}")


def make_grad_scaler(enabled: bool) -> Any:
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def load_metadata(dataset_dir: str | Path) -> dict[str, Any]:
    with open(Path(dataset_dir) / "metadata.json", "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_forward_model(requested: str, metadata: dict[str, Any]) -> str:
    value = str(requested).strip().lower()
    if value in {"", "auto"}:
        value = str(metadata.get("forward_model", "spherical_circular_fft")).strip().lower()
    aliases = {
        "spherical": "spherical_circular_fft",
        "angular_spectrum": "angular_spectrum_padded_fft",
        "asm": "angular_spectrum_padded_fft",
        "asf_linear": "asf_linear_convolution_fft",
    }
    value = aliases.get(value, value)
    allowed = {
        "spherical_circular_fft",
        "angular_spectrum_padded_fft",
        "asf_linear_convolution_fft",
    }
    if value not in allowed:
        raise ValueError(f"Unsupported forward model: {value!r}; expected one of {sorted(allowed)}")
    return value


class ExpertMaskDataset(Dataset):
    def __init__(self, dataset_dir: str | Path, split: str, *, metadata: dict[str, Any] | None = None, cache_size: int = 4) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.split = str(split)
        self.metadata = metadata if metadata is not None else load_metadata(self.dataset_dir)
        self.target_resolution = int(self.metadata.get("target_resolution", 1024))
        self.mask_resolution = int(self.metadata.get("mask_resolution", 512))
        self.cache_size = max(1, int(cache_size))
        self.files = sorted((self.dataset_dir / self.split).glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No expert shards under {self.dataset_dir / self.split}")
        self.refs: list[tuple[int, int]] = []
        for shard_idx, path in enumerate(self.files):
            target_shape, _ = read_npz_array_header(path, "target")
            mask_shape, _ = read_npz_array_header(path, "expert_mask")
            if tuple(target_shape[-2:]) != (self.target_resolution, self.target_resolution):
                raise ValueError(f"{path} target shape {target_shape} mismatches metadata target_resolution")
            if tuple(mask_shape[-2:]) != (self.mask_resolution, self.mask_resolution):
                raise ValueError(f"{path} expert_mask shape {mask_shape} mismatches metadata mask_resolution")
            for idx in range(int(target_shape[0])):
                self.refs.append((shard_idx, idx))
        self.complex_indices = [
            index for index, (shard_idx, _sample_idx) in enumerate(self.refs)
            if self.files[shard_idx].name.startswith("complex8x_")
        ]
        complex_set = set(self.complex_indices)
        self.simple_indices = [index for index in range(len(self.refs)) if index not in complex_set]
        self._cache: OrderedDict[int, dict[str, Any]] = OrderedDict()

    def _load_shard(self, shard_idx: int) -> dict[str, Any]:
        cached = self._cache.get(shard_idx)
        if cached is not None:
            self._cache.move_to_end(shard_idx)
            return cached
        path = self.files[shard_idx]
        with np.load(path) as shard:
            loaded = {
                "target": shard["target"].astype(np.float32),
                "optics": shard["optics"].astype(np.float32),
                "expert_mask": shard["expert_mask"].astype(np.float32),
                "expert_pred": shard["expert_pred"].astype(np.float32) if "expert_pred" in shard.files else None,
                "sample_id": shard["sample_id"].astype(np.int64) if "sample_id" in shard.files else np.arange(shard["target"].shape[0]),
                "pattern_type": shard["pattern_type"] if "pattern_type" in shard.files else np.asarray(["unknown"] * shard["target"].shape[0]),
                "score": shard["score"].astype(np.float32) if "score" in shard.files else np.zeros(shard["target"].shape[0], dtype=np.float32),
            }
        self._cache[shard_idx] = loaded
        self._cache.move_to_end(shard_idx)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return loaded

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        shard_idx, sample_idx = self.refs[int(index)]
        shard = self._load_shard(shard_idx)
        row = {
            "target": torch.from_numpy(shard["target"][sample_idx]).to(torch.float32),
            "optics": torch.from_numpy(shard["optics"][sample_idx]).to(torch.float32),
            "expert_mask": torch.from_numpy(shard["expert_mask"][sample_idx]).to(torch.float32),
            "sample_id": int(shard["sample_id"][sample_idx]),
            "pattern_type": str(shard["pattern_type"][sample_idx]),
            "score": float(shard["score"][sample_idx]),
        }
        if shard["expert_pred"] is not None:
            row["expert_pred"] = torch.from_numpy(shard["expert_pred"][sample_idx]).to(torch.float32)
        return row


class BalancedReplaySampler(Sampler[int]):
    """Sample a fixed mix while replaying the smaller original set.

    The appended complex set is much larger than the original set. Sampling
    with replacement keeps both distributions visible to the optimizer and
    prevents the original simple patterns from disappearing from training.
    """

    def __init__(
        self,
        dataset: ExpertMaskDataset,
        *,
        simple_ratio: float,
        samples_per_epoch: int,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 1234,
    ) -> None:
        if num_replicas < 1 or not 0 <= rank < num_replicas:
            raise ValueError("invalid sampler rank/replica configuration")
        self.dataset = dataset
        self.simple_ratio = float(np.clip(simple_ratio, 0.0, 1.0))
        self.samples_per_epoch = max(1, int(samples_per_epoch) if int(samples_per_epoch) > 0 else len(dataset))
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = int(math.ceil(self.samples_per_epoch / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        simple_pool = torch.as_tensor(self.dataset.simple_indices, dtype=torch.long)
        complex_pool = torch.as_tensor(self.dataset.complex_indices, dtype=torch.long)
        if simple_pool.numel() == 0:
            simple_count = 0
        elif complex_pool.numel() == 0:
            simple_count = self.samples_per_epoch
        else:
            simple_count = int(round(self.samples_per_epoch * self.simple_ratio))
        complex_count = self.samples_per_epoch - simple_count

        def draw(pool: torch.Tensor, count: int) -> torch.Tensor:
            if count <= 0:
                return torch.empty(0, dtype=torch.long)
            if pool.numel() == 0:
                return torch.empty(0, dtype=torch.long)
            picks = torch.randint(pool.numel(), (count,), generator=generator)
            return pool[picks]

        indices = torch.cat((draw(simple_pool, simple_count), draw(complex_pool, complex_count)))
        if indices.numel() < self.samples_per_epoch:
            fallback = simple_pool if simple_pool.numel() else complex_pool
            indices = torch.cat((indices, draw(fallback, self.samples_per_epoch - indices.numel())))
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
        if indices.numel() < self.total_size:
            repeats = (self.total_size + indices.numel() - 1) // indices.numel()
            indices = indices.repeat(repeats)[: self.total_size]
        indices = indices.tolist()
        return iter(indices[self.rank : self.total_size : self.num_replicas])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


def ensure_bchw(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    if x.dim() == 2:
        x = x[None, None]
    elif x.dim() == 3:
        x = x[None] if x.shape[0] == 1 else x[:, None]
    return x


def normalize_target(x: torch.Tensor) -> torch.Tensor:
    x = ensure_bchw(x).clamp_min(0.0)
    return x / x.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6)


def override_optics_inter_num(optics: torch.Tensor, inter_num: float) -> torch.Tensor:
    if float(inter_num) <= 0:
        return optics
    optics = optics.clone()
    if optics.dim() == 1:
        optics = optics[None]
    optics[..., 3] = float(inter_num)
    return optics


def highpass_image(image: torch.Tensor, kernel_size: int = 9) -> torch.Tensor:
    k = max(3, int(kernel_size))
    if k % 2 == 0:
        k += 1
    image = image.to(torch.float32)
    return image - F.avg_pool2d(image, k, stride=1, padding=k // 2)


def gradient_magnitude(image: torch.Tensor) -> torch.Tensor:
    image = image.to(torch.float32)
    gy = F.pad((image[..., 1:, :] - image[..., :-1, :]).abs(), (0, 0, 0, 1))
    gx = F.pad((image[..., :, 1:] - image[..., :, :-1]).abs(), (0, 1, 0, 0))
    return 0.5 * (gy + gx)


def sigma_to_time(sigma: torch.Tensor, *, sigma_min: float, sigma_max: float) -> torch.Tensor:
    log_min = math.log(max(float(sigma_min), 1e-8))
    log_max = math.log(max(float(sigma_max), 1e-8))
    sigma = sigma.to(torch.float32).clamp_min(1e-8)
    return ((sigma.log() - log_min) / max(log_max - log_min, 1e-8)).clamp(0.0, 1.0)


def sample_sigma(batch: int, *, sigma_min: float, sigma_max: float, rho: float, device: torch.device) -> torch.Tensor:
    u = torch.rand(batch, device=device)
    min_inv = float(sigma_min) ** (1.0 / float(rho))
    max_inv = float(sigma_max) ** (1.0 / float(rho))
    return (max_inv + u * (min_inv - max_inv)).pow(float(rho))


def sigma_schedule(steps: int, *, sigma_min: float, sigma_max: float, rho: float, device: torch.device) -> torch.Tensor:
    ramp = torch.linspace(0.0, 1.0, max(1, int(steps)), device=device)
    min_inv = float(sigma_min) ** (1.0 / float(rho))
    max_inv = float(sigma_max) ** (1.0 / float(rho))
    return (max_inv + ramp * (min_inv - max_inv)).pow(float(rho))


def expert_mask_to_logits(mask: torch.Tensor, logit_clip: float) -> torch.Tensor:
    mask = ensure_bchw(mask).clamp(0.0, 1.0)
    lo = torch.full_like(mask, -float(logit_clip))
    hi = torch.full_like(mask, float(logit_clip))
    return torch.where(mask >= 0.5, hi, lo)


def target_topk_initial_logits(
    target: torch.Tensor,
    *,
    mask_resolution: int,
    target_mask_mean: float,
    logit_clip: float,
) -> torch.Tensor:
    target_small = F.interpolate(normalize_target(target), size=(int(mask_resolution), int(mask_resolution)), mode="area")
    flat = target_small.flatten(1)
    kth = max(1, min(flat.shape[1], int(round(float(target_mask_mean) * flat.shape[1]))))
    top_idx = torch.topk(flat, kth, dim=1).indices
    mask_flat = torch.zeros_like(flat)
    mask_flat.scatter_(1, top_idx, 1.0)
    mask = mask_flat.view_as(target_small)
    return expert_mask_to_logits(mask, float(logit_clip))


def coordinate_maps(batch: int, size: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    axis = torch.linspace(-1.0, 1.0, int(size), device=device, dtype=dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return yy[None, None].expand(batch, 1, -1, -1), xx[None, None].expand(batch, 1, -1, -1)


def build_model_input(
    noisy_logits: torch.Tensor,
    target: torch.Tensor,
    sigma_t: torch.Tensor,
    *,
    mask_resolution: int,
    logit_clip: float,
) -> torch.Tensor:
    noisy_logits = ensure_bchw(noisy_logits)
    batch = noisy_logits.shape[0]
    target_n = normalize_target(target).to(device=noisy_logits.device, dtype=noisy_logits.dtype)
    target_down = F.interpolate(target_n, size=(int(mask_resolution), int(mask_resolution)), mode="area")
    target_hp = F.interpolate(highpass_image(target_n), size=(int(mask_resolution), int(mask_resolution)), mode="area")
    target_edge = F.interpolate(gradient_magnitude(target_n), size=(int(mask_resolution), int(mask_resolution)), mode="area")
    target_hp = target_hp / target_hp.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    target_edge = target_edge / target_edge.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    sigma_map = sigma_t.reshape(batch, 1, 1, 1).expand(batch, 1, int(mask_resolution), int(mask_resolution))
    yy, xx = coordinate_maps(batch, int(mask_resolution), noisy_logits.device, noisy_logits.dtype)
    input_scale = max(float(logit_clip), 1e-6)
    noisy_scaled = (noisy_logits / input_scale).clamp(-2.0, 2.0)
    if input_scale <= 2.0:
        noisy_prob = 0.5 * (noisy_logits.clamp(-1.0, 1.0) + 1.0)
    else:
        noisy_prob = torch.sigmoid(noisy_logits)
    return torch.cat([noisy_scaled, noisy_prob, target_down, target_hp, target_edge, sigma_map, yy, xx], dim=1)


def binary_st(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    prob = torch.sigmoid(logits / max(float(temperature), 1e-4))
    hard = (prob >= 0.5).to(prob.dtype)
    return hard.detach() - prob.detach() + prob


class FFTPlanCache:
    def __init__(
        self,
        *,
        mask_resolution: int,
        target_resolution: int,
        device: torch.device,
        max_size: int = 4,
        forward_model: str = "spherical_circular_fft",
        inter_num_override: float = 0.0,
        allow_undersampling: bool = False,
    ) -> None:
        self.mask_resolution = int(mask_resolution)
        self.target_resolution = int(target_resolution)
        self.device = device
        self.max_size = max(1, int(max_size))
        self.forward_model = str(forward_model)
        self.inter_num_override = float(inter_num_override)
        self.allow_undersampling = bool(allow_undersampling)
        self._cache: OrderedDict[tuple[float, float, float, float], SmallFFTForwardPlan] = OrderedDict()

    @staticmethod
    def _key(optics: torch.Tensor) -> tuple[float, float, float, float]:
        arr = optics.detach().cpu().to(torch.float64).reshape(-1)
        return tuple(float(v.item()) for v in arr[:4])

    def get(self, optics: torch.Tensor) -> SmallFFTForwardPlan:
        values = list(self._key(optics))
        if self.inter_num_override > 0:
            values[3] = self.inter_num_override
        key = tuple(values)
        plan = self._cache.get(key)
        if plan is not None:
            self._cache.move_to_end(key)
            return plan
        plan = SmallFFTForwardPlan(
            list(key),
            mask_resolution=self.mask_resolution,
            target_resolution=self.target_resolution,
            device=self.device,
            forward_model=self.forward_model,
            allow_undersampling=self.allow_undersampling,
        )
        self._cache[key] = plan
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)
        return plan

    def forward(self, mask: torch.Tensor, optics: torch.Tensor) -> torch.Tensor:
        outs = []
        for idx in range(mask.shape[0]):
            outs.append(self.get(optics[idx]).forward(mask[idx:idx + 1]))
        return torch.cat(outs, dim=0)


def physical_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    cosine_weight: float,
    centered_cosine_weight: float,
    l1_weight: float,
    highpass_cosine_weight: float,
    highpass_l1_weight: float,
    gradient_cosine_weight: float,
) -> torch.Tensor:
    """Field-first loss with separately controlled global and edge terms."""
    pred_n = normalize_target(pred)
    target_n = normalize_target(target)
    l1 = (pred_n - target_n).abs().mean(dim=(-3, -2, -1))
    hp_l1 = (highpass_image(pred_n) - highpass_image(target_n)).abs().mean(dim=(-3, -2, -1))
    return (
        float(cosine_weight) * (1.0 - cosine_score_image(pred_n, target_n))
        + float(centered_cosine_weight) * (1.0 - centered_cosine_score_image(pred_n, target_n))
        + float(l1_weight) * l1
        + float(highpass_cosine_weight) * (1.0 - highpass_cosine_score_image(pred_n, target_n))
        + float(highpass_l1_weight) * hp_l1
        + float(gradient_cosine_weight) * (1.0 - gradient_cosine_score_image(pred_n, target_n))
    )


def masked_mean_per_sample(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    value = ensure_bchw(value)
    mask = ensure_bchw(mask).to(device=value.device, dtype=value.dtype)
    denom = mask.sum(dim=(-3, -2, -1)).clamp_min(1.0)
    return (value * mask).sum(dim=(-3, -2, -1)) / denom


def bottom_band_bce_loss(
    pred_prob: torch.Tensor,
    expert_mask: torch.Tensor,
    target: torch.Tensor,
    *,
    mask_resolution: int,
    bottom_band_px: int,
) -> torch.Tensor:
    """Supervise the target-relative bottom foreground without changing full-image BCE.

    The band is located from each target's foreground bounding box, so translated
    and scaled pipeline targets do not all use the image's last rows.
    """
    target_bin = (target > 0.05).to(torch.float32)
    target_rows = target_bin.amax(dim=-1).squeeze(1) > 0
    target_height = int(target.shape[-2])
    row_index = torch.arange(target_height, device=target.device).view(1, -1)
    y_max = torch.where(target_rows, row_index, torch.zeros_like(row_index)).amax(dim=-1)
    has_foreground = target_rows.any(dim=-1)
    band_start = (y_max - max(1, int(bottom_band_px)) + 1).clamp_min(0)
    bottom_rows = row_index >= band_start[:, None]
    bottom_rows = bottom_rows & has_foreground[:, None]
    bottom_region = bottom_rows[:, None, :, None].to(pred_prob.dtype)
    bottom_region = F.interpolate(
        bottom_region,
        size=(int(mask_resolution), int(mask_resolution)),
        mode="nearest",
    )

    pred_prob = pred_prob.clamp(1e-4, 1.0 - 1e-4)
    pixel_bce = F.binary_cross_entropy_with_logits(
        torch.logit(pred_prob),
        expert_mask,
        reduction="none",
    )
    positive = bottom_region * expert_mask
    negative = bottom_region * (1.0 - expert_mask)
    positive_loss = (pixel_bce * positive).sum(dim=(-3, -2, -1)) / positive.sum(dim=(-3, -2, -1)).clamp_min(1.0)
    negative_loss = (pixel_bce * negative).sum(dim=(-3, -2, -1)) / negative.sum(dim=(-3, -2, -1)).clamp_min(1.0)
    # Positive pixels receive the main signal; negative pixels still prevent
    # filling the whole bottom band and creating new optical background light.
    return (positive_loss + 0.5 * negative_loss).mean()


def bottom_optical_masks(
    target: torch.Tensor,
    *,
    bottom_band_px: int,
    background_band_px: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return target-bottom foreground and below-bottom background masks.

    Rows with unusually high foreground occupancy are treated as a horizontal
    bridge. If no such bridge exists, the last foreground rows are used so the
    loss remains valid for targets made only of vertical lines.
    """
    target_bin = (target > 0.05).to(torch.float32)
    row_occupancy = target_bin.mean(dim=-1)
    row_has_foreground = row_occupancy > 0
    height = int(target.shape[-2])
    row_index = torch.arange(height, device=target.device).view(1, 1, -1)
    y_max = torch.where(
        row_has_foreground,
        row_index,
        torch.zeros_like(row_index),
    ).amax(dim=-1, keepdim=True)
    has_foreground = row_has_foreground.any(dim=-1, keepdim=True)
    near_bottom = row_index >= (y_max - max(1, int(bottom_band_px)) + 1)
    max_occupancy = row_occupancy.amax(dim=-1, keepdim=True)
    horizontal_threshold = torch.maximum(
        0.5 * max_occupancy,
        target_bin.new_tensor(0.25),
    )
    horizontal_rows = near_bottom & (row_occupancy >= horizontal_threshold)
    has_horizontal = horizontal_rows.any(dim=-1, keepdim=True)
    selected_rows = torch.where(
        has_horizontal,
        horizontal_rows,
        near_bottom & row_has_foreground,
    )

    foreground_mask = target_bin * selected_rows.unsqueeze(-1).to(target_bin.dtype)
    below_bottom = (
        (row_index > y_max)
        & (row_index <= y_max + max(1, int(background_band_px)))
        & has_foreground
    )
    background_mask = (1.0 - target_bin) * below_bottom.unsqueeze(-1).to(target_bin.dtype)
    return foreground_mask, background_mask


def bottom_endpoint_optical_loss(
    pred_n: torch.Tensor,
    target_n: torch.Tensor,
    endpoint_mask: torch.Tensor,
) -> torch.Tensor:
    """Match the optical energy of target columns at their lower endpoints.

    This handles targets whose bottom consists of separate vertical terminals
    rather than one horizontal bridge. The column profile is averaged only
    over the selected bottom rows and only scored for columns containing target
    foreground, so empty bottom columns do not create a false horizontal line.
    """
    endpoint_rows = endpoint_mask.amax(dim=-1)
    endpoint_cols = endpoint_mask.amax(dim=-2)
    row_count = endpoint_rows.sum(dim=-1, keepdim=True).clamp_min(1.0)
    pred_col = (pred_n * endpoint_rows.unsqueeze(-1)).sum(dim=-2) / row_count
    target_col = (target_n * endpoint_rows.unsqueeze(-1)).sum(dim=-2) / row_count
    col_error = (pred_col - target_col).abs().unsqueeze(-2)
    col_weight = endpoint_cols.unsqueeze(-2)
    return masked_mean_per_sample(col_error, col_weight)


def bottom_continuity_optical_loss(
    pred_n: torch.Tensor,
    target_n: torch.Tensor,
    endpoint_mask: torch.Tensor,
) -> torch.Tensor:
    """Penalize dark gaps between adjacent target bottom-line columns.

    The endpoint loss matches each column independently and can therefore
    still tolerate an isolated dark column. This loss compares the minimum
    intensity of every adjacent target-supported column pair, so one dark
    column creates a direct penalty even when the average bottom intensity is
    otherwise correct.
    """
    endpoint_rows = endpoint_mask.amax(dim=-1)
    endpoint_cols = endpoint_mask.amax(dim=-2)
    row_count = endpoint_rows.sum(dim=-1, keepdim=True).clamp_min(1.0)
    pred_col = (pred_n * endpoint_rows.unsqueeze(-1)).sum(dim=-2) / row_count
    target_col = (target_n * endpoint_rows.unsqueeze(-1)).sum(dim=-2) / row_count

    target_pair = torch.minimum(target_col[..., 1:], target_col[..., :-1])
    pred_pair = torch.minimum(pred_col[..., 1:], pred_col[..., :-1])
    pair_mask = torch.minimum(endpoint_cols[..., 1:], endpoint_cols[..., :-1])
    gap = F.relu(target_pair - pred_pair).square().unsqueeze(-2)
    return masked_mean_per_sample(gap, pair_mask.unsqueeze(-2))


def full_continuity_optical_loss(
    pred_n: torch.Tensor,
    target_n: torch.Tensor,
) -> torch.Tensor:
    """Penalize dark breaks along target-supported horizontal and vertical runs.

    Only adjacent pixels that are both foreground in the target are scored.
    Therefore an intentional target gap is not treated as a missing connection.
    The per-sample normalization keeps dense patterns from dominating the batch.
    """
    target_bin = (target_n > 0.05).to(pred_n.dtype)

    target_h = torch.minimum(target_n[..., :, 1:], target_n[..., :, :-1])
    pred_h = torch.minimum(pred_n[..., :, 1:], pred_n[..., :, :-1])
    mask_h = target_bin[..., :, 1:] * target_bin[..., :, :-1]

    target_v = torch.minimum(target_n[..., 1:, :], target_n[..., :-1, :])
    pred_v = torch.minimum(pred_n[..., 1:, :], pred_n[..., :-1, :])
    mask_v = target_bin[..., 1:, :] * target_bin[..., :-1, :]

    gap_h = F.relu(target_h - pred_h).square() * mask_h
    gap_v = F.relu(target_v - pred_v).square() * mask_v
    numerator = gap_h.sum(dim=(-3, -2, -1)) + gap_v.sum(dim=(-3, -2, -1))
    denominator = mask_h.sum(dim=(-3, -2, -1)) + mask_v.sum(dim=(-3, -2, -1))
    return numerator / denominator.clamp_min(1.0)


def visual_physical_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    background_weight: float,
    foreground_weight: float,
    row_projection_weight: float,
    col_projection_weight: float,
    bottom_foreground_weight: float,
    bottom_background_weight: float,
    bottom_endpoint_weight: float,
    bottom_continuity_weight: float,
    full_continuity_weight: float,
    visual_foreground_weight: float,
    visual_background_weight: float,
    visual_background_tail_weight: float,
    visual_speckle_weight: float,
    visual_uniformity_weight: float,
    visual_energy_weight: float,
    visual_bottleneck_weight: float,
    visual_background_tail_fraction: float,
    visual_softmin_temperature: float,
    visual_bottleneck_focal_gain: float,
    bottom_band_px: int,
    bottom_background_band_px: int,
    include_visual_quality: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pred_n = normalize_target(pred)
    target_n = normalize_target(target)
    target_bin = (target_n > 0.05).to(pred_n.dtype)
    background = 1.0 - target_bin

    background_leak = masked_mean_per_sample(pred_n.square(), background)
    foreground_l1 = masked_mean_per_sample((pred_n - target_n).abs(), target_bin)
    pred_row = pred_n.mean(dim=-1)
    target_row = target_n.mean(dim=-1)
    row_weight = (target_bin.mean(dim=-1) > 0).to(pred_n.dtype)
    row_projection = ((pred_row - target_row).abs() * row_weight).sum(dim=(-2, -1)) / row_weight.sum(dim=(-2, -1)).clamp_min(1.0)

    pred_col = pred_n.mean(dim=-2)
    target_col = target_n.mean(dim=-2)
    col_weight = (target_bin.mean(dim=-2) > 0).to(pred_n.dtype)
    col_projection = ((pred_col - target_col).abs() * col_weight).sum(dim=(-2, -1)) / col_weight.sum(dim=(-2, -1)).clamp_min(1.0)
    bottom_foreground_mask, bottom_background_mask = bottom_optical_masks(
        target_n,
        bottom_band_px=int(bottom_band_px),
        background_band_px=int(bottom_background_band_px),
    )
    bottom_foreground = masked_mean_per_sample(
        (pred_n - target_n).abs(),
        bottom_foreground_mask,
    )
    bottom_background = masked_mean_per_sample(
        pred_n.square(),
        bottom_background_mask,
    )
    bottom_endpoint = bottom_endpoint_optical_loss(
        pred_n,
        target_n,
        bottom_foreground_mask,
    )
    bottom_continuity = bottom_continuity_optical_loss(
        pred_n,
        target_n,
        bottom_foreground_mask,
    )
    full_continuity = full_continuity_optical_loss(pred_n, target_n)
    visual_quality, visual_quality_metrics = visual_quality_training_loss(
        pred_n,
        target_n,
        foreground_weight=float(visual_foreground_weight),
        background_weight=float(visual_background_weight),
        background_tail_weight=float(visual_background_tail_weight),
        speckle_weight=float(visual_speckle_weight),
        uniformity_weight=float(visual_uniformity_weight),
        energy_weight=float(visual_energy_weight),
        bottleneck_weight=float(visual_bottleneck_weight),
        background_tail_fraction=float(visual_background_tail_fraction),
        softmin_temperature=float(visual_softmin_temperature),
        bottleneck_focal_gain=float(visual_bottleneck_focal_gain),
    )

    total = (
        float(background_weight) * background_leak
        + float(foreground_weight) * foreground_l1
        + float(row_projection_weight) * row_projection
        + float(col_projection_weight) * col_projection
        + float(bottom_foreground_weight) * bottom_foreground
        + float(bottom_background_weight) * bottom_background
        + float(bottom_endpoint_weight) * bottom_endpoint
        + float(bottom_continuity_weight) * bottom_continuity
        + float(full_continuity_weight) * full_continuity
        + (visual_quality if bool(include_visual_quality) else 0.0)
    )
    metrics = {
        "background_loss": background_leak,
        "foreground_loss": foreground_l1,
        "row_projection_loss": row_projection,
        "col_projection_loss": col_projection,
        "bottom_foreground_loss": bottom_foreground,
        "bottom_background_loss": bottom_background,
        "bottom_endpoint_loss": bottom_endpoint,
        "bottom_continuity_loss": bottom_continuity,
        "full_continuity_loss": full_continuity,
    }
    metrics.update(visual_quality_metrics)
    metrics["visual_target_loss"] = visual_quality
    return total, metrics


def diffusion_alpha_bars(num_steps: int, *, beta_start: float, beta_end: float, device: torch.device) -> torch.Tensor:
    betas = torch.linspace(float(beta_start), float(beta_end), max(2, int(num_steps)), device=device).clamp(1e-6, 0.999)
    return torch.cumprod(1.0 - betas, dim=0)


def effective_t_range(train_stage: str, t_min: float, t_max: float) -> tuple[float, float]:
    stage = str(train_stage)
    if stage == "direct":
        return 0.0, 0.0
    lo = max(0.0, min(float(t_min), 1.0))
    if float(t_max) >= 0:
        hi = max(0.0, min(float(t_max), 1.0))
    elif stage == "low-noise":
        hi = 0.10
    else:
        hi = 1.0
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi


def sample_t_indices(
    batch_size: int,
    total_steps: int,
    *,
    train_stage: str,
    t_min: float,
    t_max: float,
    device: torch.device,
) -> torch.Tensor:
    if str(train_stage) == "direct":
        return torch.zeros(int(batch_size), device=device, dtype=torch.long)
    lo, hi = effective_t_range(str(train_stage), float(t_min), float(t_max))
    max_idx = max(0, int(total_steps) - 1)
    lo_idx = max(0, min(max_idx, int(round(lo * max_idx))))
    hi_idx = max(0, min(max_idx, int(round(hi * max_idx))))
    if hi_idx < lo_idx:
        lo_idx, hi_idx = hi_idx, lo_idx
    return torch.randint(lo_idx, hi_idx + 1, (int(batch_size),), device=device, dtype=torch.long)


def compute_batch_loss(
    model: torch.nn.Module,
    batch: dict[str, Any],
    fft_cache: FFTPlanCache,
    *,
    mask_resolution: int,
    sigma_min: float,
    sigma_max: float,
    sigma_rho: float,
    logit_clip: float,
    mask_temperature: float,
    denoise_weight: float,
    bce_weight: float,
    physical_weight: float,
    physical_cosine_weight: float,
    physical_centered_cosine_weight: float,
    physical_l1_weight: float,
    physical_highpass_cosine_weight: float,
    physical_highpass_l1_weight: float,
    physical_gradient_cosine_weight: float,
    cosine_score_loss_weight: float,
    background_weight: float,
    foreground_weight: float,
    row_projection_weight: float,
    col_projection_weight: float,
    bottom_supervision_weight: float,
    bottom_band_px: int,
    bottom_foreground_weight: float,
    bottom_background_weight: float,
    bottom_endpoint_weight: float,
    bottom_continuity_weight: float,
    full_continuity_weight: float,
    visual_foreground_weight: float,
    visual_background_weight: float,
    visual_background_tail_weight: float,
    visual_speckle_weight: float,
    visual_uniformity_weight: float,
    visual_energy_weight: float,
    visual_bottleneck_weight: float,
    visual_background_tail_fraction: float,
    visual_softmin_temperature: float,
    visual_bottleneck_focal_gain: float,
    visual_loss_weight: float,
    expert_foreground_weight: float,
    expert_background_weight: float,
    expert_guard_weight: float,
    expert_guard_tail_weight: float,
    expert_floor_weight: float,
    guard_band_px: int,
    expert_tail_fraction: float,
    binary_print_loss_weight: float,
    binary_print_threshold: float,
    binary_print_temperature: float,
    binary_print_dice_weight: float,
    binary_print_false_positive_weight: float,
    binary_print_false_negative_weight: float,
    binary_print_background_tail_weight: float,
    binary_print_background_tail_fraction: float,
    binary_print_worst_sample_weight: float,
    binary_print_worst_sample_fraction: float,
    binary_print_pair_continuity_weight: float,
    binary_print_window_continuity_weight: float,
    binary_print_continuity_window_px: int,
    binary_print_continuity_tail_fraction: float,
    bottom_background_band_px: int,
    mask_mean_weight: float,
    target_mask_mean: float,
    pure_noise_prob: float,
    target_only_prob: float,
    diffusion_steps: int,
    beta_start: float,
    beta_end: float,
    train_stage: str,
    prediction_type: str,
    t_min: float,
    t_max: float,
    binary_print_endpoint_weight: float = 0.0,
    binary_print_endpoint_margin: float = 0.0,
    binary_print_endpoint_tail_fraction: float = 0.10,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    stage = str(train_stage)
    pred_type = str(prediction_type)
    if stage == "direct" and pred_type != "x0":
        raise ValueError("--train-stage direct requires --prediction-type x0")
    target = normalize_target(batch["target"])
    optics = batch["optics"].to(torch.float32)
    expert_mask = ensure_bchw(batch["expert_mask"]).to(target.device).clamp(0.0, 1.0)
    batch_size = int(target.shape[0])
    alpha_bars = diffusion_alpha_bars(int(diffusion_steps), beta_start=float(beta_start), beta_end=float(beta_end), device=target.device)
    t_idx = sample_t_indices(
        batch_size,
        alpha_bars.numel(),
        train_stage=stage,
        t_min=float(t_min),
        t_max=float(t_max),
        device=target.device,
    )
    alpha_bar = alpha_bars[t_idx].view(batch_size, 1, 1, 1)
    sigma_t = t_idx.to(torch.float32) / float(max(1, alpha_bars.numel() - 1))
    clean_x0 = expert_mask * 2.0 - 1.0
    noise = torch.randn_like(clean_x0)
    if stage == "direct":
        x_t = torch.zeros_like(clean_x0)
        noise = torch.zeros_like(clean_x0)
        alpha_bar = torch.ones_like(alpha_bar)
        sigma_t = torch.zeros_like(sigma_t)
    else:
        x_t = alpha_bar.sqrt() * clean_x0 + (1.0 - alpha_bar).sqrt() * noise
    model_input = build_model_input(x_t, target, sigma_t, mask_resolution=mask_resolution, logit_clip=1.0)
    model_out = model(model_input, optics, sigma_t).float()
    if pred_type == "x0":
        pred_x0 = model_out.tanh()
        pred_noise = (x_t - alpha_bar.sqrt() * pred_x0) / (1.0 - alpha_bar).sqrt().clamp_min(1e-4)
        denoise_loss = F.mse_loss(pred_x0, clean_x0)
    elif pred_type == "epsilon":
        pred_noise = model_out
        pred_x0 = ((x_t - (1.0 - alpha_bar).sqrt() * pred_noise) / alpha_bar.sqrt().clamp_min(1e-6)).clamp(-1.0, 1.0)
        denoise_loss = F.mse_loss(pred_noise, noise)
    else:
        raise ValueError(f"Unsupported prediction_type: {prediction_type}")
    pred_prob = 0.5 * (pred_x0 + 1.0)
    pred_prob_for_bce = pred_prob.clamp(1e-4, 1.0 - 1e-4)
    bce_loss = F.binary_cross_entropy_with_logits(torch.logit(pred_prob_for_bce), expert_mask)
    bottom_bce_loss = bottom_band_bce_loss(
        pred_prob_for_bce,
        expert_mask,
        target,
        mask_resolution=int(mask_resolution),
        bottom_band_px=int(bottom_band_px),
    )
    mask_mean_loss = (pred_prob.mean(dim=(-2, -1)) - float(target_mask_mean)).square().mean()
    phys = target.new_tensor(0.0)
    background_loss = target.new_tensor(0.0)
    foreground_loss = target.new_tensor(0.0)
    row_projection_loss = target.new_tensor(0.0)
    col_projection_loss = target.new_tensor(0.0)
    bottom_foreground_loss = target.new_tensor(0.0)
    bottom_background_loss = target.new_tensor(0.0)
    bottom_endpoint_loss = target.new_tensor(0.0)
    bottom_continuity_loss = target.new_tensor(0.0)
    full_continuity_loss = target.new_tensor(0.0)
    visual_foreground_loss = target.new_tensor(0.0)
    visual_background_loss = target.new_tensor(0.0)
    visual_background_tail_loss = target.new_tensor(0.0)
    visual_speckle_loss = target.new_tensor(0.0)
    visual_uniformity_loss = target.new_tensor(0.0)
    visual_energy_loss = target.new_tensor(0.0)
    visual_bottleneck_loss = target.new_tensor(0.0)
    visual_target_loss = target.new_tensor(0.0)
    expert_foreground_loss = target.new_tensor(0.0)
    expert_background_loss = target.new_tensor(0.0)
    guard_band_loss = target.new_tensor(0.0)
    guard_band_tail_loss = target.new_tensor(0.0)
    foreground_floor_loss = target.new_tensor(0.0)
    expert_distillation_loss = target.new_tensor(0.0)
    visual_loss = target.new_tensor(0.0)
    binary_soft_dice = target.new_tensor(0.0)
    binary_dice_loss = target.new_tensor(0.0)
    binary_false_positive_loss = target.new_tensor(0.0)
    binary_false_negative_loss = target.new_tensor(0.0)
    binary_background_tail_loss = target.new_tensor(0.0)
    binary_pair_continuity_loss = target.new_tensor(0.0)
    binary_window_continuity_loss = target.new_tensor(0.0)
    binary_endpoint_margin_loss = target.new_tensor(0.0)
    binary_worst_dice_loss = target.new_tensor(0.0)
    binary_print_loss = target.new_tensor(0.0)
    cosine_score_loss = target.new_tensor(0.0)
    score = target.new_tensor(0.0)
    highpass = target.new_tensor(0.0)
    grad_score = target.new_tensor(0.0)
    hard_mean = (pred_prob >= 0.5).to(torch.float32).mean()
    use_independent_visual_loss = float(visual_loss_weight) > 0.0
    use_binary_print_loss = float(binary_print_loss_weight) > 0.0
    use_cosine_score_loss = float(cosine_score_loss_weight) > 0.0
    if float(physical_weight) > 0.0 or use_independent_visual_loss or use_binary_print_loss or use_cosine_score_loss:
        hard = (pred_prob >= 0.5).to(pred_prob.dtype)
        mask_st = hard.detach() - pred_prob.detach() + pred_prob
        pred_light = fft_cache.forward(mask_st, optics)
        phys_each = physical_loss(
            pred_light,
            target,
            cosine_weight=float(physical_cosine_weight),
            centered_cosine_weight=float(physical_centered_cosine_weight),
            l1_weight=float(physical_l1_weight),
            highpass_cosine_weight=float(physical_highpass_cosine_weight),
            highpass_l1_weight=float(physical_highpass_l1_weight),
            gradient_cosine_weight=float(physical_gradient_cosine_weight),
        ) if float(physical_weight) > 0.0 else target.new_zeros(batch_size)
        legacy_each, visual_metrics = visual_physical_loss(
            pred_light,
            target,
            background_weight=float(background_weight),
            foreground_weight=float(foreground_weight),
            row_projection_weight=float(row_projection_weight),
            col_projection_weight=float(col_projection_weight),
            bottom_foreground_weight=float(bottom_foreground_weight),
            bottom_background_weight=float(bottom_background_weight),
            bottom_endpoint_weight=float(bottom_endpoint_weight),
            bottom_continuity_weight=float(bottom_continuity_weight),
            full_continuity_weight=float(full_continuity_weight),
            visual_foreground_weight=float(visual_foreground_weight),
            visual_background_weight=float(visual_background_weight),
            visual_background_tail_weight=float(visual_background_tail_weight),
            visual_speckle_weight=float(visual_speckle_weight),
            visual_uniformity_weight=float(visual_uniformity_weight),
            visual_energy_weight=float(visual_energy_weight),
            visual_bottleneck_weight=float(visual_bottleneck_weight),
            visual_background_tail_fraction=float(visual_background_tail_fraction),
            visual_softmin_temperature=float(visual_softmin_temperature),
            visual_bottleneck_focal_gain=float(visual_bottleneck_focal_gain),
            bottom_band_px=int(bottom_band_px),
            bottom_background_band_px=int(bottom_background_band_px),
            include_visual_quality=not use_independent_visual_loss,
        )
        phys_each = phys_each + legacy_each
        phys = phys_each.mean()
        background_loss = visual_metrics["background_loss"].mean()
        foreground_loss = visual_metrics["foreground_loss"].mean()
        row_projection_loss = visual_metrics["row_projection_loss"].mean()
        col_projection_loss = visual_metrics["col_projection_loss"].mean()
        bottom_foreground_loss = visual_metrics["bottom_foreground_loss"].mean()
        bottom_background_loss = visual_metrics["bottom_background_loss"].mean()
        bottom_endpoint_loss = visual_metrics["bottom_endpoint_loss"].mean()
        bottom_continuity_loss = visual_metrics["bottom_continuity_loss"].mean()
        full_continuity_loss = visual_metrics["full_continuity_loss"].mean()
        visual_foreground_loss = visual_metrics["visual_foreground_loss"].mean()
        visual_background_loss = visual_metrics["visual_background_loss"].mean()
        visual_background_tail_loss = visual_metrics["visual_background_tail_loss"].mean()
        visual_speckle_loss = visual_metrics["visual_speckle_loss"].mean()
        visual_uniformity_loss = visual_metrics["visual_uniformity_loss"].mean()
        visual_energy_loss = visual_metrics["visual_energy_loss"].mean()
        visual_bottleneck_loss = visual_metrics["visual_bottleneck_loss"].mean()
        visual_target_loss = visual_metrics["visual_target_loss"].mean()
        if use_independent_visual_loss:
            if "expert_pred" not in batch:
                raise KeyError("Independent visual loss requires expert_pred in every dataset shard.")
            expert_each, expert_metrics = expert_optical_distillation_loss(
                pred_light,
                batch["expert_pred"],
                target,
                foreground_weight=float(expert_foreground_weight),
                background_weight=float(expert_background_weight),
                guard_weight=float(expert_guard_weight),
                guard_tail_weight=float(expert_guard_tail_weight),
                floor_weight=float(expert_floor_weight),
                guard_band_px=int(guard_band_px),
                tail_fraction=float(expert_tail_fraction),
            )
            expert_distillation_loss = expert_each.mean()
            expert_foreground_loss = expert_metrics["expert_foreground_loss"].mean()
            expert_background_loss = expert_metrics["expert_background_loss"].mean()
            guard_band_loss = expert_metrics["guard_band_loss"].mean()
            guard_band_tail_loss = expert_metrics["guard_band_tail_loss"].mean()
            foreground_floor_loss = expert_metrics["foreground_floor_loss"].mean()
            visual_loss = visual_target_loss + expert_distillation_loss
        if use_binary_print_loss:
            binary_each, binary_metrics = soft_binary_printing_loss(
                pred_light,
                target,
                threshold=float(binary_print_threshold),
                temperature=float(binary_print_temperature),
                dice_weight=float(binary_print_dice_weight),
                false_positive_weight=float(binary_print_false_positive_weight),
                false_negative_weight=float(binary_print_false_negative_weight),
                background_tail_weight=float(binary_print_background_tail_weight),
                pair_continuity_weight=float(binary_print_pair_continuity_weight),
                window_continuity_weight=float(binary_print_window_continuity_weight),
                continuity_window_px=int(binary_print_continuity_window_px),
                continuity_tail_fraction=float(binary_print_continuity_tail_fraction),
                background_tail_fraction=float(binary_print_background_tail_fraction),
                endpoint_weight=float(binary_print_endpoint_weight),
                endpoint_margin=float(binary_print_endpoint_margin),
                endpoint_tail_fraction=float(binary_print_endpoint_tail_fraction),
            )
            binary_dice_values = binary_metrics["binary_dice_loss"]
            worst_count = max(
                1,
                int(math.ceil(binary_dice_values.numel() * max(1e-4, min(1.0, float(binary_print_worst_sample_fraction))))),
            )
            binary_worst_dice_loss = binary_dice_values.topk(worst_count, largest=True, sorted=False).values.mean()
            binary_print_loss = binary_each.mean() + float(binary_print_worst_sample_weight) * binary_worst_dice_loss
            binary_soft_dice = binary_metrics["binary_soft_dice"].mean()
            binary_dice_loss = binary_dice_values.mean()
            binary_false_positive_loss = binary_metrics["binary_false_positive_loss"].mean()
            binary_false_negative_loss = binary_metrics["binary_false_negative_loss"].mean()
            binary_background_tail_loss = binary_metrics["binary_background_tail_loss"].mean()
            binary_pair_continuity_loss = binary_metrics["binary_pair_continuity_loss"].mean()
            binary_window_continuity_loss = binary_metrics["binary_window_continuity_loss"].mean()
            binary_endpoint_margin_loss = binary_metrics["binary_endpoint_margin_loss"].mean()
        score_each = cosine_score_image(pred_light, target)
        score = score_each.mean()
        cosine_score_loss = (1.0 - score_each).mean()
        highpass = highpass_cosine_score_image(pred_light, target).mean()
        grad_score = gradient_cosine_score_image(pred_light, target).mean()
    loss = (
        float(denoise_weight) * denoise_loss
        + float(bce_weight) * bce_loss
        + float(bottom_supervision_weight) * bottom_bce_loss
        + float(physical_weight) * phys
        + float(visual_loss_weight) * visual_loss
        + float(binary_print_loss_weight) * binary_print_loss
        + float(cosine_score_loss_weight) * cosine_score_loss
        + float(mask_mean_weight) * mask_mean_loss
    )
    metrics = {
        "loss": loss.detach(),
        "denoise_loss": denoise_loss.detach(),
        "bce_loss": bce_loss.detach(),
        "bottom_bce_loss": bottom_bce_loss.detach(),
        "physical_loss": phys.detach(),
        "background_loss": background_loss.detach(),
        "foreground_loss": foreground_loss.detach(),
        "row_projection_loss": row_projection_loss.detach(),
        "col_projection_loss": col_projection_loss.detach(),
        "bottom_foreground_loss": bottom_foreground_loss.detach(),
        "bottom_background_loss": bottom_background_loss.detach(),
        "bottom_endpoint_loss": bottom_endpoint_loss.detach(),
        "bottom_continuity_loss": bottom_continuity_loss.detach(),
        "full_continuity_loss": full_continuity_loss.detach(),
        "visual_foreground_loss": visual_foreground_loss.detach(),
        "visual_background_loss": visual_background_loss.detach(),
        "visual_background_tail_loss": visual_background_tail_loss.detach(),
        "visual_speckle_loss": visual_speckle_loss.detach(),
        "visual_uniformity_loss": visual_uniformity_loss.detach(),
        "visual_energy_loss": visual_energy_loss.detach(),
        "visual_bottleneck_loss": visual_bottleneck_loss.detach(),
        "visual_target_loss": visual_target_loss.detach(),
        "expert_foreground_loss": expert_foreground_loss.detach(),
        "expert_background_loss": expert_background_loss.detach(),
        "guard_band_loss": guard_band_loss.detach(),
        "guard_band_tail_loss": guard_band_tail_loss.detach(),
        "foreground_floor_loss": foreground_floor_loss.detach(),
        "expert_distillation_loss": expert_distillation_loss.detach(),
        "visual_loss": visual_loss.detach(),
        "weighted_visual_loss": (float(visual_loss_weight) * visual_loss).detach(),
        "binary_soft_dice": binary_soft_dice.detach(),
        "binary_dice_loss": binary_dice_loss.detach(),
        "binary_false_positive_loss": binary_false_positive_loss.detach(),
        "binary_false_negative_loss": binary_false_negative_loss.detach(),
        "binary_background_tail_loss": binary_background_tail_loss.detach(),
        "binary_pair_continuity_loss": binary_pair_continuity_loss.detach(),
        "binary_window_continuity_loss": binary_window_continuity_loss.detach(),
        "binary_endpoint_margin_loss": binary_endpoint_margin_loss.detach(),
        "binary_worst_dice_loss": binary_worst_dice_loss.detach(),
        "binary_print_loss": binary_print_loss.detach(),
        "weighted_binary_print_loss": (float(binary_print_loss_weight) * binary_print_loss).detach(),
        "cosine_score_loss": cosine_score_loss.detach(),
        "weighted_cosine_score_loss": (float(cosine_score_loss_weight) * cosine_score_loss).detach(),
        "mask_mean_loss": mask_mean_loss.detach(),
        "score": score.detach(),
        "highpass_score": highpass.detach(),
        "gradient_score": grad_score.detach(),
        "prob_mean": pred_prob.mean().detach(),
        "hard_mean": hard_mean.detach(),
        "expert_mean": expert_mask.mean().detach(),
        "pure_noise_frac": target.new_tensor(0.0),
        "target_only_frac": target.new_tensor(0.0),
        "t_mean": sigma_t.mean().detach(),
    }
    return loss, metrics


@torch.no_grad()
def predict_direct_logits(
    model: torch.nn.Module,
    target: torch.Tensor,
    optics: torch.Tensor,
    *,
    mask_resolution: int,
    logit_clip: float,
    prediction_type: str,
    state: torch.Tensor | None = None,
    amp: bool,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    if str(prediction_type) != "x0":
        raise ValueError("direct prediction requires prediction_type=x0")
    device = target.device
    batch = int(target.shape[0])
    if state is None:
        x = torch.zeros(batch, 1, int(mask_resolution), int(mask_resolution), device=device)
    else:
        x = ensure_bchw(state).to(device=device, dtype=target.dtype)
    sigma_t = torch.zeros(batch, device=device)
    model_input = build_model_input(x, target, sigma_t, mask_resolution=mask_resolution, logit_clip=1.0)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=bool(amp and device.type == "cuda")):
        pred_x0 = model(model_input, optics, sigma_t).float().tanh()
    return pred_x0.clamp(-1.0, 1.0) * float(logit_clip)


@torch.no_grad()
def sample_mask(
    model: torch.nn.Module,
    target: torch.Tensor,
    optics: torch.Tensor,
    *,
    mask_resolution: int,
    sigma_min: float,
    sigma_max: float,
    sigma_rho: float,
    steps: int,
    eta: float,
    logit_clip: float,
    init_mode: str,
    diffusion_steps: int,
    beta_start: float,
    beta_end: float,
    prediction_type: str,
    amp: bool,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    device = target.device
    batch = int(target.shape[0])
    alpha_bars = diffusion_alpha_bars(int(diffusion_steps), beta_start=float(beta_start), beta_end=float(beta_end), device=device)
    total_steps = alpha_bars.numel()
    sample_steps = max(1, min(int(steps), total_steps))
    indices = torch.linspace(total_steps - 1, 0, sample_steps, device=device).round().to(torch.long)
    x = torch.randn(batch, 1, int(mask_resolution), int(mask_resolution), device=device)
    for step_pos, t_idx in enumerate(indices):
        alpha_bar = alpha_bars[t_idx].view(1, 1, 1, 1)
        sigma_t = (t_idx.to(torch.float32) / float(max(1, total_steps - 1))).expand(batch)
        model_input = build_model_input(x, target, sigma_t, mask_resolution=mask_resolution, logit_clip=1.0)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=bool(amp and device.type == "cuda")):
            model_out = model(model_input, optics, sigma_t).float()
        if str(prediction_type) == "x0":
            pred_x0 = model_out.tanh()
            pred_noise = (x - alpha_bar.sqrt() * pred_x0) / (1.0 - alpha_bar).sqrt().clamp_min(1e-4)
        elif str(prediction_type) == "epsilon":
            pred_noise = model_out
            pred_x0 = ((x - (1.0 - alpha_bar).sqrt() * pred_noise) / alpha_bar.sqrt().clamp_min(1e-6)).clamp(-1.0, 1.0)
        else:
            raise ValueError(f"Unsupported prediction_type: {prediction_type}")
        if step_pos == len(indices) - 1:
            x = pred_x0
            break
        next_t = indices[step_pos + 1]
        next_alpha = alpha_bars[next_t].view(1, 1, 1, 1)
        direction = (1.0 - next_alpha).sqrt() * pred_noise
        x = next_alpha.sqrt() * pred_x0 + direction
        if float(eta) > 0:
            x = x + float(eta) * torch.randn_like(x) * (1.0 - next_alpha).sqrt()
    return x.clamp(-1.0, 1.0) * float(logit_clip)


def save_checkpoint(path: Path, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: Any | None = None, scaler: Any, epoch: int, global_step: int, model_config: dict[str, Any], metadata: dict[str, Any], args: argparse.Namespace, best_metric: float) -> None:
    ckpt = {
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else {},
        "scaler": scaler.state_dict() if hasattr(scaler, "state_dict") else {},
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_config": model_config,
        "dataset_metadata": metadata,
        "train_args": vars(args),
        "best_metric": float(best_metric),
        "mode": "expert_supervised_mask_diffusion",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write beside the destination and atomically replace it.  A direct
    # torch.save(path) can leave a multi-GB Zip64 archive unreadable if the
    # process or filesystem is interrupted during the write.
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(ckpt, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


class ScorePlateauScheduler:
    """Short per-step warmup followed by validation-score-driven LR reductions."""

    def __init__(self, optimizer: torch.optim.Optimizer, args: argparse.Namespace) -> None:
        self.optimizer = optimizer
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.warmup_steps = max(0, int(getattr(args, "warmup_steps", 0)))
        self.batch_steps = 0
        self.plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(getattr(args, "plateau_factor", 0.5)),
            patience=max(0, int(getattr(args, "plateau_patience", 2))),
            threshold=max(0.0, float(getattr(args, "plateau_threshold", 1e-3))),
            threshold_mode="abs",
            cooldown=max(0, int(getattr(args, "plateau_cooldown", 1))),
            min_lr=max(0.0, float(getattr(args, "min_lr", 0.0))),
        )
        if self.warmup_steps > 0:
            self._set_warmup_lr(1.0 / float(self.warmup_steps))

    def _set_warmup_lr(self, ratio: float) -> None:
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * ratio

    def step(self) -> None:
        """Advance only the initial warmup; plateau updates occur after validation."""
        self.batch_steps += 1
        if self.warmup_steps > 0 and self.batch_steps < self.warmup_steps:
            self._set_warmup_lr(float(self.batch_steps + 1) / float(self.warmup_steps))
        elif self.warmup_steps > 0 and self.batch_steps == self.warmup_steps:
            self._set_warmup_lr(1.0)

    def step_metric(self, score: float) -> None:
        if self.batch_steps >= self.warmup_steps:
            self.plateau.step(float(score))

    def state_dict(self) -> dict[str, Any]:
        return {
            "base_lrs": list(self.base_lrs),
            "warmup_steps": int(self.warmup_steps),
            "batch_steps": int(self.batch_steps),
            "plateau": self.plateau.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.base_lrs = [float(value) for value in state.get("base_lrs", self.base_lrs)]
        self.warmup_steps = int(state.get("warmup_steps", self.warmup_steps))
        self.batch_steps = int(state.get("batch_steps", self.batch_steps))
        plateau_state = state.get("plateau", {})
        if plateau_state:
            self.plateau.load_state_dict(plateau_state)


def build_lr_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace, *, steps_per_epoch: int) -> Any | None:
    """Build either fixed, cosine, or validation-score-driven learning-rate control."""
    scheduler_name = str(getattr(args, "lr_scheduler", "constant"))
    if scheduler_name == "constant":
        return None
    if scheduler_name == "score_plateau":
        return ScorePlateauScheduler(optimizer, args)
    total = int(getattr(args, "scheduler_total_steps", 0))
    if total <= 0:
        total = max(1, int(args.epochs) * max(1, int(steps_per_epoch)))
    warmup = max(0, int(getattr(args, "warmup_steps", 0)))
    min_ratio = max(0.0, min(1.0, float(getattr(args, "scheduler_min_ratio", 0.05))))

    def schedule(step: int) -> float:
        t = max(1, int(step) + 1)
        if warmup > 0 and t <= warmup:
            return float(t) / float(warmup)
        if total <= warmup:
            return min_ratio
        progress = max(0.0, min(1.0, (float(t) - float(warmup)) / float(total - warmup)))
        return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule)


def load_checkpoint_model(path: str | Path, *, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device)
    model = build_model_from_config(dict(ckpt["model_config"])).to(device)
    state = {str(k).replace("module.", "", 1): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, ckpt


def save_comparison_png(path: Path, *, target: torch.Tensor, expert_mask: torch.Tensor, pred_mask: torch.Tensor, pred_light: torch.Tensor, title: str) -> None:
    from PIL import Image, ImageDraw

    def panel(x: torch.Tensor, size: int = 320) -> Any:
        arr = x.detach().cpu().to(torch.float32).numpy()
        arr = arr - float(np.nanmin(arr))
        hi = float(np.nanmax(arr))
        if hi > 1e-8:
            arr = arr / hi
        img = Image.fromarray(np.uint8(np.clip(arr, 0.0, 1.0) * 255), mode="L")
        return img.resize((size, size), resample=Image.Resampling.NEAREST).convert("RGB")

    labels = ["target", "expert mask", "sampled mask", "FFT lightmap"]
    panels = [panel(target), panel(expert_mask), panel(pred_mask), panel(pred_light)]
    size = 320
    gap = 10
    title_h = 30
    label_h = 24
    canvas = Image.new("RGB", (size * 4 + gap * 3, size + title_h + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill=(0, 0, 0))
    for i, (label, img) in enumerate(zip(labels, panels)):
        x = i * (size + gap)
        draw.text((x + 8, title_h + 4), label, fill=(0, 0, 0))
        canvas.paste(img, (x, title_h + label_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def train_main(args: argparse.Namespace) -> None:
    distributed, rank, _world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    if bool(args.allow_tf32) and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(int(args.seed) + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed) + rank)

    metadata = load_metadata(args.dataset_dir)
    args.forward_model = resolve_forward_model(str(args.forward_model), metadata)
    train_set = ExpertMaskDataset(args.dataset_dir, str(args.train_split), metadata=metadata, cache_size=int(args.cache_size))
    val_set = ExpertMaskDataset(args.dataset_dir, str(args.val_split), metadata=metadata, cache_size=int(args.cache_size))
    train_sampler = BalancedReplaySampler(
        train_set,
        simple_ratio=float(args.simple_replay_ratio),
        samples_per_epoch=int(args.samples_per_epoch),
        num_replicas=_world_size if distributed else 1,
        rank=rank if distributed else 0,
        seed=int(args.seed),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=int(args.batch_size),
        shuffle=False,
        sampler=train_sampler,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=int(args.num_workers) > 0,
    )
    val_workers = max(0, min(int(args.num_workers), 2))
    val_kwargs = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": val_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
        "persistent_workers": val_workers > 0,
    }
    simple_val_set = torch.utils.data.Subset(val_set, val_set.simple_indices)
    complex_val_set = torch.utils.data.Subset(val_set, val_set.complex_indices)
    val_simple_loader = DataLoader(simple_val_set, **val_kwargs)
    val_complex_loader = DataLoader(complex_val_set, **val_kwargs)

    first = train_set[0]
    optics_dim = int(first["optics"].numel()) if torch.is_tensor(first["optics"]) else 4
    model_config = model_config_from_args(args, optics_dim=optics_dim)
    model = ExpertMaskDiffusionUNet(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    start_epoch = 1
    global_step = 0
    best_metric = -math.inf
    resume_ckpt: dict[str, Any] | None = None
    if str(args.resume_checkpoint).strip():
        resume_ckpt = torch.load(args.resume_checkpoint, map_location=device)
        state = {str(k).replace("module.", "", 1): v for k, v in resume_ckpt["model"].items()}
        model.load_state_dict(state, strict=False)
        if "optimizer" in resume_ckpt and not bool(args.resume_reset_optimizer):
            optimizer.load_state_dict(resume_ckpt["optimizer"])
        if not bool(args.resume_reset_epoch):
            start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
            global_step = int(resume_ckpt.get("global_step", 0))
            best_metric = float(resume_ckpt.get("best_metric", -math.inf))
        if bool(args.resume_reset_best_metric):
            best_metric = -math.inf

    scheduler = build_lr_scheduler(optimizer, args, steps_per_epoch=len(train_loader))
    if scheduler is not None and resume_ckpt is not None and not bool(args.resume_reset_optimizer):
        scheduler_state = resume_ckpt.get("scheduler", {})
        if scheduler_state:
            scheduler.load_state_dict(scheduler_state)

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=False,  # 避免 ReDDP 在前向传播中插入隐式集合操作
        )
    amp_dtype = parse_amp_dtype(args.amp_dtype)
    scaler = make_grad_scaler(bool(args.amp and amp_dtype == torch.float16 and device.type == "cuda"))
    autocast_enabled = bool(args.amp and device.type == "cuda")
    fft_cache = FFTPlanCache(
        mask_resolution=int(args.mask_resolution),
        target_resolution=int(args.target_resolution),
        device=device,
        max_size=int(args.fft_cache_size),
        forward_model=str(args.forward_model),
        inter_num_override=float(args.fft_inter_num),
    )
    val_fft_inter_num = (
        float(args.val_fft_inter_num)
        if float(args.val_fft_inter_num) > 0
        else float(args.fft_inter_num)
    )
    val_fft_cache = FFTPlanCache(
        mask_resolution=int(args.mask_resolution),
        target_resolution=int(args.target_resolution),
        device=device,
        max_size=int(args.fft_cache_size),
        forward_model=str(args.forward_model),
        inter_num_override=val_fft_inter_num,
    )

    out_dir = Path(args.out_dir)
    if is_main(rank):
        (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        with open(out_dir / "train_config.json", "w", encoding="utf-8") as f:
            json.dump({"mode": "expert_supervised_mask_diffusion", "model_config": model_config, "train_args": vars(args), "dataset_metadata": metadata}, f, indent=2, ensure_ascii=False)
        print(
            f"Expert diffusion training samples={len(train_set)} simple={len(train_set.simple_indices)} "
            f"complex={len(train_set.complex_indices)} val_simple={len(simple_val_set)} "
            f"val_complex={len(complex_val_set)} samples_per_epoch={len(train_sampler)} model={model_config} "
            f"forward_model={args.forward_model} train_fft_inter={args.fft_inter_num} "
            f"val_fft_inter={val_fft_inter_num}",
            flush=True,
        )

    metric_keys = (
        "loss", "denoise_loss", "bce_loss", "bottom_bce_loss", "physical_loss",
        "background_loss", "foreground_loss", "row_projection_loss", "col_projection_loss",
        "bottom_foreground_loss", "bottom_background_loss", "bottom_endpoint_loss",
        "bottom_continuity_loss", "full_continuity_loss",
        "visual_foreground_loss", "visual_background_loss", "visual_background_tail_loss",
        "visual_speckle_loss", "visual_uniformity_loss", "visual_energy_loss",
        "visual_bottleneck_loss", "visual_target_loss",
        "expert_foreground_loss", "expert_background_loss", "guard_band_loss",
        "guard_band_tail_loss", "foreground_floor_loss", "expert_distillation_loss",
        "visual_loss", "weighted_visual_loss",
        "binary_soft_dice", "binary_dice_loss", "binary_false_positive_loss",
        "binary_false_negative_loss", "binary_background_tail_loss",
        "binary_pair_continuity_loss", "binary_window_continuity_loss", "binary_endpoint_margin_loss",
        "binary_worst_dice_loss",
        "binary_print_loss", "weighted_binary_print_loss",
        "cosine_score_loss", "weighted_cosine_score_loss",
        "mask_mean_loss",
        "score", "highpass_score", "gradient_score", "prob_mean", "hard_mean",
        "expert_mean", "pure_noise_frac", "target_only_frac", "t_mean",
    )

    def evaluate_validation(loader: DataLoader, max_batches: int) -> list[dict[str, float]]:
        rows: list[dict[str, float]] = []
        if max_batches == 0:
            return rows
        with torch.no_grad():
            for val_idx, batch in enumerate(loader, start=1):
                if max_batches > 0 and val_idx > max_batches:
                    break
                batch = move_batch(batch, device)
                batch["optics"] = override_optics_inter_num(batch["optics"], float(args.override_inter_num))
                target = normalize_target(batch["target"])
                optics = batch["optics"].to(torch.float32)
                if str(args.val_mode) == "direct":
                    logits = predict_direct_logits(
                        model,
                        target,
                        optics,
                        mask_resolution=int(args.mask_resolution),
                        logit_clip=float(args.logit_clip),
                        prediction_type=str(args.prediction_type),
                        amp=bool(args.amp),
                        amp_dtype=amp_dtype,
                    )
                else:
                    logits = sample_mask(
                        model,
                        target,
                        optics,
                        mask_resolution=int(args.mask_resolution),
                        sigma_min=float(args.sigma_min),
                        sigma_max=float(args.sigma_max),
                        sigma_rho=float(args.sigma_rho),
                        steps=int(args.val_sample_steps),
                        eta=float(args.eta),
                        logit_clip=float(args.logit_clip),
                        init_mode=str(args.sample_init_mode),
                        diffusion_steps=int(args.diffusion_steps),
                        beta_start=float(args.beta_start),
                        beta_end=float(args.beta_end),
                        prediction_type=str(args.prediction_type),
                        amp=bool(args.amp),
                        amp_dtype=amp_dtype,
                    )
                mask = (torch.sigmoid(logits) >= 0.5).to(torch.float32)
                pred = val_fft_cache.forward(mask, optics)
                quality = optical_visual_quality_metrics(pred, target)
                binary_topology = binary_printing_topology_metrics(
                    pred,
                    target,
                    threshold=float(args.binary_print_threshold),
                )
                rows.append(
                    {
                        "score": float(quality["cosine_score"].mean().item()),
                        "highpass": float(quality["highpass_score"].mean().item()),
                        "gradient": float(quality["gradient_score"].mean().item()),
                        "visual_quality": float(quality["visual_quality_score"].mean().item()),
                        "foreground_mae": float(quality["foreground_mae"].mean().item()),
                        "background_p95": float(quality["background_p95"].mean().item()),
                        "uniformity": float(quality["foreground_uniformity"].mean().item()),
                        "speckle": float(quality["speckle_gradient"].mean().item()),
                        "pair_gap_rate": float(quality["pair_gap_rate"].mean().item()),
                        "binary_dice": float(binary_topology["binary_dice"].mean().item()),
                        "binary_pair_break_rate": float(binary_topology["binary_pair_break_rate"].mean().item()),
                        "binary_longest_gap": float(binary_topology["binary_longest_gap"].mean().item()),
                        "binary_topology": float(binary_topology["binary_topology"].mean().item()),
                        "mask_mean": float(mask.mean().item()),
                    }
                )
        return rows

    def summarize_validation(rows: list[dict[str, float]]) -> dict[str, float]:
        if not rows:
            return {}
        means = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
        means["binary_dice_min"] = float(np.min([row["binary_dice"] for row in rows]))
        return means

    try:
        for epoch in range(start_epoch, int(args.epochs) + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            running = {k: 0.0 for k in metric_keys}
            seen = 0
            for batch_idx, batch in enumerate(train_loader, start=1):
                batch = move_batch(batch, device)
                batch["optics"] = override_optics_inter_num(batch["optics"], float(args.override_inter_num))
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                    loss, metrics = compute_batch_loss(
                        model,
                        batch,
                        fft_cache,
                        mask_resolution=int(args.mask_resolution),
                        sigma_min=float(args.sigma_min),
                        sigma_max=float(args.sigma_max),
                        sigma_rho=float(args.sigma_rho),
                        logit_clip=float(args.logit_clip),
                        mask_temperature=float(args.mask_temperature),
                        denoise_weight=float(args.denoise_weight),
                        bce_weight=float(args.bce_weight),
                        physical_weight=float(args.physical_weight),
                        physical_cosine_weight=float(args.physical_cosine_weight),
                        physical_centered_cosine_weight=float(args.physical_centered_cosine_weight),
                        physical_l1_weight=float(args.physical_l1_weight),
                        physical_highpass_cosine_weight=float(args.physical_highpass_cosine_weight),
                        physical_highpass_l1_weight=float(args.physical_highpass_l1_weight),
                        physical_gradient_cosine_weight=float(args.physical_gradient_cosine_weight),
                        cosine_score_loss_weight=float(args.cosine_score_loss_weight),
                        background_weight=float(args.background_weight),
                        foreground_weight=float(args.foreground_weight),
                        row_projection_weight=float(args.row_projection_weight),
                        col_projection_weight=float(args.col_projection_weight),
                        bottom_supervision_weight=float(args.bottom_supervision_weight),
                        bottom_band_px=int(args.bottom_band_px),
                        bottom_foreground_weight=float(args.bottom_foreground_weight),
                        bottom_background_weight=float(args.bottom_background_weight),
                        bottom_endpoint_weight=float(args.bottom_endpoint_weight),
                        bottom_continuity_weight=float(args.bottom_continuity_weight),
                        full_continuity_weight=float(args.full_continuity_weight),
                        visual_foreground_weight=float(args.visual_foreground_weight),
                        visual_background_weight=float(args.visual_background_weight),
                        visual_background_tail_weight=float(args.visual_background_tail_weight),
                        visual_speckle_weight=float(args.visual_speckle_weight),
                        visual_uniformity_weight=float(args.visual_uniformity_weight),
                        visual_energy_weight=float(args.visual_energy_weight),
                        visual_bottleneck_weight=float(args.visual_bottleneck_weight),
                        visual_background_tail_fraction=float(args.visual_background_tail_fraction),
                        visual_softmin_temperature=float(args.visual_softmin_temperature),
                        visual_bottleneck_focal_gain=float(args.visual_bottleneck_focal_gain),
                        visual_loss_weight=float(args.visual_loss_weight),
                        expert_foreground_weight=float(args.expert_foreground_weight),
                        expert_background_weight=float(args.expert_background_weight),
                        expert_guard_weight=float(args.expert_guard_weight),
                        expert_guard_tail_weight=float(args.expert_guard_tail_weight),
                        expert_floor_weight=float(args.expert_floor_weight),
                        guard_band_px=int(args.guard_band_px),
                        expert_tail_fraction=float(args.expert_tail_fraction),
                        binary_print_loss_weight=float(args.binary_print_loss_weight),
                        binary_print_threshold=float(args.binary_print_threshold),
                        binary_print_temperature=float(args.binary_print_temperature),
                        binary_print_dice_weight=float(args.binary_print_dice_weight),
                        binary_print_false_positive_weight=float(args.binary_print_false_positive_weight),
                        binary_print_false_negative_weight=float(args.binary_print_false_negative_weight),
                        binary_print_background_tail_weight=float(args.binary_print_background_tail_weight),
                        binary_print_background_tail_fraction=float(args.binary_print_background_tail_fraction),
                        binary_print_worst_sample_weight=float(args.binary_print_worst_sample_weight),
                        binary_print_worst_sample_fraction=float(args.binary_print_worst_sample_fraction),
                        binary_print_pair_continuity_weight=float(args.binary_print_pair_continuity_weight),
                        binary_print_window_continuity_weight=float(args.binary_print_window_continuity_weight),
                        binary_print_continuity_window_px=int(args.binary_print_continuity_window_px),
                        binary_print_continuity_tail_fraction=float(args.binary_print_continuity_tail_fraction),
                        bottom_background_band_px=int(args.bottom_background_band_px),
                        mask_mean_weight=float(args.mask_mean_weight),
                        target_mask_mean=float(args.target_mask_mean),
                        pure_noise_prob=float(args.pure_noise_prob),
                        target_only_prob=float(args.target_only_prob),
                        diffusion_steps=int(args.diffusion_steps),
                        beta_start=float(args.beta_start),
                        beta_end=float(args.beta_end),
                        train_stage=str(args.train_stage),
                        prediction_type=str(args.prediction_type),
                        t_min=float(args.t_min),
                        t_max=float(args.t_max),
                        binary_print_endpoint_weight=float(args.binary_print_endpoint_weight),
                        binary_print_endpoint_margin=float(args.binary_print_endpoint_margin),
                        binary_print_endpoint_tail_fraction=float(args.binary_print_endpoint_tail_fraction),
                    )
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    if float(args.grad_clip) > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if float(args.grad_clip) > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
                    optimizer.step()
                global_step += 1
                if scheduler is not None:
                    scheduler.step()
                bs = int(batch["target"].shape[0])
                seen += bs
                for key in metric_keys:
                    running[key] += float(metrics[key].item()) * bs
                if is_main(rank) and int(args.log_every) > 0 and global_step % int(args.log_every) == 0:
                    denom = max(1, seen)
                    parts = [f"epoch={epoch}", f"step={global_step}", f"batch={batch_idx}/{len(train_loader)}"]
                    if scheduler is not None:
                        parts.append(f"lr={optimizer.param_groups[0]['lr']:.3e}")
                    parts += [f"{k}={running[k] / denom:.5f}" for k in metric_keys]
                    print(" ".join(parts), flush=True)
                if is_main(rank) and int(args.save_step_every) > 0 and global_step % int(args.save_step_every) == 0:
                    save_checkpoint(out_dir / "checkpoints" / "latest.pt", model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, epoch=epoch, global_step=global_step, model_config=model_config, metadata=metadata, args=args, best_metric=best_metric)
                # 训练循环内的定期 barrier，防止 save_checkpoint (torch.save)
                # 导致的 rank 0 CPU-GPU 同步扰乱 NCCL 通信
                if distributed and int(args.save_step_every) > 0 and global_step % int(args.save_step_every) == 0:
                    dist.barrier()

            if is_main(rank):
                save_checkpoint(out_dir / "checkpoints" / "latest.pt", model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, epoch=epoch, global_step=global_step, model_config=model_config, metadata=metadata, args=args, best_metric=best_metric)
            # 同步所有 rank，防止 save_checkpoint (torch.save) 在 rank 0 上的
            # CPU-GPU 同步扰乱 NCCL 集合操作序列
            if distributed:
                dist.barrier()

            if int(args.val_every) > 0 and epoch % int(args.val_every) == 0:
                if is_main(rank):
                    model.eval()
                    simple_limit = int(args.val_max_batches) if int(args.val_simple_max_batches) < 0 else int(args.val_simple_max_batches)
                    complex_limit = int(args.val_max_batches) if int(args.val_complex_max_batches) < 0 else int(args.val_complex_max_batches)
                    simple_rows = evaluate_validation(val_simple_loader, simple_limit)
                    complex_rows = evaluate_validation(val_complex_loader, complex_limit)
                    val_rows = simple_rows + complex_rows
                    simple_means = summarize_validation(simple_rows)
                    complex_means = summarize_validation(complex_rows)
                    combined_means = summarize_validation(val_rows)
                    if val_rows:
                        if str(args.best_metric) == "visual_quality":
                            val_score = float(np.mean([r["visual_quality"] for r in val_rows]))
                        elif str(args.best_metric) == "binary_dice":
                            val_score = float(np.mean([r["binary_dice"] for r in val_rows]))
                        elif str(args.best_metric) == "cosine_score":
                            val_score = float(np.mean([r["score"] for r in val_rows]))
                        elif str(args.best_metric) == "binary_dice_min":
                            val_score = float(np.min([r["binary_dice"] for r in val_rows]))
                        elif str(args.best_metric) == "binary_topology":
                            val_score = float(np.mean([r["binary_topology"] for r in val_rows]))
                        elif str(args.best_metric) == "balanced_binary_dice":
                            group_dice = [means["binary_dice"] for means in (simple_means, complex_means) if means]
                            val_score = float(min(group_dice)) if group_dice else -math.inf
                        else:
                            val_score = float(np.mean([r["score"] + 0.5 * r["highpass"] + 0.5 * r["gradient"] for r in val_rows]))
                    else:
                        val_score = -math.inf
                    if simple_means:
                        print("VAL_SIMPLE " + " ".join(f"{k}={v:.5f}" for k, v in simple_means.items()), flush=True)
                    if complex_means:
                        print("VAL_COMPLEX " + " ".join(f"{k}={v:.5f}" for k, v in complex_means.items()), flush=True)
                    print("VAL_COMBINED " + " ".join(f"{k}={v:.5f}" for k, v in combined_means.items()) + f" select={val_score:.5f}", flush=True)
                    if scheduler is not None and hasattr(scheduler, "step_metric"):
                        scheduler.step_metric(val_score)
                    if val_score > best_metric:
                        best_metric = val_score
                        save_checkpoint(out_dir / "checkpoints" / "best.pt", model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, epoch=epoch, global_step=global_step, model_config=model_config, metadata=metadata, args=args, best_metric=best_metric)
                if distributed:
                    if scheduler is not None and hasattr(scheduler, "step_metric"):
                        scheduler_state: list[Any] = [scheduler.state_dict() if is_main(rank) else None]
                        dist.broadcast_object_list(scheduler_state, src=0)
                        if not is_main(rank):
                            scheduler.load_state_dict(scheduler_state[0])
                    dist.barrier()
                # 验证后同步，确保所有 rank 同时进入下一个 epoch 的训练
                if distributed:
                    dist.barrier()
    finally:
        cleanup_distributed(distributed)


def batch_infer_main(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    model, ckpt = load_checkpoint_model(args.checkpoint, device=device)
    train_args = ckpt.get("train_args", {})
    mask_resolution = int(train_args.get("mask_resolution", ckpt.get("dataset_metadata", {}).get("mask_resolution", 512)))
    target_resolution = int(train_args.get("target_resolution", ckpt.get("dataset_metadata", {}).get("target_resolution", 1024)))
    metadata = load_metadata(args.dataset_dir)
    requested_forward = str(args.forward_model)
    if requested_forward in {"", "auto"}:
        requested_forward = str(train_args.get("forward_model", "auto"))
    forward_model = resolve_forward_model(requested_forward, metadata)
    fft_inter_num = float(args.fft_inter_num)
    if fft_inter_num <= 0:
        fft_inter_num = float(
            train_args.get("val_fft_inter_num", train_args.get("fft_inter_num", 0.0))
        )
    dataset = ExpertMaskDataset(args.dataset_dir, args.split, metadata=metadata, cache_size=1)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)
    amp_dtype = parse_amp_dtype(args.amp_dtype)
    fft_cache = FFTPlanCache(
        mask_resolution=mask_resolution,
        target_resolution=target_resolution,
        device=device,
        max_size=1,
        forward_model=forward_model,
        inter_num_override=fft_inter_num,
    )
    rows = []
    for idx in range(min(int(args.num_samples), len(dataset))):
        item = dataset[idx]
        target = normalize_target(item["target"]).to(device)
        optics = item["optics"].to(device=device, dtype=torch.float32)
        if optics.dim() == 1:
            optics = optics[None]
        optics = override_optics_inter_num(optics, float(args.override_inter_num))
        best = None
        t0 = time.perf_counter()
        infer_mode = str(args.infer_mode or train_args.get("val_mode", "sample"))
        prediction_type = str(args.prediction_type or train_args.get("prediction_type", "x0"))
        num_restarts = 1 if infer_mode == "direct" else max(1, int(args.restarts))
        for _ in range(num_restarts):
            if infer_mode == "direct":
                state = None
                logits = None
                direct_iters = max(1, int(args.direct_iters))
                for _iter_idx in range(direct_iters):
                    logits = predict_direct_logits(
                        model,
                        target,
                        optics,
                        mask_resolution=mask_resolution,
                        logit_clip=float(args.logit_clip or train_args.get("logit_clip", 12.0)),
                        prediction_type=prediction_type,
                        state=state,
                        amp=bool(args.amp),
                        amp_dtype=amp_dtype,
                    )
                    pred_x0 = (logits / float(args.logit_clip or train_args.get("logit_clip", 12.0))).clamp(-1.0, 1.0)
                    if str(args.direct_feedback) == "hard":
                        state = (torch.sigmoid(logits) >= 0.5).to(torch.float32) * 2.0 - 1.0
                    else:
                        state = pred_x0
                assert logits is not None
            else:
                logits = sample_mask(
                    model,
                    target,
                    optics,
                    mask_resolution=mask_resolution,
                    sigma_min=float(args.sigma_min or train_args.get("sigma_min", 0.05)),
                    sigma_max=float(args.sigma_max or train_args.get("sigma_max", 4.0)),
                    sigma_rho=float(args.sigma_rho or train_args.get("sigma_rho", 7.0)),
                    steps=int(args.sample_steps or train_args.get("sample_steps", 16)),
                    eta=float(args.eta if args.eta >= 0 else train_args.get("eta", 0.5)),
                    logit_clip=float(args.logit_clip or train_args.get("logit_clip", 12.0)),
                    init_mode=str(args.sample_init_mode or train_args.get("sample_init_mode", "noise")),
                    diffusion_steps=int(args.diffusion_steps or train_args.get("diffusion_steps", 1000)),
                    beta_start=float(args.beta_start or train_args.get("beta_start", 1e-4)),
                    beta_end=float(args.beta_end or train_args.get("beta_end", 0.02)),
                    prediction_type=prediction_type,
                    amp=bool(args.amp),
                    amp_dtype=amp_dtype,
                )
            mask = (torch.sigmoid(logits) >= 0.5).to(torch.float32)
            pred = fft_cache.forward(mask, optics)
            score = float(cosine_score_image(pred, target).mean().item())
            hp = float(highpass_cosine_score_image(pred, target).mean().item())
            grad = float(gradient_cosine_score_image(pred, target).mean().item())
            select = score + 0.5 * hp + 0.5 * grad
            candidate = {"logits": logits, "mask": mask, "pred": pred, "score": score, "highpass": hp, "gradient": grad, "select": select}
            if best is None or candidate["select"] > best["select"]:
                best = candidate
        assert best is not None
        elapsed = time.perf_counter() - t0
        row = {
            "index": idx,
            "sample_id": item.get("sample_id", idx),
            "pattern_type": item.get("pattern_type", "unknown"),
            "score": best["score"],
            "highpass_score": best["highpass"],
            "gradient_score": best["gradient"],
            "mask_mean": float(best["mask"].mean().item()),
            "elapsed_s": elapsed,
        }
        rows.append(row)
        if idx < int(args.save_images):
            sample_dir = out_dir / "images" / f"sample_{idx:04d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            expert_mask = ensure_bchw(item["expert_mask"])[0, 0]
            save_gray_png(sample_dir / "target.png", target[0, 0].cpu())
            save_gray_png(sample_dir / "expert_mask.png", expert_mask.cpu())
            save_gray_png(sample_dir / "sampled_mask.png", best["mask"][0, 0].cpu())
            save_gray_png(sample_dir / "predicted_lightmap.png", best["pred"][0, 0].cpu())
            save_comparison_png(
                sample_dir / "comparison.png",
                target=target[0, 0].cpu(),
                expert_mask=expert_mask.cpu(),
                pred_mask=best["mask"][0, 0].cpu(),
                pred_light=best["pred"][0, 0].cpu(),
                title=f"#{idx} score={best['score']:.4f} hp={best['highpass']:.4f} grad={best['gradient']:.4f}",
            )
        print(f"[{idx + 1}/{min(int(args.num_samples), len(dataset))}] score={best['score']:.4f} hp={best['highpass']:.4f} grad={best['gradient']:.4f} mask={float(best['mask'].mean().item()):.3f} time={elapsed:.1f}s", flush=True)
    scores = np.asarray([r["score"] for r in rows], dtype=np.float64)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"checkpoint": str(Path(args.checkpoint).resolve()), "score_mean": float(scores.mean()) if scores.size else 0.0, "num_samples": len(rows)}, f, indent=2, ensure_ascii=False)
    with open(out_dir / "scores.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["index"])
        writer.writeheader()
        writer.writerows(rows)


def add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input-channels", type=int, default=8)
    p.add_argument("--base-channels", type=int, default=128)
    p.add_argument("--channel-mults", default="1,2,4,8")
    p.add_argument("--emb-dim", type=int, default=256)
    p.add_argument("--kernel-size", type=int, default=3)
    p.add_argument("--use-fourier", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fourier-modes", type=int, default=16)


def build_train_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train expert-supervised diffusion mask model.")
    p.add_argument("train", nargs="?", default="train")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--train-split", default="train")
    p.add_argument("--val-split", default="val")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--cache-size", type=int, default=4)
    p.add_argument(
        "--simple-replay-ratio",
        type=float,
        default=0.50,
        help="Fraction of each training epoch drawn from original non-complex shards.",
    )
    p.add_argument(
        "--samples-per-epoch",
        type=int,
        default=240000,
        help="Total sampled examples per epoch; <=0 uses the raw dataset size.",
    )
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lr-scheduler", choices=("constant", "warmup_cosine", "score_plateau"), default="constant")
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--scheduler-total-steps", type=int, default=0)
    p.add_argument("--scheduler-min-ratio", type=float, default=0.05)
    p.add_argument("--plateau-factor", type=float, default=0.5)
    p.add_argument("--plateau-patience", type=int, default=2)
    p.add_argument("--plateau-threshold", type=float, default=1e-3)
    p.add_argument("--plateau-cooldown", type=int, default=1)
    p.add_argument("--min-lr", type=float, default=0.0)
    p.add_argument("--mask-resolution", type=int, default=512)
    p.add_argument("--target-resolution", type=int, default=1024)
    p.add_argument("--override-inter-num", type=float, default=2.0)
    p.add_argument(
        "--forward-model",
        choices=("auto", "spherical_circular_fft", "angular_spectrum_padded_fft", "asf_linear_convolution_fft"),
        default="auto",
        help="FFT propagation model; auto reads dataset metadata.",
    )
    p.add_argument(
        "--fft-inter-num",
        type=float,
        default=0.0,
        help="Numerical sampling used only by the training FFT; <=0 uses optics without changing model input.",
    )
    p.add_argument(
        "--val-fft-inter-num",
        type=float,
        default=0.0,
        help="Independent validation FFT sampling; <=0 reuses --fft-inter-num.",
    )
    p.add_argument("--train-stage", choices=("direct", "low-noise", "full"), default="full")
    p.add_argument("--prediction-type", choices=("x0", "epsilon"), default="x0")
    p.add_argument("--t-min", type=float, default=0.0)
    p.add_argument("--t-max", type=float, default=-1.0, help="Normalized max diffusion timestep; -1 uses the stage default.")
    p.add_argument("--val-mode", choices=("direct", "sample"), default="sample")
    p.add_argument("--sigma-min", type=float, default=0.05)
    p.add_argument("--sigma-max", type=float, default=4.0)
    p.add_argument("--sigma-rho", type=float, default=7.0)
    p.add_argument("--sample-steps", type=int, default=16)
    p.add_argument("--val-sample-steps", type=int, default=8)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--sample-init-mode", choices=("noise", "zero", "target"), default="noise")
    p.add_argument("--diffusion-steps", type=int, default=1000)
    p.add_argument("--beta-start", type=float, default=1e-4)
    p.add_argument("--beta-end", type=float, default=0.02)
    p.add_argument("--logit-clip", type=float, default=12.0)
    p.add_argument("--mask-temperature", type=float, default=1.0)
    p.add_argument("--denoise-weight", type=float, default=1.0)
    p.add_argument("--bce-weight", type=float, default=0.5)
    p.add_argument("--physical-weight", type=float, default=0.05)
    p.add_argument("--physical-cosine-weight", type=float, default=1.0)
    p.add_argument("--physical-centered-cosine-weight", type=float, default=0.0)
    p.add_argument("--physical-l1-weight", type=float, default=0.25)
    p.add_argument("--physical-highpass-cosine-weight", type=float, default=0.30)
    p.add_argument("--physical-highpass-l1-weight", type=float, default=0.15)
    p.add_argument("--physical-gradient-cosine-weight", type=float, default=0.20)
    p.add_argument("--cosine-score-loss-weight", type=float, default=0.0, help="Independent multiplier for 1 - optical cosine score.")
    p.add_argument("--background-weight", type=float, default=0.0)
    p.add_argument("--foreground-weight", type=float, default=0.0)
    p.add_argument("--row-projection-weight", type=float, default=0.0)
    p.add_argument("--col-projection-weight", type=float, default=0.0)
    p.add_argument("--bottom-supervision-weight", type=float, default=0.0)
    p.add_argument("--bottom-band-px", type=int, default=6)
    p.add_argument("--bottom-foreground-weight", type=float, default=0.0)
    p.add_argument("--bottom-background-weight", type=float, default=0.0)
    p.add_argument("--bottom-endpoint-weight", type=float, default=0.0)
    p.add_argument("--bottom-continuity-weight", type=float, default=0.0)
    p.add_argument("--full-continuity-weight", type=float, default=0.0)
    p.add_argument("--visual-foreground-weight", type=float, default=0.0)
    p.add_argument("--visual-background-weight", type=float, default=0.0)
    p.add_argument("--visual-background-tail-weight", type=float, default=0.0)
    p.add_argument("--visual-speckle-weight", type=float, default=0.0)
    p.add_argument("--visual-uniformity-weight", type=float, default=0.0)
    p.add_argument("--visual-energy-weight", type=float, default=0.0)
    p.add_argument("--visual-bottleneck-weight", type=float, default=0.0)
    p.add_argument("--visual-background-tail-fraction", type=float, default=0.05)
    p.add_argument("--visual-softmin-temperature", type=float, default=0.05)
    p.add_argument("--visual-bottleneck-focal-gain", type=float, default=2.0)
    p.add_argument("--visual-loss-weight", type=float, default=0.0, help="Independent multiplier for visual target and expert-field losses.")
    p.add_argument("--expert-foreground-weight", type=float, default=0.0)
    p.add_argument("--expert-background-weight", type=float, default=0.0)
    p.add_argument("--expert-guard-weight", type=float, default=0.0)
    p.add_argument("--expert-guard-tail-weight", type=float, default=0.0)
    p.add_argument("--expert-floor-weight", type=float, default=0.0)
    p.add_argument("--guard-band-px", type=int, default=6)
    p.add_argument("--expert-tail-fraction", type=float, default=0.05)
    p.add_argument("--binary-print-loss-weight", type=float, default=0.0, help="Independent multiplier for the differentiable thresholded-print loss.")
    p.add_argument("--binary-print-threshold", type=float, default=0.55, help="Relative FFT threshold used by the binary-print loss and Dice validation.")
    p.add_argument("--binary-print-temperature", type=float, default=0.04)
    p.add_argument("--binary-print-dice-weight", type=float, default=0.0)
    p.add_argument("--binary-print-false-positive-weight", type=float, default=0.0)
    p.add_argument("--binary-print-false-negative-weight", type=float, default=0.0)
    p.add_argument("--binary-print-background-tail-weight", type=float, default=0.0)
    p.add_argument("--binary-print-background-tail-fraction", type=float, default=0.01)
    p.add_argument("--binary-print-worst-sample-weight", type=float, default=0.0)
    p.add_argument("--binary-print-worst-sample-fraction", type=float, default=0.50)
    p.add_argument("--binary-print-pair-continuity-weight", type=float, default=0.0)
    p.add_argument("--binary-print-window-continuity-weight", type=float, default=0.0)
    p.add_argument("--binary-print-continuity-window-px", type=int, default=5)
    p.add_argument("--binary-print-continuity-tail-fraction", type=float, default=0.003)
    p.add_argument("--binary-print-endpoint-weight", type=float, default=0.0)
    p.add_argument("--binary-print-endpoint-margin", type=float, default=0.0)
    p.add_argument("--binary-print-endpoint-tail-fraction", type=float, default=0.10)
    p.add_argument("--bottom-background-band-px", type=int, default=6)
    p.add_argument("--mask-mean-weight", type=float, default=2.0)
    p.add_argument("--target-mask-mean", type=float, default=0.47)
    p.add_argument("--pure-noise-prob", type=float, default=0.0)
    p.add_argument("--target-only-prob", type=float, default=0.0)
    p.add_argument("--fft-cache-size", type=int, default=1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    p.add_argument("--allow-tf32", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-step-every", type=int, default=500)
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--val-max-batches", type=int, default=8)
    p.add_argument("--val-simple-max-batches", type=int, default=-1, help="Original-shard validation limit; -1 uses --val-max-batches.")
    p.add_argument("--val-complex-max-batches", type=int, default=-1, help="Complex-shard validation limit; -1 uses --val-max-batches.")
    p.add_argument(
        "--best-metric",
        choices=("legacy", "visual_quality", "cosine_score", "binary_dice", "binary_dice_min", "binary_topology", "balanced_binary_dice"),
        default="legacy",
    )
    p.add_argument("--resume-checkpoint", default="")
    p.add_argument("--resume-reset-optimizer", action="store_true")
    p.add_argument("--resume-reset-epoch", action="store_true")
    p.add_argument("--resume-reset-best-metric", action="store_true")
    add_model_args(p)
    return p


def build_batch_infer_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch infer expert-supervised diffusion mask model.")
    p.add_argument("batch-infer", nargs="?", default="batch-infer")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--sample-steps", type=int, default=0)
    p.add_argument("--restarts", type=int, default=4)
    p.add_argument("--sample-init-mode", choices=("", "noise", "zero", "target"), default="")
    p.add_argument("--infer-mode", choices=("", "direct", "sample"), default="")
    p.add_argument("--direct-iters", type=int, default=1)
    p.add_argument("--direct-feedback", choices=("soft", "hard"), default="soft")
    p.add_argument("--prediction-type", choices=("", "x0", "epsilon"), default="")
    p.add_argument("--sigma-min", type=float, default=0.0)
    p.add_argument("--sigma-max", type=float, default=0.0)
    p.add_argument("--sigma-rho", type=float, default=0.0)
    p.add_argument("--diffusion-steps", type=int, default=0)
    p.add_argument("--beta-start", type=float, default=0.0)
    p.add_argument("--beta-end", type=float, default=0.0)
    p.add_argument("--eta", type=float, default=-1.0)
    p.add_argument("--logit-clip", type=float, default=0.0)
    p.add_argument("--override-inter-num", type=float, default=2.0)
    p.add_argument(
        "--forward-model",
        choices=("auto", "spherical_circular_fft", "angular_spectrum_padded_fft", "asf_linear_convolution_fft"),
        default="auto",
    )
    p.add_argument("--fft-inter-num", type=float, default=0.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    p.add_argument("--save-images", type=int, default=10)
    return p


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "batch-infer":
        args = build_batch_infer_parser().parse_args()
        batch_infer_main(args)
    else:
        args = build_train_parser().parse_args()
        train_main(args)


if __name__ == "__main__":
    main()
