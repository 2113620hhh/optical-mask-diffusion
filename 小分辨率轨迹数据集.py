#!/usr/bin/env python3
from __future__ import annotations

import json
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.format import read_array_header_1_0, read_array_header_2_0, read_magic
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


OPTICS_SCHEMA = ("L0", "lamda", "distance", "inter_num")


def load_metadata(dataset_dir: str | Path) -> dict[str, Any]:
    path = Path(dataset_dir) / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"metadata.json not found under {dataset_dir}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_npz_array_header(path: Path, key: str) -> tuple[tuple[int, ...], np.dtype]:
    member = f"{key}.npy"
    with zipfile.ZipFile(path) as archive:
        if member not in archive.namelist():
            raise KeyError(f"{path} missing {key}")
        with archive.open(member) as f:
            version = read_magic(f)
            if version == (1, 0):
                shape, _fortran_order, dtype = read_array_header_1_0(f)
            else:
                shape, _fortran_order, dtype = read_array_header_2_0(f)
    return tuple(int(v) for v in shape), np.dtype(dtype)


def trajectory_logit_clip_from_metadata(metadata: dict[str, Any]) -> float:
    traj = metadata.get("trajectory", {})
    if isinstance(traj, dict):
        return float(traj.get("logit_clip", 12.0))
    return 12.0


def decode_trajectory_logits(values: np.ndarray, logit_clip: float) -> np.ndarray:
    if values.dtype == np.int8:
        return values.astype(np.float32) * (float(logit_clip) / 127.0)
    return values.astype(np.float32)


def decode_trajectory_tensor(values: np.ndarray, logit_clip: float) -> torch.Tensor:
    if values.dtype == np.int8:
        return torch.from_numpy(values.astype(np.float32)).mul_(float(logit_clip) / 127.0)
    return torch.from_numpy(values.astype(np.float32))


def coord_maps(height: int = 256, width: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.linspace(-1.0, 1.0, int(height), dtype=torch.float32)
    x = torch.linspace(-1.0, 1.0, int(width), dtype=torch.float32)
    yy = y.view(1, height, 1).expand(1, height, width).contiguous()
    xx = x.view(1, 1, width).expand(1, height, width).contiguous()
    return yy, xx


def ensure_bchw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        x = x[None, None]
    elif x.dim() == 3:
        x = x[None] if x.shape[0] == 1 else x[:, None]
    elif x.dim() != 4:
        raise ValueError(f"Expected 2D/3D/4D tensor, got {tuple(x.shape)}")
    if x.shape[1] != 1:
        raise ValueError(f"Expected single-channel tensor, got {tuple(x.shape)}")
    return x


def resize_target_to_mask(target: torch.Tensor, size: tuple[int, int], mode: str = "area") -> torch.Tensor:
    target = ensure_bchw(target)
    if target.shape[-2:] == tuple(size):
        return target
    if mode == "nearest":
        return F.interpolate(target.to(torch.float32), size=size, mode="nearest")
    if mode == "bilinear":
        return F.interpolate(target.to(torch.float32), size=size, mode="bilinear", align_corners=False)
    return F.interpolate(target.to(torch.float32), size=size, mode="area")


def build_full_trajectory_input(
    *,
    current_logit: torch.Tensor,
    target: torch.Tensor,
    step_frac: torch.Tensor | float,
    tau: torch.Tensor | float,
    coord_y: torch.Tensor | None = None,
    coord_x: torch.Tensor | None = None,
    prev_delta: torch.Tensor | None = None,
    target_resize_mode: str = "area",
) -> torch.Tensor:
    """Build the 10-channel full-image trajectory-distillation input.

    Channels:
      current_prob, current_hard, target_up, zero_pred, zero_residual,
      step_map, tau_map, prev_delta, coord_y, coord_x.
    """
    if current_logit.dim() == 3:
        current_logit = current_logit[:, None]
    current_logit = ensure_bchw(current_logit).to(torch.float32)
    batch, _, height, width = current_logit.shape
    device = current_logit.device
    dtype = current_logit.dtype

    target = ensure_bchw(target).to(device=device, dtype=dtype)
    if target.shape[0] == 1 and batch > 1:
        target = target.expand(batch, -1, -1, -1)
    target_up = resize_target_to_mask(target, (height, width), mode=target_resize_mode).to(dtype)

    current_prob = torch.sigmoid(current_logit)
    current_hard = (current_prob >= 0.5).to(dtype)
    zero_pred = torch.zeros_like(target_up)
    zero_residual = target_up

    if not torch.is_tensor(step_frac):
        step_frac = torch.full((batch,), float(step_frac), device=device, dtype=dtype)
    else:
        step_frac = step_frac.to(device=device, dtype=dtype).reshape(-1)
        if step_frac.numel() == 1 and batch > 1:
            step_frac = step_frac.expand(batch)
    if not torch.is_tensor(tau):
        tau = torch.full((batch,), float(tau), device=device, dtype=dtype)
    else:
        tau = tau.to(device=device, dtype=dtype).reshape(-1)
        if tau.numel() == 1 and batch > 1:
            tau = tau.expand(batch)
    step_map = step_frac.view(batch, 1, 1, 1).expand(batch, 1, height, width)
    tau_map = tau.view(batch, 1, 1, 1).expand(batch, 1, height, width)

    if prev_delta is None:
        prev_delta_ch = torch.zeros_like(current_logit)
    else:
        if prev_delta.dim() == 3:
            prev_delta = prev_delta[:, None]
        prev_delta_ch = ensure_bchw(prev_delta).to(device=device, dtype=dtype)
        if prev_delta_ch.shape[-2:] != (height, width):
            prev_delta_ch = F.interpolate(prev_delta_ch, size=(height, width), mode="area")
        if prev_delta_ch.shape[0] == 1 and batch > 1:
            prev_delta_ch = prev_delta_ch.expand(batch, -1, -1, -1)

    if coord_y is None or coord_x is None:
        cy, cx = coord_maps(height, width)
        coord_y = cy[None]
        coord_x = cx[None]
    elif coord_y.dim() == 3:
        coord_y = coord_y[:, None] if coord_y.shape[0] != 1 else coord_y[None]
        coord_x = coord_x[:, None] if coord_x.shape[0] != 1 else coord_x[None]
    coord_y = ensure_bchw(coord_y).to(device=device, dtype=dtype)
    coord_x = ensure_bchw(coord_x).to(device=device, dtype=dtype)
    if coord_y.shape[0] == 1 and batch > 1:
        coord_y = coord_y.expand(batch, -1, -1, -1)
        coord_x = coord_x.expand(batch, -1, -1, -1)

    return torch.cat(
        [
            current_prob,
            current_hard,
            target_up,
            zero_pred,
            zero_residual,
            step_map,
            tau_map,
            prev_delta_ch,
            coord_y,
            coord_x,
        ],
        dim=1,
    )


class SmallTrajectoryDataset(Dataset):
    """Full-image trajectory pair dataset for 128 target / 256 mask experiments."""

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str = "train",
        *,
        metadata: dict[str, Any] | None = None,
        cache_size: int = 2,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.split = str(split)
        self.metadata = metadata if metadata is not None else load_metadata(self.dataset_dir)
        self.logit_clip = trajectory_logit_clip_from_metadata(self.metadata)
        self.cache_size = max(1, int(cache_size))
        self.target_resolution = int(self.metadata.get("target_resolution", 128))
        self.mask_resolution = int(self.metadata.get("mask_resolution", 256))

        split_dir = self.dataset_dir / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        self.files = sorted(split_dir.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz shards found under: {split_dir}")

        self.sample_refs: list[tuple[int, int]] = []
        self.item_refs: list[tuple[int, int]] = []
        self.steps: np.ndarray | None = None
        self.tau: np.ndarray | None = None
        self.max_step = 1
        self.num_states: int | None = None

        for shard_idx, path in enumerate(self.files):
            target_shape, _target_dtype = read_npz_array_header(path, "target")
            traj_shape, _traj_dtype = read_npz_array_header(path, "trajectory_logits_q")
            count = int(target_shape[0])
            states = int(traj_shape[1])
            if states < 2:
                raise ValueError(f"{path} must contain at least two trajectory states")
            with np.load(path) as shard:
                missing = [
                    key
                    for key in ("target", "trajectory_logits_q", "trajectory_steps", "trajectory_tau", "optics")
                    if key not in shard.files
                ]
                if missing:
                    raise KeyError(f"{path} missing required fields: {missing}")
                steps = shard["trajectory_steps"].astype(np.int64)
                tau = shard["trajectory_tau"].astype(np.float32)
                if len(steps) != states or len(tau) != states:
                    raise ValueError(f"{path} trajectory length mismatch")
            if self.steps is None:
                self.steps = steps
                self.tau = tau
                self.max_step = max(int(steps[-1]), 1)
                self.num_states = states
            elif not np.array_equal(self.steps, steps):
                raise ValueError(f"{path} uses different trajectory_steps from previous shards")
            for sample_idx in range(count):
                sample_ref = len(self.sample_refs)
                self.sample_refs.append((shard_idx, sample_idx))
                for pair_idx in range(states - 1):
                    self.item_refs.append((sample_ref, pair_idx))

        self.coord_y, self.coord_x = coord_maps(self.mask_resolution, self.mask_resolution)
        self._cache: OrderedDict[int, dict[str, Any]] = OrderedDict()

    def _load_shard(self, shard_idx: int) -> dict[str, Any]:
        cached = self._cache.get(shard_idx)
        if cached is not None:
            self._cache.move_to_end(shard_idx)
            return cached
        path = self.files[shard_idx]
        with np.load(path) as shard:
            target = shard["target"].astype(np.float32)
            traj = shard["trajectory_logits_q"].copy()
            optics = shard["optics"].astype(np.float32)
            sample_id = shard["sample_id"].astype(np.int64) if "sample_id" in shard.files else np.arange(target.shape[0])
            pattern_type = shard["pattern_type"] if "pattern_type" in shard.files else np.asarray(["unknown"] * target.shape[0])
            score = shard["score"].astype(np.float32) if "score" in shard.files else np.full((target.shape[0],), np.nan, dtype=np.float32)
        loaded = {
            "target": target,
            "traj": traj,
            "optics": optics,
            "sample_id": sample_id,
            "pattern_type": pattern_type,
            "score": score,
        }
        self._cache[shard_idx] = loaded
        self._cache.move_to_end(shard_idx)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return loaded

    def __len__(self) -> int:
        return len(self.item_refs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample_ref, pair_idx = self.item_refs[int(idx)]
        shard_idx, sample_idx = self.sample_refs[sample_ref]
        shard = self._load_shard(shard_idx)

        target = torch.from_numpy(shard["target"][sample_idx]).to(torch.float32)
        current = decode_trajectory_tensor(shard["traj"][sample_idx, pair_idx], self.logit_clip)
        nxt = decode_trajectory_tensor(shard["traj"][sample_idx, pair_idx + 1], self.logit_clip)
        delta = nxt - current
        assert self.steps is not None and self.tau is not None
        step_frac = float(self.steps[pair_idx]) / float(self.max_step)
        tau_value = float(self.tau[pair_idx])

        return {
            "target": target,
            "current_logit": current,
            "next_logit": nxt,
            "delta_target": delta,
            "step_frac": torch.tensor(step_frac, dtype=torch.float32),
            "tau": torch.tensor(tau_value, dtype=torch.float32),
            "coord_y": self.coord_y.clone(),
            "coord_x": self.coord_x.clone(),
            "optics": torch.from_numpy(shard["optics"][sample_idx]).to(torch.float32),
            "sample_id": int(shard["sample_id"][sample_idx]),
            "pair_idx": int(pair_idx),
            "score": float(shard["score"][sample_idx]),
            "pattern_type": str(shard["pattern_type"][sample_idx]),
        }


def load_split_samples(dataset_dir: str | Path, split: str, limit: int | None = None) -> list[dict[str, Any]]:
    """Load one record per optical sample, useful for evaluation/inference loops."""
    dataset_dir = Path(dataset_dir)
    metadata = load_metadata(dataset_dir)
    logit_clip = trajectory_logit_clip_from_metadata(metadata)
    rows: list[dict[str, Any]] = []
    for path in sorted((dataset_dir / split).glob("*.npz")):
        with np.load(path) as shard:
            target = shard["target"].astype(np.float32)
            traj = decode_trajectory_logits(shard["trajectory_logits_q"], logit_clip)
            optics = shard["optics"].astype(np.float32)
            sample_id = shard["sample_id"].astype(np.int64) if "sample_id" in shard.files else np.arange(target.shape[0])
            pattern_type = shard["pattern_type"] if "pattern_type" in shard.files else np.asarray(["unknown"] * target.shape[0])
            score = shard["score"].astype(np.float32) if "score" in shard.files else np.full((target.shape[0],), np.nan, dtype=np.float32)
            mask_binary = shard["mask_binary"].astype(np.uint8) if "mask_binary" in shard.files else None
            for i in range(target.shape[0]):
                rows.append(
                    {
                        "target": target[i],
                        "initial_logit": traj[i, 0],
                        "final_logit": traj[i, -1],
                        "optics": optics[i],
                        "sample_id": int(sample_id[i]),
                        "pattern_type": str(pattern_type[i]),
                        "expert_score": float(score[i]),
                        "mask_binary": mask_binary[i] if mask_binary is not None else None,
                    }
                )
                if limit is not None and len(rows) >= int(limit):
                    return rows
    return rows
