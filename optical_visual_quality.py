#!/usr/bin/env python3
"""Differentiable optical-image losses and visual-quality metrics.

All photometric terms are measured relative to the mean target-supported
foreground intensity. This makes background leakage and foreground variation
comparable across targets with different fill ratios.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from 光学FFT前向传播_小分辨率 import (
    cosine_score_image,
    gradient_cosine_score_image,
    highpass_cosine_score_image,
)


def ensure_bchw(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    if x.dim() == 2:
        x = x[None, None]
    elif x.dim() == 3:
        x = x[None] if x.shape[0] == 1 else x[:, None]
    if x.dim() != 4 or x.shape[1] != 1:
        raise ValueError(f"Expected a single-channel BCHW tensor, got {tuple(x.shape)}")
    return x


def masked_mean_per_sample(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    value = ensure_bchw(value)
    mask = ensure_bchw(mask).to(device=value.device, dtype=value.dtype)
    denominator = mask.sum(dim=(-3, -2, -1)).clamp_min(1.0)
    return (value * mask).sum(dim=(-3, -2, -1)) / denominator


def foreground_relative_images(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    foreground_threshold: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pred = ensure_bchw(pred).clamp_min(0.0)
    target = ensure_bchw(target).to(device=pred.device).clamp_min(0.0)
    if target.shape[-2:] != pred.shape[-2:]:
        target = F.interpolate(target, size=pred.shape[-2:], mode="area")
    foreground = (target > float(foreground_threshold)).to(pred.dtype)
    background = 1.0 - foreground
    pred_mean = masked_mean_per_sample(pred, foreground).view(-1, 1, 1, 1).clamp_min(1e-6)
    target_mean = masked_mean_per_sample(target, foreground).view(-1, 1, 1, 1).clamp_min(1e-6)
    return pred / pred_mean, target / target_mean, foreground, background, pred_mean, target_mean


def masked_top_fraction_mean(
    value: torch.Tensor,
    mask: torch.Tensor,
    *,
    fraction: float,
) -> torch.Tensor:
    value = ensure_bchw(value)
    mask = ensure_bchw(mask).to(device=value.device, dtype=torch.bool)
    outputs = []
    for sample_value, sample_mask in zip(value, mask):
        selected = sample_value[sample_mask]
        if selected.numel() == 0:
            outputs.append(sample_value.new_tensor(0.0))
            continue
        count = max(1, int(math.ceil(selected.numel() * max(1e-4, min(1.0, float(fraction))))))
        outputs.append(selected.topk(count, largest=True, sorted=False).values.mean())
    return torch.stack(outputs)


def foreground_speckle_gradient(
    pred_relative: torch.Tensor,
    foreground: torch.Tensor,
) -> torch.Tensor:
    pair_h = foreground[..., :, 1:] * foreground[..., :, :-1]
    pair_v = foreground[..., 1:, :] * foreground[..., :-1, :]
    grad_h = (pred_relative[..., :, 1:] - pred_relative[..., :, :-1]).square()
    grad_v = (pred_relative[..., 1:, :] - pred_relative[..., :-1, :]).square()
    sum_h = (grad_h * pair_h).sum(dim=(-3, -2, -1))
    sum_v = (grad_v * pair_v).sum(dim=(-3, -2, -1))
    count_h = pair_h.sum(dim=(-3, -2, -1))
    count_v = pair_v.sum(dim=(-3, -2, -1))
    return (sum_h + sum_v) / (count_h + count_v).clamp_min(1.0)


def soft_minimum(a: torch.Tensor, b: torch.Tensor, temperature: float) -> torch.Tensor:
    tau = max(float(temperature), 1e-4)
    return -tau * torch.logsumexp(torch.stack([-a / tau, -b / tau], dim=0), dim=0)


def bottleneck_continuity_loss(
    pred_relative: torch.Tensor,
    target_relative: torch.Tensor,
    foreground: torch.Tensor,
    *,
    softmin_temperature: float = 0.05,
    focal_gain: float = 2.0,
) -> torch.Tensor:
    pair_h = foreground[..., :, 1:] * foreground[..., :, :-1]
    pair_v = foreground[..., 1:, :] * foreground[..., :-1, :]

    pred_h = soft_minimum(pred_relative[..., :, 1:], pred_relative[..., :, :-1], softmin_temperature)
    target_h = soft_minimum(target_relative[..., :, 1:], target_relative[..., :, :-1], softmin_temperature)
    pred_v = soft_minimum(pred_relative[..., 1:, :], pred_relative[..., :-1, :], softmin_temperature)
    target_v = soft_minimum(target_relative[..., 1:, :], target_relative[..., :-1, :], softmin_temperature)

    gap_h = F.relu(target_h - pred_h)
    gap_v = F.relu(target_v - pred_v)
    weight_h = 1.0 + float(focal_gain) * gap_h.detach()
    weight_v = 1.0 + float(focal_gain) * gap_v.detach()
    numerator = (weight_h * gap_h.square() * pair_h).sum(dim=(-3, -2, -1))
    numerator = numerator + (weight_v * gap_v.square() * pair_v).sum(dim=(-3, -2, -1))
    denominator = pair_h.sum(dim=(-3, -2, -1)) + pair_v.sum(dim=(-3, -2, -1))
    mean_loss = numerator / denominator.clamp_min(1.0)

    tail_losses = []
    weighted_h = weight_h * gap_h.square()
    weighted_v = weight_v * gap_v.square()
    for sample_h, sample_v, mask_h, mask_v in zip(weighted_h, weighted_v, pair_h, pair_v):
        selected = torch.cat([sample_h[mask_h.to(torch.bool)], sample_v[mask_v.to(torch.bool)]])
        if selected.numel() == 0:
            tail_losses.append(sample_h.new_tensor(0.0))
            continue
        count = max(1, int(math.ceil(selected.numel() * 0.05)))
        tail_losses.append(selected.topk(count, largest=True, sorted=False).values.mean())
    tail_loss = torch.stack(tail_losses)
    return 0.5 * mean_loss + 0.5 * tail_loss


def visual_quality_training_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    foreground_weight: float,
    background_weight: float,
    background_tail_weight: float,
    speckle_weight: float,
    uniformity_weight: float,
    energy_weight: float,
    bottleneck_weight: float,
    foreground_threshold: float = 0.05,
    background_tail_fraction: float = 0.05,
    softmin_temperature: float = 0.05,
    bottleneck_focal_gain: float = 2.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pred_rel, target_rel, foreground, background, pred_mean, target_mean = foreground_relative_images(
        pred,
        target,
        foreground_threshold=foreground_threshold,
    )
    difference = pred_rel - target_rel
    foreground_loss = masked_mean_per_sample(torch.sqrt(difference.square() + 1e-6) - 1e-3, foreground)
    background_loss = masked_mean_per_sample(pred_rel.square(), background)
    background_tail_loss = masked_top_fraction_mean(
        pred_rel.square(),
        background,
        fraction=background_tail_fraction,
    )
    speckle_loss = foreground_speckle_gradient(pred_rel, foreground)
    uniformity_loss = masked_mean_per_sample((pred_rel - 1.0).abs(), foreground)
    energy_loss = ((pred_mean.flatten() - target_mean.flatten()) / target_mean.flatten().clamp_min(1e-6)).square()
    bottleneck_loss = bottleneck_continuity_loss(
        pred_rel,
        target_rel,
        foreground,
        softmin_temperature=softmin_temperature,
        focal_gain=bottleneck_focal_gain,
    )
    total = (
        float(foreground_weight) * foreground_loss
        + float(background_weight) * background_loss
        + float(background_tail_weight) * background_tail_loss
        + float(speckle_weight) * speckle_loss
        + float(uniformity_weight) * uniformity_loss
        + float(energy_weight) * energy_loss
        + float(bottleneck_weight) * bottleneck_loss
    )
    return total, {
        "visual_foreground_loss": foreground_loss,
        "visual_background_loss": background_loss,
        "visual_background_tail_loss": background_tail_loss,
        "visual_speckle_loss": speckle_loss,
        "visual_uniformity_loss": uniformity_loss,
        "visual_energy_loss": energy_loss,
        "visual_bottleneck_loss": bottleneck_loss,
    }


def guard_band_mask(
    target: torch.Tensor,
    *,
    radius_px: int,
    foreground_threshold: float = 0.05,
) -> torch.Tensor:
    """Return the target-background ring immediately surrounding foreground lines."""
    target = ensure_bchw(target).clamp_min(0.0)
    foreground = (target > float(foreground_threshold)).to(target.dtype)
    radius = max(0, int(radius_px))
    if radius == 0:
        return torch.zeros_like(foreground)
    kernel_size = 2 * radius + 1
    dilated = F.max_pool2d(foreground, kernel_size=kernel_size, stride=1, padding=radius)
    return (dilated > 0).to(target.dtype) * (1.0 - foreground)


def expert_optical_distillation_loss(
    pred: torch.Tensor,
    expert_pred: torch.Tensor,
    target: torch.Tensor,
    *,
    foreground_weight: float,
    background_weight: float,
    guard_weight: float,
    guard_tail_weight: float,
    floor_weight: float,
    guard_band_px: int,
    tail_fraction: float = 0.05,
    foreground_threshold: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Distill the realizable optical field produced by an expert mask.

    The target remains the physical objective, while the expert field supplies
    an attainable local reference for background darkness and line quality.
    Every field is normalized by its own target-foreground energy so the loss
    focuses on spatial quality instead of absolute exposure.
    """
    pred_rel, _, foreground, background, _, _ = foreground_relative_images(
        pred,
        target,
        foreground_threshold=foreground_threshold,
    )
    expert_pred = ensure_bchw(expert_pred).to(device=pred_rel.device, dtype=pred_rel.dtype).clamp_min(0.0)
    if expert_pred.shape[-2:] != pred_rel.shape[-2:]:
        expert_pred = F.interpolate(expert_pred, size=pred_rel.shape[-2:], mode="area")
    expert_mean = masked_mean_per_sample(expert_pred, foreground).view(-1, 1, 1, 1).clamp_min(1e-6)
    expert_rel = expert_pred / expert_mean

    foreground_loss = masked_mean_per_sample((pred_rel - expert_rel).abs(), foreground)
    excess_background = F.relu(pred_rel - expert_rel).square()
    background_loss = masked_mean_per_sample(excess_background, background)

    guard = guard_band_mask(
        target,
        radius_px=int(guard_band_px),
        foreground_threshold=foreground_threshold,
    ).to(device=pred_rel.device, dtype=pred_rel.dtype)
    guard_loss = masked_mean_per_sample(excess_background, guard)
    guard_tail_loss = masked_top_fraction_mean(excess_background, guard, fraction=tail_fraction)

    # The largest deficits are the darkest local parts of otherwise continuous lines.
    foreground_deficit = F.relu(expert_rel - pred_rel).square()
    floor_loss = masked_top_fraction_mean(foreground_deficit, foreground, fraction=tail_fraction)

    total = (
        float(foreground_weight) * foreground_loss
        + float(background_weight) * background_loss
        + float(guard_weight) * guard_loss
        + float(guard_tail_weight) * guard_tail_loss
        + float(floor_weight) * floor_loss
    )
    return total, {
        "expert_foreground_loss": foreground_loss,
        "expert_background_loss": background_loss,
        "guard_band_loss": guard_loss,
        "guard_band_tail_loss": guard_tail_loss,
        "foreground_floor_loss": floor_loss,
    }


def _masked_parts_mean_and_tail(
    parts: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate differently shaped masked losses while preserving per-sample tails."""
    batch_size = int(parts[0][0].shape[0])
    numerator = parts[0][0].new_zeros(batch_size)
    denominator = parts[0][0].new_zeros(batch_size)
    tails = []
    for value, mask in parts:
        numerator = numerator + (value * mask).sum(dim=(-3, -2, -1))
        denominator = denominator + mask.sum(dim=(-3, -2, -1))
    for sample_index in range(batch_size):
        selected = torch.cat([
            value[sample_index][mask[sample_index].to(torch.bool)]
            for value, mask in parts
        ])
        if selected.numel() == 0:
            tails.append(numerator.new_tensor(0.0))
            continue
        count = max(1, int(math.ceil(selected.numel() * max(1e-4, min(1.0, float(fraction))))))
        tails.append(selected.topk(count, largest=True, sorted=False).values.mean())
    return numerator / denominator.clamp_min(1.0), torch.stack(tails)


def soft_print_connectivity_losses(
    soft_print: torch.Tensor,
    foreground: torch.Tensor,
    *,
    window_px: int,
    tail_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalize isolated and multi-pixel breaks on target-supported wire runs."""
    pair_h_mask = foreground[..., :, 1:] * foreground[..., :, :-1]
    pair_v_mask = foreground[..., 1:, :] * foreground[..., :-1, :]
    pair_h_deficit = 1.0 - torch.minimum(soft_print[..., :, 1:], soft_print[..., :, :-1])
    pair_v_deficit = 1.0 - torch.minimum(soft_print[..., 1:, :], soft_print[..., :-1, :])
    pair_mean, pair_tail = _masked_parts_mean_and_tail(
        [(pair_h_deficit, pair_h_mask), (pair_v_deficit, pair_v_mask)],
        fraction=tail_fraction,
    )
    pair_loss = 0.5 * (pair_mean + pair_tail)

    length = max(2, int(window_px))
    window_parts: list[tuple[torch.Tensor, torch.Tensor]] = []
    if soft_print.shape[-1] >= length:
        horizontal_min = -F.max_pool2d(-soft_print, kernel_size=(1, length), stride=1)
        horizontal_mask = (F.avg_pool2d(foreground, kernel_size=(1, length), stride=1) >= 0.999).to(soft_print.dtype)
        window_parts.append((1.0 - horizontal_min, horizontal_mask))
    if soft_print.shape[-2] >= length:
        vertical_min = -F.max_pool2d(-soft_print, kernel_size=(length, 1), stride=1)
        vertical_mask = (F.avg_pool2d(foreground, kernel_size=(length, 1), stride=1) >= 0.999).to(soft_print.dtype)
        window_parts.append((1.0 - vertical_min, vertical_mask))
    if not window_parts:
        return pair_loss, pair_loss.new_zeros(pair_loss.shape)
    window_mean, window_tail = _masked_parts_mean_and_tail(window_parts, fraction=tail_fraction)
    return pair_loss, 0.5 * (window_mean + window_tail)


def target_axis_endpoint_mask(foreground: torch.Tensor) -> torch.Tensor:
    """Return the first/last foreground pixel of every non-empty row and column.

    The union is rotation invariant for the axis-aligned circuit patterns used by
    this project. It covers line terminals and U-shaped bottom bridges without
    treating an intentional interior gap as foreground.
    """
    foreground_bool = ensure_bchw(foreground).to(torch.bool)
    top = foreground_bool & (foreground_bool.cumsum(dim=-2) == 1)
    bottom = foreground_bool & (foreground_bool.flip(-2).cumsum(dim=-2).flip(-2) == 1)
    left = foreground_bool & (foreground_bool.cumsum(dim=-1) == 1)
    right = foreground_bool & (foreground_bool.flip(-1).cumsum(dim=-1).flip(-1) == 1)
    return (top | bottom | left | right).to(foreground.dtype)


def soft_binary_printing_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    threshold: float,
    temperature: float,
    dice_weight: float,
    false_positive_weight: float,
    false_negative_weight: float,
    background_tail_weight: float,
    pair_continuity_weight: float,
    window_continuity_weight: float,
    continuity_window_px: int,
    continuity_tail_fraction: float,
    background_tail_fraction: float = 0.01,
    foreground_threshold: float = 0.05,
    endpoint_weight: float = 0.0,
    endpoint_margin: float = 0.0,
    endpoint_tail_fraction: float = 0.10,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Optimize the eventual thresholded optical image with a smooth surrogate.

    ``sigmoid((field - threshold) / temperature)`` approximates a hard binary
    print while retaining gradients through the FFT field and the predicted mask.
    The target foreground is the desired white class; background tail loss is
    deliberately applied to the brightest would-be white defects.
    """
    pred_relative, _, foreground, background, _, _ = foreground_relative_images(
        pred,
        target,
        foreground_threshold=foreground_threshold,
    )
    softness = max(float(temperature), 1e-4)
    soft_print = torch.sigmoid((pred_relative - float(threshold)) / softness)
    target_binary = foreground
    intersection = (soft_print * target_binary).sum(dim=(-3, -2, -1))
    denominator = soft_print.sum(dim=(-3, -2, -1)) + target_binary.sum(dim=(-3, -2, -1))
    soft_dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    dice_loss = 1.0 - soft_dice
    false_positive_loss = masked_mean_per_sample(soft_print, background)
    false_negative_loss = masked_mean_per_sample(1.0 - soft_print, foreground)
    background_tail_loss = masked_top_fraction_mean(
        soft_print,
        background,
        fraction=background_tail_fraction,
    )
    pair_continuity_loss, window_continuity_loss = soft_print_connectivity_losses(
        soft_print,
        foreground,
        window_px=continuity_window_px,
        tail_fraction=continuity_tail_fraction,
    )
    endpoint_mask = target_axis_endpoint_mask(foreground)
    endpoint_floor = float(threshold) + max(0.0, float(endpoint_margin))
    endpoint_deficit = F.relu(endpoint_floor - pred_relative)
    endpoint_mean = masked_mean_per_sample(endpoint_deficit, endpoint_mask)
    endpoint_tail = masked_top_fraction_mean(
        endpoint_deficit,
        endpoint_mask,
        fraction=endpoint_tail_fraction,
    )
    endpoint_margin_loss = 0.5 * (endpoint_mean + endpoint_tail)
    total = (
        float(dice_weight) * dice_loss
        + float(false_positive_weight) * false_positive_loss
        + float(false_negative_weight) * false_negative_loss
        + float(background_tail_weight) * background_tail_loss
        + float(pair_continuity_weight) * pair_continuity_loss
        + float(window_continuity_weight) * window_continuity_loss
        + float(endpoint_weight) * endpoint_margin_loss
    )
    return total, {
        "binary_soft_dice": soft_dice,
        "binary_dice_loss": dice_loss,
        "binary_false_positive_loss": false_positive_loss,
        "binary_false_negative_loss": false_negative_loss,
        "binary_background_tail_loss": background_tail_loss,
        "binary_pair_continuity_loss": pair_continuity_loss,
        "binary_window_continuity_loss": window_continuity_loss,
        "binary_endpoint_margin_loss": endpoint_margin_loss,
    }


@torch.no_grad()
def binary_printing_dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    threshold: float,
    foreground_threshold: float = 0.05,
) -> torch.Tensor:
    """Return actual Dice after the same normalized hard threshold used at inference."""
    pred_relative, _, foreground, _, _, _ = foreground_relative_images(
        pred,
        target,
        foreground_threshold=foreground_threshold,
    )
    printed = (pred_relative >= float(threshold)).to(pred_relative.dtype)
    intersection = (printed * foreground).sum(dim=(-3, -2, -1))
    denominator = printed.sum(dim=(-3, -2, -1)) + foreground.sum(dim=(-3, -2, -1))
    return (2.0 * intersection + 1e-6) / (denominator + 1e-6)


@torch.no_grad()
def hard_print_topology_metrics(
    printed: torch.Tensor,
    target_foreground: torch.Tensor,
    *,
    longest_gap_penalty: float = 0.10,
) -> dict[str, torch.Tensor]:
    """Measure binary-print continuity only inside intended foreground runs.

    A break is counted only when two adjacent target pixels are both white but
    at least one of them prints black. Therefore intentional black spacings in
    the target never become false continuity errors.
    """
    printed = ensure_bchw(printed).to(torch.bool)
    target = ensure_bchw(target_foreground).to(device=printed.device, dtype=torch.bool)
    intersection = (printed & target).sum(dim=(-3, -2, -1)).to(torch.float32)
    denominator = (printed.sum(dim=(-3, -2, -1)) + target.sum(dim=(-3, -2, -1))).to(torch.float32)
    dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)

    target_pair_h = target[..., :, 1:] & target[..., :, :-1]
    target_pair_v = target[..., 1:, :] & target[..., :-1, :]
    printed_pair_h = printed[..., :, 1:] & printed[..., :, :-1]
    printed_pair_v = printed[..., 1:, :] & printed[..., :-1, :]
    broken_pairs = (
        (target_pair_h & ~printed_pair_h).sum(dim=(-3, -2, -1))
        + (target_pair_v & ~printed_pair_v).sum(dim=(-3, -2, -1))
    ).to(torch.float32)
    all_pairs = (
        target_pair_h.sum(dim=(-3, -2, -1))
        + target_pair_v.sum(dim=(-3, -2, -1))
    ).to(torch.float32)
    pair_break_rate = broken_pairs / all_pairs.clamp_min(1.0)

    longest_values: list[torch.Tensor] = []
    missing = target & ~printed
    for target_sample, missing_sample in zip(target[:, 0], missing[:, 0]):
        longest = 0
        for target_line, missing_line in list(zip(target_sample, missing_sample)) + list(zip(target_sample.T, missing_sample.T)):
            current = 0
            for is_target, is_missing in zip(target_line.tolist(), missing_line.tolist()):
                if is_target and is_missing:
                    current += 1
                    longest = max(longest, current)
                else:
                    current = 0
        longest_values.append(dice.new_tensor(float(longest)))
    longest_gap = torch.stack(longest_values)
    topology_score = dice * (1.0 - pair_break_rate).clamp(0.0, 1.0) * torch.exp(
        -float(longest_gap_penalty) * longest_gap
    )
    return {
        "binary_dice": dice,
        "binary_pair_break_rate": pair_break_rate,
        "binary_longest_gap": longest_gap,
        "binary_topology": topology_score,
    }


@torch.no_grad()
def binary_printing_topology_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    threshold: float,
    foreground_threshold: float = 0.05,
    longest_gap_penalty: float = 0.10,
) -> dict[str, torch.Tensor]:
    """Evaluate Dice and continuity after the deployed normalized threshold."""
    pred_relative, _, foreground, _, _, _ = foreground_relative_images(
        pred,
        target,
        foreground_threshold=foreground_threshold,
    )
    printed = pred_relative >= float(threshold)
    return hard_print_topology_metrics(
        printed,
        foreground,
        longest_gap_penalty=longest_gap_penalty,
    )


def _masked_quantile_per_sample(value: torch.Tensor, mask: torch.Tensor, quantile: float) -> torch.Tensor:
    outputs = []
    bool_mask = mask.to(torch.bool)
    for sample_value, sample_mask in zip(value, bool_mask):
        selected = sample_value[sample_mask]
        outputs.append(
            torch.quantile(selected, float(quantile)) if selected.numel() else sample_value.new_tensor(0.0)
        )
    return torch.stack(outputs)


@torch.no_grad()
def optical_visual_quality_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    foreground_threshold: float = 0.05,
    gap_threshold: float = 0.50,
    background_false_positive_threshold: float = 0.30,
    background_tail_fraction: float = 0.01,
) -> dict[str, torch.Tensor]:
    pred = ensure_bchw(pred)
    target = ensure_bchw(target).to(device=pred.device)
    pred_rel, target_rel, foreground, background, pred_mean, target_mean = foreground_relative_images(
        pred,
        target,
        foreground_threshold=foreground_threshold,
    )
    foreground_mae = masked_mean_per_sample((pred_rel - target_rel).abs(), foreground)
    foreground_rmse = masked_mean_per_sample((pred_rel - target_rel).square(), foreground).sqrt()
    foreground_p05 = _masked_quantile_per_sample(pred_rel, foreground, 0.05)
    background_mean = masked_mean_per_sample(pred_rel, background)
    background_p95 = _masked_quantile_per_sample(pred_rel, background, 0.95)
    background_top_mean = masked_top_fraction_mean(pred_rel, background, fraction=background_tail_fraction)
    background_false_positive_rate = masked_mean_per_sample(
        (pred_rel >= float(background_false_positive_threshold)).to(pred_rel.dtype),
        background,
    )
    speckle = foreground_speckle_gradient(pred_rel, foreground)
    uniformity = masked_mean_per_sample((pred_rel - 1.0).abs(), foreground)
    gap_rate = masked_mean_per_sample((pred_rel < float(gap_threshold)).to(pred_rel.dtype), foreground)
    foreground_deficit = F.relu(float(gap_threshold) - pred_rel) / max(float(gap_threshold), 1e-6)
    foreground_deficit_top1 = masked_top_fraction_mean(foreground_deficit, foreground, fraction=0.01)

    pair_h = foreground[..., :, 1:] * foreground[..., :, :-1]
    pair_v = foreground[..., 1:, :] * foreground[..., :-1, :]
    pred_pair_h = torch.minimum(pred_rel[..., :, 1:], pred_rel[..., :, :-1])
    pred_pair_v = torch.minimum(pred_rel[..., 1:, :], pred_rel[..., :-1, :])
    pair_gap_h = (pred_pair_h < float(gap_threshold)).to(pred_rel.dtype) * pair_h
    pair_gap_v = (pred_pair_v < float(gap_threshold)).to(pred_rel.dtype) * pair_v
    pair_gap_rate = (
        pair_gap_h.sum(dim=(-3, -2, -1)) + pair_gap_v.sum(dim=(-3, -2, -1))
    ) / (pair_h.sum(dim=(-3, -2, -1)) + pair_v.sum(dim=(-3, -2, -1))).clamp_min(1.0)

    cosine = cosine_score_image(pred, target).clamp(0.0, 1.0)
    highpass = highpass_cosine_score_image(pred, target).clamp(0.0, 1.0)
    gradient = gradient_cosine_score_image(pred, target).clamp(0.0, 1.0)
    background_quality = torch.exp(-(0.6 * background_p95 + 0.4 * background_top_mean))
    continuity_quality = torch.exp(-foreground_deficit_top1) * (1.0 - pair_gap_rate).clamp(0.0, 1.0)
    quality_parts: dict[str, tuple[torch.Tensor, float]] = {
        "cosine": (cosine, 0.15),
        "highpass": (highpass, 0.10),
        "gradient": (gradient, 0.10),
        "foreground": (torch.exp(-foreground_mae), 0.15),
        "background": (background_quality, 0.20),
        "uniformity": (torch.exp(-uniformity), 0.10),
        "speckle": (torch.exp(-speckle.clamp_min(0.0).sqrt()), 0.05),
        "continuity": (continuity_quality, 0.15),
    }
    log_quality = cosine.new_zeros(cosine.shape)
    for component, weight in quality_parts.values():
        log_quality = log_quality + float(weight) * component.clamp(1e-6, 1.0).log()
    visual_quality = log_quality.exp()

    return {
        "visual_quality_score": visual_quality,
        "cosine_score": cosine,
        "highpass_score": highpass,
        "gradient_score": gradient,
        "foreground_mae": foreground_mae,
        "foreground_rmse": foreground_rmse,
        "foreground_p05": foreground_p05,
        "foreground_uniformity": uniformity,
        "speckle_gradient": speckle,
        "background_mean": background_mean,
        "background_p95": background_p95,
        "background_top1_mean": background_top_mean,
        "background_false_positive_rate": background_false_positive_rate,
        "foreground_gap_rate": gap_rate,
        "foreground_deficit_top1": foreground_deficit_top1,
        "pair_gap_rate": pair_gap_rate,
        "foreground_energy_ratio": pred_mean.flatten() / target_mean.flatten().clamp_min(1e-6),
    }


def longest_foreground_gap(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    foreground_threshold: float = 0.05,
    gap_threshold: float = 0.50,
) -> int:
    pred_rel, _, foreground, _, _, _ = foreground_relative_images(
        pred,
        target,
        foreground_threshold=foreground_threshold,
    )
    target_mask = foreground[0, 0].detach().cpu().to(torch.bool)
    gap_mask = (pred_rel[0, 0] < float(gap_threshold)).detach().cpu() & target_mask

    longest = 0
    for target_line, gap_line in list(zip(target_mask, gap_mask)) + list(zip(target_mask.T, gap_mask.T)):
        current = 0
        for is_target, is_gap in zip(target_line.tolist(), gap_line.tolist()):
            if is_target and is_gap:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
    return int(longest)


def scalar_metrics(metrics: dict[str, torch.Tensor], index: int = 0) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in metrics.items():
        flat = value.detach().cpu().reshape(-1)
        result[key] = float(flat[min(int(index), flat.numel() - 1)].item())
    return result


OVQS_DEFINITION: dict[str, Any] = {
    "name": "Optical Visual Quality Score (OVQS)",
    "range": [0.0, 1.0],
    "higher_is_better": True,
    "aggregation": "weighted geometric mean",
    "weights": {
        "cosine": 0.15,
        "highpass": 0.10,
        "gradient": 0.10,
        "foreground_fidelity": 0.15,
        "background_p95_and_top1": 0.20,
        "foreground_uniformity": 0.10,
        "speckle_gradient": 0.05,
        "pair_continuity_and_worst_deficit": 0.15,
    },
}
