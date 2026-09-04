#!/usr/bin/env python3
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_channel_mults(value: Any) -> tuple[int, ...]:
    if isinstance(value, (tuple, list)):
        parts = [int(v) for v in value]
    else:
        parts = [int(p.strip()) for p in str(value).split(",") if p.strip()]
    parts = [p for p in parts if p > 0]
    return tuple(parts) if parts else (1, 2, 4, 8)


def group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if int(channels) % groups == 0:
            return groups
    return 1


def normalize_optics_tensor(optics: torch.Tensor) -> torch.Tensor:
    if optics.dim() == 1:
        optics = optics[None]
    optics = optics.to(torch.float32)
    if optics.shape[-1] < 4:
        return optics
    out = optics.clone()
    out[..., 0] = torch.log10(out[..., 0].abs().clamp_min(1e-12)) + 6.0
    out[..., 1] = torch.log10(out[..., 1].abs().clamp_min(1e-12)) + 9.0
    out[..., 2] = torch.log10(out[..., 2].abs().clamp_min(1e-12)) + 6.0
    out[..., 3] = out[..., 3] / 4.0
    if out.shape[-1] > 4:
        out[..., 4:] = torch.tanh(out[..., 4:])
    return out


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = int(kernel_size) // 2
        self.norm1 = nn.GroupNorm(group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, int(kernel_size), padding=padding)
        self.norm2 = nn.GroupNorm(group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, int(kernel_size), padding=padding)
        self.emb = nn.Linear(emb_dim, out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(F.silu(emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class FourierBlock(nn.Module):
    def __init__(self, channels: int, modes: int = 16) -> None:
        super().__init__()
        self.channels = int(channels)
        self.modes = int(modes)
        scale = 1.0 / max(1, self.channels * self.channels)
        self.weights = nn.Parameter(
            scale * torch.randn(self.channels, self.channels, self.modes, self.modes, dtype=torch.cfloat)
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        x_ft = torch.fft.rfft2(x.to(torch.float32))
        out_ft = torch.zeros(batch, channels, height, width // 2 + 1, dtype=torch.cfloat, device=x.device)
        modes = min(self.modes, height, width // 2 + 1)
        out_ft[:, :, :modes, :modes] = torch.einsum(
            "b i h w, i o h w -> b o h w",
            x_ft[:, :, :modes, :modes],
            self.weights[:channels, :channels, :modes, :modes],
        )
        return x + self.gamma * torch.fft.irfft2(out_ft, s=(height, width)).to(x.dtype)


class ExpertMaskDiffusionUNet(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int = 8,
        base_channels: int = 128,
        channel_mults: tuple[int, ...] = (1, 2, 4, 8),
        emb_dim: int = 256,
        optics_dim: int = 4,
        kernel_size: int = 3,
        use_fourier: bool = True,
        fourier_modes: int = 16,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.base_channels = int(base_channels)
        self.channel_mults = tuple(int(v) for v in channel_mults)
        self.emb_dim = int(emb_dim)
        self.optics_dim = int(optics_dim)
        self.kernel_size = int(kernel_size)
        self.use_fourier = bool(use_fourier)
        self.fourier_modes = int(fourier_modes)

        self.cond_emb = nn.Sequential(
            nn.Linear(self.optics_dim + 1, self.emb_dim),
            nn.SiLU(),
            nn.Linear(self.emb_dim, self.emb_dim),
        )
        self.in_conv = nn.Conv2d(self.input_channels, self.base_channels, 3, padding=1)
        channels = [self.base_channels * mult for mult in self.channel_mults]
        self.down_blocks = nn.ModuleList()
        prev = self.base_channels
        for level, ch in enumerate(channels):
            self.down_blocks.append(
                nn.ModuleDict(
                    {
                        "block1": ResBlock(prev, ch, self.emb_dim, self.kernel_size),
                        "block2": ResBlock(ch, ch, self.emb_dim, self.kernel_size),
                        "down": Downsample(ch) if level < len(channels) - 1 else nn.Identity(),
                    }
                )
            )
            prev = ch

        bot_ch = channels[-1]
        self.bot1 = ResBlock(bot_ch, bot_ch, self.emb_dim, self.kernel_size)
        self.bot_fourier = FourierBlock(bot_ch, modes=self.fourier_modes) if self.use_fourier else nn.Identity()
        self.bot2 = ResBlock(bot_ch, bot_ch, self.emb_dim, self.kernel_size)

        self.up_blocks = nn.ModuleList()
        prev = bot_ch
        for level, ch in enumerate(reversed(channels)):
            is_last = level == len(channels) - 1
            self.up_blocks.append(
                nn.ModuleDict(
                    {
                        "block1": ResBlock(prev + ch, ch, self.emb_dim, self.kernel_size),
                        "block2": ResBlock(ch, ch, self.emb_dim, self.kernel_size),
                        "up": Upsample(ch) if not is_last else nn.Identity(),
                    }
                )
            )
            prev = ch

        self.out_norm = nn.GroupNorm(group_count(self.base_channels), self.base_channels)
        self.out_conv = nn.Conv2d(self.base_channels, 1, 3, padding=1)

    def forward(self, x: torch.Tensor, optics: torch.Tensor, sigma_t: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected BCHW input, got {tuple(x.shape)}")
        if optics.dim() == 1:
            optics = optics[None].expand(x.shape[0], -1)
        if optics.shape[0] == 1 and x.shape[0] > 1:
            optics = optics.expand(x.shape[0], -1)
        sigma_t = sigma_t.to(device=x.device, dtype=torch.float32).reshape(-1, 1)
        if sigma_t.shape[0] == 1 and x.shape[0] > 1:
            sigma_t = sigma_t.expand(x.shape[0], -1)
        cond = torch.cat([normalize_optics_tensor(optics).to(x.device), sigma_t], dim=1)
        emb = self.cond_emb(cond).to(dtype=x.dtype)

        h = self.in_conv(x)
        skips: list[torch.Tensor] = []
        for block in self.down_blocks:
            h = block["block1"](h, emb)
            h = block["block2"](h, emb)
            skips.append(h)
            h = block["down"](h)

        h = self.bot1(h, emb)
        h = self.bot_fourier(h)
        h = self.bot2(h, emb)

        for idx, block in enumerate(self.up_blocks):
            skip = skips[-(idx + 1)]
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            h = block["block1"](h, emb)
            h = block["block2"](h, emb)
            h = block["up"](h)
        return self.out_conv(F.silu(self.out_norm(h)))


def model_config_from_args(args: Any, *, optics_dim: int = 4) -> dict[str, Any]:
    return {
        "input_channels": int(getattr(args, "input_channels", 8)),
        "base_channels": int(getattr(args, "base_channels", 128)),
        "channel_mults": tuple(parse_channel_mults(getattr(args, "channel_mults", "1,2,4,8"))),
        "emb_dim": int(getattr(args, "emb_dim", 256)),
        "optics_dim": int(optics_dim),
        "kernel_size": int(getattr(args, "kernel_size", 3)),
        "use_fourier": bool(getattr(args, "use_fourier", True)),
        "fourier_modes": int(getattr(args, "fourier_modes", 16)),
    }


def build_model_from_config(config: dict[str, Any]) -> ExpertMaskDiffusionUNet:
    return ExpertMaskDiffusionUNet(
        input_channels=int(config.get("input_channels", 8)),
        base_channels=int(config.get("base_channels", 128)),
        channel_mults=parse_channel_mults(config.get("channel_mults", (1, 2, 4, 8))),
        emb_dim=int(config.get("emb_dim", 256)),
        optics_dim=int(config.get("optics_dim", 4)),
        kernel_size=int(config.get("kernel_size", 3)),
        use_fourier=bool(config.get("use_fourier", True)),
        fourier_modes=int(config.get("fourier_modes", 16)),
    )
