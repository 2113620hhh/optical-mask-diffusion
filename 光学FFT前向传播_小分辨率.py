#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy.fft import next_fast_len
except Exception:  # pragma: no cover
    def next_fast_len(n: int) -> int:
        n = max(1, int(n))
        return 1 << (n - 1).bit_length()


OPTICS_SCHEMA = ("L0", "lamda", "distance", "inter_num")


def _unit_scale(unit: str | None) -> float:
    if unit is None:
        return 1.0
    text = str(unit).strip().lower()
    table = {
        "m": 1.0,
        "meter": 1.0,
        "meters": 1.0,
        "metre": 1.0,
        "metres": 1.0,
        "mm": 1e-3,
        "millimeter": 1e-3,
        "millimeters": 1e-3,
        "um": 1e-6,
        "μm": 1e-6,
        "micron": 1e-6,
        "microns": 1e-6,
        "micrometer": 1e-6,
        "micrometers": 1e-6,
        "nm": 1e-9,
        "nanometer": 1e-9,
        "nanometers": 1e-9,
    }
    if text not in table:
        raise ValueError(f"Unsupported optics unit: {unit!r}")
    return table[text]


def _as_float(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().reshape(-1)[0].item())
    if isinstance(value, np.ndarray):
        return float(value.reshape(-1)[0])
    return float(value)


def _extract_optics_values(optics: Any) -> dict[str, float]:
    if isinstance(optics, OpticsConfig):
        return {
            "L0": float(optics.L0),
            "lamda": float(optics.lamda),
            "distance": float(optics.distance),
            "inter_num": float(optics.inter_num),
        }
    if isinstance(optics, dict):
        keys = {
            "L0": ("L0", "l0", "mask_size", "mask_length"),
            "lamda": ("lamda", "lambda", "wavelength", "wave_length"),
            "distance": ("distance", "dist", "z"),
            "inter_num": ("inter_num", "inter", "upsample_factor"),
        }
        values: dict[str, float] = {}
        for out_key, aliases in keys.items():
            for key in aliases:
                if key in optics:
                    values[out_key] = _as_float(optics[key])
                    break
            if out_key not in values:
                raise KeyError(f"Optics is missing {out_key}; expected schema {OPTICS_SCHEMA}")
        return values
    if torch.is_tensor(optics):
        arr = optics.detach().cpu().to(torch.float32).reshape(-1).numpy()
    else:
        arr = np.asarray(optics, dtype=np.float32).reshape(-1)
    if arr.size < 4:
        raise ValueError(f"Optics vector must contain at least 4 values: {OPTICS_SCHEMA}")
    return {
        "L0": float(arr[0]),
        "lamda": float(arr[1]),
        "distance": float(arr[2]),
        "inter_num": float(arr[3]),
    }


def normalize_optics(
    optics: Any,
    optics_units: str | dict[str, str] | None = None,
    *,
    auto_unit: bool = False,
) -> tuple[dict[str, float], dict[str, Any]]:
    values = _extract_optics_values(optics)
    if isinstance(optics, dict) and optics_units is None:
        optics_units = optics.get("optics_units", optics.get("units"))

    applied_units: dict[str, str] = {}
    if isinstance(optics_units, dict):
        for key in ("L0", "lamda", "distance"):
            unit = optics_units.get(key, optics_units.get(key.lower(), "m"))
            values[key] *= _unit_scale(unit)
            applied_units[key] = str(unit)
    elif optics_units is not None:
        for key in ("L0", "lamda", "distance"):
            values[key] *= _unit_scale(str(optics_units))
            applied_units[key] = str(optics_units)
    elif auto_unit:
        # Useful for quick experiments that pass [um, nm, um, inter].
        # Explicit optics_units is preferred for reproducible datasets.
        heuristics = {
            "L0": "um" if values["L0"] > 1e-3 else "m",
            "lamda": "nm" if values["lamda"] > 1e-6 else "m",
            "distance": "um" if values["distance"] > 1e-3 else "m",
        }
        for key, unit in heuristics.items():
            values[key] *= _unit_scale(unit)
            applied_units[key] = unit
    else:
        applied_units = {"L0": "m", "lamda": "m", "distance": "m"}

    values["inter_num"] = float(max(1, int(round(values["inter_num"]))))
    metadata = {
        "optics_schema": list(OPTICS_SCHEMA),
        "optics_units_input": applied_units,
        "optics_units_internal": {"L0": "m", "lamda": "m", "distance": "m", "inter_num": "1"},
        "optics_values_m": dict(values),
    }
    return values, metadata


@dataclass(frozen=True)
class OpticsConfig:
    L0: float = 256 * 20e-9
    lamda: float = 13.5e-9
    distance: float = 150e-6
    inter_num: int = 2
    mask_resolution: int = 256
    target_resolution: int = 128

    @property
    def pixel_pitch(self) -> float:
        return float(self.L0) / float(self.mask_resolution)

    def as_vector(self) -> np.ndarray:
        return np.asarray([self.L0, self.lamda, self.distance, float(self.inter_num)], dtype=np.float32)

    @classmethod
    def from_any(
        cls,
        optics: Any,
        *,
        mask_resolution: int = 256,
        target_resolution: int = 128,
        optics_units: str | dict[str, str] | None = None,
        auto_unit: bool = False,
    ) -> "OpticsConfig":
        values, _ = normalize_optics(optics, optics_units=optics_units, auto_unit=auto_unit)
        return cls(
            L0=float(values["L0"]),
            lamda=float(values["lamda"]),
            distance=float(values["distance"]),
            inter_num=max(1, int(round(values["inter_num"]))),
            mask_resolution=int(mask_resolution),
            target_resolution=int(target_resolution),
        )


def ensure_bchw(x: torch.Tensor | np.ndarray, *, device: torch.device | None = None) -> torch.Tensor:
    if not torch.is_tensor(x):
        x = torch.from_numpy(np.asarray(x))
    if device is not None:
        x = x.to(device)
    x = x.to(torch.float32)
    if x.dim() == 2:
        x = x[None, None]
    elif x.dim() == 3:
        x = x[None] if x.shape[0] == 1 else x[:, None]
    elif x.dim() != 4:
        raise ValueError(f"Expected mask/target with 2, 3, or 4 dims, got shape {tuple(x.shape)}")
    if x.shape[1] != 1:
        raise ValueError(f"Expected single-channel BCHW tensor, got shape {tuple(x.shape)}")
    return x.contiguous()


class SmallFFTForwardPlan:
    """Cached spherical-PSF FFT forward plan for 256 -> 128 experiments.

    This follows the large-scale random point-flip code path: repeat the binary
    mask onto the optical grid, convolve with exp(i*k*r)/r by FFT on that grid,
    ifftshift, then crop the central ROI.
    """

    def __init__(
        self,
        optics: Any,
        *,
        mask_resolution: int = 256,
        target_resolution: int = 128,
        device: torch.device | str | None = None,
        optics_units: str | dict[str, str] | None = None,
        auto_unit: bool = False,
        forward_model: str = "spherical_circular_fft",
        allow_undersampling: bool = False,
    ) -> None:
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.mask_resolution = int(mask_resolution)
        self.target_resolution = int(target_resolution)
        self.forward_model = str(forward_model).lower()
        values, metadata = normalize_optics(optics, optics_units=optics_units, auto_unit=auto_unit)
        self.metadata = metadata
        self.L0 = float(values["L0"])
        self.lamda = float(values["lamda"])
        self.distance = float(values["distance"])
        self.inter = max(1, int(round(values["inter_num"])))

        n_mask = self.mask_resolution
        n_target = self.target_resolution
        self.pixel_pitch = self.L0 / float(n_mask)
        self.dx = self.pixel_pitch / float(self.inter)
        self.n_up = n_mask * self.inter
        self.output_at_optical_resolution = n_target > n_mask
        if self.output_at_optical_resolution and n_target > self.n_up:
            raise ValueError(
                "target_resolution can be larger than mask_resolution only up to "
                f"mask_resolution * inter_num ({self.n_up}); got target_resolution={n_target}"
            )
        self.roi = n_target if self.output_at_optical_resolution else n_target * self.inter
        self.roi_start = (self.n_up - self.roi) // 2

        if self.forward_model in {
            "spherical",
            "spherical_circular_fft",
            "large_scale",
            "random_point_kernel_fft",
            "random_point_compatible_fft",
        }:
            axis = torch.linspace(
                -self.L0 / 2.0,
                self.L0 / 2.0,
                self.n_up,
                dtype=torch.float32,
                device=self.device,
            )
            yy, xx = torch.meshgrid(axis, axis, indexing="ij")
            k = 2.0 * math.pi / self.lamda
            r = torch.sqrt(xx.square() + yy.square() + self.distance ** 2)
            psf = torch.exp(1j * k * r) / r.clamp_min(1e-30)
            self.psf_fft = torch.fft.fft2(psf.to(torch.complex64), dim=(-2, -1))
            self.fft_shape = (self.n_up, self.n_up)
            self.metadata.update(
                {
                    "forward_model": "random_point_kernel_fft"
                    if self.forward_model in {"random_point_kernel_fft", "random_point_compatible_fft"}
                    else "spherical_circular_fft",
                    "mask_resolution": self.mask_resolution,
                    "target_resolution": self.target_resolution,
                    "pixel_pitch_m": self.pixel_pitch,
                    "dx_m": self.dx,
                    "optical_resolution": self.n_up,
                    "roi_resolution": self.roi,
                    "output_at_optical_resolution": self.output_at_optical_resolution,
                    "fft_shape": list(self.fft_shape),
                }
            )
            return

        if self.forward_model in {"angular_spectrum", "angular_spectrum_padded_fft", "asm"}:
            # The legacy spherical branch samples exp(i k r) in the spatial
            # domain and circularly convolves it. Its rapidly varying phase is
            # aliased at the legacy grid spacing. Angular-spectrum propagation
            # samples the exact free-space transfer function on the FFT
            # frequency grid instead. Zero-padding makes the finite-mask
            # boundary non-periodic over the observed central field of view.
            self.pad_factor = 2
            self.padded_resolution = self.n_up * self.pad_factor
            self.pad_before = (self.padded_resolution - self.n_up) // 2
            fy = torch.fft.fftfreq(
                self.padded_resolution,
                d=self.dx,
                dtype=torch.float32,
                device=self.device,
            ).view(-1, 1)
            fx = torch.fft.fftfreq(
                self.padded_resolution,
                d=self.dx,
                dtype=torch.float32,
                device=self.device,
            ).view(1, -1)
            spectral_argument = 1.0 - (self.lamda * fx).square() - (self.lamda * fy).square()
            wave_number = 2.0 * math.pi / self.lamda
            propagating = spectral_argument >= 0.0
            propagating_transfer = torch.exp(
                1j * wave_number * self.distance * torch.sqrt(spectral_argument.clamp_min(0.0))
            )
            evanescent_transfer = torch.exp(
                -wave_number * self.distance * torch.sqrt((-spectral_argument).clamp_min(0.0))
            ).to(torch.complex64)
            self.angular_spectrum_transfer = torch.where(
                propagating,
                propagating_transfer,
                evanescent_transfer,
            ).to(torch.complex64)
            self.fft_shape = (self.padded_resolution, self.padded_resolution)
            self.metadata.update(
                {
                    "forward_model": "angular_spectrum_padded_fft",
                    "mask_resolution": self.mask_resolution,
                    "target_resolution": self.target_resolution,
                    "pixel_pitch_m": self.pixel_pitch,
                    "dx_m": self.dx,
                    "optical_resolution": self.n_up,
                    "padded_optical_resolution": self.padded_resolution,
                    "padding_factor": self.pad_factor,
                    "roi_resolution": self.roi,
                    "output_at_optical_resolution": self.output_at_optical_resolution,
                    "fft_shape": list(self.fft_shape),
                }
            )
            return

        asf_n = n_mask * 2 * self.inter - 1 - ((n_mask - n_target) // 2 * self.inter)
        self.asf_n = max(1, int(asf_n))
        self.L_asf = self.asf_n * self.dx

        f_max = (self.L_asf / 2.0) / (
            self.lamda * math.sqrt((self.L_asf / 2.0) ** 2 + self.distance ** 2)
        )
        self.nyquist_dx = 1.0 / (2.0 * f_max)
        if self.dx > self.nyquist_dx * (1.0 + 1e-6) and not bool(allow_undersampling):
            raise RuntimeError("采样不足，需要增大 inter_num 或减小 pixel_pitch")

        x = torch.linspace(-self.L_asf / 2.0, self.L_asf / 2.0, self.asf_n, dtype=torch.float32, device=self.device)
        y = torch.linspace(self.L_asf / 2.0, -self.L_asf / 2.0, self.asf_n, dtype=torch.float32, device=self.device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        k = 2.0 * math.pi / self.lamda
        r = torch.sqrt(xx.square() + yy.square() + self.distance ** 2)
        asf = self.distance / (1j * self.lamda) * torch.exp(1j * k * r) / r.square().clamp_min(1e-30)
        asf = (asf / torch.sqrt(torch.sum(torch.abs(asf).square())).clamp_min(1e-12)).to(torch.complex64)

        fft_h = next_fast_len(self.n_up + self.asf_n - 1)
        fft_w = next_fast_len(self.n_up + self.asf_n - 1)
        self.fft_shape = (int(fft_h), int(fft_w))
        self.asf_fft = torch.fft.fft2(asf, s=self.fft_shape, dim=(-2, -1))
        self.same_start = (self.asf_n - 1) // 2
        self.metadata.update(
            {
                "forward_model": "ASF_linear_convolution_fft",
                "mask_resolution": self.mask_resolution,
                "target_resolution": self.target_resolution,
                "pixel_pitch_m": self.pixel_pitch,
                "dx_m": self.dx,
                "asf_n": self.asf_n,
                "optical_resolution": self.n_up,
                "roi_resolution": self.roi,
                "output_at_optical_resolution": self.output_at_optical_resolution,
                "fft_shape": list(self.fft_shape),
                "sampling": {"f_max": f_max, "nyquist_dx_m": self.nyquist_dx},
            }
        )

    def forward(self, mask: torch.Tensor | np.ndarray) -> torch.Tensor:
        mask_t = ensure_bchw(mask, device=self.device)
        if mask_t.shape[-2:] != (self.mask_resolution, self.mask_resolution):
            raise ValueError(
                f"Expected mask shape [...,{self.mask_resolution},{self.mask_resolution}], "
                f"got {tuple(mask_t.shape)}"
            )
        if self.inter > 1:
            mask_up = mask_t.repeat_interleave(self.inter, dim=-2).repeat_interleave(self.inter, dim=-1)
        else:
            mask_up = mask_t

        if self.forward_model in {
            "spherical",
            "spherical_circular_fft",
            "large_scale",
            "random_point_kernel_fft",
            "random_point_compatible_fft",
        }:
            field_full = torch.fft.ifftshift(
                torch.fft.ifft2(
                    torch.fft.fft2(mask_up.to(torch.complex64), dim=(-2, -1)) * self.psf_fft,
                    dim=(-2, -1),
                ),
                dim=(-2, -1),
            )
            rh = rw = self.roi_start
            field_roi = field_full[..., rh:rh + self.roi, rw:rw + self.roi]
        elif self.forward_model in {"angular_spectrum", "angular_spectrum_padded_fft", "asm"}:
            padded_mask = F.pad(
                mask_up,
                (
                    self.pad_before,
                    self.padded_resolution - self.n_up - self.pad_before,
                    self.pad_before,
                    self.padded_resolution - self.n_up - self.pad_before,
                ),
                mode="constant",
                value=0.0,
            )
            field_full = torch.fft.ifft2(
                torch.fft.fft2(padded_mask.to(torch.complex64), dim=(-2, -1)) * self.angular_spectrum_transfer,
                dim=(-2, -1),
            )
            start = self.pad_before + self.roi_start
            field_roi = field_full[..., start:start + self.roi, start:start + self.roi]
        else:
            mask_fft = torch.fft.fft2(mask_up.to(torch.complex64), s=self.fft_shape, dim=(-2, -1))
            field_full = torch.fft.ifft2(mask_fft * self.asf_fft, s=self.fft_shape, dim=(-2, -1))

            sh = sw = self.same_start
            field_same = field_full[..., sh:sh + self.n_up, sw:sw + self.n_up]
            rh = rw = self.roi_start
            field_roi = field_same[..., rh:rh + self.roi, rw:rw + self.roi]

        intensity = field_roi.real.square() + field_roi.imag.square()
        intensity = intensity / intensity.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        if self.inter > 1 and not self.output_at_optical_resolution:
            intensity = F.interpolate(
                intensity.to(torch.float32),
                size=(self.target_resolution, self.target_resolution),
                mode="area",
            )
            intensity = intensity / intensity.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        return intensity.to(torch.float32)


def light_forward_fft(
    mask: torch.Tensor | np.ndarray,
    optics: Any,
    *,
    mask_resolution: int = 256,
    target_resolution: int = 128,
    optics_units: str | dict[str, str] | None = None,
    auto_unit: bool = False,
    forward_model: str = "spherical_circular_fft",
    return_metadata: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    device = mask.device if torch.is_tensor(mask) else None
    plan = SmallFFTForwardPlan(
        optics,
        mask_resolution=mask_resolution,
        target_resolution=target_resolution,
        device=device,
        optics_units=optics_units,
        auto_unit=auto_unit,
        forward_model=forward_model,
    )
    out = plan.forward(mask)
    if return_metadata:
        return out, dict(plan.metadata)
    return out


def center_pad_or_crop_to_size(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    out_h, out_w = int(size[0]), int(size[1])
    h, w = image.shape[-2:]
    y0 = max(0, (h - out_h) // 2)
    x0 = max(0, (w - out_w) // 2)
    cropped = image[..., y0:y0 + min(h, out_h), x0:x0 + min(w, out_w)]
    pad_h = out_h - cropped.shape[-2]
    pad_w = out_w - cropped.shape[-1]
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return F.pad(cropped, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)


def make_raw_asf_kernel(
    optics: Any,
    *,
    mask_resolution: int = 256,
    target_resolution: int = 128,
    device: torch.device | str | None = None,
    optics_units: str | dict[str, str] | None = None,
) -> torch.Tensor:
    cfg = OpticsConfig.from_any(
        optics,
        mask_resolution=mask_resolution,
        target_resolution=target_resolution,
        optics_units=optics_units,
    )
    dev = torch.device(device) if device is not None else torch.device("cpu")
    n = int(cfg.mask_resolution)
    l0 = float(cfg.L0)
    x = torch.linspace(-l0 / 2.0, l0 / 2.0, n, dtype=torch.float32, device=dev)
    y = torch.linspace(l0 / 2.0, -l0 / 2.0, n, dtype=torch.float32, device=dev)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    k = 2.0 * math.pi / float(cfg.lamda)
    r = torch.sqrt(xx.square() + yy.square() + float(cfg.distance) ** 2)
    asf = float(cfg.distance) / (1j * float(cfg.lamda)) * torch.exp(1j * k * r) / r.square().clamp_min(1e-30)
    return (asf / torch.sqrt(torch.sum(torch.abs(asf).square())).clamp_min(1e-12)).to(torch.complex64)


def make_spherical_psf_kernel(
    optics: Any,
    *,
    optical_resolution: int,
    mask_resolution: int = 256,
    target_resolution: int = 128,
    device: torch.device | str | None = None,
    optics_units: str | dict[str, str] | None = None,
) -> torch.Tensor:
    cfg = OpticsConfig.from_any(
        optics,
        mask_resolution=mask_resolution,
        target_resolution=target_resolution,
        optics_units=optics_units,
    )
    dev = torch.device(device) if device is not None else torch.device("cpu")
    axis = torch.linspace(-float(cfg.L0) / 2.0, float(cfg.L0) / 2.0, int(optical_resolution), dtype=torch.float32, device=dev)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    k = 2.0 * math.pi / float(cfg.lamda)
    r = torch.sqrt(xx.square() + yy.square() + float(cfg.distance) ** 2)
    return (torch.exp(1j * k * r) / r.clamp_min(1e-30)).to(torch.complex64)


@torch.no_grad()
def wiener_initial_probability(
    target: torch.Tensor | np.ndarray,
    optics: Any,
    *,
    mask_resolution: int = 256,
    target_resolution: int = 128,
    snr: float = 10.0,
    optics_units: str | dict[str, str] | None = None,
    forward_model: str = "spherical_circular_fft",
) -> torch.Tensor:
    target_t = ensure_bchw(target)
    device = target_t.device
    model = str(forward_model).lower()
    if model in {"spherical", "spherical_circular_fft", "large_scale"}:
        cfg = OpticsConfig.from_any(
            optics,
            mask_resolution=mask_resolution,
            target_resolution=target_resolution,
            optics_units=optics_units,
        )
        inter = max(1, int(cfg.inter_num))
        optical_resolution = int(mask_resolution) * inter
        psf = make_spherical_psf_kernel(
            optics,
            optical_resolution=optical_resolution,
            mask_resolution=mask_resolution,
            target_resolution=target_resolution,
            device=device,
            optics_units=optics_units,
        )
        if int(target_resolution) > int(mask_resolution):
            target_up = center_pad_or_crop_to_size(
                target_t.to(torch.float32),
                (optical_resolution, optical_resolution),
            )[:, 0]
        else:
            target_up = center_pad_or_crop_to_size(
                target_t.to(torch.float32).repeat_interleave(inter, dim=-2).repeat_interleave(inter, dim=-1),
                (optical_resolution, optical_resolution),
            )[:, 0]
        h_f = torch.fft.fft2(psf, dim=(-2, -1))
        target_f = torch.fft.fft2(torch.fft.fftshift(target_up, dim=(-2, -1)).to(torch.complex64), dim=(-2, -1))
        inv = torch.conj(h_f) / (torch.abs(h_f).square() + 1.0 / max(float(snr), 1e-6))
        amp_up = torch.abs(torch.fft.ifft2(target_f * inv, dim=(-2, -1))).to(torch.float32)
        if inter > 1:
            amp = F.interpolate(amp_up[:, None], size=(mask_resolution, mask_resolution), mode="area")[:, 0]
        else:
            amp = amp_up
    else:
        asf = make_raw_asf_kernel(
            optics,
            mask_resolution=mask_resolution,
            target_resolution=target_resolution,
            device=device,
            optics_units=optics_units,
        )
        target_mask = center_pad_or_crop_to_size(target_t.to(torch.float32), (mask_resolution, mask_resolution))[:, 0]
        h_f = torch.fft.fft2(torch.fft.ifftshift(asf), s=(mask_resolution, mask_resolution), dim=(-2, -1))
        target_f = torch.fft.fft2(target_mask.to(torch.complex64), s=(mask_resolution, mask_resolution), dim=(-2, -1))
        inv = torch.conj(h_f) / (torch.abs(h_f).square() + 1.0 / max(float(snr), 1e-6))
        amp = torch.abs(torch.fft.ifft2(target_f * inv, s=(mask_resolution, mask_resolution), dim=(-2, -1))).to(torch.float32)
    lo = amp.flatten(1).quantile(0.02, dim=1).view(-1, 1, 1)
    hi = amp.flatten(1).quantile(0.98, dim=1).view(-1, 1, 1)
    prob = ((amp - lo) / (hi - lo).clamp_min(1e-6)).clamp(1e-4, 1.0 - 1e-4)
    return prob[:, None]


@torch.no_grad()
def wiener_initial_logits(
    target: torch.Tensor | np.ndarray,
    optics: Any,
    *,
    mask_resolution: int = 256,
    target_resolution: int = 128,
    snr: float = 10.0,
    logit_clip: float = 12.0,
    optics_units: str | dict[str, str] | None = None,
    forward_model: str = "spherical_circular_fft",
) -> torch.Tensor:
    prob = wiener_initial_probability(
        target,
        optics,
        mask_resolution=mask_resolution,
        target_resolution=target_resolution,
        snr=snr,
        optics_units=optics_units,
        forward_model=forward_model,
    )
    return torch.logit(prob.clamp(1e-4, 1.0 - 1e-4)).clamp(-float(logit_clip), float(logit_clip))


def tau_at_step(step: int | float, steps: int, tau_start: float, tau_end: float) -> float:
    decay = -math.log(max(float(tau_end), 1e-6) / max(float(tau_start), 1e-6)) / max(int(steps), 1)
    return max(float(tau_end), float(tau_start) * math.exp(-decay * float(step)))


def cosine_score_image(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_t = ensure_bchw(pred, device=pred.device if torch.is_tensor(pred) else None)
    target_t = ensure_bchw(target, device=pred_t.device)
    if target_t.shape[-2:] != pred_t.shape[-2:]:
        target_t = F.interpolate(target_t.to(torch.float32), size=pred_t.shape[-2:], mode="area")
    pred_flat = pred_t.to(torch.float32).flatten(1)
    target_flat = target_t.to(torch.float32).flatten(1)
    numerator = (pred_flat * target_flat).sum(dim=1)
    denominator = torch.sqrt(pred_flat.square().sum(dim=1) * target_flat.square().sum(dim=1) + 1e-12)
    return numerator / denominator


def centered_cosine_score_image(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_t = ensure_bchw(pred, device=pred.device if torch.is_tensor(pred) else None)
    target_t = ensure_bchw(target, device=pred_t.device)
    if target_t.shape[-2:] != pred_t.shape[-2:]:
        target_t = F.interpolate(target_t.to(torch.float32), size=pred_t.shape[-2:], mode="area")
    pred_flat = pred_t.to(torch.float32).flatten(1)
    target_flat = target_t.to(torch.float32).flatten(1)
    pred_flat = pred_flat - pred_flat.mean(dim=1, keepdim=True)
    target_flat = target_flat - target_flat.mean(dim=1, keepdim=True)
    numerator = (pred_flat * target_flat).sum(dim=1)
    denominator = torch.sqrt(pred_flat.square().sum(dim=1) * target_flat.square().sum(dim=1) + 1e-12)
    return numerator / denominator


def highpass_cosine_score_image(pred: torch.Tensor, target: torch.Tensor, *, kernel_size: int = 9) -> torch.Tensor:
    pred_t = ensure_bchw(pred, device=pred.device if torch.is_tensor(pred) else None)
    target_t = ensure_bchw(target, device=pred_t.device)
    if target_t.shape[-2:] != pred_t.shape[-2:]:
        target_t = F.interpolate(target_t.to(torch.float32), size=pred_t.shape[-2:], mode="area")
    k = max(3, int(kernel_size))
    if k % 2 == 0:
        k += 1
    pad = k // 2
    pred_hp = pred_t.to(torch.float32) - F.avg_pool2d(pred_t.to(torch.float32), k, stride=1, padding=pad)
    target_hp = target_t.to(torch.float32) - F.avg_pool2d(target_t.to(torch.float32), k, stride=1, padding=pad)
    pred_flat = pred_hp.flatten(1)
    target_flat = target_hp.flatten(1)
    numerator = (pred_flat * target_flat).sum(dim=1)
    denominator = torch.sqrt(pred_flat.square().sum(dim=1) * target_flat.square().sum(dim=1) + 1e-12)
    return numerator / denominator


def gradient_cosine_score_image(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_t = ensure_bchw(pred, device=pred.device if torch.is_tensor(pred) else None)
    target_t = ensure_bchw(target, device=pred_t.device)
    if target_t.shape[-2:] != pred_t.shape[-2:]:
        target_t = F.interpolate(target_t.to(torch.float32), size=pred_t.shape[-2:], mode="area")
    pred_t = pred_t.to(torch.float32)
    target_t = target_t.to(torch.float32)
    pred_grad = torch.cat(
        [
            (pred_t[..., 1:, :] - pred_t[..., :-1, :]).flatten(1),
            (pred_t[..., :, 1:] - pred_t[..., :, :-1]).flatten(1),
        ],
        dim=1,
    )
    target_grad = torch.cat(
        [
            (target_t[..., 1:, :] - target_t[..., :-1, :]).flatten(1),
            (target_t[..., :, 1:] - target_t[..., :, :-1]).flatten(1),
        ],
        dim=1,
    )
    numerator = (pred_grad * target_grad).sum(dim=1)
    denominator = torch.sqrt(pred_grad.square().sum(dim=1) * target_grad.square().sum(dim=1) + 1e-12)
    return numerator / denominator


def save_gray_png(path: str | Path, array: np.ndarray | torch.Tensor) -> None:
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required to save PNG outputs") from exc
    if torch.is_tensor(array):
        arr = array.detach().cpu().numpy()
    else:
        arr = np.asarray(array)
    arr = np.squeeze(arr).astype(np.float32)
    arr = arr - float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if hi > 1e-12:
        arr = arr / hi
    img = (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(img, mode="L").save(str(path))
