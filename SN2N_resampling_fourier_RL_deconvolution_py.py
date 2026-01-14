#!/usr/bin/env python
"""
SOFI data preprocessing pipeline.
Functions for diagonal sampling, Fourier upsampling, and RL deconvolution.
"""

import numpy as np
import tifffile as tiff
from pathlib import Path
from typing import Dict, Optional

try:
    from skimage.restoration import richardson_lucy as rl_ski
except ImportError:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-image"])
    from skimage.restoration import richardson_lucy as rl_ski


# Default parameters
UPSCALE = 2
FWHM = 2.7
RL_ITERS = 10
FLOOR = 1e-3


def block2d(img2d: np.ndarray):
    """
    Create two views via checkerboard sampling:
      left  = (ul + dr)/2
      right = (ur + dl)/2
    where ul/ur/dr/dl are checkerboard sub-samples.
    """
    img2d = img2d.astype(np.float32)

    # force even height/width so the sub-samples align
    h, w = img2d.shape
    h_even, w_even = h - (h % 2), w - (w % 2)
    if (h_even != h) or (w_even != w):
        img2d = img2d[:h_even, :w_even]

    ul = img2d[0::2, 0::2]
    ur = img2d[0::2, 1::2]
    dr = img2d[1::2, 1::2]
    dl = img2d[1::2, 0::2]

    left = 0.5 * (ul + dr)
    right = 0.5 * (ur + dl)
    return left, right


def fourier_zoom(img2d: np.ndarray, scale: int = 2):
    """Fourier zero-pad to increase sampling density by `scale` (default 2x)."""
    img = img2d.astype(np.float32)
    h, w = img.shape
    H, W = int(h * scale), int(w * scale)

    # forward FFT and shift to center
    F = np.fft.fftshift(np.fft.fft2(img))

    # symmetric zero padding in frequency domain
    pad_h = H - h
    pad_w = W - w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    F_pad = np.pad(F, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant")

    # inverse FFT and scale to preserve intensity
    out = np.fft.ifft2(np.fft.ifftshift(F_pad))
    out = np.real(out) * (scale * scale)
    return out


def run_pipeline(
    tiff_path: str,
    upscale: int = UPSCALE,
    fwhm: float = FWHM,
    rl_iters: int = RL_ITERS,
    floor: float = FLOOR,
) -> Dict[str, np.ndarray]:
    """
    End-to-end preprocessing pipeline:
      1) Load TIFF (first frame if stack)
      2) Checkerboard split (block2d)
      3) Fourier zero-pad upsample by `upscale`
      4) RL deconvolution with Gaussian PSF
      
    Returns dict with all intermediate and final arrays.
    """
    path = Path(tiff_path)
    if not path.exists():
        raise FileNotFoundError(f"TIFF not found at {path}")

    img = tiff.imread(path)
    if img.ndim > 2:
        img = img[0]

    # Split into two views
    left, right = block2d(img)

    # Fourier upsample
    left_up = fourier_zoom(left, scale=upscale)
    right_up = fourier_zoom(right, scale=upscale)

    # Create PSF from FWHM
    sigma = fwhm / 2.355
    psf_size = int(np.ceil(sigma * 6))
    if psf_size % 2 == 0:
        psf_size += 1
    ax = np.arange(psf_size) - psf_size // 2
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    psf /= psf.sum()

    def rl_norm(view: np.ndarray, iters: int) -> np.ndarray:
        """Normalize, apply RL deconvolution, then rescale back."""
        v = view.astype(np.float32)
        vmax = max(v.max(), floor)
        v_n = np.clip(v / vmax, 0, None) + floor
        out = rl_ski(v_n, psf, num_iter=iters, clip=False)
        return np.clip(out, 0, None) * vmax

    left_rl = rl_norm(left_up, rl_iters)
    right_rl = rl_norm(right_up, rl_iters)

    return {
        "img": img,
        "left": left,
        "right": right,
        "left_up": left_up,
        "right_up": right_up,
        "left_rl": left_rl,
        "right_rl": right_rl,
        "psf": psf,
        "sigma": sigma,
        "psf_size": psf_size,
        "upscale": upscale,
        "fwhm": fwhm,
        "rl_iters": rl_iters,
        "floor": floor,
    }


# SN2N sample preparation functions:

def make_sn2n_sample(left_rl: np.ndarray, right_rl: np.ndarray) -> np.ndarray:
    """
    Concatenate left/right RL outputs into one (H, 2W) float32 array for SN2N training.

    - Squeezes singleton dims if present
    - Validates both are 2D and same shape
    - Shared-scale normalization to [0,1]: scale = max(left/right max, 1e-6)
    """
    left = np.squeeze(left_rl)
    right = np.squeeze(right_rl)

    if left.ndim != 2 or right.ndim != 2:
        raise ValueError(f"Inputs must be 2D after squeeze; got left.ndim={left.ndim}, right.ndim={right.ndim}")
    if left.shape != right.shape:
        raise ValueError(f"Inputs must have same shape; got left={left.shape}, right={right.shape}")

    # Shared normalization to keep relative scaling
    left = left.astype(np.float32)
    right = right.astype(np.float32)
    scale = max(float(left.max()), float(right.max()), 1e-6)
    left_n = left / scale
    right_n = right / scale

    sample = np.concatenate([left_n, right_n], axis=1)  # shape (H, 2W)
    return sample


def save_sn2n_sample(sample: np.ndarray, out_path: str | Path) -> None:
    """Save SN2N sample array to TIFF file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(out_path, sample.astype(np.float32))


def build_sn2n_sample(tiff_path: str, save_path: Optional[str] = None):
    """
    Run full pipeline then package RL outputs into a single (H, 2W) sample.

    Returns (sample, res) where sample is float32 (H, 2W) and res is the pipeline dict.
    If save_path is given, writes the sample to disk.
    """
    res = run_pipeline(tiff_path)
    sample = make_sn2n_sample(res["left_rl"], res["right_rl"])
    if save_path is not None:
        save_sn2n_sample(sample, save_path)
    return sample, res
