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


def rl_deconvolve(
    image: np.ndarray,
    psf: np.ndarray,
    iters: int = RL_ITERS,
    floor: float = FLOOR
) -> np.ndarray:
    """
    Apply Richardson-Lucy deconvolution to an image with normalization.
    
    Parameters:
    -----------
    image : np.ndarray
        Input image to deconvolve
    psf : np.ndarray
        Point spread function
    iters : int
        Number of RL iterations (default: RL_ITERS)
    floor : float
        Small floor value to prevent division by zero (default: FLOOR)
    
    Returns:
    --------
    np.ndarray
        Deconvolved image in original intensity range
    """
    img = image.astype(np.float32)
    img_max = max(img.max(), floor)
    img_norm = np.clip(img / img_max, 0, None) + floor
    deconv = rl_ski(img_norm, psf, num_iter=iters, clip=False)
    return np.clip(deconv, 0, None) * img_max


import numpy as np
import random

def interchange_single(patch_size, img):
    img = img.copy()

    h, w = patch_size
    H, W = img.shape

    if h >= H or w >= W:
        return img

    x1 = random.randint(0, H - h)
    y1 = random.randint(0, W - w)
    x2 = random.randint(0, H - h)
    y2 = random.randint(0, W - w)

    patch1 = img[x1:x1+h, y1:y1+w].copy()
    patch2 = img[x2:x2+h, y2:y2+w].copy()

    img[x1:x1+h, y1:y1+w] = patch2
    img[x2:x2+h, y2:y2+w] = patch1

    return img


def interchange_multiple(patch_size, imga, imgb, ifdirect=False):
    imga = imga.copy()

    h, w = patch_size
    Ha, Wa = imga.shape
    Hb, Wb = imgb.shape

    if h >= min(Ha, Hb) or w >= min(Wa, Wb):
        return imga

    xa = random.randint(0, Ha - h)
    ya = random.randint(0, Wa - w)

    if ifdirect:
        patch = imgb[xa:xa+h, ya:ya+w]
    else:
        xb = random.randint(0, Hb - h)
        yb = random.randint(0, Wb - w)
        patch = imgb[xb:xb+h, yb:yb+w]

    imga[xa:xa+h, ya:ya+w] = patch
    return imga


def random_interchange(mode, patch_size, imga, imgb=None):
    """
    mode:
      0 = no P2P
      2 = single-image P2P
      3 = cross-image P2P
    """
    if mode == 0:
        return imga

    if imgb is None:
        mode = 2

    if mode == 2:
        return interchange_single(patch_size, imga)

    elif mode == 3:
        return interchange_multiple(patch_size, imga, imgb, ifdirect=False)

    else:
        return imga
    

def sliding_window_2d(img, patch_size, stride=64,
                      mode=0, filter_val=0):
    """
    Extract patches using sliding window + intensity thresholding.

    Parameters
    ----------
    img : 2D numpy array
        Input image (uint8/uint16/float).
    patch_size : tuple (H, W)
        Patch size.
    stride : int
        Sliding window step.
    mode : int
        0 = fixed threshold
        1 = mean(img) + filter_val
    filter_val : float
        Threshold offset.

    Returns
    -------
    patches : numpy array (N, H, W)
    """

    img = img.astype(np.float32)

    # normalize to 0–255 like original code
    if img.max() > 0:
        img = 255.0 * img / img.max()

    H, W = img.shape
    ph, pw = patch_size

    # threshold definition 
    if mode == 0:
        threshold = filter_val
    else:
        threshold = img.mean() + filter_val

    patches = []

    nx = (H - ph) // stride + 1
    ny = (W - pw) // stride + 1

    for i in range(nx):
        for j in range(ny):
            x = i * stride
            y = j * stride

            patch = img[x:x+ph, y:y+pw]

            # reject dark / empty patches
            if patch.sum() > ph * pw * threshold:
                patches.append(patch)

    if len(patches) == 0:
        return np.empty((0, ph, pw), dtype=np.float32)

    return np.stack(patches, axis=0)


def basic_augment(img, mode):
    """
    Apply flip/rotation augmentation.

    mode:
      0 = original
      1 = rot90 + flipud
      2 = flipud
      3 = fliplr
      4 = rot90 + fliplr
      5 = rot90
      6 = rot180
      7 = rot270
    """

    if mode == 0:
        return img

    elif mode == 1:
        return np.flipud(np.rot90(img, 1))

    elif mode == 2:
        return np.flipud(img)

    elif mode == 3:
        return np.fliplr(img)

    elif mode == 4:
        return np.fliplr(np.rot90(img, 1))

    elif mode == 5:
        return np.rot90(img, 1)

    elif mode == 6:
        return np.rot90(img, 2)

    elif mode == 7:
        return np.rot90(img, 3)

    else:
        return img



def run_pipeline(
    tiff_path: str,
    upscale: int = UPSCALE,
    fwhm: float = FWHM,
    rl_iters: int = RL_ITERS,
    floor: float = FLOOR,

    # --- augmentation controls ---
    p2p_mode: int = 0,
    p2p_patch_size: tuple = (128, 128),

    use_sliding: bool = False,
    slide_patch_size: tuple = (128, 128),
    slide_stride: int = 64,

    basic_aug_mode: Optional[int] = None,
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

    img = img.astype(np.float32)

    img = random_interchange(
        mode=p2p_mode,
        patch_size=p2p_patch_size,
        imga=img
    )
 
    if use_sliding:
        patches = sliding_window_2d(
            img,
            patch_size=slide_patch_size,
            stride=slide_stride
        )

        # if no valid patch, skip file
        if patches.shape[0] == 0:
            raise RuntimeError("No valid patches extracted by sliding window.")

        # randomly choose ONE patch for this pipeline pass
        img = patches[np.random.randint(0, patches.shape[0])]

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

    # Apply RL deconvolution to both upsampled views
    left_rl = rl_deconvolve(left_up, psf, iters=rl_iters, floor=floor)
    right_rl = rl_deconvolve(right_up, psf, iters=rl_iters, floor=floor)


    if basic_aug_mode is not None:
        left_rl = basic_augment(left_rl, basic_aug_mode)
        right_rl = basic_augment(right_rl, basic_aug_mode)

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

import numpy as np

def make_sn2n_sample(
    left_rl: np.ndarray,
    right_rl: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.5,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Concatenate left/right RL outputs into one (H, 2W) float32 array for SN2N training.

    Normalization:
    - Fit a single (p_low, p_high) on BOTH views (shared parameters)
    - Apply (x - lo) / (hi - lo) and clip to [0,1]
    """
    left = np.squeeze(left_rl).astype(np.float32)
    right = np.squeeze(right_rl).astype(np.float32)

    if left.ndim != 2 or right.ndim != 2:
        raise ValueError(f"Inputs must be 2D after squeeze; got left.ndim={left.ndim}, right.ndim={right.ndim}")
    if left.shape != right.shape:
        raise ValueError(f"Inputs must have same shape; got left={left.shape}, right={right.shape}")

    # Fit ONCE on both views combined (unified normalization)
    both = np.concatenate([left.ravel(), right.ravel()])
    lo = np.percentile(both, p_low)
    hi = np.percentile(both, p_high)
    if (hi - lo) < eps:
        hi = lo + eps

    left_n = (left - lo) / (hi - lo)
    right_n = (right - lo) / (hi - lo)

    left_n = np.clip(left_n, 0.0, 1.0)
    right_n = np.clip(right_n, 0.0, 1.0)

    sample = np.concatenate([left_n, right_n], axis=1).astype(np.float32)  # (H, 2W)
    return sample


def save_sn2n_sample(sample: np.ndarray, out_path: str | Path) -> None:
    """Save SN2N sample array to TIFF file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(out_path, sample.astype(np.float32))


def build_sn2n_sample(
    tiff_path: str,
    save_path: Optional[str] = None,
    p_low: float = 1,
    p_high: float = 99.5,

    p2p_mode: int = 0,
    basic_aug_mode: Optional[int] = None,
):
    res = run_pipeline(
    tiff_path,
    p2p_mode=p2p_mode,
    basic_aug_mode=basic_aug_mode,
    use_sliding=True,   # normally ON for training
)
    sample = make_sn2n_sample(res["left_rl"], res["right_rl"], p_low=p_low, p_high=p_high)
    if save_path is not None:
        save_sn2n_sample(sample, save_path)
    return sample, res
