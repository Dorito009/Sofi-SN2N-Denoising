#!/usr/bin/env python
# coding: utf-8

# ## SOFI data preprocessing

# In[22]:


import numpy as np
get_ipython().run_line_magic('pip', 'install tifffile')
import tifffile as tiff
import matplotlib.pyplot as plt
from pathlib import Path
import glob, os

# Global configuration: update paths/params here
C2_PATH = Path(r"C:\Users\ntpar\Downloads\Capstone\data\07_Dens3000_b32_filtWilly\001_Training_\Noisy_SOFI_Results_sub500sl1fwhm3\1_100f__iter10_recon0_subseqlength100_start1_end100_20250828_1708\01_noisyvid__mctsofi_c2.tif")
UPSCALE = 2
FWHM = 2.7
RL_ITERS = 10
FLOOR = 1e-3


# ## Diagonal Sampling & Fourier upsampling
# Functions  (mostly taken from the github page)

# In[23]:


def block2d(img2d: np.ndarray):
    """
    Create two views:
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

# Load image
if not C2_PATH.exists():
    raise FileNotFoundError(f"TIFF not found at {C2_PATH}. Update C2_PATH to an existing file.")

img = tiff.imread(C2_PATH)
if img.ndim > 2:
    # if it’s a stack, take the first frame for testing
    img = img[0]

print("Loaded:", C2_PATH)
print("Original shape:", img.shape, "dtype:", img.dtype)

left, right = block2d(img)
print("After block2d left/right shape:", left.shape, right.shape)

left_up = fourier_zoom(left, scale=UPSCALE)
right_up = fourier_zoom(right, scale=UPSCALE)
print("After Fourier upsample left/right:", left_up.shape, right_up.shape)

# visualize
fig, axs = plt.subplots(2, 3, figsize=(12, 7))
axs[0,0].imshow(img, cmap="gray");      axs[0,0].set_title("Original c2")
axs[0,1].imshow(left, cmap="gray");     axs[0,1].set_title("Left (subsampled)")
axs[0,2].imshow(right, cmap="gray");    axs[0,2].set_title("Right (subsampled)")
axs[1,1].imshow(left_up, cmap="gray");  axs[1,1].set_title("Left Fourier-up 2x")
axs[1,2].imshow(right_up, cmap="gray"); axs[1,2].set_title("Right Fourier-up 2x")

# show difference to see if structure aligns
axs[1,0].imshow(np.abs(left_up - right_up), cmap="gray")
axs[1,0].set_title("|Left_up - Right_up|")

for ax in axs.ravel():
    ax.axis("off")
plt.tight_layout()
plt.show()


# ## Checking if fourier upsampling is correct

# In[24]:


# Diagnostics: check Fourier 2x quality
print("--- Fourier diagnostics (using left/left_up) ---")

# shape/summary stats
print("left shape:", left.shape, "left_up shape:", left_up.shape)
print("left mean/std:", float(np.mean(left)), float(np.std(left)))
print("left_up mean/std:", float(np.mean(left_up)), float(np.std(left_up)))

# downsample 2x-up back to original grid
left_down = left_up[::UPSCALE, ::UPSCALE]
mae = float(np.mean(np.abs(left_down - left)))
max_err = float(np.max(np.abs(left_down - left)))
print("downsampled-back shape:", left_down.shape)
print("MAE (left vs left_up[::2]):", mae)
print("Max err (left vs left_up[::2]):", max_err)

# frequency-domain check: log-magnitude spectra
F_orig = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(left))))
F_up = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(left_up))))

fig, axs = plt.subplots(2, 3, figsize=(12, 7))
axs[0,0].imshow(left, cmap="gray");        axs[0,0].set_title("Left (orig)")
axs[0,1].imshow(left_up, cmap="gray");     axs[0,1].set_title("Left 2x up")
axs[0,2].imshow(left_down, cmap="gray");   axs[0,2].set_title("Left 2x up -> 1x")
axs[1,0].imshow(np.abs(left - left_down), cmap="magma"); axs[1,0].set_title("|orig - down|")
axs[1,1].imshow(F_orig, cmap="magma");     axs[1,1].set_title("FFT log-mag (orig)")
axs[1,2].imshow(F_up, cmap="magma");       axs[1,2].set_title("FFT log-mag (2x)")
for ax in axs.ravel():
    ax.axis("off")
plt.tight_layout()
plt.show()


# ## RL-deconvolution

# In[25]:


# Richardson–Lucy via scikit-image on 2x upsampled views (normalized per-view)
try:
    from skimage.restoration import richardson_lucy as rl_ski
except ImportError:
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-image"])
    from skimage.restoration import richardson_lucy as rl_ski

# PSF from global FWHM
psf_sigma = FWHM / 2.355  # ~1.147
psf_size = int(np.ceil(psf_sigma * 6))  # cover ~±3 sigma
if psf_size % 2 == 0:
    psf_size += 1

ax = np.arange(psf_size) - psf_size // 2
xx, yy = np.meshgrid(ax, ax)
psf = np.exp(-(xx**2 + yy**2) / (2 * psf_sigma**2))
psf /= psf.sum()

rl_iters = RL_ITERS
floor = FLOOR

# normalize each view to [0,1] to make RL behave consistently
lu = left_up.astype(np.float32)
ru = right_up.astype(np.float32)
lu_max = max(lu.max(), floor)
ru_max = max(ru.max(), floor)
lu_n = np.clip(lu / lu_max, 0, None) + floor
ru_n = np.clip(ru / ru_max, 0, None) + floor

left_rl_ski = rl_ski(lu_n, psf, num_iter=rl_iters, clip=False)
right_rl_ski = rl_ski(ru_n, psf, num_iter=rl_iters, clip=False)

# clamp and rescale back to original dynamic range
left_rl_ski = np.clip(left_rl_ski, 0, None) * lu_max
right_rl_ski = np.clip(right_rl_ski, 0, None) * ru_max

print("scikit-image RL done:", "left", left_rl_ski.shape, "right", right_rl_ski.shape)
print(f"FWHM={FWHM} px, sigma={psf_sigma:.4f}, psf_size={psf_size}, iterations={rl_iters}")

fig, axs = plt.subplots(2, 3, figsize=(12, 7))
axs[0,0].imshow(left_up, cmap="gray");          axs[0,0].set_title("Left 2x up")
axs[0,1].imshow(left_rl_ski, cmap="gray");      axs[0,1].set_title(f"Left RL norm ({rl_iters} it)")
axs[0,2].imshow(np.abs(left_rl_ski - left_up), cmap="magma"); axs[0,2].set_title("|RL - up|")
axs[1,0].imshow(right_up, cmap="gray");         axs[1,0].set_title("Right 2x up")
axs[1,1].imshow(right_rl_ski, cmap="gray");     axs[1,1].set_title(f"Right RL norm ({rl_iters} it)")
axs[1,2].imshow(np.abs(right_rl_ski - right_up), cmap="magma"); axs[1,2].set_title("|RL - up|")
for ax in axs.ravel():
    ax.axis("off")
plt.tight_layout()
plt.show()


# # Pipeline-function for the pre-processing

# In[26]:


# Pipeline function:
from typing import Dict

def run_pipeline(
    tiff_path: str,
    upscale: int = UPSCALE,
    fwhm: float = FWHM,
    rl_iters: int = RL_ITERS,
    floor: float = FLOOR,
) -> Dict[str, np.ndarray]:
    """
    End-to-end pipeline:
      1) load TIFF (first frame if stack)
      2) checkerboard split (block2d)
      3) Fourier zero-pad upsample by `upscale`
      4) RL deconv with Gaussian PSF (default FWHM set up top)
      Returns arrays for reuse in a downstream pipeline.
    """
    path = Path(tiff_path)
    if not path.exists():
        raise FileNotFoundError(f"TIFF not found at {path}")

    img = tiff.imread(path)
    if img.ndim > 2:
        img = img[0]

    # split
    left, right = block2d(img)

    # Fourier upsample
    left_up = fourier_zoom(left, scale=upscale)
    right_up = fourier_zoom(right, scale=upscale)

    # PSF from FWHM
    sigma = fwhm / 2.355
    psf_size = int(np.ceil(sigma * 6))
    if psf_size % 2 == 0:
        psf_size += 1
    ax = np.arange(psf_size) - psf_size // 2
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    psf /= psf.sum()

    def rl_norm(view: np.ndarray, iters: int) -> np.ndarray:
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

# Example usage (set your path):
res = run_pipeline(str(C2_PATH))
left_rl = res["left_rl"]
right_rl = res["right_rl"]


# In[27]:


# Run pipeline and show RL outputs
path_to_use = str(C2_PATH)
res = run_pipeline(path_to_use)
left_rl = res["left_rl"]
right_rl = res["right_rl"]

fig, axs = plt.subplots(1, 3, figsize=(14, 5))
axs[0].imshow(left_rl, cmap="gray");  axs[0].set_title("Left RL")
axs[1].imshow(right_rl, cmap="gray"); axs[1].set_title("Right RL")
axs[2].imshow(np.abs(left_rl - right_rl), cmap="magma"); axs[2].set_title("|Left RL - Right RL|")
for ax in axs: ax.axis("off")
plt.tight_layout()
plt.show()


# In[28]:


# Pack left/right RL into (H, 2W) sample for net2D.load_batch2d()
import tifffile as tiff

def make_sn2n_sample(left_rl: np.ndarray, right_rl: np.ndarray) -> np.ndarray:
    """Concatenate left/right RL outputs into one (H, 2W) float32 array.

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

    # shared normalization to keep relative scaling
    left = left.astype(np.float32)
    right = right.astype(np.float32)
    scale = max(float(left.max()), float(right.max()), 1e-6)
    left_n = left / scale
    right_n = right / scale

    sample = np.concatenate([left_n, right_n], axis=1)  # shape (H, 2W)
    return sample


def save_sn2n_sample(sample: np.ndarray, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(out_path, sample.astype(np.float32))


# Sanity check: pack then split
_sample = make_sn2n_sample(left_rl, right_rl)
H, W2 = _sample.shape
W = W2 // 2
_left_back = _sample[:, :W]
_right_back = _sample[:, W:]
assert _left_back.shape == left_rl.squeeze().shape
assert _right_back.shape == right_rl.squeeze().shape
print("sample shape (H,2W):", _sample.shape)
print("left/right restored shapes:", _left_back.shape, _right_back.shape)


# In[29]:


# Combined helper: run pipeline -> pack to (H, 2W) sample
from typing import Optional

def build_sn2n_sample(tiff_path: str, save_path: Optional[str] = None):
    """Run pipeline then package RL outputs into a single (H, 2W) sample.

    Returns (sample, res) where sample is float32 (H, 2W) and res is the pipeline dict.
    If save_path is given, writes the sample via save_sn2n_sample.
    """
    res = run_pipeline(tiff_path)
    sample = make_sn2n_sample(res["left_rl"], res["right_rl"])
    if save_path is not None:
        save_sn2n_sample(sample, save_path)
    return sample, res

# Example: build and optionally save
sample, res = build_sn2n_sample(str(C2_PATH))
print("Built sample shape:", sample.shape)


# In[ ]:




