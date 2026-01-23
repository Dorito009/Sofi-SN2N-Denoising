"""SQUIRREL-style metric helpers extracted from the notebook."""
from __future__ import annotations
import numpy as np
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim

EPS = 1e-6
P_LOW = 1.0
P_HIGH = 99.5

def fit_affine_percentiles(x, p_low=P_LOW, p_high=P_HIGH):
    x = x.astype(np.float32)
    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)
    if hi - lo < EPS:
        hi = lo + EPS
    return float(lo), float(hi)

def apply_affine(x, lo, hi):
    x = x.astype(np.float32)
    return np.clip((x - lo) / (hi - lo + EPS), 0.0, 1.0)

def fit_alpha_beta(x, y, eps=1e-12):
    """Fit y ≈ alpha*x + beta in least squares sense."""
    x = x.astype(np.float64).ravel()
    y = y.astype(np.float64).ravel()
    A = np.stack([x, np.ones_like(x)], axis=1)
    (alpha, beta), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(alpha), float(beta)

def squirrel_rsp_rse(sr, gt, eps=1e-12, intensity_fit=True):
    """Compare SR directly to GT: affine match (optional), then Pearson + RMSE."""
    sr = sr.astype(np.float32, copy=False)
    gt = gt.astype(np.float32, copy=False)

    if intensity_fit:
        alpha, beta = fit_alpha_beta(sr, gt)
        pred = alpha * sr + beta
    else:
        alpha, beta = 1.0, 0.0
        pred = sr

    a = pred.ravel().astype(np.float64)
    b = gt.ravel().astype(np.float64)

    if np.std(a) < eps or np.std(b) < eps:
        rsp = 0.0
    else:
        rsp, _ = pearsonr(a, b)
        rsp = float(rsp)

    rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
    return rsp, rmse, alpha, beta

def squirrel_ssim(sr, gt, eps=1e-12, intensity_fit=True, clip_to_gt=True):
    """SSIM(SR, GT) with optional affine match y≈alpha*x+beta."""
    sr = sr.astype(np.float32, copy=False)
    gt = gt.astype(np.float32, copy=False)

    if intensity_fit:
        alpha, beta = fit_alpha_beta(sr, gt)
        pred = alpha * sr + beta
    else:
        alpha, beta = 1.0, 0.0
        pred = sr

    if clip_to_gt:
        pred = np.clip(pred, gt.min(), gt.max())

    data_range = float(gt.max() - gt.min())
    if data_range < eps:
        return 0.0, alpha, beta

    val = ssim(gt, pred, data_range=data_range)
    return float(val), alpha, beta

def add_metrics_labels_ssim(ax, rsp, rse, ssim_val=None):
    text = f"RSP: {rsp:.4f}\nRSE: {rse:.6f}"
    if ssim_val is not None:
        text += f"\nSSIM: {ssim_val:.4f}"
    ax.text(0.95, 0.05, text, color='white', transform=ax.transAxes,
            fontsize=10, ha='right', va='bottom', bbox=dict(facecolor='black', alpha=0.8))

__all__ = ['EPS','P_LOW','P_HIGH','fit_affine_percentiles','apply_affine','fit_alpha_beta','squirrel_rsp_rse','squirrel_ssim','add_metrics_labels_ssim']
