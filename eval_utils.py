"""Evaluation + mapping helpers extracted from the notebook."""

from __future__ import annotations

from pathlib import Path

import numpy as np
# Imports
import os
import sys
import gc
import random
import importlib
import torch
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

import matplotlib.pyplot as plt
import tifffile
import torch
from tqdm.auto import tqdm
from datetime import datetime
import improved_trainer
from improved_trainer import net2D  
from get_options import trainer2D


from metrics_utils import squirrel_rsp_rse, squirrel_ssim, apply_affine, fit_affine_percentiles, fit_alpha_beta, add_metrics_labels_ssim



def load_and_standardize_mapping(mapping_csv: Path) -> pd.DataFrame:
    """
    Loads a mapping CSV and standardizes columns to:
      - img_path   (str)
      - img_file   (str)
      - gt_path    (str)
      - gt_exists  (bool)

    Accepts common variants:
      image path:  test_path, path, input_path, file_path, sofi_path, img_path
      image file:  test_file, file, filename, img_file
      gt path:     gt_path, gt
      gt exists:   gt_exists, exists, has_gt
    """
    mapping_csv = Path(mapping_csv)
    if not mapping_csv.exists():
        raise FileNotFoundError(f"Mapping CSV not found: {mapping_csv}")

    df = pd.read_csv(mapping_csv)

    def pick(cols):
        for c in cols:
            if c in df.columns:
                return c
        return None

    img_path_col = pick(["test_path", "path", "input_path", "file_path", "sofi_path", "img_path"])
    img_file_col = pick(["test_file", "file", "filename", "img_file"])
    gt_path_col  = pick(["gt_path", "gt"])
    gt_exists_col = pick(["gt_exists", "exists", "has_gt"])

    if img_path_col is None:
        raise ValueError(f"Could not find an image-path column in CSV. Columns found: {list(df.columns)}")
    if gt_path_col is None:
        raise ValueError(f"Could not find a gt-path column in CSV. Columns found: {list(df.columns)}")

    out = pd.DataFrame()
    out["img_path"] = df[img_path_col].astype(str)

    if img_file_col is not None:
        out["img_file"] = df[img_file_col].astype(str)
    else:
        out["img_file"] = out["img_path"].apply(lambda p: Path(p).name)

    out["gt_path"] = df[gt_path_col].astype(str)

    if gt_exists_col is not None:
        if df[gt_exists_col].dtype == bool:
            out["gt_exists"] = df[gt_exists_col]
        else:
            out["gt_exists"] = df[gt_exists_col].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        out["gt_exists"] = out["gt_path"].apply(lambda p: Path(p).exists())

    return out

from pathlib import Path

def process_test_set_with_model(df_mapping, sn2n_model, verbose=True, sigma=1.0):
    """
    Process all test images: make predictions and calculate SQUIRREL + SSIM metrics.

    This version is robust to different column names in df_mapping.
    It standardizes to:
        - img_path
        - img_file  (derived from img_path if needed)
        - gt_path
        - gt_exists
    """

    import numpy as np
    import pandas as pd
    import tifffile
    from pathlib import Path

    # standardize mapping DataFrame
    def _standardize_df_mapping(df: pd.DataFrame) -> pd.DataFrame:
        cols = list(df.columns)

        # If already standardized, just copy and return
        required = {"img_path", "img_file", "gt_path", "gt_exists"}
        if required.issubset(df.columns):
            return df.copy()

        def pick(candidates):
            for c in candidates:
                if c in df.columns:
                    return c
            return None

        img_path_col = pick(["img_path", "test_path", "path", "input_path", "file_path", "sofi_path"])
        gt_path_col  = pick(["gt_path", "gt", "gtfile", "gt_file"])
        gt_exists_col = pick(["gt_exists", "exists", "has_gt"])

        if img_path_col is None:
            raise ValueError(
                f"Could not find an image-path column in df_mapping. "
                f"Available columns: {cols}"
            )
        if gt_path_col is None:
            raise ValueError(
                f"Could not find a GT-path column in df_mapping. "
                f"Available columns: {cols}"
            )

        out = pd.DataFrame()
        out["img_path"] = df[img_path_col].astype(str)
        out["img_file"] = out["img_path"].apply(lambda p: Path(p).name)
        out["gt_path"] = df[gt_path_col].astype(str)

        if gt_exists_col is not None:
            s = df[gt_exists_col]
            if s.dtype == bool:
                out["gt_exists"] = s
            else:
                out["gt_exists"] = s.astype(str).str.lower().isin(["true", "1", "yes", "y"])
        else:
            # fallback: GT exists if path is non-empty
            out["gt_exists"] = out["gt_path"].astype(str).str.strip() != ""

        return out

    df_std = _standardize_df_mapping(df_mapping)

    # main evaluation loop
    results = []

    # Filter rows that actually have GT
    valid_pairs = df_std[df_std["gt_exists"] == True].copy()

    if verbose:
        print(f"Processing {len(valid_pairs)} test-GT pairs...")

    # If the SN2N wrapper has an underlying torch model, put it in eval()
    if hasattr(sn2n_model, "model") and hasattr(sn2n_model.model, "eval"):
        sn2n_model.model.eval()

    for idx, row in valid_pairs.iterrows():
        test_path = Path(row["img_path"])
        test_file = row["img_file"]
        gt_path = Path(row["gt_path"])

        try:
            # Load test image (SOFI+RL)
            test_img = tifffile.imread(test_path)
            if test_img.ndim > 2:
                test_img = test_img[0]

            # Load GT
            gt_img = tifffile.imread(gt_path)
            if gt_img.ndim > 2:
                gt_img = gt_img[0]

            # SN2N prediction
            pred_t = sn2n_model.test(test_img)  # the wrapper expects a 2D np array
            pred = np.squeeze(pred_t.detach().cpu().numpy() if hasattr(pred_t, "detach") else pred_t)

            # Affine fit on GT; test/pred already normalized upstream
            lo, hi = fit_affine_percentiles(gt_img)
            gt_norm = apply_affine(gt_img, lo, hi).astype(np.float32)
            test_norm = test_img.astype(np.float32)
            pred_norm = pred.astype(np.float32)

            # SQUIRREL metrics
            sofi_s_rsp, sofi_s_rse, sofi_a, sofi_b = squirrel_rsp_rse(test_norm, gt_norm)
            sn2n_s_rsp, sn2n_s_rse, sn2n_a, sn2n_b = squirrel_rsp_rse(pred_norm, gt_norm)

            # SSIM metrics
            sofi_ssim, sofi_ssim_a, sofi_ssim_b = squirrel_ssim(
                test_norm, gt_norm, intensity_fit=True, clip_to_gt=True
            )
            sn2n_ssim, sn2n_ssim_a, sn2n_ssim_b = squirrel_ssim(
                pred_norm, gt_norm, intensity_fit=True, clip_to_gt=True
            )

            results.append({
                "test_file": test_file,
                "test_path": str(test_path),
                "gt_path": str(gt_path),
                "test_shape": test_img.shape,
                "gt_shape": gt_img.shape,
                # SQUIRREL
                "sofi_s_rsp": sofi_s_rsp,
                "sofi_s_rse": sofi_s_rse,
                "sofi_alpha": sofi_a,
                "sofi_beta": sofi_b,
                "sn2n_s_rsp": sn2n_s_rsp,
                "sn2n_s_rse": sn2n_s_rse,
                "sn2n_alpha": sn2n_a,
                "sn2n_beta": sn2n_b,
                # SSIM
                "sofi_ssim": sofi_ssim,
                "sofi_ssim_alpha": sofi_ssim_a,
                "sofi_ssim_beta": sofi_ssim_b,
                "sn2n_ssim": sn2n_ssim,
                "sn2n_ssim_alpha": sn2n_ssim_a,
                "sn2n_ssim_beta": sn2n_ssim_b,
                # Improvements
                "s_rsp_improvement": sn2n_s_rsp - sofi_s_rsp,
                "s_rse_improvement": sofi_s_rse - sn2n_s_rse,
                "ssim_improvement": sn2n_ssim - sofi_ssim,
                "status": "success",
            })

            if verbose and (len(results) % 10 == 0):
                print(f"  Processed {len(results)}/{len(valid_pairs)}...")

        except Exception as e:
            results.append({
                "test_file": test_file,
                "test_path": str(test_path),
                "gt_path": str(gt_path),
                "status": f"error: {e}",
            })
            if verbose:
                print(f"  Error processing {test_file}: {e}")

    df_results = pd.DataFrame(results)

    if verbose:
        print(f"\nCompleted: {len(results)} images processed")
        successful = df_results[df_results["status"] == "success"]

        if len(successful) > 0:
            print(f"\nAverage SQUIRREL metrics across test set (sigma={sigma}):")
            print(f"  SOFI+RL - RSP: {successful['sofi_s_rsp'].mean():.4f}, "
                  f"RSE: {successful['sofi_s_rse'].mean():.6f}")
            print(f"  SN2N    - RSP: {successful['sn2n_s_rsp'].mean():.4f}, "
                  f"RSE: {successful['sn2n_s_rse'].mean():.6f}")

            print(f"\nAverage SSIM across test set:")
            print(f"  SOFI+RL - SSIM: {successful['sofi_ssim'].mean():.4f}")
            print(f"  SN2N    - SSIM: {successful['sn2n_ssim'].mean():.4f}")

            print(f"\nAverage improvements (SQUIRREL/SSIM):")
            print(f"  RSP: {successful['s_rsp_improvement'].mean():+.4f}")
            print(f"  RSE: {successful['s_rse_improvement'].mean():+.6f} (positive = better)")
            print(f"  SSIM: {successful['ssim_improvement'].mean():+.4f} (positive = better)")

    return df_results




def process_val2_with_model(df_val2_std: pd.DataFrame, sn2n_model, verbose: bool = True) -> pd.DataFrame:
    """
    Evaluate SQUIRREL RSP/RSE + SSIM on VAL2.
    df_val2_std must have: img_path, img_file, gt_path, gt_exists
    """
    results = []
    valid = df_val2_std[df_val2_std["gt_exists"] == True].copy()

    if verbose:
        print(f"[val2] Processing {len(valid)} image-GT pairs...")

    # Put underlying torch model in eval mode if present
    if hasattr(sn2n_model, "model") and hasattr(sn2n_model.model, "eval"):
        sn2n_model.model.eval()

    rows = valid.itertuples(index=False)
    if verbose:
        rows = tqdm(list(rows), total=len(valid), desc="Eval val2", leave=False)

    for row in rows:
        img_path = Path(row.img_path)
        gt_path = Path(row.gt_path)
        img_file = row.img_file

        try:
            inp = tifffile.imread(img_path)
            if inp.ndim > 2:
                inp = inp[0]

            gt = tifffile.imread(gt_path)
            if gt.ndim > 2:
                gt = gt[0]

            pred_t = sn2n_model.test(inp)  # expects np 2D
            pred = np.squeeze(pred_t.detach().cpu().numpy())

            # Normalize GT via affine; assume inp/pred already normalized upstream
            lo, hi = fit_affine_percentiles(gt)
            gt_norm = apply_affine(gt, lo, hi).astype(np.float32)
            inp_norm = inp.astype(np.float32)
            pred_norm = pred.astype(np.float32)

            # SQUIRREL metrics
            inp_rsp, inp_rse, inp_a, inp_b = squirrel_rsp_rse(inp_norm, gt_norm)
            pred_rsp, pred_rse, pred_a, pred_b = squirrel_rsp_rse(pred_norm, gt_norm)

            # SSIM metrics
            inp_ssim, inp_ssim_a, inp_ssim_b = squirrel_ssim(inp_norm, gt_norm, intensity_fit=True, clip_to_gt=True)
            pred_ssim, pred_ssim_a, pred_ssim_b = squirrel_ssim(pred_norm, gt_norm, intensity_fit=True, clip_to_gt=True)

            results.append({
                "split": "val2",
                "file": str(img_file),
                "img_path": str(img_path),
                "gt_path": str(gt_path),

                "sofi_s_rsp": float(inp_rsp),
                "sofi_s_rse": float(inp_rse),
                "sofi_ssim": float(inp_ssim),

                "sn2n_s_rsp": float(pred_rsp),
                "sn2n_s_rse": float(pred_rse),
                "sn2n_ssim": float(pred_ssim),

                # Improvements (+ better)
                "rsp_improvement": float(pred_rsp - inp_rsp),
                "rse_improvement": float(inp_rse - pred_rse),
                "ssim_improvement": float(pred_ssim - inp_ssim),

                "status": "success",
            })

        except Exception as e:
            results.append({
                "split": "val2",
                "file": str(img_file),
                "img_path": str(img_path),
                "gt_path": str(gt_path),
                "status": f"error: {e}",
            })

    df_results = pd.DataFrame(results)

    if verbose:
        df_s = df_results[df_results["status"] == "success"]
        print(f"[val2] Done: {len(df_results)} processed | {len(df_s)} success | {len(df_results) - len(df_s)} errors")
        if len(df_s) > 0:
            print(f"[val2] Means:")
            print(f"  SOFI+RL  RSP {df_s['sofi_s_rsp'].mean():.4f} | RSE {df_s['sofi_s_rse'].mean():.6f} | SSIM {df_s['sofi_ssim'].mean():.4f}")
            print(f"  SN2N     RSP {df_s['sn2n_s_rsp'].mean():.4f} | RSE {df_s['sn2n_s_rse'].mean():.6f} | SSIM {df_s['sn2n_ssim'].mean():.4f}")
            print(f"  Improve  RSP {df_s['rsp_improvement'].mean():+.4f} | RSE {df_s['rse_improvement'].mean():+.6f} | SSIM {df_s['ssim_improvement'].mean():+.4f}")

    return df_results

def summarize_metrics(df_results: pd.DataFrame) -> Optional[dict]:
    df_s = df_results[df_results["status"] == "success"]
    if len(df_s) == 0:
        return None
    return {
        "n_eval": int(len(df_s)),
        "sofi_rsp_mean": float(df_s["sofi_s_rsp"].mean()),
        "sofi_rse_mean": float(df_s["sofi_s_rse"].mean()),
        "sofi_ssim_mean": float(df_s["sofi_ssim"].mean()),
        "sn2n_rsp_mean": float(df_s["sn2n_s_rsp"].mean()),
        "sn2n_rse_mean": float(df_s["sn2n_s_rse"].mean()),
        "sn2n_ssim_mean": float(df_s["sn2n_ssim"].mean()),
        "rsp_impr_mean": float(df_s["rsp_improvement"].mean()),
        "rse_impr_mean": float(df_s["rse_improvement"].mean()),
        "ssim_impr_mean": float(df_s["ssim_improvement"].mean()),
    }

def score_from_metrics(m: dict, w_rsp=2.0, w_rse=2.0, w_ssim=1.0) -> float:
    return w_rsp * m["rsp_impr_mean"] + w_rse * m["rse_impr_mean"] + w_ssim * m["ssim_impr_mean"]

def hyperparameter_search_train_val1_eval_val2(
    train_patches_dir: str,
    val1_patches_dir: str,
    df_val2_std: pd.DataFrame,
    lr_list,
    sn2n_list,
    bs_list,
    lambda_grad_list=(0.0,),
    q_list=(0.8,),
    epochs=3,
    patience=10,
    score_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    save_dir: Optional[Path] = None,
    verbose_eval: bool = False,
):
    w_rsp, w_rse, w_ssim = score_weights
    runs = []

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    for lr in lr_list:
        for sn2n in sn2n_list:
            for bs in bs_list:
                for lam_g in lambda_grad_list:
                    for q in q_list:
                        print(f"\nRun: lr={lr}, sn2n={sn2n}, bs={bs}, lambda_grad={lam_g}, q={q}")
                        print("  - training on TRAIN")
                        print("  - val loss on VAL1")
                        print("  - metrics on VAL2")

                        model = net2D(
                            img_path=train_patches_dir,
                            lr=lr,
                            sn2n_loss=sn2n,
                            bs=bs,
                            epochs=epochs,
                            val_path=val1_patches_dir,   # VAL1 for loss during training
                            val_bs=bs,
                            early_stop=True,
                            patience=patience,
                            lambda_grad=lam_g,
                            q=q,
                        )

                        train_loss, val1_loss = model.train()

                        df_eval = process_val2_with_model(df_val2_std, model, verbose=verbose_eval)
                        m = summarize_metrics(df_eval)

                        run = {
                            "lr": lr,
                            "sn2n_loss": sn2n,
                            "bs": bs,
                            "lambda_grad": lam_g,
                            "q": q,
                            "train_loss": train_loss,
                            "val1_loss": val1_loss,
                        }

                        if m is None:
                            run["status"] = "no_successful_val2_pairs"
                        else:
                            run.update(m)
                            run["score"] = float(score_from_metrics(m, w_rsp=w_rsp, w_rse=w_rse, w_ssim=w_ssim))
                            run["status"] = "ok"

                        runs.append(run)

                        if save_dir is not None:
                            tag = f"lr{lr}_sn2n{sn2n}_bs{bs}_lg{lam_g}_q{q}".replace(".", "p")
                            df_eval.to_csv(save_dir / f"val2_metrics_{tag}.csv", index=False)

                        del model
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()

    df_runs = pd.DataFrame(runs).sort_values("score", ascending=False, na_position="last")
    return df_runs



__all__ = ['load_and_standardize_mapping','process_test_set_with_model','process_val2_with_model','summarize_metrics','score_from_metrics','hyperparameter_search_train_val1_eval_val2']

