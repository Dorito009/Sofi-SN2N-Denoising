# -*- coding: utf-8 -*-

import os
import datetime
from glob import glob
from pathlib import Path
from typing import Optional
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from skimage.io import imread as sk_imread
from tqdm import tqdm

from models import Unet_2d



# Cached Sobel kernels (per device/dtype)
_BASE_SOBEL_X = torch.tensor(
    [[[-1, 0, 1],
      [-2, 0, 2],
      [-1, 0, 1]]],
    dtype=torch.float32
).view(1, 1, 3, 3)

_BASE_SOBEL_Y = torch.tensor(
    [[[-1, -2, -1],
      [ 0,  0,  0],
      [ 1,  2,  1]]],
    dtype=torch.float32
).view(1, 1, 3, 3)

_SOBEL_CACHE = {}  


def _sobel_kernels_like(x: torch.Tensor):
    key = (x.device, x.dtype)
    if key not in _SOBEL_CACHE:
        _SOBEL_CACHE[key] = (
            _BASE_SOBEL_X.to(device=x.device, dtype=x.dtype),
            _BASE_SOBEL_Y.to(device=x.device, dtype=x.dtype),
        )
    return _SOBEL_CACHE[key]


def gradient_loss(pred, target, proxy, tau_thr, softness=0.1, blur_kernel_size=3):
    """Sobel gradient L1 with a soft mask from a blurred proxy image.

    pred, target: Bx1xHxW
    proxy: same shape, typically 0.5*(inputs+labels)
    tau_thr: per-batch threshold (Bx1x1x1) computed from the proxy
    """
    sobel_x, sobel_y = _sobel_kernels_like(pred)

    grad_px = F.conv2d(pred, sobel_x, padding=1)
    grad_py = F.conv2d(pred, sobel_y, padding=1)
    grad_tx = F.conv2d(target, sobel_x, padding=1)
    grad_ty = F.conv2d(target, sobel_y, padding=1)

    pad = blur_kernel_size // 2
    proxy_blur = F.avg_pool2d(proxy, kernel_size=blur_kernel_size, stride=1, padding=pad)

    # Soft mask avoids flicker from tiny intensity changes.
    scale = softness * torch.clamp(torch.abs(tau_thr), min=1e-6)
    mask = torch.sigmoid((proxy_blur - tau_thr) / scale)

    return (
        F.l1_loss(grad_px * mask, grad_tx * mask) +
        F.l1_loss(grad_py * mask, grad_ty * mask)
    )


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.best = float("inf")
        self.counter = 0
        self.stop = False

    def step(self, metric: float):
        if metric is None:
            return
        if metric < self.best:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


class net2D:
    def __init__(
        self,
        img_path,
        sn2n_loss=1,
        bs=32,
        lr=2e-4,
        epochs=100,
        img_patch="128",
        if_alr=True,
        val_path=None,
        val_bs=None,
        early_stop: bool = False,
        patience: int = 10,
        lambda_grad: float = 0.1,
        q: float = 0.80,
        use_amp: Optional[bool] = None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_path = img_path
        self.parent_dir = os.path.dirname(img_path)

        candidate_path = img_path
        if len(glob(os.path.join(candidate_path, "*.tif"))) == 0:
            legacy_path = os.path.join(self.parent_dir, "datasets")
            candidate_path = legacy_path

        self.dataset_path = candidate_path
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)

        self.val_path = val_path
        self.val_bs = val_bs if val_bs is not None else bs

        self.model_save_path = os.path.join(self.parent_dir, "models")
        os.makedirs(self.model_save_path, exist_ok=True)

        self.images_path = os.path.join(self.parent_dir, "images")
        os.makedirs(self.images_path, exist_ok=True)

        self.sn2n_loss = sn2n_loss
        self.model = Unet_2d(n_channels=1, n_classes=1, bilinear=True).to(self.device)

        self.bs = bs
        self.epochs = epochs
        self.lr = lr
        self.lambda_grad = lambda_grad
        self.img_patch = (int(img_patch),) * 2
        self.q = q

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.constrained = torch.nn.L1Loss(reduction="mean")
        self.criterion = torch.nn.L1Loss(reduction="mean")

        self.if_alr = if_alr
        if self.if_alr:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=10, verbose=True
            )

        self.early_stop = early_stop
        self.early_stopper = EarlyStopping(patience=patience) if self.early_stop else None

        # AMP
        if use_amp is None:
            use_amp = torch.cuda.is_available()
        self.use_amp = bool(use_amp)
        self.scaler = GradScaler(enabled=self.use_amp)

    # Epoch-accurate batching (no replacement)
    def load_batch2d(self, traindata_path, batch_size=None, shuffle=True, augment=True, drop_last=False):
        paths = glob(os.path.join(traindata_path, "*.tif"))
        if batch_size is None:
            batch_size = self.bs

        if shuffle:
            np.random.shuffle(paths)

        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start:start + batch_size]
            if drop_last and len(batch_paths) < batch_size:
                continue

            imgs_As, imgs_Bs = [], []
            for batch_tem in batch_paths:
                img = sk_imread(batch_tem)
                h, w = img.shape
                half_w = int(w / 2)

                img_data = img[:, :half_w]
                img_label = img[:, half_w:]

                if augment:
                    a = np.random.random()
                    b = np.random.random()
                    if a < 0.5:
                        img_data = np.fliplr(img_data)
                        img_label = np.fliplr(img_label)
                    else:
                        img_data = np.flipud(img_data)
                        img_label = np.flipud(img_label)

                    if b < 0.33:
                        img_data = np.rot90(img_data, 1)
                        img_label = np.rot90(img_label, 1)
                    elif b < 0.66:
                        img_data = np.rot90(img_data, 2)
                        img_label = np.rot90(img_label, 2)
                    else:
                        img_data = np.rot90(img_data, 3)
                        img_label = np.rot90(img_label, 3)

                img_data = img_data.astype("float32").reshape(1, h, half_w)
                img_label = img_label.astype("float32").reshape(1, h, half_w)

                imgs_As.append(img_data)
                imgs_Bs.append(img_label)

            yield np.array(imgs_As), np.array(imgs_Bs)

    def compute_initial_train_loss(self):
        """Compute the training loss over all training patches with the
        current (untrained) model, without updating weights."""
        train_files = glob(os.path.join(self.dataset_path, "*.tif"))
        if len(train_files) == 0:
            print(f"No training patches found at {self.dataset_path}; "
                f"cannot compute initial training loss.")
            return None

        self.model.eval()
        losses = []
        with torch.no_grad():
            # same loader as training, but without shuffle/augment
            for inputs_np, labels_np in self.load_batch2d(
                self.dataset_path,
                batch_size=self.bs,
                shuffle=False,
                augment=False,
                drop_last=False,
            ):
                inputs = torch.from_numpy(inputs_np).to(self.device, dtype=torch.float32)
                labels = torch.from_numpy(labels_np).to(self.device, dtype=torch.float32)

                with autocast("cuda", enabled=self.use_amp):
                    inputs_pred1 = self.model(inputs)
                    loss1 = self.criterion(inputs_pred1, labels)

                    # same SN2N + gradient logic as in train()
                    with torch.no_grad():
                        proxy = 0.5 * (inputs + labels)
                        proxy_blur = F.avg_pool2d(proxy, kernel_size=3, stride=1, padding=1)
                        flat = proxy_blur.float().view(proxy_blur.size(0), -1)
                        tau_thr = torch.quantile(flat, self.q, dim=1).view(-1, 1, 1, 1).to(proxy_blur.dtype)

                    if self.sn2n_loss != 0:
                        labels_pred1 = self.model(labels)

                        loss_grad = gradient_loss(
                            inputs_pred1,
                            labels_pred1.detach(),
                            proxy,
                            tau_thr,
                            softness=0.1,
                            blur_kernel_size=3,
                        )

                        loss2 = self.criterion(labels_pred1, inputs)
                        loss3 = self.constrained(labels_pred1, inputs_pred1)
                        loss_sn2n = (loss1 + loss2 + self.sn2n_loss * loss3) / (2 + self.sn2n_loss)

                        loss = loss_sn2n + self.lambda_grad * loss_grad
                    else:
                        loss = loss1

                losses.append(loss.item())

        return float(np.mean(losses)) if losses else None


    def train(self):
        print(f"The path for the raw images used for training is located under:\n{self.img_path}")
        print(f"The training dataset is being saved under:\n{self.dataset_path}")
        print(f"Models is being saved under:\n{self.model_save_path}")
        print(f"Training temporary prediction images is being saved under:\n{self.images_path}")

        if self.val_path:
            val_count = len(glob(os.path.join(self.val_path, "*.tif")))
            print(f"Validation path: {self.val_path} | patches found: {val_count}")
        else:
            print("Validation path not set; skipping validation.")

        start_time = datetime.datetime.now()

        train_epoch_history = []
        val_history = []
        lr_ms = []
        best_val = float("inf")

        # check training data exists 
        train_files = glob(os.path.join(self.dataset_path, "*.tif"))
        if len(train_files) == 0:
            raise RuntimeError(
                f"No training patches found. Expected .tif files in '{self.dataset_path}'. "
                f"img_path provided='{self.img_path}'."
            )

        #  initial training loss (epoch 0, before any updates) 
        print("Computing initial training loss (epoch 0, before training)...")
        init_train_loss = self.compute_initial_train_loss()
        if init_train_loss is not None:
            train_epoch_history.append(init_train_loss)
            print(f"[Epoch 0] initial train_loss(avg): {init_train_loss * 100:.4f}")
        else:
            print("[Epoch 0] initial train_loss(avg): skipped (no train batches)")

        #  initial validation loss (epoch 0, before any updates) 
        if self.val_path:
            print("Computing initial validation loss (epoch 0, before training)...")
            init_val_loss = self.validate()
            if init_val_loss is not None:
                val_history.append(init_val_loss)
                best_val = init_val_loss
                print(f"[Epoch 0] initial val_loss: {init_val_loss * 100:.4f}")
            else:
                print("[Epoch 0] initial val_loss: skipped (no val batches)")
        else:
            print("[Epoch 0] initial val_loss: skipped (val_path not set)")


        for epoch in range(self.epochs):
            self.model.train()

            # Recompute batch count each epoch (no replacement; includes remainder batch)
            n_train = len(train_files)
            n_batches = int(np.ceil(n_train / self.bs))

            epoch_losses = []
            batch_iter = tqdm(
                enumerate(self.load_batch2d(self.dataset_path, batch_size=self.bs, shuffle=True, augment=True, drop_last=False)),
                total=n_batches,
                desc=f"Epoch {epoch+1}/{self.epochs}",
                leave=False,
            )

            for i, (inputs_np, labels_np) in batch_iter:
                inputs = torch.from_numpy(inputs_np).to(self.device, dtype=torch.float32)
                labels = torch.from_numpy(labels_np).to(self.device, dtype=torch.float32)

                self.optimizer.zero_grad(set_to_none=True)

                with autocast('cuda', enabled=self.use_amp):
                    inputs_pred1 = self.model(inputs)
                    loss1 = self.criterion(inputs_pred1, labels)

                    # threshold from blurred proxy (compute quantile in float32 for stability)
                    with torch.no_grad():
                        proxy = 0.5 * (inputs + labels)
                        proxy_blur = F.avg_pool2d(proxy, kernel_size=3, stride=1, padding=1)
                        flat = proxy_blur.float().view(proxy_blur.size(0), -1)
                        tau_thr = torch.quantile(flat, self.q, dim=1).view(-1, 1, 1, 1).to(proxy_blur.dtype)

                    if self.sn2n_loss != 0:
                        labels_pred1 = self.model(labels)

                        loss_grad = gradient_loss(
                            inputs_pred1,
                            labels_pred1.detach(),
                            proxy,
                            tau_thr,
                            softness=0.1,
                            blur_kernel_size=3,
                        )

                        loss2 = self.criterion(labels_pred1, inputs)
                        loss3 = self.constrained(labels_pred1, inputs_pred1)
                        loss_sn2n = (loss1 + loss2 + self.sn2n_loss * loss3) / (2 + self.sn2n_loss)

                        loss = loss_sn2n + self.lambda_grad * loss_grad
                    else:
                        loss = loss1

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                epoch_losses.append(loss.item())

            train_epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else None
            train_epoch_history.append(train_epoch_loss if train_epoch_loss is not None else np.nan)

            if train_epoch_loss is not None:
                print(f"[Epoch {epoch+1}/{self.epochs}] train_loss(avg): {train_epoch_loss * 100:.4f}")
            else:
                print(f"[Epoch {epoch+1}/{self.epochs}] train_loss(avg): skipped (no train batches)")

            # Validation
            val_loss_epoch = None
            if self.val_path:
                val_loss_epoch = self.validate()
                if val_loss_epoch is not None:
                    val_history.append(val_loss_epoch)
                    best_val = min(best_val, val_loss_epoch)
                    print(f"[Epoch {epoch+1}/{self.epochs}] val_loss: {val_loss_epoch * 100:.4f}")
                else:
                    print(f"[Epoch {epoch+1}/{self.epochs}] val_loss: skipped (no val batches)")
            else:
                print(f"[Epoch {epoch+1}/{self.epochs}] val_loss: skipped (val_path not set)")

            # Scheduler metric: prefer validation, else epoch-average train loss
            scheduler_metric = val_loss_epoch if val_loss_epoch is not None else train_epoch_loss
            if self.if_alr and scheduler_metric is not None:
                self.scheduler.step(scheduler_metric)

            lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
            lr_ms.append(np.array(lr))

            if self.early_stop and scheduler_metric is not None:
                self.early_stopper.step(scheduler_metric)
                if self.early_stopper.stop:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

            # Optional preview prediction (kept as in your original logic)
            raw_path = glob(os.path.join(self.img_path, "*.tif"))
            if raw_path:
                test_img = tifffile.imread(raw_path[0])
                if len(test_img.shape) == 3:
                    test_img = test_img[0, :, :]
                test_img = np.squeeze(test_img)

                test_pred = self.test(test_img)
                test_pred = test_pred.to(torch.device("cpu")).numpy()
                for _, item in enumerate(test_pred):
                    item = item.astype(np.float32, copy=False)
                    tifffile.imwrite(os.path.join(self.images_path, f"epoch_{epoch}.tif"), item)

            # Save checkpoint every 10 epochs (state_dict instead of full model)
            if epoch % 10 == 0:
                ckpt_path = os.path.join(
                    self.model_save_path,
                    f"model_{datetime.datetime.now().month}_{datetime.datetime.now().day}_{epoch}.pth",
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "lr": lr,
                        "train_loss_avg": train_epoch_loss,
                        "val_loss": val_loss_epoch,
                        "use_amp": self.use_amp,
                    },
                    ckpt_path,
                )

        # Final save (state_dict)
        final_path = os.path.join(
            self.model_save_path,
            f"model_{datetime.datetime.now().month}_{datetime.datetime.now().day}_full.pth",
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "use_amp": self.use_amp,
            },
            final_path,
        )

        # Persist epoch-level averages (as before, but cleaner inputs)
        if train_epoch_history:
            with open(os.path.join(self.parent_dir, "loss.txt"), "w") as f1:
                for v in train_epoch_history:
                    f1.write(str(v) + "\r\n")

        if val_history:
            with open(os.path.join(self.parent_dir, "val_loss.txt"), "w") as f2:
                for v in val_history:
                    f2.write(str(v) + "\r\n")

        return (train_epoch_history[-1] if train_epoch_history else None,
                best_val if val_history else None)

    def validate(self):
        if not self.val_path:
            return None

        val_files = glob(os.path.join(self.val_path, "*.tif"))
        if len(val_files) == 0:
            print(f"No validation patches found at {self.val_path}; skipping validation.")
            return None

        self.model.eval()
        losses = []
        with torch.no_grad():
            for start in range(0, len(val_files), self.val_bs):
                batch_paths = val_files[start:start + self.val_bs]

                imgs_As, imgs_Bs = [], []
                for batch_tem in batch_paths:
                    img = sk_imread(batch_tem)
                    h, w = img.shape
                    half_w = int(w / 2)

                    img_data = img[:, :half_w].astype("float32").reshape(1, h, half_w)
                    img_label = img[:, half_w:].astype("float32").reshape(1, h, half_w)

                    imgs_As.append(img_data)
                    imgs_Bs.append(img_label)

                inputs = torch.from_numpy(np.array(imgs_As)).to(self.device, dtype=torch.float32)
                labels = torch.from_numpy(np.array(imgs_Bs)).to(self.device, dtype=torch.float32)

                with autocast('cuda', enabled=self.use_amp):
                    preds = self.model(inputs)
                    loss1 = self.criterion(preds, labels)

                    proxy = 0.5 * (inputs + labels)
                    proxy_blur = F.avg_pool2d(proxy, kernel_size=3, stride=1, padding=1)
                    flat = proxy_blur.float().view(proxy_blur.size(0), -1)
                    tau_thr = torch.quantile(flat, self.q, dim=1).view(-1, 1, 1, 1).to(proxy_blur.dtype)

                    if self.sn2n_loss != 0:
                        labels_pred1 = self.model(labels)

                        loss_grad = gradient_loss(
                            preds,
                            labels_pred1.detach(),
                            proxy,
                            tau_thr,
                            softness=0.1,
                            blur_kernel_size=3,
                        )

                        loss2 = self.criterion(labels_pred1, inputs)
                        loss3 = self.constrained(labels_pred1, preds)
                        loss_sn2n = (loss1 + loss2 + self.sn2n_loss * loss3) / (2 + self.sn2n_loss)
                        loss = loss_sn2n + self.lambda_grad * loss_grad
                    else:
                        loss = loss1

                losses.append(loss.item())

        return float(np.mean(losses)) if losses else None

    def test(self, test_img_np):
        self.model.eval()
        with torch.no_grad():
            for data_np in self.load_test_batch2d(test_img_np):
                data = torch.from_numpy(data_np).to(self.device, dtype=torch.float32)
                y_pred = self.model(data)
                return y_pred

    def load_test_batch2d(self, img_tem):
        img_tem = np.squeeze(img_tem).astype(np.float32)
        h, w = img_tem.shape
        imgs_A = np.zeros((1, 1, h, w), dtype=np.float32)
        imgs_A[0, 0, :, :] = img_tem
        yield imgs_A
