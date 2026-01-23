"""Plotting helpers extracted from the notebook."""
from __future__ import annotations
import matplotlib.pyplot as plt

def plot_loss_curve(train_loss, val_loss=None):
    # Plot loss curves if your trainer returns them.
    if train_loss is None:
        print('No loss curves available (trainer did not return losses).')
        return

    plt.figure(figsize=(7, 4))
    plt.plot(train_loss, label='train')
    if val_loss is not None:
        plt.plot(val_loss, label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

__all__ = ['plot_loss_curve']
