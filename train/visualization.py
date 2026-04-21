from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_training_history(history: list[dict[str, float]], output_path: Path) -> None:
    if not history:
        return
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    epochs = frame["epoch"]

    axes[0, 0].plot(epochs, frame["train_loss"], label="train")
    axes[0, 0].plot(epochs, frame["valid_loss"], label="valid")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(epochs, frame["train_p_win_acc"], label="train")
    axes[0, 1].plot(epochs, frame["valid_p_win_acc"], label="valid")
    axes[0, 1].set_title("P(win) Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(epochs, frame["train_ret_mu_mae"], label="train ret")
    axes[1, 0].plot(epochs, frame["valid_ret_mu_mae"], label="valid ret")
    axes[1, 0].plot(epochs, frame["train_risk_dd_mae"], label="train dd")
    axes[1, 0].plot(epochs, frame["valid_risk_dd_mae"], label="valid dd")
    axes[1, 0].set_title("Regression MAE")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(epochs, frame["train_rank_score_mae"], label="train")
    axes[1, 1].plot(epochs, frame["valid_rank_score_mae"], label="valid")
    axes[1, 1].set_title("Rank Score MAE")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_business_history(history: list[dict[str, float]], output_path: Path) -> None:
    if not history:
        return
    frame = pd.DataFrame(history)
    epochs = frame["epoch"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    if "valid_business_score" in frame:
        axes[0].plot(epochs, frame["valid_business_score"], label="valid business")
    if "train_business_score" in frame:
        axes[0].plot(epochs, frame["train_business_score"], label="train business", alpha=0.7)
    axes[0].set_title("Business Score")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    if "valid_topk_utility" in frame:
        axes[1].plot(epochs, frame["valid_topk_utility"], label="valid utility")
    if "train_topk_utility" in frame:
        axes[1].plot(epochs, frame["train_topk_utility"], label="train utility", alpha=0.7)
    axes[1].set_title("Top-K Utility")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_topk_metrics(history: list[dict[str, float]], output_path: Path) -> None:
    if not history:
        return
    frame = pd.DataFrame(history)
    epochs = frame["epoch"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    candidates = {
        "valid_topk_avg_ret_10d": (0, 0, "Top-K Avg Return"),
        "valid_topk_bigloss_rate_10d": (0, 1, "Top-K Bigloss Rate"),
        "valid_topk_avg_drawdown_10d": (1, 0, "Top-K Avg Drawdown"),
        "valid_topk_utility_10d": (1, 1, "Top-K Utility"),
    }
    for column, (row, col, title) in candidates.items():
        if column in frame:
            axes[row, col].plot(epochs, frame[column], label=column.replace("valid_", ""))
        axes[row, col].set_title(title)
        axes[row, col].grid(alpha=0.3)
        axes[row, col].legend()
    for axis in axes[-1, :]:
        axis.set_xlabel("Epoch")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
