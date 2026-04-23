"""Plotting utilities — all static PNGs saved under reports/figures/."""
from pathlib import Path
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import SETTINGS, resolve_path
from src.utils.helpers import ensure_dir

FIG_DIR = ensure_dir(resolve_path(SETTINGS["paths"]["reports"]) / "figures")

sns.set_theme(style="whitegrid", context="talk")
BRAND_PALETTE = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"]


def plot_confusion_matrix(
    cm: List[List[int]], labels: List[str], title: str, filename: str
) -> Path:
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar=False,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    path = FIG_DIR / filename
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_loss_curves(history: Dict, title: str, filename: str) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.get("loss", []), label="train", color=BRAND_PALETTE[0])
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="val", color=BRAND_PALETTE[3])
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    if "accuracy" in history or "acc" in history:
        acc_key = "accuracy" if "accuracy" in history else "acc"
        val_acc_key = "val_accuracy" if "val_accuracy" in history else "val_acc"
        axes[1].plot(history[acc_key], label="train", color=BRAND_PALETTE[0])
        if val_acc_key in history:
            axes[1].plot(history[val_acc_key], label="val", color=BRAND_PALETTE[3])
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("epoch")
        axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    path = FIG_DIR / filename
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_residuals(y_true, y_pred, title: str, filename: str) -> Path:
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals, alpha=0.6, color=BRAND_PALETTE[0])
    ax.axhline(0, color="black", linestyle="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    ax.set_title(title)
    fig.tight_layout()
    path = FIG_DIR / filename
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_metric_bars(
    metric_dict: Dict[str, float], title: str, filename: str, ylabel: str
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    names = list(metric_dict.keys())
    values = [metric_dict[n] for n in names]
    bars = ax.bar(names, values, color=BRAND_PALETTE[: len(names)])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    fig.tight_layout()
    path = FIG_DIR / filename
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_cluster_scatter(
    X_2d: np.ndarray, labels: np.ndarray, title: str, filename: str
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    unique = np.unique(labels)
    for i, lbl in enumerate(unique):
        mask = labels == lbl
        ax.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            s=40,
            alpha=0.7,
            color=BRAND_PALETTE[i % len(BRAND_PALETTE)],
            label=f"Segment {lbl}",
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    path = FIG_DIR / filename
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
