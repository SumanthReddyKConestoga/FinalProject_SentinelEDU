"""DVC stage 4: train SLP, MLP, Fine-Tuned ANN, and 1D CNN.

Demonstrates gradient descent via loss curves + lr comparison.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.config import SETTINGS, resolve_path
from src.preprocessing.preprocessor import Preprocessor
from src.preprocessing.sequence_builder import build_sequences
from src.models.deep import build_slp, build_mlp, build_tuned_ann, tuning_configs
from src.models.cnn import build_cnn1d
from src.evaluation.metrics import classification_metrics
from src.evaluation.plots import plot_confusion_matrix, plot_loss_curves, plot_metric_bars
from src.utils.helpers import ensure_dir, write_json
from src.utils.logging import get_logger

log = get_logger(__name__)

PROCESSED = resolve_path(SETTINGS["paths"]["data_processed"])
MODELS = resolve_path(SETTINGS["paths"]["models"])
REPORTS = resolve_path(SETTINGS["paths"]["reports"])
LABELS = ["High", "Medium", "Low"]


def _encode_labels(y, labels=LABELS):
    mapping = {lbl: i for i, lbl in enumerate(labels)}
    return np.array([mapping[v] for v in y])


def _load_data():
    train = pd.read_parquet(PROCESSED / "static_train.parquet")
    val = pd.read_parquet(PROCESSED / "static_val.parquet")
    test = pd.read_parquet(PROCESSED / "static_test.parquet")
    return train, val, test


def train_tabular_deep():
    import tensorflow as tf

    train, val, test = _load_data()
    pre = Preprocessor.load()
    X_train = pre.transform(train).astype(np.float32)
    X_val = pre.transform(val).astype(np.float32)
    X_test = pre.transform(test).astype(np.float32)
    y_train = _encode_labels(train["risk_class"].values)
    y_val = _encode_labels(val["risk_class"].values)
    y_test = _encode_labels(test["risk_class"].values)

    input_dim = X_train.shape[1]
    n_classes = len(LABELS)
    epochs = SETTINGS["training"]["deep"]["epochs"]
    batch_size = SETTINGS["training"]["deep"]["batch_size"]
    patience = SETTINGS["training"]["deep"]["patience"]

    cb = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        )
    ]

    results = {}
    f1_bars = {}

    def _train_and_record(model, name, out_subdir):
        log.info(f"Training {name}")
        hist = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=cb,
            verbose=0,
        )
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        metrics = classification_metrics(
            [LABELS[i] for i in y_test],
            [LABELS[i] for i in y_pred],
            y_proba=y_pred_proba,
            labels=LABELS,
        )
        out = ensure_dir(MODELS / out_subdir)
        model.save(out / f"{name}.keras")
        plot_loss_curves(
            hist.history, f"Training curves — {name}", f"loss_{name}.png"
        )
        plot_confusion_matrix(
            metrics["confusion_matrix"], LABELS,
            f"Confusion Matrix — {name}", f"cm_{name}.png"
        )
        results[name] = metrics
        f1_bars[name] = metrics["f1_macro"]
        return hist

    # --- SLP ---
    _train_and_record(build_slp(input_dim, n_classes), "slp", "deep")

    # --- MLP ---
    _train_and_record(build_mlp(input_dim, n_classes), "mlp", "deep")

    # --- Fine-Tuned ANN — 3 configs ---
    best_f1 = -1.0
    best_config_name = None
    for cfg in tuning_configs():
        model = build_tuned_ann(input_dim, n_classes, cfg)
        _train_and_record(model, cfg["name"], "deep")
        if f1_bars[cfg["name"]] > best_f1:
            best_f1 = f1_bars[cfg["name"]]
            best_config_name = cfg["name"]
    log.info(f"Best tuned ANN config: {best_config_name} (F1={best_f1:.3f})")

    # --- Gradient Descent demo (lr comparison on a clone of MLP) ---
    gd_histories = {}
    for lr_label, lr in [("lr_0.1_high", 0.1), ("lr_0.001_good", 0.001)]:
        log.info(f"Gradient descent demo: {lr_label}")
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(input_dim,)),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(n_classes, activation="softmax"),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        hist = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=30,
            batch_size=batch_size,
            verbose=0,
        )
        gd_histories[lr_label] = hist.history
        plot_loss_curves(
            hist.history,
            f"Gradient Descent ({lr_label})",
            f"gd_{lr_label}.png",
        )

    plot_metric_bars(
        f1_bars,
        "Deep Models — Macro F1 Comparison",
        "deep_f1.png",
        "Macro F1",
    )
    write_json(REPORTS / "deep_metrics.json", results)
    return results


def train_cnn():
    import tensorflow as tf

    log.info("=== Training 1D CNN ===")
    train, val, test = _load_data()
    weekly = pd.read_parquet(PROCESSED / "weekly.parquet")
    static_all = pd.read_parquet(PROCESSED / "static_all.parquet")

    # Use full static_all for label mapping, but filter sequences by split membership
    train_ids = set(train["student_id"])
    val_ids = set(val["student_id"])
    test_ids = set(test["student_id"])

    weekly_train = weekly[weekly["student_id"].isin(train_ids)]
    weekly_val = weekly[weekly["student_id"].isin(val_ids)]
    weekly_test = weekly[weekly["student_id"].isin(test_ids)]

    X_train, y_train_str, _ = build_sequences(weekly_train, static_all)
    X_val, y_val_str, _ = build_sequences(weekly_val, static_all)
    X_test, y_test_str, _ = build_sequences(weekly_test, static_all)

    y_train = _encode_labels(y_train_str)
    y_val = _encode_labels(y_val_str)
    y_test = _encode_labels(y_test_str)

    window = X_train.shape[1]
    n_features = X_train.shape[2]
    n_classes = len(LABELS)

    model = build_cnn1d(window, n_features, n_classes)
    cb = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=SETTINGS["training"]["deep"]["patience"],
            restore_best_weights=True,
        )
    ]

    hist = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=SETTINGS["training"]["deep"]["epochs"],
        batch_size=SETTINGS["training"]["deep"]["batch_size"],
        callbacks=cb,
        verbose=0,
    )

    out = ensure_dir(MODELS / "cnn")
    model.save(out / "cnn1d.keras")

    # eval
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    metrics = classification_metrics(
        [LABELS[i] for i in y_test],
        [LABELS[i] for i in y_pred],
        y_proba=y_pred_proba,
        labels=LABELS,
    )

    plot_loss_curves(hist.history, "1D CNN Training Curves", "loss_cnn1d.png")
    plot_confusion_matrix(
        metrics["confusion_matrix"], LABELS,
        "Confusion Matrix — 1D CNN", "cm_cnn1d.png"
    )
    write_json(REPORTS / "cnn_metrics.json", metrics)
    log.info(f"CNN F1 macro: {metrics['f1_macro']:.3f}")
    return metrics


def main():
    train_tabular_deep()
    train_cnn()


if __name__ == "__main__":
    main()
