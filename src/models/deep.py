"""Deep learning model factories.

Covers required topics:
  - Single Layer Perceptron (SLP)
  - Multi Layer Perceptron (MLP)
  - Fine-Tuning ANN (3 manual hyperparameter configs)
  - Gradient Descent (via Adam, loss curves logged)
"""
from typing import Tuple, List, Dict
import numpy as np

# Lazy TF import - heavy module
def _tf():
    import tensorflow as tf
    return tf


def build_slp(input_dim: int, n_classes: int):
    """Single Layer Perceptron — one linear layer + softmax."""
    tf = _tf()
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(n_classes, activation="softmax"),
        ],
        name="SLP",
    )
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_mlp(input_dim: int, n_classes: int, dropout: float = 0.3):
    """Multi Layer Perceptron — 2 hidden layers with ReLU + dropout."""
    tf = _tf()
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(n_classes, activation="softmax"),
        ],
        name="MLP",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_tuned_ann(
    input_dim: int, n_classes: int, config: Dict
):
    """Fine-tuned ANN — hyperparameters provided via config dict.

    config keys: hidden_sizes (list of int), learning_rate, dropout, activation
    """
    tf = _tf()
    layers = [tf.keras.layers.Input(shape=(input_dim,))]
    for hs in config["hidden_sizes"]:
        layers.append(tf.keras.layers.Dense(hs, activation=config["activation"]))
        layers.append(tf.keras.layers.Dropout(config["dropout"]))
    layers.append(tf.keras.layers.Dense(n_classes, activation="softmax"))
    model = tf.keras.Sequential(layers, name=config.get("name", "TunedANN"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def tuning_configs() -> List[Dict]:
    """Three hyperparameter configs to try for fine-tuning demo."""
    return [
        {
            "name": "TunedANN_A_small",
            "hidden_sizes": [32, 16],
            "learning_rate": 0.001,
            "dropout": 0.2,
            "activation": "relu",
        },
        {
            "name": "TunedANN_B_medium",
            "hidden_sizes": [64, 32, 16],
            "learning_rate": 0.001,
            "dropout": 0.3,
            "activation": "relu",
        },
        {
            "name": "TunedANN_C_wide",
            "hidden_sizes": [128, 64],
            "learning_rate": 0.0005,
            "dropout": 0.4,
            "activation": "relu",
        },
    ]
