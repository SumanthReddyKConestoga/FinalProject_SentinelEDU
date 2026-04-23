"""1D CNN for sequential behavioral features.

Input shape: (window_size, n_features)
Covers required topic: CNN / Vanilla Deep Learning
"""


def _tf():
    import tensorflow as tf
    return tf


def build_cnn1d(window: int, n_features: int, n_classes: int):
    tf = _tf()
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(window, n_features)),
            tf.keras.layers.Conv1D(
                filters=32, kernel_size=3, padding="same", activation="relu"
            ),
            tf.keras.layers.Conv1D(
                filters=64, kernel_size=3, padding="same", activation="relu"
            ),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(n_classes, activation="softmax"),
        ],
        name="CNN1D",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
