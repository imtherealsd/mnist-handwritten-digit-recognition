"""
src/model_utils.py
==================
Model loading, building, and inference utilities for Handwritten Digit Recognition.
"""

import os
from typing import Any, Dict, Optional
import numpy as np


def build_cnn_model():
    """
    Construct the standard Convolutional Neural Network (CNN) architecture
    for MNIST digit classification (0-9).

    Architecture:
    -------------
    1. Input: (28, 28, 1) grayscale image
    2. Conv Block 1:
       - Conv2D (32 filters, 3x3, ReLU, padding='same')
       - BatchNormalization
       - Conv2D (32 filters, 3x3, ReLU)
       - BatchNormalization
       - MaxPooling2D (2x2)
       - Dropout (0.25)
    3. Conv Block 2:
       - Conv2D (64 filters, 3x3, ReLU, padding='same')
       - BatchNormalization
       - Conv2D (64 filters, 3x3, ReLU)
       - BatchNormalization
       - MaxPooling2D (2x2)
       - Dropout (0.25)
    4. Classifier Head:
       - Flatten
       - Dense (128 units, ReLU)
       - BatchNormalization
       - Dropout (0.50)
       - Dense (10 units, Softmax) for classes 0 to 9

    Returns
    -------
    keras.Model
        Uncompiled Keras Sequential model.
    """
    import keras
    from keras import layers

    model = keras.Sequential(
        [
            layers.Input(shape=(28, 28, 1), name="input_image"),
            # --- Conv Block 1 ---
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same", name="conv1_1"),
            layers.BatchNormalization(name="bn1_1"),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu", name="conv1_2"),
            layers.BatchNormalization(name="bn1_2"),
            layers.MaxPooling2D(pool_size=(2, 2), name="pool1"),
            layers.Dropout(0.25, name="dropout1"),
            # --- Conv Block 2 ---
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same", name="conv2_1"),
            layers.BatchNormalization(name="bn2_1"),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu", name="conv2_2"),
            layers.BatchNormalization(name="bn2_2"),
            layers.MaxPooling2D(pool_size=(2, 2), name="pool2"),
            layers.Dropout(0.25, name="dropout2"),
            # --- Dense Head ---
            layers.Flatten(name="flatten"),
            layers.Dense(128, activation="relu", name="dense_128"),
            layers.BatchNormalization(name="bn_dense"),
            layers.Dropout(0.50, name="dropout_dense"),
            layers.Dense(10, activation="softmax", name="output_digits"),
        ],
        name="mnist_cnn",
    )
    return model


def load_mnist_model(model_path: str = "mnist_cnn.keras"):
    """
    Load the saved trained Keras model.

    Parameters
    ----------
    model_path : str
        Relative or absolute path to the `.keras` model file.

    Returns
    -------
    keras.Model
        Loaded Keras model.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist at the specified path.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file '{model_path}' not found.\n"
            f"Please train the model using 'notebooks/MNIST_Handwritten_Digit_Recognition.ipynb' "
            f"in Google Colab (or run 'python train.py') and save the trained model as '{model_path}'."
        )

    import keras

    # Load model using modern Keras format
    model = keras.models.load_model(model_path)
    return model


def predict_digit(model: Any, preprocessed_image: np.ndarray) -> Dict[str, Any]:
    """
    Perform digit prediction and calculate probability distributions.

    Parameters
    ----------
    model : Any
        A trained Keras model or compatible predictor with `.predict()` method.
    preprocessed_image : np.ndarray
        Preprocessed image array with shape (1, 28, 28, 1) and values in [0.0, 1.0].

    Returns
    -------
    dict
        {
            "digit": int (0-9),
            "confidence": float (0.0 to 1.0),
            "probabilities": list of 10 floats (one per digit 0-9)
        }
    """
    if preprocessed_image is None:
        raise ValueError("preprocessed_image cannot be None.")

    arr = np.array(preprocessed_image, dtype=np.float32)

    # Ensure shape is (1, 28, 28, 1)
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=(0, -1))
    elif arr.ndim == 3:
        if arr.shape == (28, 28, 1):
            arr = np.expand_dims(arr, axis=0)
        elif arr.shape == (1, 28, 28):
            arr = np.expand_dims(arr, axis=-1)

    if arr.shape != (1, 28, 28, 1):
        raise ValueError(f"Expected image shape (1, 28, 28, 1), but got {arr.shape}.")

    # Inference without progress print
    preds = model.predict(arr, verbose=0)
    probs = preds[0]
    predicted_digit = int(np.argmax(probs))
    confidence = float(probs[predicted_digit])

    return {
        "digit": predicted_digit,
        "confidence": confidence,
        "probabilities": [float(p) for p in probs],
    }
