"""
train.py
========
Standalone model training and evaluation script for MNIST Handwritten Digit Recognition.
This script can be run locally or adapted for Google Colab:
1. Loads MNIST dataset (60,000 train, 10,000 test).
2. Splits train set into 50,000 training and 10,000 validation sets.
3. Normalizes pixel intensities to [0.0, 1.0] float32.
4. One-hot encodes targets into 10 classes.
5. Builds and compiles the Convolutional Neural Network (CNN).
6. Trains with EarlyStopping and ReduceLROnPlateau.
7. Evaluates strictly on the hold-out test set (10,000 images).
8. Prints classification report and confusion matrix.
9. Saves the model to `mnist_cnn.keras`.
"""

import os
import random
import numpy as np
import tensorflow as tf
import keras
from keras import utils, callbacks
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.model_utils import build_cnn_model

# Reproducibility seeds
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def train():
    print("=" * 60)
    print("MNIST Handwritten Digit Recognition - Model Training")
    print("=" * 60)

    # 1. Load MNIST dataset
    print("\n[1/6] Loading MNIST dataset...")
    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()
    print(f"  Full Training data: {x_train_full.shape}, Labels: {y_train_full.shape}")
    print(f"  Test data:          {x_test.shape}, Labels: {y_test.shape}")

    # 2. Preprocess & Split
    print("\n[2/6] Normalizing pixels and splitting training / validation...")
    x_train_full = x_train_full.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Reshape to (N, 28, 28, 1)
    x_train_full = np.expand_dims(x_train_full, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # One-hot encoding
    num_classes = 10
    y_train_full_cat = utils.to_categorical(y_train_full, num_classes)
    y_test_cat = utils.to_categorical(y_test, num_classes)

    # Split 50k train / 10k val (keeping test set completely out of training & hyperparameter tuning)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full_cat,
        test_size=10000,
        random_state=SEED,
        stratify=y_train_full,
    )
    print(f"  Train split: {x_train.shape[0]} samples")
    print(f"  Val split:   {x_val.shape[0]} samples")
    print(f"  Test split:  {x_test.shape[0]} samples")

    # 3. Build Model
    print("\n[3/6] Building CNN architecture...")
    model = build_cnn_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # 4. Train Model
    print("\n[4/6] Training model with EarlyStopping...")
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-5,
        verbose=1,
    )

    batch_size = 128
    epochs = 8  # 8 epochs is plenty to reach ~99%+ on MNIST while keeping CPU runtime short

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )

    # 5. Evaluate on Hold-out Test Set
    print("\n[5/6] Evaluating strictly on unseen test set (10,000 samples)...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test_cat, verbose=0)
    print(f"  --> Final Test Loss:     {test_loss:.4f}")
    print(f"  --> Final Test Accuracy: {test_accuracy * 100:.2f}%")

    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)

    print("\n--- Classification Report ---")
    report = classification_report(y_test, y_pred_classes, digits=4)
    print(report)

    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred_classes)
    print(cm)

    # 6. Save Model
    save_path = "mnist_cnn.keras"
    print(f"\n[6/6] Saving trained model to '{save_path}'...")
    model.save(save_path)
    print(f"  Model successfully saved to '{os.path.abspath(save_path)}'.")
    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)

    return test_loss, test_accuracy, report, cm


if __name__ == "__main__":
    train()
