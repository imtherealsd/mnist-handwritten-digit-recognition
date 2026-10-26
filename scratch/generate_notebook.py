import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Handwritten Digit Recognition using CNN (MNIST)\n",
    "### Machine Learning Internship Assignment\n",
    "**Assigned by:** Horizon Intern  \n",
    "**Objective:** Build, train, evaluate, and export a Convolutional Neural Network (CNN) that classifies handwritten digits from 0 to 9 using the classic MNIST dataset.\n",
    "\n",
    "---\n",
    "### Pipeline Overview:\n",
    "1. **Environment Setup & Imports**: Setting seeds, loading TensorFlow/Keras and numerical libraries.\n",
    "2. **Dataset Loading & Exploration**: Inspecting dataset shapes, data types, pixel value ranges, and class balance.\n",
    "3. **Sample Visualization**: Displaying a grid of real handwritten digits with their ground truth labels.\n",
    "4. **Data Preprocessing**: Normalization, channel reshaping `(28, 28, 1)`, one-hot encoding, and train/validation splitting.\n",
    "5. **CNN Architecture Construction**: Designing convolutional blocks, pooling, dropout, batch normalization, and dense layers.\n",
    "6. **Model Compilation & Training**: Training with Adam optimizer, EarlyStopping, and ReduceLROnPlateau.\n",
    "7. **Training History Visualization**: Plotting loss and accuracy curves across training and validation sets.\n",
    "8. **Evaluation on Unseen Test Set**: Evaluating strictly on 10,000 holdout test images.\n",
    "9. **Performance Analysis**: Confusion matrix heatmap, classification report (precision, recall, f1-score), and misclassified sample inspection.\n",
    "10. **Model Export**: Saving as `mnist_cnn.keras` for local Streamlit deployment."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Environment Setup & Library Imports"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. Environment Setup & Library Imports\n",
    "import os\n",
    "import random\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.metrics import classification_report, confusion_matrix\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "import tensorflow as tf\n",
    "import keras\n",
    "from keras import layers, callbacks, utils\n",
    "\n",
    "# Set random seeds for reproducibility\n",
    "SEED = 42\n",
    "os.environ['PYTHONHASHSEED'] = str(SEED)\n",
    "random.seed(SEED)\n",
    "np.random.seed(SEED)\n",
    "tf.random.set_seed(SEED)\n",
    "\n",
    "print(f\"TensorFlow Version: {tf.__version__}\")\n",
    "print(f\"Keras Version:      {keras.__version__}\")\n",
    "print(f\"GPU Available:       {bool(tf.config.list_physical_devices('GPU'))}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Load and Explore the MNIST Dataset\n",
    "The MNIST dataset contains 70,000 grayscale images of handwritten digits (0–9) sized at 28×28 pixels.  \n",
    "Standard split: 60,000 for training, 10,000 for testing."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load standard MNIST dataset\n",
    "(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = keras.datasets.mnist.load_data()\n",
    "\n",
    "print(\"--- Dataset Dimensions ---\")\n",
    "print(f\"Raw Training Images: {x_train_raw.shape}, Labels: {y_train_raw.shape}\")\n",
    "print(f\"Raw Test Images:     {x_test_raw.shape}, Labels: {y_test_raw.shape}\")\n",
    "print(f\"Image Data Type:     {x_train_raw.dtype}\")\n",
    "print(f\"Pixel Intensity:     Min={x_train_raw.min()}, Max={x_train_raw.max()}\")\n",
    "print(f\"Unique Class Labels: {np.unique(y_train_raw)}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Class Distribution Verification\n",
    "Let's confirm the dataset is well-balanced across all 10 digits."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot distribution of classes in training set\n",
    "digit_counts = np.bincount(y_train_raw)\n",
    "\n",
    "plt.figure(figsize=(9, 4))\n",
    "bars = plt.bar(range(10), digit_counts, color=\"#3B82F6\", edgecolor=\"#1E3A8A\", alpha=0.85)\n",
    "plt.title(\"Class Distribution in MNIST Training Set (60,000 Samples)\", fontsize=13, pad=12)\n",
    "plt.xlabel(\"Digit Class (0-9)\", fontsize=11)\n",
    "plt.ylabel(\"Number of Images\", fontsize=11)\n",
    "plt.xticks(range(10))\n",
    "plt.grid(axis=\"y\", linestyle=\"--\", alpha=0.6)\n",
    "\n",
    "# Add values on top of bars\n",
    "for bar in bars:\n",
    "    height = bar.get_height()\n",
    "    plt.annotate(f\"{height}\",\n",
    "                 xy=(bar.get_x() + bar.get_width() / 2, height),\n",
    "                 xytext=(0, 3), textcoords=\"offset points\",\n",
    "                 ha=\"center\", va=\"bottom\", fontsize=9)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Sample Digit Visualization\n",
    "Display a 5x5 grid of random handwritten digit samples from the training set."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize 25 random samples from training set\n",
    "plt.figure(figsize=(9, 9))\n",
    "for i in range(25):\n",
    "    plt.subplot(5, 5, i + 1)\n",
    "    idx = random.randint(0, len(x_train_raw) - 1)\n",
    "    plt.imshow(x_train_raw[idx], cmap=\"gray\")\n",
    "    plt.title(f\"Label: {y_train_raw[idx]}\", fontsize=10)\n",
    "    plt.axis(\"off\")\n",
    "plt.suptitle(\"Sample Handwritten Digits from MNIST Training Data\", fontsize=14, y=0.92)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Data Preprocessing\n",
    "Before feeding images into a Convolutional Neural Network:\n",
    "1. **Normalization**: Rescale integer pixel values `[0, 255]` to floating point `[0.0, 1.0]` to ensure stable gradients and faster convergence.\n",
    "2. **Reshaping**: Add channel dimension: `(N, 28, 28) -> (N, 28, 28, 1)` since Keras Conv2D expects 4D tensors `(batch, height, width, channels)`.\n",
    "3. **One-Hot Encoding**: Convert integer targets (e.g. `3`) to 10-dimensional binary vectors (`[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`).\n",
    "4. **Train-Validation Split**: Split the 60,000 training set into 50,000 for training and 10,000 for validation.  \n",
    "   *Crucial rule*: The 10,000 test set remains completely untouched until final evaluation."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. Normalize pixel values\n",
    "x_train_norm = x_train_raw.astype(\"float32\") / 255.0\n",
    "x_test_norm = x_test_raw.astype(\"float32\") / 255.0\n",
    "\n",
    "# 2. Reshape to include channel dimension (28, 28, 1)\n",
    "x_train_norm = np.expand_dims(x_train_norm, axis=-1)\n",
    "x_test_norm = np.expand_dims(x_test_norm, axis=-1)\n",
    "\n",
    "# 3. One-hot encode targets\n",
    "NUM_CLASSES = 10\n",
    "y_train_cat = utils.to_categorical(y_train_raw, NUM_CLASSES)\n",
    "y_test_cat = utils.to_categorical(y_test_raw, NUM_CLASSES)\n",
    "\n",
    "# 4. Stratified validation split (50k train / 10k val)\n",
    "x_train, x_val, y_train, y_val = train_test_split(\n",
    "    x_train_norm,\n",
    "    y_train_cat,\n",
    "    test_size=10000,\n",
    "    random_state=SEED,\n",
    "    stratify=y_train_raw\n",
    ")\n",
    "\n",
    "print(f\"Training set:   {x_train.shape}, Labels: {y_train.shape}\")\n",
    "print(f\"Validation set: {x_val.shape}, Labels: {y_val.shape}\")\n",
    "print(f\"Test set:       {x_test_norm.shape}, Labels: {y_test_cat.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. CNN Architecture Construction\n",
    "We construct a modern, compact Convolutional Neural Network designed specifically for digit recognition:\n",
    "- **Block 1**: 2× Conv2D (32 filters, 3×3) + Batch Normalization + MaxPooling (2×2) + Dropout (0.25)\n",
    "- **Block 2**: 2× Conv2D (64 filters, 3×3) + Batch Normalization + MaxPooling (2×2) + Dropout (0.25)\n",
    "- **Dense Classifier**: Flatten + Dense (128 units, ReLU) + Batch Normalization + Dropout (0.50) + Dense (10 units, Softmax)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def create_mnist_cnn():\n",
    "    model = keras.Sequential([\n",
    "        layers.Input(shape=(28, 28, 1), name=\"input_image\"),\n",
    "        \n",
    "        # Conv Block 1\n",
    "        layers.Conv2D(32, kernel_size=(3, 3), activation=\"relu\", padding=\"same\", name=\"conv1_1\"),\n",
    "        layers.BatchNormalization(name=\"bn1_1\"),\n",
    "        layers.Conv2D(32, kernel_size=(3, 3), activation=\"relu\", name=\"conv1_2\"),\n",
    "        layers.BatchNormalization(name=\"bn1_2\"),\n",
    "        layers.MaxPooling2D(pool_size=(2, 2), name=\"pool1\"),\n",
    "        layers.Dropout(0.25, name=\"dropout1\"),\n",
    "        \n",
    "        # Conv Block 2\n",
    "        layers.Conv2D(64, kernel_size=(3, 3), activation=\"relu\", padding=\"same\", name=\"conv2_1\"),\n",
    "        layers.BatchNormalization(name=\"bn2_1\"),\n",
    "        layers.Conv2D(64, kernel_size=(3, 3), activation=\"relu\", name=\"conv2_2\"),\n",
    "        layers.BatchNormalization(name=\"bn2_2\"),\n",
    "        layers.MaxPooling2D(pool_size=(2, 2), name=\"pool2\"),\n",
    "        layers.Dropout(0.25, name=\"dropout2\"),\n",
    "        \n",
    "        # Classification Head\n",
    "        layers.Flatten(name=\"flatten\"),\n",
    "        layers.Dense(128, activation=\"relu\", name=\"dense_128\"),\n",
    "        layers.BatchNormalization(name=\"bn_dense\"),\n",
    "        layers.Dropout(0.50, name=\"dropout_dense\"),\n",
    "        layers.Dense(10, activation=\"softmax\", name=\"output_digits\")\n",
    "    ], name=\"mnist_cnn\")\n",
    "    return model\n",
    "\n",
    "model = create_mnist_cnn()\n",
    "model.summary()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Model Compilation & Training\n",
    "- **Optimizer**: Adam (learning rate = 0.001)\n",
    "- **Loss Function**: Categorical Crossentropy\n",
    "- **Metric**: Accuracy\n",
    "- **Callbacks**:\n",
    "  - `EarlyStopping`: Prevents overfitting by monitoring `val_loss` with patience of 3 epochs and restores best weights.\n",
    "  - `ReduceLROnPlateau`: Halves learning rate when validation loss plateaus."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(\n",
    "    optimizer=keras.optimizers.Adam(learning_rate=0.001),\n",
    "    loss=\"categorical_crossentropy\",\n",
    "    metrics=[\"accuracy\"]\n",
    ")\n",
    "\n",
    "callbacks_list = [\n",
    "    callbacks.EarlyStopping(\n",
    "        monitor=\"val_loss\",\n",
    "        patience=3,\n",
    "        restore_best_weights=True,\n",
    "        verbose=1\n",
    "    ),\n",
    "    callbacks.ReduceLROnPlateau(\n",
    "        monitor=\"val_loss\",\n",
    "        factor=0.5,\n",
    "        patience=2,\n",
    "        min_lr=1e-5,\n",
    "        verbose=1\n",
    "    )\n",
    "]\n",
    "\n",
    "BATCH_SIZE = 128\n",
    "EPOCHS = 10\n",
    "\n",
    "history = model.fit(\n",
    "    x_train,\n",
    "    y_train,\n",
    "    batch_size=BATCH_SIZE,\n",
    "    epochs=EPOCHS,\n",
    "    validation_data=(x_val, y_val),\n",
    "    callbacks=callbacks_list,\n",
    "    verbose=1\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Training History Visualization\n",
    "Plotting loss and accuracy over epochs to evaluate learning dynamics and check for signs of overfitting or underfitting."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n",
    "\n",
    "# Accuracy Curves\n",
    "ax1.plot(history.history[\"accuracy\"], label=\"Training Accuracy\", color=\"#2563EB\", lw=2)\n",
    "ax1.plot(history.history[\"val_accuracy\"], label=\"Validation Accuracy\", color=\"#16A34A\", lw=2, linestyle=\"--\")\n",
    "ax1.set_title(\"Model Accuracy vs. Epochs\", fontsize=13)\n",
    "ax1.set_xlabel(\"Epoch\", fontsize=11)\n",
    "ax1.set_ylabel(\"Accuracy\", fontsize=11)\n",
    "ax1.legend(loc=\"lower right\")\n",
    "ax1.grid(True, linestyle=\"--\", alpha=0.6)\n",
    "\n",
    "# Loss Curves\n",
    "ax2.plot(history.history[\"loss\"], label=\"Training Loss\", color=\"#DC2626\", lw=2)\n",
    "ax2.plot(history.history[\"val_loss\"], label=\"Validation Loss\", color=\"#D97706\", lw=2, linestyle=\"--\")\n",
    "ax2.set_title(\"Model Loss vs. Epochs\", fontsize=13)\n",
    "ax2.set_xlabel(\"Epoch\", fontsize=11)\n",
    "ax2.set_ylabel(\"Loss\", fontsize=11)\n",
    "ax2.legend(loc=\"upper right\")\n",
    "ax2.grid(True, linestyle=\"--\", alpha=0.6)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. Evaluation on Hold-Out Test Set\n",
    "We evaluate strictly on the 10,000 unseen images from the test set."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Evaluate on hold-out test set\n",
    "test_loss, test_accuracy = model.evaluate(x_test_norm, y_test_cat, verbose=0)\n",
    "print(\"=\" * 45)\n",
    "print(f\"Hold-out Test Loss:     {test_loss:.4f}\")\n",
    "print(f\"Hold-out Test Accuracy: {test_accuracy * 100:.2f}%\")\n",
    "print(\"=\" * 45)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 9. Performance Analysis: Confusion Matrix & Classification Report"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Generate predictions\n",
    "y_pred_probs = model.predict(x_test_norm, verbose=0)\n",
    "y_pred = np.argmax(y_pred_probs, axis=1)\n",
    "\n",
    "# Classification Report\n",
    "print(\"--- Classification Report ---\")\n",
    "print(classification_report(y_test_raw, y_pred, digits=4))\n",
    "\n",
    "# Confusion Matrix Heatmap\n",
    "cm = confusion_matrix(y_test_raw, y_pred)\n",
    "\n",
    "plt.figure(figsize=(9, 7))\n",
    "sns.heatmap(cm, annot=True, fmt=\"d\", cmap=\"Blues\", cbar=False,\n",
    "            xticklabels=range(10), yticklabels=range(10))\n",
    "plt.title(\"Confusion Matrix on MNIST Test Set (10,000 Samples)\", fontsize=13, pad=12)\n",
    "plt.xlabel(\"Predicted Digit\", fontsize=11)\n",
    "plt.ylabel(\"True Digit\", fontsize=11)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Misclassified Samples Inspection\n",
    "Examining where the model made errors provides crucial insight into edge cases (e.g., ambiguous handwriting, overlapping strokes)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Find misclassified indices\n",
    "misclassified_idx = np.where(y_test_raw != y_pred)[0]\n",
    "print(f\"Total misclassified test samples: {len(misclassified_idx)} / {len(y_test_raw)}\")\n",
    "\n",
    "plt.figure(figsize=(10, 5))\n",
    "for i, idx in enumerate(misclassified_idx[:10]):\n",
    "    plt.subplot(2, 5, i + 1)\n",
    "    plt.imshow(x_test_raw[idx], cmap=\"gray\")\n",
    "    plt.title(f\"True: {y_test_raw[idx]}\\nPred: {y_pred[idx]}\", color=\"red\", fontsize=10)\n",
    "    plt.axis(\"off\")\n",
    "plt.suptitle(\"Sample Misclassified Digits on Test Set\", fontsize=13)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 10. Save and Export Model for Streamlit Web App\n",
    "We save the complete trained model in modern Keras v3 format as `mnist_cnn.keras`.\n",
    "If running in Google Colab, the snippet below automatically triggers a browser download."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Save model to disk\n",
    "MODEL_FILENAME = \"mnist_cnn.keras\"\n",
    "model.save(MODEL_FILENAME)\n",
    "print(f\"Model successfully saved as '{MODEL_FILENAME}' (Size: {os.path.getsize(MODEL_FILENAME) / (1024*1024):.2f} MB).\")\n",
    "\n",
    "# If running in Google Colab, download the file automatically\n",
    "try:\n",
    "    from google.colab import files\n",
    "    files.download(MODEL_FILENAME)\n",
    "    print(\"Triggered Colab file download. Move this file to the root of your local project.\")\n",
    "except ImportError:\n",
    "    print(\"Running locally. Place 'mnist_cnn.keras' in your repository root.\")"
   ]
  }
 ],
 "metadata": {
  "colab": {
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

os.makedirs(r"d:\Horizon Internship\ML-I\notebooks", exist_ok=True)
notebook_path = r"d:\Horizon Internship\ML-I\notebooks\MNIST_Handwritten_Digit_Recognition.ipynb"
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Notebook generated successfully at:", notebook_path)
