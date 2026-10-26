"""
tests/test_model_utils.py
=========================
Unit tests for model loading, building, and inference utilities.
These tests use lightweight mock/dummy model interfaces to avoid retraining.
"""

import numpy as np
import pytest

from src.model_utils import load_mnist_model, predict_digit


class DummyPredictor:
    """Mock model implementing .predict() for unit testing without loading weights."""

    def __init__(self, target_digit: int = 7):
        self.target_digit = target_digit

    def predict(self, x, verbose=0):
        batch_size = x.shape[0]
        # Return probability distribution with peak at target_digit
        probs = np.full((batch_size, 10), 0.02, dtype=np.float32)
        probs[:, self.target_digit] = 0.82
        return probs


def test_load_mnist_model_missing_file():
    """Verify load_mnist_model raises FileNotFoundError with helpful guidance if file is missing."""
    with pytest.raises(FileNotFoundError) as exc_info:
        load_mnist_model("non_existent_model_file.keras")
    assert "not found" in str(exc_info.value).lower()
    assert "colab" in str(exc_info.value).lower()


def test_predict_digit_mock_model():
    """Verify predict_digit parses predictions, confidence, and probabilities properly."""
    dummy_model = DummyPredictor(target_digit=3)
    dummy_image = np.zeros((1, 28, 28, 1), dtype=np.float32)

    result = predict_digit(dummy_model, dummy_image)

    assert isinstance(result, dict)
    assert result["digit"] == 3
    assert pytest.approx(result["confidence"], 0.01) == 0.82
    assert len(result["probabilities"]) == 10
    assert pytest.approx(sum(result["probabilities"]), 0.01) == 1.0


def test_predict_digit_handles_flexible_shapes():
    """Verify predict_digit accepts (28, 28) and (28, 28, 1) and reshapes correctly."""
    dummy_model = DummyPredictor(target_digit=5)

    # 2D shape (28, 28)
    img_2d = np.zeros((28, 28), dtype=np.float32)
    res_2d = predict_digit(dummy_model, img_2d)
    assert res_2d["digit"] == 5

    # 3D shape (28, 28, 1)
    img_3d = np.zeros((28, 28, 1), dtype=np.float32)
    res_3d = predict_digit(dummy_model, img_3d)
    assert res_3d["digit"] == 5


def test_predict_digit_invalid_inputs():
    """Verify predict_digit raises ValueError for invalid shapes or None input."""
    dummy_model = DummyPredictor()

    with pytest.raises(ValueError):
        predict_digit(dummy_model, None)

    with pytest.raises(ValueError):
        invalid_shape_img = np.zeros((32, 32), dtype=np.float32)
        predict_digit(dummy_model, invalid_shape_img)
