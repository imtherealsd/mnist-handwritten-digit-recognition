"""
tests/test_preprocess.py
========================
Unit tests for the MNIST image preprocessing pipeline.
Verifies empty canvas detection, output dimensions, data types,
pixel value normalization, aspect-ratio scaling, centering,
and multi-digit segmentation.
"""

import numpy as np
import pytest
from PIL import Image

from src.preprocess import (
    is_canvas_empty,
    preprocess_image,
    shift_image_to_center_of_mass,
    extract_digit_bounding_boxes,
    segment_and_preprocess_digits,
)


def test_is_canvas_empty_on_none_and_blank():
    """Verify is_canvas_empty handles None and all-zero canvas arrays correctly."""
    assert is_canvas_empty(None) is True

    # Empty 280x280 RGBA array (black canvas, zero alpha)
    blank_rgba = np.zeros((280, 280, 4), dtype=np.uint8)
    assert is_canvas_empty(blank_rgba) is True

    # Empty 280x280 grayscale
    blank_gray = np.zeros((280, 280), dtype=np.uint8)
    assert is_canvas_empty(blank_gray) is True


def test_is_canvas_empty_with_strokes():
    """Verify is_canvas_empty detects user drawings."""
    canvas = np.zeros((280, 280, 4), dtype=np.uint8)
    canvas[:, :, 3] = 255  # Fully opaque

    # Draw a line of white pixels
    canvas[100:150, 140:145, :3] = 255
    assert is_canvas_empty(canvas) is False


def test_preprocess_output_shape_and_dtype():
    """Verify output shape is (1, 28, 28, 1) and dtype is float32."""
    canvas = np.zeros((280, 280, 4), dtype=np.uint8)
    canvas[:, :, 3] = 255
    # Draw a vertical stroke simulating the digit '1'
    canvas[50:220, 135:145, :3] = 255

    model_input, preview = preprocess_image(canvas)

    assert model_input.shape == (1, 28, 28, 1)
    assert model_input.dtype == np.float32
    assert preview.shape == (28, 28)
    assert preview.dtype == np.uint8


def test_preprocess_pixel_value_range():
    """Verify all pixel values in model_input are strictly normalized between 0.0 and 1.0."""
    canvas = np.zeros((200, 200, 3), dtype=np.uint8)
    # Draw an arbitrary stroke
    canvas[60:140, 60:140] = 220

    model_input, preview = preprocess_image(canvas)

    assert np.all(model_input >= 0.0)
    assert np.all(model_input <= 1.0)
    assert float(np.max(model_input)) > 0.5


def test_preprocess_centering_and_scaling():
    """
    Verify that an off-center digit drawn in the bottom-right corner
    is properly cropped, scaled to 20x20, and centered into the 28x28 canvas.
    """
    canvas = np.zeros((300, 300), dtype=np.uint8)
    # Draw a block in bottom-right corner (y: 240-280, x: 240-280)
    canvas[240:280, 240:280] = 255

    model_input, preview = preprocess_image(canvas)

    # In the 28x28 output, the center region (e.g. y: 8-20, x: 8-20) must now have high intensity
    center_region = preview[8:20, 8:20]
    assert np.max(center_region) > 100


def test_preprocess_inversion_of_dark_on_white():
    """
    Verify that black ink on a white background (typical photo/scan)
    is automatically inverted to white stroke on black background.
    """
    # White background (255) with a black stroke (0)
    white_page = np.full((100, 100), 255, dtype=np.uint8)
    white_page[30:70, 45:55] = 0

    model_input, preview = preprocess_image(white_page)

    # Bounded stroke should become bright (>0.5) in model_input
    assert np.max(model_input) > 0.5
    # The edges should be dark (0.0) in MNIST style
    assert model_input[0, 0, 0, 0] == 0.0


def test_preprocess_blank_returns_zeros():
    """Verify preprocessing a blank canvas returns an all-zero array of correct shape."""
    blank = np.zeros((280, 280), dtype=np.uint8)
    model_input, preview = preprocess_image(blank)

    assert model_input.shape == (1, 28, 28, 1)
    assert np.all(model_input == 0.0)
    assert np.all(preview == 0)


def test_multi_digit_segmentation_two_digits():
    """
    Verify that segment_and_preprocess_digits detects and extracts multiple
    digits drawn side-by-side in left-to-right order.
    """
    wide_canvas = np.zeros((200, 450), dtype=np.uint8)
    # Draw first digit stroke on left (x: 50-80, y: 50-150)
    wide_canvas[50:150, 50:80] = 255
    # Draw second digit stroke on right (x: 220-250, y: 50-150)
    wide_canvas[50:150, 220:250] = 255

    digits = segment_and_preprocess_digits(wide_canvas)

    assert len(digits) == 2
    # Ensure sorted left to right
    assert digits[0]["bbox"][2] < digits[1]["bbox"][2]
    # Each digit must have (1, 28, 28, 1) float32 shape
    for d in digits:
        assert d["model_input"].shape == (1, 28, 28, 1)
        assert d["model_input"].dtype == np.float32
        assert d["preview_28x28"].shape == (28, 28)
        assert np.max(d["model_input"]) > 0.5


def test_multi_digit_segmentation_single_digit():
    """Verify single digit returns a list with exactly 1 element."""
    canvas = np.zeros((200, 200), dtype=np.uint8)
    canvas[40:120, 80:110] = 255

    digits = segment_and_preprocess_digits(canvas)

    assert len(digits) == 1
    assert digits[0]["model_input"].shape == (1, 28, 28, 1)


def test_multi_digit_empty_returns_empty_list():
    """Verify blank input returns empty list."""
    blank = np.zeros((200, 400), dtype=np.uint8)
    assert segment_and_preprocess_digits(blank) == []
    assert segment_and_preprocess_digits(None) == []


def test_multi_digit_2d_scatter_preserves_independent_digits():
    """
    Verify that digits placed above/below each other with overlapping X coordinates
    are NOT erroneously merged into a single digit.
    """
    canvas_2d = np.zeros((400, 300), dtype=np.uint8)
    # Digit 1 at top-left: y in [30, 90], x in [40, 80]
    canvas_2d[30:90, 40:80] = 255
    # Digit 2 at bottom-left: y in [220, 280], x in [40, 80] (same x-column!)
    canvas_2d[220:280, 40:80] = 255

    digits = segment_and_preprocess_digits(canvas_2d)
    assert len(digits) == 2, f"Expected 2 separate digits, got {len(digits)}"


def test_color_distance_on_white_background():
    """
    Verify that bright colored strokes (e.g. yellow or cyan) on a white page
    are preserved rather than faded out by naive luminance thresholding.
    """
    white_page = np.full((120, 120, 3), 255, dtype=np.uint8)
    # Bright yellow stroke (255, 230, 0)
    white_page[30:90, 45:65] = [255, 230, 0]

    digits = segment_and_preprocess_digits(white_page)
    assert len(digits) == 1
    assert float(np.max(digits[0]["model_input"])) > 0.5

