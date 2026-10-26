import os
import sys
sys.path.insert(0, os.path.abspath("."))
import numpy as np
from PIL import Image

def to_standard_grayscale_array(image_input, invert_colors=False):
    if isinstance(image_input, np.ndarray):
        arr = image_input
    elif isinstance(image_input, Image.Image):
        arr = np.array(image_input)
    else:
        raise ValueError("Unsupported input")

    if arr.ndim == 3 and arr.shape[-1] == 4:
        alpha = arr[:, :, 3]
        rgb = arr[:, :, :3]
        if np.all(alpha == 0):
            return np.zeros(arr.shape[:2], dtype=np.float32)
        edge_alpha = np.concatenate([alpha[0, :], alpha[-1, :], alpha[:, 0], alpha[:, -1]])
        if np.mean(edge_alpha) < 50.0:
            gray = np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float32)
            gray = np.where(alpha > 20, gray, 0.0)
            gray[gray < 25.0] = 0.0
            return gray
        else:
            arr = rgb

    if arr.ndim == 3 and arr.shape[-1] >= 3:
        arr_rgb = arr[:, :, :3].astype(np.float32)
        edge_pixels = np.concatenate([arr_rgb[0, :, :], arr_rgb[-1, :, :], arr_rgb[:, 0, :], arr_rgb[:, -1, :]], axis=0)
        bg_color = np.median(edge_pixels, axis=0)
        if np.max(bg_color) < 15.0 and not invert_colors:
            img_arr = np.max(arr_rgb, axis=-1)
        else:
            img_arr = np.max(np.abs(arr_rgb - bg_color), axis=-1)
    elif arr.ndim == 2:
        arr_2d = arr.astype(np.float32)
        edge_pixels = np.concatenate([arr_2d[0, :], arr_2d[-1, :], arr_2d[:, 0], arr_2d[:, -1]])
        bg_val = float(np.median(edge_pixels))
        if bg_val < 15.0 and not invert_colors:
            img_arr = arr_2d
        else:
            img_arr = np.abs(arr_2d - bg_val)
    else:
        img_arr = arr.astype(np.float32)

    img_arr[img_arr < 25.0] = 0.0
    return img_arr

from src.preprocess import extract_digit_bounding_boxes

samples = [
    ("multidigit_42.png", 2),
    ("multidigit_73.png", 2),
    ("multidigit_508.png", 3),
    ("multidigit_91_dark.png", 2),
    ("user_uploaded_collage.png", 36),
    ("test_dark_printed.png", 5),
    ("typed_numbers_test.png", 15),
]

for name, expected in samples:
    img = Image.open("sample_images/" + name)
    gray = to_standard_grayscale_array(img)
    boxes = extract_digit_bounding_boxes(gray)
    status = "OK" if len(boxes) == expected else f"MISMATCH (got {len(boxes)})"
    print(f"{name:30}: expected {expected:2}, got {len(boxes):2} -> {status}")
