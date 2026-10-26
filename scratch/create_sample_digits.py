"""
scratch/create_sample_digits.py
===============================
Generates multiple test images containing handwritten multiple digits:
- sample_images/multidigit_42.png (42 on white paper with natural ink)
- sample_images/multidigit_73.png (73 on white paper)
- sample_images/multidigit_508.png (508 on white paper)
- sample_images/multidigit_91_dark.png (91 on dark canvas)
"""

import os
import numpy as np
from PIL import Image, ImageFilter
import keras

os.makedirs("sample_images", exist_ok=True)

# Load real MNIST digits from test set for authentic handwriting
(_, _), (x_test, y_test) = keras.datasets.mnist.load_data()

def create_multidigit_image(digit_list, filename, bg_color=255, canvas_h=130):
    """
    Stitch real MNIST handwritten digits onto a realistic paper canvas with natural spacing.
    """
    digit_size = 56
    gap = 24
    padding = 35
    total_w = padding * 2 + len(digit_list) * digit_size + (len(digit_list) - 1) * gap
    canvas = np.full((canvas_h, total_w), bg_color, dtype=np.uint8)

    cur_x = padding
    for d in digit_list:
        # Find a real MNIST sample of this digit
        indices = np.where(y_test == d)[0]
        chosen_idx = indices[3 if len(indices) > 3 else 0]
        mnist_digit = x_test[chosen_idx]  # 28x28, 0=bg, 255=digit

        # Resize to 56x56
        d_pil = Image.fromarray(mnist_digit).resize((digit_size, digit_size), Image.Resampling.BICUBIC)
        d_arr = np.array(d_pil, dtype=np.float32)

        paste_y = (canvas_h - digit_size) // 2

        if bg_color == 255:
            # White background, dark stroke
            stroke_strength = d_arr / 255.0
            patch = canvas[paste_y : paste_y + digit_size, cur_x : cur_x + digit_size]
            canvas[paste_y : paste_y + digit_size, cur_x : cur_x + digit_size] = np.clip(
                patch * (1.0 - stroke_strength), 0, 255
            ).astype(np.uint8)
        else:
            # Black background, white stroke
            patch = canvas[paste_y : paste_y + digit_size, cur_x : cur_x + digit_size]
            canvas[paste_y : paste_y + digit_size, cur_x : cur_x + digit_size] = np.clip(
                patch + d_arr, 0, 255
            ).astype(np.uint8)

        cur_x += digit_size + gap

    pil_img = Image.fromarray(canvas)
    pil_img = pil_img.filter(ImageFilter.SMOOTH_MORE)
    pil_img.save(filename)
    print(f"Generated: {filename} (Target Number: {''.join(str(d) for d in digit_list)})")
    return pil_img

# 1. Multi-digit "42" (White background, black ink)
create_multidigit_image([4, 2], "sample_images/multidigit_42.png", bg_color=255)

# 2. Multi-digit "73" (White background, black ink)
create_multidigit_image([7, 3], "sample_images/multidigit_73.png", bg_color=255)

# 3. Multi-digit "508" (White background, black ink)
create_multidigit_image([5, 0, 8], "sample_images/multidigit_508.png", bg_color=255)

# 4. Multi-digit "91" (Black background, white ink)
create_multidigit_image([9, 1], "sample_images/multidigit_91_dark.png", bg_color=0)

print("\nAll sample test images generated in 'sample_images/' directory!")
