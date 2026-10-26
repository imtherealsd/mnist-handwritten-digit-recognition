import os
import sys
sys.path.insert(0, os.path.abspath("."))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from src.preprocess import segment_and_preprocess_digits
from src.model_utils import load_mnist_model, predict_digit

# Test 1: Light text on dark background (inverted printed text)
img1 = Image.new("RGB", (400, 150), (20, 20, 30))
draw1 = ImageDraw.Draw(img1)
try:
    font = ImageFont.truetype("arial.ttf", 36)
except:
    font = ImageFont.load_default()
draw1.text((30, 50), "98765", fill=(240, 240, 240), font=font)
img1.save("sample_images/test_dark_printed.png")

# Test 2: Multi-line typed phone number / date
img2 = Image.new("RGB", (450, 220), (245, 245, 240))
draw2 = ImageDraw.Draw(img2)
draw2.text((40, 30), "Call: 800 555 0199", fill=(10, 10, 10), font=font)
draw2.text((40, 110), "Date: 2026-10-21", fill=(10, 10, 10), font=font)
img2.save("sample_images/test_multiline_typed.png")

# Test 3: Receipt style monospace numbers
img3 = Image.new("RGB", (350, 200), (255, 255, 255))
draw3 = ImageDraw.Draw(img3)
try:
    mono_font = ImageFont.truetype("consola.ttf", 32)
except:
    mono_font = font
draw3.text((30, 30), "QTY: 42", fill=(40, 40, 40), font=mono_font)
draw3.text((30, 80), "TOTAL: 1890", fill=(40, 40, 40), font=mono_font)
img3.save("sample_images/test_receipt_mono.png")

print("Created 3 challenging test images!")

model = load_mnist_model("mnist_cnn.keras")

for name, img in [("Dark Background", img1), ("Multi-line Typed", img2), ("Receipt Style", img3)]:
    digits = segment_and_preprocess_digits(img)
    preds = [str(predict_digit(model, d["model_input"])["digit"]) for d in digits]
    print(f"{name}: found {len(digits)} digits -> {' '.join(preds)}")
