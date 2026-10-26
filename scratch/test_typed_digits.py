import os
import sys
sys.path.insert(0, os.path.abspath("."))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from src.preprocess import segment_and_preprocess_digits
from src.model_utils import load_mnist_model, predict_digit

# Let's create an image with typed / printed numbers in various standard fonts
img = Image.new("RGB", (700, 300), (255, 255, 255))
draw = ImageDraw.Draw(img)

# Try to use standard windows fonts
font_paths = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/times.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "C:/Windows/Fonts/consola.ttf",
]

text_samples = [
    ("2024", 50, 40),
    ("58913", 50, 120),
    ("47 62 80", 50, 200),
]

for text, x, y in text_samples:
    try:
        font = ImageFont.truetype("arial.ttf", 48)
    except:
        font = ImageFont.load_default()
    draw.text((x, y), text, fill=(0, 0, 0), font=font)

test_path = "sample_images/typed_numbers_test.png"
img.save(test_path)
print(f"Saved {test_path}")

# Now test segmentation and prediction
digits = segment_and_preprocess_digits(img)
print(f"Total digits extracted from typed text: {len(digits)}")

model = load_mnist_model("mnist_cnn.keras")
preds = []
for idx, d in enumerate(digits):
    res = predict_digit(model, d["model_input"])
    preds.append(f"#{idx+1}: {res['digit']} ({res['confidence']*100:.1f}%) [box={d['bbox']}]")

print("Predictions on typed numbers:")
for p in preds:
    print(p)
