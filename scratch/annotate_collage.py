import os
import sys
sys.path.insert(0, os.path.abspath("."))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from src.preprocess import to_standard_grayscale_array, crop_scale_and_center_digit
from src.model_utils import load_mnist_model, predict_digit
from scratch.test_2d_boxes import extract_2d_boxes

img_path = r'C:\Users\sdoff\.gemini\antigravity-ide\brain\b2dc7757-0c94-47c0-84d9-7aadc17fb6c4\.user_uploaded\media_1792569826524.png'
orig_img = Image.open(img_path).convert("RGB")
gray = to_standard_grayscale_array(orig_img)
binary = (gray > 25.0).astype(np.uint8)

boxes = extract_2d_boxes(binary, min_area=30, min_dim=10)
print(f"Total digits extracted: {len(boxes)}")

model = load_mnist_model('mnist_cnn.keras')

draw = ImageDraw.Draw(orig_img)
results = []
for idx, (ymin, ymax, xmin, xmax) in enumerate(boxes):
    cropped = gray[ymin:ymax, xmin:xmax]
    inp, preview = crop_scale_and_center_digit(cropped)
    res = predict_digit(model, inp)
    d = res['digit']
    conf = res['confidence']
    results.append((idx + 1, d, conf, (ymin, ymax, xmin, xmax)))
    
    # Draw bounding box
    draw.rectangle([xmin, ymin, xmax, ymax], outline="#2563EB", width=3)
    # Draw label tag
    label = f"#{idx+1}: {d} ({int(conf*100)}%)"
    draw.rectangle([xmin, max(0, ymin - 18), xmin + len(label) * 8, ymin], fill="#2563EB")
    draw.text((xmin + 2, max(0, ymin - 17)), label, fill="#FFFFFF")

output_annotated_path = r'C:\Users\sdoff\.gemini\antigravity-ide\brain\b2dc7757-0c94-47c0-84d9-7aadc17fb6c4\annotated_collage.png'
orig_img.save(output_annotated_path)
print(f"Annotated image saved to: {output_annotated_path}")

for item in results:
    print(f"Digit #{item[0]}: {item[1]} (conf={item[2]*100:.1f}%) at box {item[3]}")
