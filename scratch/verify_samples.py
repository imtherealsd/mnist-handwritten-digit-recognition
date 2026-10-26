import sys
sys.path.insert(0, ".")
from PIL import Image
from src.preprocess import segment_and_preprocess_digits
from src.model_utils import load_mnist_model, predict_digit

model = load_mnist_model("mnist_cnn.keras")
test_files = [
    ("sample_images/multidigit_42.png", "42"),
    ("sample_images/multidigit_73.png", "73"),
    ("sample_images/multidigit_508.png", "508"),
    ("sample_images/multidigit_91_dark.png", "91"),
]

print("=" * 60)
print("MULTI-DIGIT ACCURACY VERIFICATION REPORT")
print("=" * 60)
for path, expected in test_files:
    img = Image.open(path)
    digits = segment_and_preprocess_digits(img)
    preds = [predict_digit(model, d["model_input"]) for d in digits]
    predicted_number = "".join(str(p["digit"]) for p in preds)
    confs = [f"{p['confidence']*100:.1f}%" for p in preds]
    status = "EXACT MATCH" if predicted_number == expected else "MISMATCH"
    print(f"File: {path}")
    print(f"  Target Number:    {expected}")
    print(f"  Predicted Number: {predicted_number} [{status}]")
    print(f"  Segmented Digits: {[p['digit'] for p in preds]}")
    print(f"  Confidences:      {confs}")
    print("-" * 50)
print("=" * 60)
