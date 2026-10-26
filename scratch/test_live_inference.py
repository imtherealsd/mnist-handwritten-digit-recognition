import sys
sys.path.insert(0, ".")
import keras
from src.model_utils import load_mnist_model, predict_digit

# Load test dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Load our genuine trained model
model = load_mnist_model("mnist_cnn.keras")

# Test on the first 5 test samples
print("=" * 50)
print("Live Inference Test on Genuine Trained Model")
print("=" * 50)

for i in range(5):
    img = x_test[i].astype("float32") / 255.0
    true_label = int(y_test[i])
    result = predict_digit(model, img)
    pred_label = result["digit"]
    confidence = result["confidence"] * 100.0
    status = "CORRECT" if pred_label == true_label else "MISMATCH"
    print(f"Sample #{i+1}: True={true_label} | Pred={pred_label} | Confidence={confidence:.2f}% | [{status}]")

print("=" * 50)
