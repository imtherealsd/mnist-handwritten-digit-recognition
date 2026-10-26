import sys
sys.path.insert(0, ".")
import numpy as np
import scipy.ndimage as ndi
from PIL import Image
from src.model_utils import load_mnist_model, predict_digit
from src.preprocess import shift_image_to_center_of_mass

def extract_digit_bounding_boxes(img_arr, min_area=30, min_dim=6):
    binary = (img_arr > 25.0).astype(np.uint8)
    if not np.any(binary):
        return []

    # Small closing/dilation to group strokes within same digit
    struct = np.ones((7, 7), dtype=np.uint8)
    dilated = ndi.binary_dilation(binary, structure=struct)

    labeled, num_features = ndi.label(dilated)
    slices = ndi.find_objects(labeled)

    raw_boxes = []
    for sl in slices:
        if sl is None:
            continue
        y_sl, x_sl = sl
        ymin, ymax = y_sl.start, y_sl.stop
        xmin, xmax = x_sl.start, x_sl.stop
        h = ymax - ymin
        w = xmax - xmin
        if h < min_dim or w < min_dim:
            continue
        if np.sum(binary[ymin:ymax, xmin:xmax]) < min_area:
            continue
        raw_boxes.append([ymin, ymax, xmin, xmax])

    if not raw_boxes:
        return []

    raw_boxes.sort(key=lambda b: b[2])

    merged = []
    for box in raw_boxes:
        if not merged:
            merged.append(box)
            continue
        prev = merged[-1]
        overlap = min(prev[3], box[3]) - max(prev[2], box[2])
        gap = box[2] - prev[3]
        avg_h = ((prev[1] - prev[0]) + (box[1] - box[0])) / 2.0
        if overlap > 0 or gap < (avg_h * 0.15):
            prev[0] = min(prev[0], box[0])
            prev[1] = max(prev[1], box[1])
            prev[2] = min(prev[2], box[2])
            prev[3] = max(prev[3], box[3])
        else:
            merged.append(box)

    return merged

def process_cropped_digit(cropped, target_box_size=20, canvas_size=28):
    crop_h, crop_w = cropped.shape
    if crop_h > crop_w:
        scale = target_box_size / float(crop_h)
        new_h = target_box_size
        new_w = max(1, int(round(crop_w * scale)))
    else:
        scale = target_box_size / float(crop_w)
        new_w = target_box_size
        new_h = max(1, int(round(crop_h * scale)))

    cropped_pil = Image.fromarray(cropped.astype(np.uint8))
    resized_pil = cropped_pil.resize((new_w, new_h), Image.Resampling.BICUBIC)
    resized_arr = np.array(resized_pil, dtype=np.float32)

    canvas_28x28 = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    start_y = (canvas_size - new_h) // 2
    start_x = (canvas_size - new_w) // 2
    canvas_28x28[start_y : start_y + new_h, start_x : start_x + new_w] = resized_arr

    final_28x28 = shift_image_to_center_of_mass(canvas_28x28)
    final_28x28 = np.clip(final_28x28, 0.0, 255.0)

    normalized = (final_28x28 / 255.0).astype(np.float32)
    model_input = np.expand_dims(normalized, axis=(0, -1))
    return model_input, final_28x28.astype(np.uint8)

# Load real MNIST digits from test set and stitch them side-by-side as "42"
import keras
(_, _), (x_test, y_test) = keras.datasets.mnist.load_data()

# Find a '4' and a '2'
idx_4 = np.where(y_test == 4)[0][0]
idx_2 = np.where(y_test == 2)[0][0]

digit_4 = x_test[idx_4]
digit_2 = x_test[idx_2]

# Create a wide canvas of 100 x 250 and place '4' and '2' side by side with gap
wide_canvas = np.zeros((100, 250), dtype=np.float32)
wide_canvas[30:58, 40:68] = digit_4
wide_canvas[30:58, 140:168] = digit_2

boxes = extract_digit_bounding_boxes(wide_canvas)
print(f"Detected {len(boxes)} digits on canvas!")

model = load_mnist_model("mnist_cnn.keras")

digits_detected = []
for i, b in enumerate(boxes):
    crop = wide_canvas[b[0]:b[1], b[2]:b[3]]
    model_input, preview = process_cropped_digit(crop)
    res = predict_digit(model, model_input)
    digits_detected.append(str(res["digit"]))
    print(f"  Digit #{i+1}: Predicted={res['digit']} (Confidence: {res['confidence']*100:.1f}%)")

full_number = "".join(digits_detected)
print(f"--> Combined Multidigit Prediction: {full_number}")
