import os
import sys
sys.path.insert(0, os.path.abspath("."))
import numpy as np
import scipy.ndimage as ndi
from PIL import Image
from src.preprocess import to_standard_grayscale_array, crop_scale_and_center_digit
from src.model_utils import load_mnist_model, predict_digit

def extract_2d_boxes(binary, min_area=30, min_dim=8):
    labeled, num_features = ndi.label(binary)
    slices = ndi.find_objects(labeled)
    
    boxes = []
    for sl in slices:
        if sl is None: continue
        ymin, ymax = sl[0].start, sl[0].stop
        xmin, xmax = sl[1].start, sl[1].stop
        h = ymax - ymin
        w = xmax - xmin
        if h < min_dim or w < min_dim: continue
        if np.sum(binary[ymin:ymax, xmin:xmax]) < min_area: continue
        boxes.append([ymin, ymax, xmin, xmax])
        
    changed = True
    while changed:
        changed = False
        new_boxes = []
        skip = set()
        for i in range(len(boxes)):
            if i in skip: continue
            b1 = boxes[i]
            for j in range(i + 1, len(boxes)):
                if j in skip: continue
                b2 = boxes[j]
                
                y_overlap = min(b1[1], b2[1]) - max(b1[0], b2[0])
                x_overlap = min(b1[3], b2[3]) - max(b1[2], b2[2])
                
                y_gap = max(0, max(b1[0], b2[0]) - min(b1[1], b2[1]))
                min_h = min(b1[1] - b1[0], b2[1] - b2[0])
                min_w = min(b1[3] - b1[2], b2[3] - b2[2])
                
                should_merge = False
                if x_overlap > 0 and y_overlap > 0:
                    iou_area = x_overlap * y_overlap
                    area1 = (b1[1] - b1[0]) * (b1[3] - b1[2])
                    area2 = (b2[1] - b2[0]) * (b2[3] - b2[2])
                    if iou_area / min(area1, area2) > 0.15:
                        should_merge = True
                elif y_gap < (min_h * 0.15) and x_overlap > (min_w * 0.5):
                    should_merge = True
                    
                if should_merge:
                    b1 = [min(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), max(b1[3], b2[3])]
                    skip.add(j)
                    changed = True
            new_boxes.append(b1)
        boxes = new_boxes

    if len(boxes) > 1:
        y_centers = [(b[0] + b[1]) / 2.0 for b in boxes]
        avg_h = np.mean([b[1] - b[0] for b in boxes])
        if np.std(y_centers) < (avg_h * 0.5):
            boxes.sort(key=lambda b: b[2])
        else:
            row_height = max(15.0, avg_h * 0.75)
            boxes.sort(key=lambda b: (round(((b[0] + b[1]) / 2.0) / row_height), b[2]))
            
    return boxes

model = load_mnist_model('mnist_cnn.keras')

samples = [
    'sample_images/multidigit_42.png',
    'sample_images/multidigit_73.png',
    'sample_images/multidigit_508.png',
    'sample_images/multidigit_91_dark.png'
]

for path in samples:
    img = Image.open(path)
    gray = to_standard_grayscale_array(img)
    binary = (gray > 25.0).astype(np.uint8)
    boxes = extract_2d_boxes(binary)
    preds = []
    for b in boxes:
        cropped = gray[b[0]:b[1], b[2]:b[3]]
        inp, _ = crop_scale_and_center_digit(cropped)
        res = predict_digit(model, inp)
        preds.append(str(res['digit']))
    print(f"{os.path.basename(path)}: boxes={len(boxes)}, predicted={''.join(preds)}")
