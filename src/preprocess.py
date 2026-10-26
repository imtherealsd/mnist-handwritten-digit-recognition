"""
src/preprocess.py
=================
Preprocessing utilities for Handwritten Digit Recognition (MNIST).

Supports:
1. Single-digit standard MNIST preprocessing (LeCun et al. standard).
2. Multi-digit automatic segmentation (detects and extracts multiple digits
   drawn side-by-side like '42', '100', '73' from left to right).
3. Auto-polarity inversion for uploaded photos on white paper.
4. Bounding box detection, aspect-ratio preserving 20x20 scaling,
   and Center of Mass alignment inside a 28x28 frame.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
from PIL import Image
import scipy.ndimage as ndi
from scipy.ndimage import center_of_mass


def is_canvas_empty(image_input, threshold: float = 10.0) -> bool:
    """
    Check if the drawing canvas or image contains any user strokes.

    Parameters
    ----------
    image_input : np.ndarray or PIL.Image.Image
        Input image from canvas or file upload.
    threshold : float
        Minimum pixel intensity threshold to consider as content.

    Returns
    -------
    bool
        True if the image is considered empty/blank, False otherwise.
    """
    if image_input is None:
        return True

    if isinstance(image_input, Image.Image):
        arr = np.array(image_input)
    else:
        arr = np.array(image_input)

    if arr.size == 0:
        return True

    # If image has alpha channel (e.g. RGBA from canvas)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        alpha = arr[:, :, 3]
        rgb = arr[:, :, :3]
        if np.max(alpha) == 0:
            return True
        rgb_mean = np.mean(rgb, axis=-1)
        if np.ptp(rgb_mean) < threshold and np.max(rgb_mean) < threshold:
            return True
        return False
    elif arr.ndim == 3:
        rgb_mean = np.mean(arr, axis=-1)
        return float(np.ptp(rgb_mean)) < threshold and float(np.max(rgb_mean)) < threshold
    else:
        return float(np.ptp(arr)) < threshold and float(np.max(arr)) < threshold


def shift_image_to_center_of_mass(img_28x28: np.ndarray) -> np.ndarray:
    """
    Shift a 28x28 grayscale image so its center of mass aligns at (14, 14),
    replicating the exact preprocessing method used for the original MNIST dataset.

    Parameters
    ----------
    img_28x28 : np.ndarray
        2D array of shape (28, 28) with values in range [0, 255].

    Returns
    -------
    np.ndarray
        Shifted 2D array of shape (28, 28).
    """
    cy, cx = center_of_mass(img_28x28)

    if np.isnan(cx) or np.isnan(cy):
        return img_28x28

    shift_x = int(np.round(14.0 - cx))
    shift_y = int(np.round(14.0 - cy))

    shifted = np.zeros_like(img_28x28)
    src_y_start = max(0, -shift_y)
    src_y_end = min(28, 28 - shift_y)
    src_x_start = max(0, -shift_x)
    src_x_end = min(28, 28 - shift_x)

    dst_y_start = max(0, shift_y)
    dst_y_end = min(28, 28 + shift_y)
    dst_x_start = max(0, shift_x)
    dst_x_end = min(28, 28 + shift_x)

    shifted[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = img_28x28[
        src_y_start:src_y_end, src_x_start:src_x_end
    ]
    return shifted


def to_standard_grayscale_array(image_input, invert_colors: bool = False) -> np.ndarray:
    """
    Convert canvas RGBA or uploaded RGB/Grayscale image to a clean 2D float32 array
    where 0 is black background and 255 is white stroke (MNIST convention).

    Supports:
    - Dark strokes on white/light background (auto-inverts).
    - Multi-colored strokes/fonts on white background (via background color distance).
    - White strokes on black background (standard canvas / MNIST format).
    """
    if isinstance(image_input, np.ndarray):
        arr = image_input
    elif isinstance(image_input, Image.Image):
        arr = np.array(image_input)
    else:
        raise ValueError("Unsupported image input type. Expected numpy.ndarray or PIL.Image.")

    # 1. Canvas RGBA input with potential alpha channel
    if arr.ndim == 3 and arr.shape[-1] == 4:
        alpha = arr[:, :, 3]
        rgb = arr[:, :, :3]
        if np.all(alpha == 0):
            return np.zeros(arr.shape[:2], dtype=np.float32)

        edge_alpha = np.concatenate([alpha[0, :], alpha[-1, :], alpha[:, 0], alpha[:, -1]])
        if np.mean(edge_alpha) < 50.0:
            # Drawing canvas with transparent background: strokes are colored/white on transparent
            gray = np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float32)
            gray = np.where(alpha > 20, gray, 0.0)
            img_arr = gray
            img_arr[img_arr < 25.0] = 0.0
            return img_arr
        else:
            arr = rgb

    # 2. RGB 3-channel input (e.g. photos, scans, colored graphics)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        arr_rgb = arr[:, :, :3].astype(np.float32)
        edge_pixels = np.concatenate(
            [arr_rgb[0, :, :], arr_rgb[-1, :, :], arr_rgb[:, 0, :], arr_rgb[:, -1, :]],
            axis=0,
        )
        bg_color = np.median(edge_pixels, axis=0)

        # If background is already pure black (like dark canvas exported as RGB)
        if float(np.max(bg_color)) < 15.0 and not invert_colors:
            img_arr = np.max(arr_rgb, axis=-1)
        else:
            # Color distance from background captures strokes on any background (white, dark, or colored)
            img_arr = np.max(np.abs(arr_rgb - bg_color), axis=-1)

    # 3. Grayscale 2D input
    elif arr.ndim == 2:
        arr_2d = arr.astype(np.float32)
        edge_pixels = np.concatenate(
            [arr_2d[0, :], arr_2d[-1, :], arr_2d[:, 0], arr_2d[:, -1]]
        )
        bg_val = float(np.median(edge_pixels))
        if bg_val < 15.0 and not invert_colors:
            img_arr = arr_2d
        else:
            img_arr = np.abs(arr_2d - bg_val)
    else:
        img_arr = arr.astype(np.float32)

    # Zero out faint background noise
    img_arr[img_arr < 25.0] = 0.0
    return img_arr


def extract_digit_bounding_boxes(
    img_arr: np.ndarray,
    min_area: int = 20,
    min_dim: int = 5
) -> List[Tuple[int, int, int, int]]:
    """
    Identify and extract bounding boxes for all distinct digits in an image.
    Uses 2D connected component analysis with intelligent stroke merging
    and natural reading order sorting (handles horizontal rows like '42',
    2D scattered digits or collages, and multi-line typed/printed text).

    Parameters
    ----------
    img_arr : np.ndarray
        2D grayscale array where 0 is background and values > 0 are strokes.
    min_area : int
        Minimum number of non-zero pixels required to consider as a digit.
    min_dim : int
        Minimum height or width in pixels.

    Returns
    -------
    List[Tuple[int, int, int, int]]
        List of bounding boxes (ymin, ymax, xmin, xmax) sorted in reading order.
    """
    binary = (img_arr > 25.0).astype(np.uint8)
    if not np.any(binary):
        return []

    labeled, num_features = ndi.label(binary)
    slices = ndi.find_objects(labeled)

    boxes = []
    total_area = img_arr.shape[0] * img_arr.shape[1]
    for sl in slices:
        if sl is None:
            continue
        ymin, ymax = sl[0].start, sl[0].stop
        xmin, xmax = sl[1].start, sl[1].stop
        h = ymax - ymin
        w = xmax - xmin
        box_area = h * w
        if h < min_dim or w < min_dim:
            continue
        if box_area > (total_area * 0.95):
            continue
        # Filter flat horizontal lines / hyphens / underlines
        if (w / max(1, h)) > 3.0 and h < 20:
            continue
        if np.sum(binary[ymin:ymax, xmin:xmax]) < min_area:
            continue
        boxes.append([ymin, ymax, xmin, xmax])

    if not boxes:
        return []

    # Iterative 2D stroke merging:
    # Only merge two components if they actually intersect in 2D or are
    # tightly stacked parts of the same glyph (e.g. crossbar of 4, dot, hat of 5).
    # Never merge components simply because they happen to share an X-coordinate.
    changed = True
    while changed:
        changed = False
        new_boxes = []
        skip = set()
        for i in range(len(boxes)):
            if i in skip:
                continue
            b1 = boxes[i]
            for j in range(i + 1, len(boxes)):
                if j in skip:
                    continue
                b2 = boxes[j]

                y_overlap = min(b1[1], b2[1]) - max(b1[0], b2[0])
                x_overlap = min(b1[3], b2[3]) - max(b1[2], b2[2])

                y_gap = max(0, max(b1[0], b2[0]) - min(b1[1], b2[1]))
                x_gap = max(0, max(b1[2], b2[2]) - min(b1[3], b2[3]))

                min_h = min(b1[1] - b1[0], b2[1] - b2[0])
                min_w = min(b1[3] - b1[2], b2[3] - b2[2])

                should_merge = False
                # Case 1: True 2D intersection or enclosure
                if x_overlap > 0 and y_overlap > 0:
                    iou_area = x_overlap * y_overlap
                    area1 = (b1[1] - b1[0]) * (b1[3] - b1[2])
                    area2 = (b2[1] - b2[0]) * (b2[3] - b2[2])
                    if (iou_area / min(area1, area2)) > 0.15:
                        should_merge = True

                # Case 2: Vertically stacked stroke fragments (e.g. top of 5 or 7, or crossbars)
                elif y_gap < (min_h * 0.15) and x_overlap > (min_w * 0.5):
                    should_merge = True

                if should_merge:
                    b1 = [
                        min(b1[0], b2[0]),
                        max(b1[1], b2[1]),
                        min(b1[2], b2[2]),
                        max(b1[3], b2[3]),
                    ]
                    skip.add(j)
                    changed = True
            new_boxes.append(b1)
        boxes = new_boxes

    # Sort in human reading order:
    if len(boxes) > 1:
        y_centers = [(b[0] + b[1]) / 2.0 for b in boxes]
        avg_h = np.mean([b[1] - b[0] for b in boxes])
        # If all digits are aligned in a single horizontal row:
        if float(np.std(y_centers)) < (avg_h * 0.45):
            boxes.sort(key=lambda b: b[2])
        else:
            # Multi-line or 2D scattered digits: sort top-to-bottom, left-to-right
            row_height = max(15.0, avg_h * 0.75)
            boxes.sort(key=lambda b: (round(((b[0] + b[1]) / 2.0) / row_height), b[2]))

    return [(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in boxes]


def crop_scale_and_center_digit(
    cropped_arr: np.ndarray,
    target_box_size: int = 20,
    canvas_size: int = 28
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standardize a single cropped digit array into 28x28 MNIST format.
    """
    crop_h, crop_w = cropped_arr.shape
    if crop_h == 0 or crop_w == 0:
        blank = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        return np.zeros((1, canvas_size, canvas_size, 1), dtype=np.float32), blank

    if crop_h > crop_w:
        scale = target_box_size / float(crop_h)
        new_h = target_box_size
        new_w = max(1, int(round(crop_w * scale)))
    else:
        scale = target_box_size / float(crop_w)
        new_w = target_box_size
        new_h = max(1, int(round(crop_h * scale)))

    cropped_pil = Image.fromarray(cropped_arr.astype(np.uint8))
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
    preview_28x28 = final_28x28.astype(np.uint8)

    return model_input, preview_28x28


def segment_and_preprocess_digits(
    image_input,
    target_box_size: int = 20,
    canvas_size: int = 28,
    invert_colors: bool = False
) -> List[Dict[str, Any]]:
    """
    Segment one or multiple digits from an input image and preprocess each
    into standard MNIST specifications (1, 28, 28, 1).

    Parameters
    ----------
    image_input : np.ndarray or PIL.Image.Image
        Raw canvas image or uploaded image.
    target_box_size : int, default=20
        Dimension of inner box to fit digit (MNIST standard is 20).
    canvas_size : int, default=28
        Total output width/height (MNIST standard is 28).
    invert_colors : bool, default=False
        Force color inversion if requested.

    Returns
    -------
    List[Dict[str, Any]]
        List of detected digits from left to right:
        [
            {
                "model_input": np.ndarray (1, 28, 28, 1) float32,
                "preview_28x28": np.ndarray (28, 28) uint8,
                "bbox": (ymin, ymax, xmin, xmax)
            },
            ...
        ]
    """
    if is_canvas_empty(image_input):
        return []

    img_arr = to_standard_grayscale_array(image_input, invert_colors=invert_colors)
    boxes = extract_digit_bounding_boxes(img_arr)

    # Fallback to single bounding box if extraction found nothing
    if not boxes:
        coords = np.argwhere(img_arr > 20.0)
        if coords.size == 0:
            return []
        ymin, xmin = coords.min(axis=0)
        ymax, xmax = coords.max(axis=0)
        boxes = [(int(ymin), int(ymax + 1), int(xmin), int(xmax + 1))]

    results = []
    for ymin, ymax, xmin, xmax in boxes:
        cropped = img_arr[ymin:ymax, xmin:xmax]
        model_input, preview = crop_scale_and_center_digit(
            cropped,
            target_box_size=target_box_size,
            canvas_size=canvas_size
        )
        results.append({
            "model_input": model_input,
            "preview_28x28": preview,
            "bbox": (ymin, ymax, xmin, xmax)
        })

    return results


def preprocess_image(
    image_input,
    target_box_size: int = 20,
    canvas_size: int = 28,
    invert_colors: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Legacy single-digit preprocessor function preserved for backward compatibility.
    """
    digits = segment_and_preprocess_digits(
        image_input,
        target_box_size=target_box_size,
        canvas_size=canvas_size,
        invert_colors=invert_colors
    )
    if not digits:
        blank = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        return blank.reshape(1, canvas_size, canvas_size, 1).astype(np.float32), blank
    return digits[0]["model_input"], digits[0]["preview_28x28"]
