"""
app.py
======
Interactive Streamlit Web Application for Handwritten Digit Recognition (MNIST).
Features:
- Real-time drawing canvas (using streamlit-drawable-canvas)
- Multi-digit automatic segmentation: recognizes both single digits ('7')
  and multi-digit numbers ('42', '100', '73') from left to right!
- Alternative image uploader (PNG / JPG / JPEG) for robustness
- Preprocessing visualization ("What the model sees" in 28x28 per digit)
- Confidence score & 10-class probability distribution
- Comprehensive sidebar with architecture, methodology, and limitations
- Graceful error handling if model weights are missing
"""

import os
from typing import Optional
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import streamlit as st

from src.preprocess import (
    is_canvas_empty,
    preprocess_image,
    segment_and_preprocess_digits,
)
from src.model_utils import load_mnist_model, predict_digit

# Try to import streamlit-drawable-canvas
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except Exception:
    CANVAS_AVAILABLE = False


# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Handwritten Digit Recognition | MNIST CNN",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Styling & CSS ---
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #FFFFFF !important;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #CBD5E1 !important;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .multidigit-hero {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border: 2px solid #3B82F6;
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .multidigit-number {
        font-size: 4.5rem;
        font-weight: 900;
        color: #1D4ED8;
        letter-spacing: 4px;
        line-height: 1.1;
        margin: 0.3rem 0;
    }
    .digit-display {
        font-size: 4rem;
        font-weight: 800;
        color: #2563EB;
        line-height: 1;
        margin: 0.5rem 0;
    }
    .confidence-text {
        font-size: 1.1rem;
        color: #059669;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_cached_model(model_path: str = "mnist_cnn.keras"):
    """Cache loaded Keras model to avoid reloading on every rerun."""
    return load_mnist_model(model_path)


def render_sidebar():
    """Render the explanatory sidebar with model specifications and guidelines."""
    with st.sidebar:
        st.markdown("### 📌 Project Information")
        st.markdown("**Handwritten Digit Recognition (MNIST)**")
        st.markdown("---")

        st.markdown("### 🔢 Supported Inputs")
        st.markdown(
            """
            - **Single Digits**: `0` to `9`
            - **Multi-Digit Numbers**: `42`, `100`, `73`, etc.
            - *Automatic left-to-right segmentation parses each digit independently.*
            """
        )
        st.markdown("---")

        st.markdown("### 🧠 CNN Architecture")
        st.markdown(
            """
            - **Input**: `28 × 28 × 1` Grayscale
            - **Conv Block 1**: 2× Conv2D (32, 3x3) + BatchNorm + MaxPool + Dropout(0.25)
            - **Conv Block 2**: 2× Conv2D (64, 3x3) + BatchNorm + MaxPool + Dropout(0.25)
            - **Classifier**: Flatten + Dense (128, ReLU) + BatchNorm + Dropout(0.50) + Dense (10, Softmax)
            - **Optimizer**: Adam (lr=0.001) | **Loss**: Categorical Crossentropy
            - **Test Accuracy**: **99.34%** (on 10,000 test samples)
            """
        )
        st.markdown("---")

        st.markdown("### ⚙️ MNIST Preprocessing")
        st.markdown(
            """
            Raw inputs are standardized via Yann LeCun's benchmark pipeline:
            1. **Connected Component Bounding Box**: Isolates each digit stroke.
            2. **Aspect Rescale**: Resizes each digit into `20 × 20`.
            3. **Center of Mass**: Positions center of mass at `(14, 14)` inside `28 × 28`.
            4. **Normalization**: Rescales `[0, 255]` to `[0.0, 1.0]` float32.
            """
        )
        st.markdown("---")

        st.markdown("### 💡 Tips for Best Results")
        st.markdown(
            """
            - **Single Digit**: Draw centered with clear strokes.
            - **Multi-Digit**: Leave a small natural space between adjacent digits.
            - Use a moderate stroke width (16–22 px).
            - Close loops clearly (e.g., on `0`, `6`, `8`, `9`).
            """
        )


def draw_bounding_boxes_on_image(
    image_input,
    detected_digits: list
) -> Optional[Image.Image]:
    """
    Overlay blue bounding boxes and prediction tags on the input image
    so the user can visually verify every segmented digit.
    """
    if image_input is None or not detected_digits:
        return None

    if isinstance(image_input, np.ndarray):
        if image_input.ndim == 3 and image_input.shape[-1] == 4:
            base_img = Image.fromarray(image_input[:, :, :3].astype(np.uint8))
        elif image_input.ndim == 3:
            base_img = Image.fromarray(image_input.astype(np.uint8))
        else:
            base_img = Image.fromarray(image_input.astype(np.uint8)).convert("RGB")
    elif isinstance(image_input, Image.Image):
        base_img = image_input.convert("RGB")
    else:
        return None

    annotated = base_img.copy()
    draw = ImageDraw.Draw(annotated)

    for idx, item in enumerate(detected_digits):
        ymin, ymax, xmin, xmax = item["bbox"]
        pred = item.get("pred", {})
        d_val = pred.get("digit", "?")
        conf = pred.get("confidence", 0.0)

        # Draw bounding box
        draw.rectangle([xmin, ymin, xmax, ymax], outline="#2563EB", width=3)

        # Draw tag badge
        tag = f"#{idx + 1}: {d_val} ({int(conf * 100)}%)"
        tag_w = len(tag) * 7 + 4
        tag_h = 16
        tag_y0 = max(0, ymin - tag_h)
        draw.rectangle([xmin, tag_y0, xmin + tag_w, ymin], fill="#2563EB")
        draw.text((xmin + 2, tag_y0 + 1), tag, fill="#FFFFFF")

    return annotated


def main():
    render_sidebar()

    st.markdown('<div class="main-title">✍️ Handwritten Digit Recognition</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Draw or upload any single digit or multi-digit number (e.g., <b>7</b>, <b>42</b>, <b>100</b>) to classify it using a CNN trained on MNIST.</div>',
        unsafe_allow_html=True,
    )

    # Check for model weights
    model_path = "mnist_cnn.keras"
    model_exists = os.path.exists(model_path)
    model = None

    if model_exists:
        try:
            model = get_cached_model(model_path)
            st.success("✅ Model loaded successfully (`mnist_cnn.keras` | 99.34% Test Accuracy)", icon="🎯")
        except Exception as e:
            st.error(f"Error loading model weights: {e}")
            model = None
    else:
        st.warning(
            f"⚠️ **Model file `{model_path}` not found in the workspace root.**  \n\n"
            "To test live predictions:  \n"
            "1. Open `notebooks/MNIST_Handwritten_Digit_Recognition.ipynb` in **Google Colab** and run all cells.  \n"
            "2. Download `mnist_cnn.keras` from Colab and place it in this project's root folder.  \n"
            "*(Alternatively, if running locally with TensorFlow installed, run `python train.py`)*.",
            icon="ℹ️",
        )

    # Input method selection: Drawing Canvas or File Upload
    input_mode = st.radio(
        "**Choose Input Method**",
        ["📁 Upload Digit Image (Any image with numbers: handwritten, typed, printed)", "🖌️ Interactive Drawing Canvas"],
        horizontal=True,
    )

    active_image = None
    source_name = ""
    predict_canvas_clicked = False

    if "Upload Digit Image" in input_mode:
        st.markdown(
            "Upload an image containing **any numbers** — supports handwritten, typed, printed, "
            "receipts, documents, screenshots, and multi-digit collages."
        )

        uploaded_file = st.file_uploader(
            "Upload any image containing digits or numbers",
            type=["png", "jpg", "jpeg", "webp", "bmp", "tiff", "jfif", "gif", "PNG", "JPG", "JPEG", "WEBP", "BMP"],
            help="Supports dark ink on light paper, white on black, or colored numbers.",
            key="digit_image_uploader",
        )

        if uploaded_file is not None:
            try:
                pil_upload = Image.open(uploaded_file)

                if pil_upload.mode in ("P", "CMYK", "1"):
                    pil_upload = pil_upload.convert("RGB")

                # Auto-downscale excessively large phone camera shots to keep inference instant
                if max(pil_upload.size) > 1200:
                    pil_upload.thumbnail((1200, 1200), Image.Resampling.LANCZOS)

                active_image = pil_upload
                source_name = f"Uploaded File ({uploaded_file.name})"

                col_u1, col_u2 = st.columns([1, 2])
                with col_u1:
                    st.image(pil_upload, caption=f"Uploaded: {uploaded_file.name}", width=230)
                with col_u2:
                    st.success(
                        f"✅ **{uploaded_file.name}** loaded successfully!  \n"
                        f"Dimensions: {pil_upload.width} × {pil_upload.height} px | Mode: {pil_upload.mode}"
                    )
            except Exception as ex:
                st.error(f"Could not load image: {ex}")
    else:
        if CANVAS_AVAILABLE:
            st.markdown(
                "Draw one or more digits inside the canvas below. "
                "The canvas has a **black background** with **white ink** to mirror MNIST. "
                "*(Leave a small gap between digits when writing numbers like `42`)*."
            )
            col_ctrl1, col_ctrl2 = st.columns([1, 2])
            with col_ctrl1:
                stroke_width = st.slider("Stroke Thickness", min_value=10, max_value=32, value=18, step=2)
            with col_ctrl2:
                st.info("💡 Multi-digit support: Draw multiple digits side-by-side from left to right!")

            # Wide canvas to accommodate single or multi-digit numbers comfortably
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0)",
                stroke_width=stroke_width,
                stroke_color="#FFFFFF",
                background_color="#000000",
                height=250,
                width=520,
                drawing_mode="freedraw",
                key="canvas",
                display_toolbar=True,
            )

            if canvas_result is not None and canvas_result.image_data is not None:
                if not is_canvas_empty(canvas_result.image_data):
                    active_image = canvas_result.image_data
                    source_name = "Canvas Drawing"

            col_btn, _ = st.columns([1, 3])
            with col_btn:
                predict_canvas_clicked = st.button("🚀 Predict Drawn Digit(s)", type="primary", use_container_width=True)
        else:
            st.warning(
                "Canvas package (`streamlit-drawable-canvas`) is not detected. "
                "Please use the **Upload Digit Image** mode."
            )

    st.markdown("---")

    # Auto-predict for uploaded files, button-triggered for drawing canvas
    should_predict = False
    if "Upload Digit Image" in input_mode:
        if active_image is not None and not is_canvas_empty(active_image):
            should_predict = True
    else:
        if predict_canvas_clicked:
            should_predict = True

    if should_predict:

        if model is None:
            st.error(
                "❌ Cannot predict because `mnist_cnn.keras` is missing. "
                "Please train the model via the notebook or `python train.py` first."
            )
            return

        # Preprocess and segment input into individual digits
        try:
            detected_digits = segment_and_preprocess_digits(active_image)
        except Exception as e:
            st.error(f"Error during image segmentation & preprocessing: {e}")
            return

        if not detected_digits:
            st.warning("⚠️ Could not extract distinct digits. Please ensure strokes are clear.")
            return

        num_digits = len(detected_digits)

        # Run model inference on all segmented digits
        try:
            for digit_item in detected_digits:
                digit_item["pred"] = predict_digit(model, digit_item["model_input"])
        except Exception as e:
            st.error(f"Error during model prediction: {e}")
            return

        # --- SCENARIO A: SINGLE DIGIT ---
        if num_digits == 1:
            d0 = detected_digits[0]
            predicted_digit = d0["pred"]["digit"]
            confidence = d0["pred"]["confidence"]
            probabilities = d0["pred"]["probabilities"]

            col_view, col_res = st.columns([1, 2])

            with col_view:
                st.markdown("### 🔍 Model Perception")
                st.markdown(
                    "Here is the normalized, centered **28 × 28 grayscale image** that the neural network evaluates:"
                )
                st.image(
                    d0["preview_28x28"],
                    caption="Preprocessed 28x28 Input (Enlarged)",
                    width=180,
                    clamp=True,
                )
                st.caption(
                    f"Source: **{source_name}** | Single Digit Mode  \n"
                    f"Min pixel: {d0['model_input'].min():.2f} | Max pixel: {d0['model_input'].max():.2f}"
                )

            with col_res:
                st.markdown("### 🏆 Prediction Output")

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div style="font-size: 0.95rem; color: #475569; font-weight: 600;">PREDICTED DIGIT</div>
                            <div class="digit-display">{predicted_digit}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with c2:
                    conf_pct = confidence * 100
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div style="font-size: 0.95rem; color: #475569; font-weight: 600;">CONFIDENCE SCORE</div>
                            <div class="digit-display" style="color: #059669; font-size: 3rem;">{conf_pct:.1f}%</div>
                            <div class="confidence-text">{"High Certainty" if conf_pct > 80 else "Moderate Certainty"}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("#### 📊 Probability Distribution (Digits 0–9)")
                prob_df = pd.DataFrame(
                    {
                        "Digit": [str(d) for d in range(10)],
                        "Probability": [p * 100 for p in probabilities],
                    }
                ).set_index("Digit")

                st.bar_chart(prob_df, y="Probability", height=230)

        # --- SCENARIO B: MULTI-DIGIT (e.g. '42', '100', '73' or 2D collage/sheets) ---
        else:
            avg_conf = np.mean([d["pred"]["confidence"] for d in detected_digits]) * 100
            annotated_preview = draw_bounding_boxes_on_image(active_image, detected_digits)

            # Case 1: Standard number sequence (2 to 6 digits, like '42', '100', '508')
            if num_digits <= 6:
                combined_number_str = "".join(str(d["pred"]["digit"]) for d in detected_digits)

                # Grand Multi-Digit Hero Banner
                st.markdown(
                    f"""
                    <div class="multidigit-hero">
                        <div style="font-size: 1.1rem; color: #1E40AF; font-weight: 700; letter-spacing: 1px;">
                            🎯 MULTI-DIGIT RECOGNITION RESULT
                        </div>
                        <div class="multidigit-number">{combined_number_str}</div>
                        <div style="font-size: 1.1rem; color: #047857; font-weight: 600;">
                            Detected {num_digits} Digits (Left-to-Right) &nbsp;|&nbsp; Average Confidence: {avg_conf:.1f}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if annotated_preview is not None:
                    with st.expander("🖼️ View Detected Bounding Boxes on Original Input", expanded=False):
                        st.image(
                            annotated_preview,
                            caption="Identified Digit Bounding Boxes Overlaid on Input",
                            use_container_width=True,
                        )

                st.markdown("### 🧩 Per-Digit Breakdown & Perception")
                st.markdown(
                    "Each digit was individually isolated, normalized, and centered using Yann LeCun's pipeline:"
                )

                cols = st.columns(num_digits)
                for idx, (col, digit_item) in enumerate(zip(cols, detected_digits)):
                    with col:
                        d_pred = digit_item["pred"]["digit"]
                        d_conf = digit_item["pred"]["confidence"] * 100
                        d_probs = digit_item["pred"]["probabilities"]

                        st.markdown(f"#### Digit #{idx + 1}")
                        st.image(
                            digit_item["preview_28x28"],
                            caption=f"Pos #{idx + 1} (28×28)",
                            width=120,
                            clamp=True,
                        )
                        st.markdown(
                            f"""
                            <div class="metric-card" style="padding: 0.8rem; margin-top: 0.5rem;">
                                <div style="font-size: 2.2rem; font-weight: 800; color: #2563EB;">{d_pred}</div>
                                <div style="font-size: 0.9rem; color: #059669; font-weight: 600;">{d_conf:.1f}% conf</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        with st.expander(f"Digit #{idx + 1} Probabilities"):
                            p_df = pd.DataFrame(
                                {
                                    "Digit": [str(d) for d in range(10)],
                                    "Prob": [p * 100 for p in d_probs],
                                }
                            ).set_index("Digit")
                            st.bar_chart(p_df, y="Prob", height=180)

            # Case 2: Multi-digit sheet or 2D collage (> 6 digits)
            else:
                all_digits_str = " ".join(str(d["pred"]["digit"]) for d in detected_digits)

                st.markdown(
                    f"""
                    <div class="multidigit-hero">
                        <div style="font-size: 1.1rem; color: #1E40AF; font-weight: 700; letter-spacing: 1px;">
                            🎯 2D MULTI-DIGIT / COLLAGE DETECTION RESULT
                        </div>
                        <div style="font-size: 2.2rem; font-weight: 800; color: #1D4ED8; margin: 0.5rem 0; word-break: break-all;">
                            {all_digits_str}
                        </div>
                        <div style="font-size: 1.1rem; color: #047857; font-weight: 600;">
                            Detected {num_digits} Digits Across Canvas &nbsp;|&nbsp; Average Confidence: {avg_conf:.1f}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Display the annotated image prominently
                if annotated_preview is not None:
                    st.markdown("### 🖼️ Detected Digit Bounding Boxes")
                    st.markdown(
                        "All identified digit regions are boxed in blue with their position number, predicted digit, and confidence:"
                    )
                    st.image(
                        annotated_preview,
                        caption=f"Detected {num_digits} Digits Overlaid on Uploaded Image",
                        use_container_width=True,
                    )

                st.markdown("### 🧩 Detected Digits Gallery")
                st.markdown(
                    "Each character normalized to 28×28 and evaluated by the MNIST CNN:"
                )

                # Responsive grid in rows of 6
                batch_size = 6
                for i in range(0, num_digits, batch_size):
                    batch = detected_digits[i : i + batch_size]
                    cols = st.columns(len(batch))
                    for b_idx, (col, digit_item) in enumerate(zip(cols, batch)):
                        global_idx = i + b_idx + 1
                        with col:
                            d_pred = digit_item["pred"]["digit"]
                            d_conf = digit_item["pred"]["confidence"] * 100
                            st.markdown(f"**Digit #{global_idx}**")
                            st.image(
                                digit_item["preview_28x28"],
                                caption=f"#{global_idx} (28×28)",
                                width=95,
                                clamp=True,
                            )
                            st.markdown(
                                f"""
                                <div class="metric-card" style="padding: 0.5rem; margin-top: 0.3rem;">
                                    <div style="font-size: 1.8rem; font-weight: 800; color: #2563EB;">{d_pred}</div>
                                    <div style="font-size: 0.85rem; color: #059669; font-weight: 600;">{d_conf:.1f}%</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                # Summary Table
                with st.expander("📋 View Full Detection & Prediction Table"):
                    table_data = []
                    for idx, d_item in enumerate(detected_digits):
                        table_data.append(
                            {
                                "Index": f"#{idx + 1}",
                                "Predicted Digit": d_item["pred"]["digit"],
                                "Confidence": f"{d_item['pred']['confidence'] * 100:.1f}%",
                                "Bounding Box (ymin, ymax, xmin, xmax)": str(d_item["bbox"]),
                            }
                        )
                    st.dataframe(pd.DataFrame(table_data), use_container_width=True)

    # Educational Expander
    with st.expander("📚 How does Multi-Digit Recognition work with an MNIST model?"):
        st.markdown(
            """
            The classic MNIST dataset contains only single digit images (28×28). To recognize multi-digit numbers like **42** or **100**:
            1. **Connected Component Analysis**: We identify distinct stroke groups across the canvas using morphological dilation and label grouping.
            2. **Left-to-Right Sorting**: Detected digit bounding boxes are sorted horizontally by their `x` coordinate.
            3. **Aspect Normalization & Centering**: Each individual digit is scaled into `20 × 20` and centered inside a `28 × 28` frame via Center-of-Mass alignment.
            4. **Individual Inference**: The CNN evaluates each segmented digit independently.
            5. **Sequence Assembly**: The predicted digits are assembled sequentially into the full number.
            """
        )


if __name__ == "__main__":
    main()
