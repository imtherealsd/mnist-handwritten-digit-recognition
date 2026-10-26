# Horizon Intern Submission Checklist

**Project**: Handwritten Digit Recognition using CNN (MNIST)  
**Internship**: Machine Learning Internship  
**Organization**: Horizon Intern  

Use this checklist to ensure all submission guidelines and technical requirements are completely satisfied before submitting your portfolio link to Horizon Intern.

---

### 1. Codebase & Repository Integrity
- [x] Standard folder structure adhered to (`app.py`, `train.py`, `requirements.txt`, `notebooks/`, `src/`, `tests/`, `screenshots/`).
- [x] `.gitignore` present to exclude virtual environments, cache files, and system artifacts.
- [x] `LICENSE` (MIT) included.
- [x] Clean imports, descriptive variable naming, and beginner-to-intermediate friendly inline documentation.
- [x] Zero hardcoded absolute paths; works across Windows, macOS, Linux, and Cloud.

---

### 2. Model Training & Notebook
- [x] Jupyter Notebook created at `notebooks/MNIST_Handwritten_Digit_Recognition.ipynb`.
- [x] Data exploration, class distribution histogram, and 5×5 sample visualization included.
- [x] Normalization to `[0.0, 1.0]`, reshaping to `(28, 28, 1)`, and categorical one-hot encoding implemented.
- [x] Hold-out test set (10,000 samples) kept strictly out of training and validation.
- [x] EarlyStopping and ReduceLROnPlateau callbacks configured.
- [x] Training history (Loss & Accuracy) visualized.
- [x] Test set evaluation with Confusion Matrix heatmap and Classification Report.
- [x] Google Colab automated download snippet included.

---

### 3. Model Weights (`mnist_cnn.keras`)
- [x] Trained model saved as `mnist_cnn.keras` in the repository root.
- [x] Model file verified using `load_mnist_model()`.
- [x] Model parameter count verified (272,490 parameters, 3.35 MB).

---

### 4. Interactive Web Application (`app.py`)
- [x] Interactive drawing canvas configured (black background, white strokes, stroke width slider).
- [x] Robust alternative image upload tab implemented.
- [x] MNIST-standard preprocessing (bounding box crop, 20×20 aspect-preserving resize, Center-of-Mass alignment).
- [x] "Model Perception" 28×28 display showing the exact processed input.
- [x] Predicted digit card and confidence percentage displayed.
- [x] 10-class horizontal bar chart showing complete probability distribution.
- [x] Empty canvas validation and graceful missing-model error handling.

---

### 5. Unit Testing & Quality Assurance
- [x] Pytest suite in `tests/` with 11 unit tests.
- [x] Tests cover empty canvas check, output shapes `(1, 28, 28, 1)`, value ranges `[0.0, 1.0]`, center-of-mass shift, and mock prediction handling.
- [x] No unit tests require slow retraining.
- [x] Run `python -m pytest tests/` and confirm 100% pass rate.

---

### 6. Screenshots & Media
- [x] Capture `01_colab_training.png` (Colab notebook training cells).
- [x] Capture `02_evaluation_matrix.png` (Confusion matrix & classification report).
- [x] Capture `03_streamlit_overview.png` (Streamlit homepage and sidebar).
- [x] Capture `04_canvas_prediction_high_conf.png` (Live canvas prediction with high confidence).
- [x] Capture `05_image_upload_test.png` (Image upload prediction).
- [x] Place images inside the `screenshots/` directory.

---

### 7. Video Demonstration (1–2 Minutes)
- [x] Review `demo_script.md` for speaking points and timing.
- [x] Record walkthrough demo video (`demo_recording.webp` generated and linked).
- [x] Demonstrate:
  1. Introduction & CNN architecture overview.
  2. Drawing a digit, preprocessing perception, and confidence score.
  3. Testing a second digit.
  4. Switching to the image upload tab.
  5. Showing test results (`pytest`).
- [ ] Upload video (or link) as specified by Horizon Intern submission instructions.

---

### 8. Final Review & Submission
- [ ] Verify `README.md` contains accurate, real evaluation metrics.
- [ ] Push clean commits to your GitHub repository.
- [ ] Test cloning and running on a clean setup:
  ```bash
  git clone <your-repo-url>
  cd mnist-handwritten-digit-recognition
  pip install -r requirements.txt
  streamlit run app.py
  ```
