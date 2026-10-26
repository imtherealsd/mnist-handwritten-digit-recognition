# 1–2 Minute Video Demo Script

**Project**: Handwritten Digit Recognition using CNN (MNIST)  
**Internship**: Machine Learning Internship — Project 1 of 2  
**Organization**: Horizon Intern  
**Target Duration**: 90 to 120 seconds  

---

### [0:00 – 0:15] Introduction
- **Visual**: Screen recording showing the Streamlit application at `http://localhost:8501` or GitHub repository overview.
- **Spoken Audio**:
  > *"Hello everyone! My name is [Your Name], and this is Project 1 of my Machine Learning Internship with Horizon Intern: **Handwritten Digit Recognition using a Convolutional Neural Network on the MNIST dataset**."*

---

### [0:15 – 0:40] Problem & Architecture Overview
- **Visual**: Show the sidebar of the Streamlit app or the Google Colab notebook showing the model summary and confusion matrix.
- **Spoken Audio**:
  > *"Our objective was to classify handwritten digits from 0 to 9 with high accuracy and deploy an interactive drawing interface.  
  > I constructed a custom CNN featuring two convolutional blocks with 32 and 64 filters, Batch Normalization for gradient stability, Dropout to prevent overfitting, and a 10-class Softmax dense output.  
  > The model was trained using Adam optimizer with Early Stopping, achieving over 99% accuracy on the unseen 10,000-image test set."*

---

### [0:40 – 1:20] Live Interactive Demo (Single & Multi-Digit)
- **Visual**: Focus on the drawing canvas. Select brush width (e.g., 18 px) and draw a single digit (e.g., `7`).
- **Action**: Click the blue **"🚀 Predict Digit(s)"** button.
- **Spoken Audio**:
  > *"Here is our Streamlit web application. Let's draw the digit '7' on the black canvas. Notice that when I click 'Predict Digit(s)', the app applies Yann LeCun's standardization pipeline: cropping the bounding box, scaling to 20x20, and centering at (14, 14) via Center of Mass. The model predicts 7 with high confidence."*
- **Action**: Clear the canvas, now draw a two-digit number (e.g., `42`), and click Predict again.
- **Spoken Audio**:
  > *"Beyond single digits, I also engineered an automatic connected-component segmentation pipeline. When I draw a multi-digit number like '42', the system segments each digit from left to right, processes both through the CNN, and combines them into the final number '42' while showing per-digit perception and confidence breakdowns."*

---

### [1:20 – 1:35] Alternative Input & Preprocessing Robustness
- **Visual**: Switch to the **"📁 Upload Digit Image"** tab. Upload a sample digit image or photo.
- **Spoken Audio**:
  > *"We also built an image upload feature that automatically detects dark ink on white paper, inverts the polarity to match MNIST, and accurately classifies uploaded files."*

---

### [1:35 – 1:50] Code Quality, Testing & Wrap-Up
- **Visual**: Show terminal running `pytest tests/` (all 14 tests passing).
- **Spoken Audio**:
  > *"The codebase is structured into reusable modules under `src/` and fully covered by 14 unit tests using pytest for preprocessing, segmentation, and inference.  
  > Thank you Horizon Intern for this assignment opportunity. This completes my demonstration of Project 1!"*

---

### Recording Tips
1. Use an open-source tool like **OBS Studio** or Windows Game Bar (`Win + G`) for screen recording.
2. Speak clearly at a moderate, confident pace.
3. Keep mouse movements smooth and intentional.
