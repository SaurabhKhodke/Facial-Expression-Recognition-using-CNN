# Facial Expression Recognition using CNN

A deep learning–based **Facial Expression Recognition (FER)** system built using **Convolutional Neural Networks (CNNs)** with TensorFlow/Keras.  
The model classifies facial images into six emotion categories and provides a strong baseline with scope for future improvement using transfer learning.

---

## Project Overview

Facial Expression Recognition is a key problem in computer vision and human–computer interaction.  
This project implements a complete FER pipeline including:

- Dataset loading and preprocessing  
- CNN-based model training  
- Performance evaluation using standard metrics  
- Visualization of training behavior and predictions  

The focus is on building a reliable baseline model and analyzing its strengths and limitations.

---

## Emotion Classes

The model predicts the following emotions:

- Angry  
- Fear  
- Happy  
- Neutral  
- Sad  
- Surprise  

---

## Dataset Structure

The dataset follows a folder-based structure compatible with Keras:

Training/Training/<emotion_name>/
Testing/Testing/<emotion_name>/


Each emotion class contains facial images corresponding to that label.  
The dataset is downloaded automatically using **KaggleHub**.

---

## Tech Stack

- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib & Seaborn  
- Scikit-learn  

---

## Model Summary

- Input size: `48 × 48` grayscale images  
- Multiple convolutional blocks with:
  - Convolution layers
  - Batch Normalization
  - Max Pooling
  - Dropout
- Fully connected dense layers
- Softmax output layer for multi-class classification  

**Loss Function:** Sparse Categorical Crossentropy  
**Optimizer:** Adam  

---

## Results

### Overall Performance

- **Test Accuracy:** 67%  
- **Weighted F1-score:** 0.67  
- **Test Samples:** 7,067 images  

### Classification Highlights

- Strong performance on **Happy** and **Surprise**
- Moderate confusion between **Fear**, **Neutral**, and **Sad**, which is common in FER datasets

---

## Training Curves

Training and validation accuracy/loss during model training:

![Training Curves](assets/training%20curve.png)

---

## Confusion Matrix

Confusion matrix showing class-wise prediction performance:

![Confusion Matrix](assets/Confusion%20matrix.png)

This visualization helps identify misclassification patterns across emotion classes.

---

## Saved Model

The trained model is saved as: emotion_model_v3.h5


This file can be reused for inference or further fine-tuning.

---

## Future Improvements

The current model is trained from scratch using grayscale images.  
Accuracy can be further improved by:

- Using **EfficientNetB0 with RGB input**
- Applying transfer learning with pretrained ImageNet weights
- Using class weighting to improve recall for underperforming classes (e.g., Fear)
- Fine-tuning deeper layers

Expected accuracy after EfficientNet upgrade: **75–80%**

---

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt



