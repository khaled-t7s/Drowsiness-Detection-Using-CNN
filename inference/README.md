# 🚗 Drowsiness Detection Using Convolutional Neural Networks (CNN)

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📌 Overview

This project presents a real-time driver drowsiness detection system based on **Convolutional Neural Networks (CNN)**.

The system classifies eye images into **Open** or **Closed** states and continuously monitors the driver's eyes using a webcam. If the eyes remain closed for more than a predefined period, an alarm is triggered to warn the driver.

This project was developed as part of a Deep Learning course.

---

## ✨ Features

- Eye state classification using CNN
- Real-time webcam monitoring
- OpenCV Haar Cascade eye detection
- Audio alarm for drowsiness
- TensorFlow / Keras implementation
- Data augmentation
- Early stopping
- Dropout regularization

---

## 🛠 Technologies

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- Pygame

---

## 📂 Project Structure

```
Drowsiness-Detection-Using-CNN
│
├── train/
│   ├── deep_learning_project.ipynb
│   └── deep_learning_project.py
│
├── inference/
│   ├── test_live.py
│   ├── my_best_eye_model.h5
│   ├── haarcascade_eye.xml
│   └── WAKY WAKY.wav
│
├── images/
│   ├── accuracy_loss.png
│   └── confusion_matrix.png
│
├── report/
│   └── ProjectReport.pdf
│
├── dataset/
│
├── requirements.txt
└── README.md
```

---

## 🧠 Model Architecture

The CNN consists of:

- 3 Convolutional Layers
- MaxPooling Layers
- Dropout Layers
- Fully Connected Layer
- Sigmoid Output Layer

Training techniques include:

- Data Augmentation
- Early Stopping
- Adam Optimizer
- Binary Cross Entropy Loss

---

## 📊 Results

### Accuracy & Loss

![Accuracy](images/accuracy_loss.png)

### Confusion Matrix

![Confusion Matrix](images/confusion_matrix.png)

**Final Validation Accuracy:** **99.87%**

---

## 📁 Dataset

Dataset used:

**MRL Eye Dataset**

https://www.kaggle.com/datasets/prasadvpatil/mrl-dataset

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/khaled-t7s/Drowsiness-Detection-Using-CNN.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the real-time detector

```bash
python inference/test_live.py
```

---

## 📖 Report

The complete project report is available in:

```
report/ProjectReport.pdf
```

---

## 📌 Future Improvements

- Mobile deployment
- Raspberry Pi implementation
- YOLO-based eye detection
- Face Mesh integration
- Driver fatigue level estimation

---

## 👨‍💻 Author

**Khaled Mahyoub**

Information Systems Engineering

Sakarya University

GitHub:
https://github.com/khaled-t7s
