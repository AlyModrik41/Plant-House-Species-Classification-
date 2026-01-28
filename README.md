# 🌿 House Plant Species Classification

## 📌 Project Description
This project aims to classify different house plant species using images from the **House Plant Species** dataset available on Kaggle. The task is formulated as a multi-class image classification problem using a Convolutional Neural Network (CNN) built from scratch.

---

## 📊 Dataset Information
- **Source:** Kaggle – House Plant Species Dataset  
- **Total Images:** ~14,790  
- **Number of Classes:** 47 plant species  
- **Image Type:** RGB images with varying resolutions and backgrounds  
- **Structure:** Each class is stored in a separate folder

---

## 🧠 Model Architecture
- Custom CNN implemented using **TensorFlow / Keras**
- Input image size: **224 × 224 × 3**
- Layers used:
  - Convolutional layers
  - MaxPooling layers
  - Fully connected (Dense) layers
- Output layer uses **Softmax** activation for multi-class classification

---

## ⚙️ Training Details
- Optimizer: **Adam**
- Loss Function: **Categorical Crossentropy**
- Evaluation Metric: **Accuracy**
- Early Stopping used to reduce overfitting
- Data loaded using image generators

---

## 📈 Results
The model achieved validation and test accuracy significantly higher than random guessing (≈2%), demonstrating effective learning despite the large number of classes and the use of a simple CNN architecture.

---

## 🛠️ Tools & Libraries
- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- Scikit-learn  

---

## 📚 Notes
This project was developed as part of an introductory machine learning course. Advanced techniques such as transfer learning or fine-tuning pretrained models were intentionally avoided.
