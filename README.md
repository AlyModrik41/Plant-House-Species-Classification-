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
## 🏆 Model Comparison: From Scratch vs Fine-Tuned

To evaluate the performance of our plant species classifier, we trained two versions of the model:

1. **From Scratch:** A CNN trained with random weights.
   
   <img width="583" height="455" alt="download" src="https://github.com/user-attachments/assets/2de2bc63-3749-46ee-b30c-693125402774" />
3. **Fine-Tuned Transfer Learning):** A pre-trained backbone with custom dense layers for plant classification.
   
   <img width="596" height="455" alt="download" src="https://github.com/user-attachments/assets/2b564592-0800-4674-a985-2c72cd0e09bc" />


### 🔹 Best Epoch Comparison
The best epoch for each model was selected based on the **lowest validation loss**.

| Model | Best Epoch | Validation Loss | Validation Accuracy |
|-------|------------|----------------|-------------------|
| From Scratch | 7 | 3.79 | 0.07 |
| Fine-Tuned  | 5 | 1.12 | 0.31 |

### 🔹 Observations

- **From Scratch Model:**
  - Very low validation accuracy (~7%) even at best epoch.
  - Overfits easily and converges slowly due to random initialization.
  - Training is resource-intensive and less effective on this dataset.

- **Fine-Tuned Model:**
  - Significantly higher validation accuracy (~31%) at best epoch.
  - Converges faster and extracts more meaningful features from images.
  - Suitable for real-world deployment, especially when converted to TFLite.

### 🔹 Conclusion
Transfer learning drastically improves performance on the house plant dataset. Selecting the **best epoch based on validation loss** ensures optimal model weights, reduces overfitting, and gives the most accurate predictions. This is why the **TFLite fine-tuned model** was chosen for deployment in the Streamlit application.



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
