# Pneumonia Detection in Chest X-Rays using Deep Learning 🩺📸  

## 🚀 Project Overview  
This project develops a deep learning model using **PyTorch** to classify chest X-ray images into two categories:  
✅ **Normal**  
❌ **Pneumonia**  

The model is built on **ResNet18**, a pretrained convolutional neural network (CNN), and fine-tuned for pneumonia detection. It achieves an accuracy of **~96%** on the test dataset. The project applies **data augmentation, dropout regularization, and learning rate scheduling** to improve generalization and prevent overfitting.  

---

## 📂 Dataset  
We used chest X-ray datasets from **Kaggle**, combining multiple sources to ensure diversity and robustness. The images were manually sorted into two classes:  
- **Normal**: Healthy lung X-ray images  
- **Pneumonia**: X-rays indicating pneumonia infection  

📌 **Dataset Sources:**  
- [Pediatric Pneumonia Chest X-ray](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  
- [Pneumonia X-ray Images](https://www.kaggle.com/datasets/kmader/pneumonia-x-ray-images)  
- [Chest X-ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  

---

## 🏗️ Model Architecture  
We utilize a **pretrained ResNet18** model, modifying its final classification layer for binary classification:  

- **Base Model:** ResNet18 (pretrained on ImageNet)  
- **Final Layer:** Fully Connected Layer → `nn.Linear(512, 2)`  
- **Activation Function:** Softmax  
- **Optimizer:** Adam (`lr=0.001`, `weight_decay=1e-4`)  
- **Loss Function:** Cross-Entropy Loss  

🔹 **Regularization Techniques Applied:**  
✔ **Dropout (0.5)** in the final layer  
✔ **Data Augmentation:** Random Rotation, Gaussian Blur, Horizontal Flip  
✔ **Learning Rate Decay:** StepLR (reducing learning rate every 7 epochs)  

---

## 🔧 Installation & Setup  
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/yourusername/Pneumonia-Detection.git
cd Pneumonia-Detection
