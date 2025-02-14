# Pneumonia Detection in Chest X-Rays using Deep Learning 🩺📸

## 🚀 Project Overview

This project develops a deep learning model using **PyTorch** to classify chest X-ray images into two categories:\
🇹 **Normal**\
🛡️ **Pneumonia**

The model is built on **ResNet18**, a pretrained convolutional neural network (CNN), and fine-tuned for pneumonia detection. It achieves an accuracy of **\~96%** on the test dataset. The project applies **data augmentation, dropout regularization, and learning rate scheduling** to improve generalization and prevent overfitting.

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

## 🏰 Model Architecture

We utilize a **pretrained ResNet18** model, modifying its final classification layer for binary classification:

- **Base Model:** ResNet18 (pretrained on ImageNet)
- **Final Layer:** Fully Connected Layer → `nn.Linear(512, 2)`
- **Activation Function:** Softmax
- **Optimizer:** Adam (`lr=0.001`, `weight_decay=1e-4`)
- **Loss Function:** Cross-Entropy Loss

🔹 **Regularization Techniques Applied:**\
👉 **Dropout (0.5)** in the final layer\
👉 **Data Augmentation:** Random Rotation, Gaussian Blur, Horizontal Flip\
👉 **Learning Rate Decay:** StepLR (reducing learning rate every 7 epochs)

---

## 🔧 Installation & Setup

### **1️⃣ Clone the Repository**

```bash
git clone https://github.com/yourusername/Pneumonia-Detection.git
cd Pneumonia-Detection
```

### **2️⃣ Install Dependencies**

```bash
pip install -r requirements.txt
```

### **3️⃣ Download Dataset (Kaggle)**

Ensure you have the dataset downloaded inside the `data/` directory. You can manually add it or use Kaggle API:

```bash
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data/
unzip data/chest-xray-pneumonia.zip -d data/
```

---

## 🏅 Training the Model

To train the model, run:

```bash
python train.py --epochs 20 --batch_size 32 --lr 0.001
```

**Optional Arguments:**

- `--epochs` (default: `20`)
- `--batch_size` (default: `32`)
- `--lr` (default: `0.001`)

---

## 📊 Model Evaluation

To evaluate the trained model on the test set:

```bash
python evaluate.py --model pneumonia_model.pth
```

The evaluation script computes:\
👉 **Test Accuracy**\
👉 **Confusion Matrix**\
👉 **Precision, Recall, and F1-Score**

---

## 🎨 Example Predictions

You can run inference on a new chest X-ray image:

```bash
python predict.py --image sample_xray.jpg --model pneumonia_model.pth
```

👉 **Output:**

```
Prediction: Pneumonia (Confidence: 97.3%)
```

---

## 📈 Results

### **📌 Training vs. Test Accuracy Curve**



### **📌 Confusion Matrix**



---

## 🚀 Future Enhancements

👉 **Try deeper architectures** (ResNet50, DenseNet121)\
👉 **Integrate Grad-CAM** for explainable AI in medical imaging\
👉 **Optimize inference time** for real-world deployment\
👉 **Deploy as a web app** using Flask or FastAPI

---

## 🤝 Contributors

- **Your Name** - [GitHub](https://github.com/yourusername)
- **Other Contributors (If any)**

---

## 📝 License

This project is open-source under the **MIT License**.

---

## 🌟 Acknowledgments

Special thanks to **Kaggle** for providing the dataset and the open-source deep learning community for continuous support!

```
```
