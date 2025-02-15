# Pneumonia Detection in Chest X-Rays using Deep Learning 🩺📸

## 🚀 Project Overview

This project develops a deep learning model using **PyTorch** to classify chest X-ray images into two categories:\
✅ **Normal**\
❌ **Pneumonia**

The model is built on **ResNet18**, a pretrained convolutional neural network (CNN), and fine-tuned for pneumonia detection. It achieves an accuracy of **\~96%** on the test dataset. The project applies **data augmentation, dropout regularization, and learning rate scheduling** to improve generalization and prevent overfitting.

---

## 📂 Dataset

We used chest X-ray datasets from **Kaggle**, combining multiple sources to ensure diversity, robustness, and ovvercome unbalanced classes. The images were manually sorted into two classes:

- **Normal**: Healthy lung X-ray images
- **Pneumonia**: X-rays indicating pneumonia infection

📌 **Dataset Sources:**

- [Chest X-ray (Covid-19 & Pneumonia)](https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia)
- [Pneumonia X-Ray Images](https://www.kaggle.com/datasets/pcbreviglieri/pneumonia-xray-images)
- [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

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

## 🏅 Training the Model

- The model undergoes 20 epochs,
- and the number of images processed in each iteration (batch_size) is 32 images.

---

## 📊 Model Evaluation

Evaluate the trained model on the test set:

👉 **Test Accuracy**\
(https://www.kaggleusercontent.com/kf/213373195/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..MrXkw3OY7MlNHXCV-vco2A.tJPsCI6rV9gUPsyJQZCsWOxZoydZWe_5EwvifqeofdO_S2KmOGsAAcWiii-7Ma6a5OrlEXVr3Gue0Ik3l3jsocVyRTycXXCrcrSNXnHRwnwAHuNgThrL8HfClXfDryacBYLbY2xh4943RwLgJVzJ543p6jcCaLkDBUI768DSnkJeyWLkxMZh0jw4mQ5OADbJaMaQqoYd38QAit0uZB2gXmnBOlC1GS9KQWCFnv68Fcbix3xOYgwih0shaU3wNbXMpNl31X9RO6M2to51PShdK-M4kPusc-gB9JBWq3rrC5Q1ohTsZ74vGhVrnuZyYREeyTuPkyqaoCvM_jV4iIRfASLkTGfSa7kbbboTJbd9OkOtL2pehTZcIA6r8sPlmSl5EhNatbLK6abXuSwQZh1ZEjCui4Z6qtAxHmeNzdEVSqaW4H-c3vCpUN26lsT4Dk08fCGFVS560MI3-B0nZHJmYFfRcbv7f1Rh08c5wXkAw9leUiwAGo9qtHkFthvyTAZfHBkYUQuJ3o7SEqFHuzTg4J06b3lxT-883-Gw_JG-VqCnLwErULqCrNoaWy8CmQdt2O61XSCsyerJ0KQW63OPktocQuP1w7VzZFeMHlFJJicFvpc_IQlmD4wF3Diu8ALyaCOh64qDSkX2IDqUHz6azG-s2_LVrfSQnnBs2jDwUD8.gOi9Rak8O9eZECN4ltfpFA/__results___files/__results___6_0.png)
---

## 🎨 Example Predictions

You can run inference on a new chest X-ray image:

```bash
python predict.py --image sample_xray.jpg --model pneumonia_model.pth
```

👉 **Output:**

```
Prediction: Pneumonia (Confidence: ~96%)
```

---

## 📈 Results

### **📌 Training vs. Test Accuracy Curve**



### **📌 Confusion Matrix**




---

## 🌟 Acknowledgments

Special thanks to **Kaggle** for providing the dataset and the open-source deep learning community for continuous support!

```
```
