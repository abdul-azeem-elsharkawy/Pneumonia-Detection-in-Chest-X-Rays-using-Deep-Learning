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

- **Train-Test Accuracy Curve:**

![alt text](https://www.kaggleusercontent.com/kf/213373195/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..MrXkw3OY7MlNHXCV-vco2A.tJPsCI6rV9gUPsyJQZCsWOxZoydZWe_5EwvifqeofdO_S2KmOGsAAcWiii-7Ma6a5OrlEXVr3Gue0Ik3l3jsocVyRTycXXCrcrSNXnHRwnwAHuNgThrL8HfClXfDryacBYLbY2xh4943RwLgJVzJ543p6jcCaLkDBUI768DSnkJeyWLkxMZh0jw4mQ5OADbJaMaQqoYd38QAit0uZB2gXmnBOlC1GS9KQWCFnv68Fcbix3xOYgwih0shaU3wNbXMpNl31X9RO6M2to51PShdK-M4kPusc-gB9JBWq3rrC5Q1ohTsZ74vGhVrnuZyYREeyTuPkyqaoCvM_jV4iIRfASLkTGfSa7kbbboTJbd9OkOtL2pehTZcIA6r8sPlmSl5EhNatbLK6abXuSwQZh1ZEjCui4Z6qtAxHmeNzdEVSqaW4H-c3vCpUN26lsT4Dk08fCGFVS560MI3-B0nZHJmYFfRcbv7f1Rh08c5wXkAw9leUiwAGo9qtHkFthvyTAZfHBkYUQuJ3o7SEqFHuzTg4J06b3lxT-883-Gw_JG-VqCnLwErULqCrNoaWy8CmQdt2O61XSCsyerJ0KQW63OPktocQuP1w7VzZFeMHlFJJicFvpc_IQlmD4wF3Diu8ALyaCOh64qDSkX2IDqUHz6azG-s2_LVrfSQnnBs2jDwUD8.gOi9Rak8O9eZECN4ltfpFA/__results___files/__results___6_0.png)
---

## 🎨 Example Predictions

👉 **Inputs:**

- Normal:
![alt text](https://media-hosting.imagekit.io//22d1e13fcb364511/NORMAL2-IM-1438-0001.jpeg?Expires=1834220017&Key-Pair-Id=K2ZIVPTIP2VGHC&Signature=PW02uqXhsY1UspodGAAZJV-~z0rdNQ9WObYe0aBhnLRRObtDCMXJnYXRtH~wCzIhyU5uMV0uYWJHQhUqOjD5ujAwUpvQhh237IF3wG0lwyHAixmvJSic5OQbu8eG-YBDtZFYkjBS77y8KPMZLWCNy8UUCFSW6NaKaDsQjTH-VA96~YIkPZ9uzo2Q~RFnuFQqmIMcYGqbnUEiQtK2uc~irGFf6eeoqixAtbzLX48wxwWlj5WDjXTgbfl041syiG6WsMq4j40wynBIacB0ySgFKVPtBiPm3ko5UNwodob5WzO2Cxprr8SYF8sZrs89O3PIKosxEoftsbhMp8kRRk~Atg__)

- Pneumonia:
![alt text](https://media-hosting.imagekit.io//473a0ca656914ee1/person1951_bacteria_4882.jpeg?Expires=1834220181&Key-Pair-Id=K2ZIVPTIP2VGHC&Signature=g7Q9yfWqKBFZwVWrL1dH75KCNn5KrBBTeLBAnwzohu97rCBAMb4cemjSn1ohvEUpfX8j-kdQjeWlROl-iLKNnPsLacVLfzil~nBEjy0YJ5TOqGk27CWiQww4hAb52kVjRwfm0RSd2jFHGAD9LBROvAvsajsG0IxhgS5PZ9aCtqTFpVTbp1SuED-2WJ8a~iASDUb7Lzc056eQHYnXxx7b64xj-g6EwyAVx1W~6d3HcVwqiYa2sfvENoZbzWN9r2PYiL35OG5zfvL6qwMJyTKzG7wGHK8VC3ZM9yUdA9CAklInj8wRhZIqpFHWHnWi1d2zFktsKpK~yX5ql~wh28RXaQ__)


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
