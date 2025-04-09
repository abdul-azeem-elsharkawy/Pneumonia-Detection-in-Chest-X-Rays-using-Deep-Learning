# Pneumonia Detection in Chest X-Rays using Deep Learning 🩺📸

## 🚀 Project Overview

This project develops a deep learning model using **PyTorch** to classify chest X-ray images into two categories:\
✅ **Normal**\
❌ **Pneumonia**

The model is built on **ResNet18**, a pretrained convolutional neural network (CNN), and fine-tuned for pneumonia detection. It achieves an accuracy of **\~96%** on the test dataset. The project applies **data augmentation, dropout regularization, and learning rate scheduling** to improve generalization and prevent overfitting.

---

## 📂 Dataset

We used chest X-ray datasets from **Kaggle**, combining multiple sources to ensure diversity, robustness, and overcome unbalanced classes. The images were manually sorted into two classes:

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
- **Final Layer:** Fully Connected Layer → `nn.Linear(model.fc.in_features, 2)`
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

Evaluate the trained model on the test set of Chest X-Ray Images (Pneumonia) dataset:

- **Train-Test Accuracy Curve:**

![alt text](https://i.postimg.cc/qMzK5ppv/download.png)

- **Classification Report:**

![alt text](https://i.postimg.cc/wvkYqTJq/Screenshot-2025-04-04-175910.png)

- **Confusion Matrix:**

![alt text](https://i.postimg.cc/V6wcPbJZ/Screenshot-2025-04-04-175557.png)
---

## 🎨 Example Predictions

👉 **Inputs:**

- Normal:
![alt text](https://media-hosting.imagekit.io//22d1e13fcb364511/NORMAL2-IM-1438-0001.jpeg?Expires=1834220017&Key-Pair-Id=K2ZIVPTIP2VGHC&Signature=PW02uqXhsY1UspodGAAZJV-~z0rdNQ9WObYe0aBhnLRRObtDCMXJnYXRtH~wCzIhyU5uMV0uYWJHQhUqOjD5ujAwUpvQhh237IF3wG0lwyHAixmvJSic5OQbu8eG-YBDtZFYkjBS77y8KPMZLWCNy8UUCFSW6NaKaDsQjTH-VA96~YIkPZ9uzo2Q~RFnuFQqmIMcYGqbnUEiQtK2uc~irGFf6eeoqixAtbzLX48wxwWlj5WDjXTgbfl041syiG6WsMq4j40wynBIacB0ySgFKVPtBiPm3ko5UNwodob5WzO2Cxprr8SYF8sZrs89O3PIKosxEoftsbhMp8kRRk~Atg__)


- Pneumonia:
![alt text](https://media-hosting.imagekit.io//473a0ca656914ee1/person1951_bacteria_4882.jpeg?Expires=1834220181&Key-Pair-Id=K2ZIVPTIP2VGHC&Signature=g7Q9yfWqKBFZwVWrL1dH75KCNn5KrBBTeLBAnwzohu97rCBAMb4cemjSn1ohvEUpfX8j-kdQjeWlROl-iLKNnPsLacVLfzil~nBEjy0YJ5TOqGk27CWiQww4hAb52kVjRwfm0RSd2jFHGAD9LBROvAvsajsG0IxhgS5PZ9aCtqTFpVTbp1SuED-2WJ8a~iASDUb7Lzc056eQHYnXxx7b64xj-g6EwyAVx1W~6d3HcVwqiYa2sfvENoZbzWN9r2PYiL35OG5zfvL6qwMJyTKzG7wGHK8VC3ZM9yUdA9CAklInj8wRhZIqpFHWHnWi1d2zFktsKpK~yX5ql~wh28RXaQ__)




👉 **Outputs:**

![alt text](https://media-hosting.imagekit.io//826ae31686914d39/Screenshot%202025-02-15%20113203.png?Expires=1834220519&Key-Pair-Id=K2ZIVPTIP2VGHC&Signature=HrXzgBECcK1Qd8Xvhl64y~Lc9MkkWYKt8tt7nOmAWbQaGfnnDLlwpwbS8-OXwFD0eJwPo5ItY~8WeUhBNujLGpxkacEthkSUGmZZIe-Pa7NE2RMwKBu5jfcK~~QIHTO-EjGWgvEOWzxUm-eivgvqlh-FIdH~bOjxMdb-4Pefwnnj9d1SkUI5PiL5XUpHHoCSmL4Hh6awfYriBNhxuYziSuzGIo5dtPYUn~BZrOn-JhgwgXcyXu1mchceJr9f-ybOZHBdsVz4~4HMVUDzhe0vKwN~yqPxUWHIcMh2e80LKMCsZPnHvkYbL~4xySPAkfN8CBltPUpo4ouqgtxhUpagDw__)


![alt text](https://media-hosting.imagekit.io//b7d30dfc358e4e5a/Screenshot%202025-02-15%20113220.png?Expires=1834220504&Key-Pair-Id=K2ZIVPTIP2VGHC&Signature=W2igp1~CoZdB5bSx8U2BA6vNyJvZ0xY9if0nm56uPmDFXVXIQxFOhBvSYHQa2DXAuIHo5MW046PeT7BD5ka9oEBVntKZLCY0IaJwdtspl2fua8C4rieCPauYPHB7zDfOeH4qtWuZ0v1IsQtwekmuIOi5sD55yluel-R5e~yJsnnJTkob95aunuMAb04nu~y~CKmzI~NMyrJqjxh5MBqEHgRFMiMI7PJB7e7yg3IuOHSXjQlLjWOxO9Wxp67Jd7jdFl7U-0WCoHUm~st7NoIL8g7ekFNK2McG8Hp8HbfTzNc0hvhtb5BTAXjWFTf2QRXETA8qtDKKVCQgNjYwPqMdww__)
