# 🧠 CIFAR-10 Image Classification with MLOps Pipeline

This project demonstrates a complete **MLOps workflow** where a CNN model is trained on the **CIFAR-10** dataset using **PyTorch**, visualized via **Streamlit**, containerized using **Docker**, deployed to **Render**, and version controlled via **GitHub**.

---

## 🚀 Live App

🔗 [Click here to try the deployed app](https://classification-cifar1o-mlops.onrender.com)

---

## 🧰 Project Features

| Module               | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| 🧠 Model Training     | CNN model trained using PyTorch with **dropout** and **early stopping**     |
| 📊 Metrics Plotting   | Train & validation accuracy/loss visualization                             |
| 🧪 Inference UI       | Streamlit app to upload an image and get predictions                        |
| 🐳 Docker             | Dockerfile to containerize the entire app                                   |
| 🔁 CI/CD Pipeline     | Automatic deployment using GitHub + Render integration                      |
| 💾 Model Saving       | Best model checkpoint saved using `torch.save()`                            |

---

## 🧪 Dataset

- Dataset: **CIFAR-10**
- Classes: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`

---

## 🏗️ Project Structure

Classification-CIFAR10-MLOps/

├── streamlit_app/

│ ├── app.py # Streamlit UI

│ ├── model/

│ │ ├── cnn_model.py # CNN model architecture

│ │ └── utils.py # Preprocessing and helper functions

│ ├── saved_models/

│ │ └── best_model.pth # Saved PyTorch model

│ └── assets/ # Sample images (optional)

├── Dockerfile # Docker build configuration

├── requirements.txt # Python dependencies

└── README.md # Project info (you’re reading it!)

---

## 🧑‍💻 How to Run Locally

### 1️⃣ Clone the repo

```bash
git clone https://github.com/your-username/Classification-CIFAR10-MLOps.git
cd Classification-CIFAR10-MLOps
```

### 2️⃣ Build Docker image

```bash
docker build -t cnn-streamlit-app .
```

### 3️⃣ Run Docker container

```bash
docker run -p 8501:8080 cnn-streamlit-app
```
App will be live at: http://localhost:8501

---

## ⚙️ Technologies Used
-  🐍 Python 3.9+
-  🔥 PyTorch
-  🧪 Streamlit
-  🐳 Docker
-  🧾 Git + GitHub
-  ☁️ Render (for deployment)
-  ✅ CI/CD (via Render’s GitHub auto-deploy)

---

## 🧠 CNN Architecture

```text
Conv2d -> ReLU -> MaxPool
Conv2d -> ReLU -> MaxPool
Flatten -> Dropout -> Linear -> ReLU -> Dropout -> Linear -> Softmax
```

---

## 📈 Metrics Visualization
Training and validation metrics are plotted and shown in the app:
  -  📉 Loss vs Epoch
  -  📈 Accuracy vs Epoch

---

## 📸 Sample Prediction Screenshot
![image](https://github.com/user-attachments/assets/2e0aa325-c97a-4cb5-9970-1597e5e094b3)

---

## 🔄 Future Improvements
 - Integrate MLflow or Weights & Biases for experiment tracking
 - Add FastAPI-based inference API
 - Improve model performance with data augmentation
 - Add unit tests for utils and model files
 - Add automatic retraining pipeline via GitHub Actions

---

## 🙌 Acknowledgments
 - CIFAR-10 dataset from Kaggle
 - PyTorch & Streamlit documentation
 - Render for hosting

---

## 📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

