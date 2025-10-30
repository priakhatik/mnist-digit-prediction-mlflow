# 🔢 MNIST Digit Prediction and Model Deployment with MLflow

## Overview
This project demonstrates an end-to-end **machine learning workflow** for handwritten digit recognition using the **MNIST dataset**.  
It includes data preprocessing, model development using deep learning, and experiment tracking using **MLflow** to monitor and manage model performance.  
The project emphasizes **reproducibility, performance tracking, and scalable model deployment**, aligning with modern MLOps practices.

---

## Objectives
- Build and train a neural network to classify handwritten digits (0–9).  
- Track experiments, metrics, and parameters using **MLflow**.  
- Evaluate and compare model performance on test data.  
- Demonstrate model saving and reproducibility for deployment.  

---

## Dataset
- **Dataset Name:** MNIST Handwritten Digit Dataset  
- **Source:** [Yann LeCun’s MNIST Dataset](http://yann.lecun.com/exdb/mnist/)  
- **Records:** 70,000 grayscale images (60,000 for training, 10,000 for testing)  
- **Image Size:** 28 × 28 pixels  
- **Classes:** Digits 0–9  

---

## 🧰 Tools & Technologies
- Python (3.10+)  
- TensorFlow / Keras  
- Scikit-learn  
- Matplotlib, Seaborn  
- NumPy, Pandas  
- MLflow for experiment tracking  
- Jupyter Notebook  

---

## Methodology

### 1. Data Preprocessing
- Loaded and normalized MNIST image data.  
- Flattened pixel values and encoded labels.  
- Split data into training and testing subsets.  

### 2. Model Development
- Built a **Neural Network (NN)** or **Convolutional Neural Network (CNN)** architecture.  
- Compiled model using **Adam optimizer** and **categorical crossentropy loss**.  
- Trained for multiple epochs to optimize accuracy.  

### 3. Model Tracking with MLflow
- Logged model parameters, metrics, and artifacts using MLflow.  
- Tracked performance across different model configurations.  
- Saved the trained model for reproducibility and deployment.  

### 4. Evaluation
- Evaluated accuracy, precision, recall, and F1-score.  
- Visualized confusion matrix and misclassified samples.  
- Compared performance across tracked experiments.  

---

## 📊 Results
| Model Type | Accuracy | Notes |
|-------------|-----------|-------|
| Simple Neural Network | 97.8% | Fast training, small architecture |
| CNN Model | **99.2%** | High accuracy and robust to variations |

**Key Insights**
- CNN models outperform fully connected architectures for image data.  
- MLflow enables transparent experiment tracking and reproducibility.  
- Model artifacts can be easily deployed or reloaded for inference.  

---

## 🌍 Deployment
- The trained model and metrics were logged to **MLflow** for versioning and tracking.  
- Model artifacts can be deployed locally or to a cloud service (e.g., AWS S3, Azure ML, or Databricks MLflow).  
- Example MLflow workflow:
  ```bash
  mlflow ui
  - Then open http://127.0.0.1:5000 in a browser to view tracked experiments.
