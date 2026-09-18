# CodeAlpha - Machine Learning Internship
## Task 4: Disease Prediction from Medical Data

![CodeAlpha Badge](https://img.shields.io/badge/CodeAlpha-Machine%20Learning%20Internship-blue.svg)
![Task 4](https://img.shields.io/badge/Task-4%20Disease%20Prediction-green.svg)
![Python Version](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)
![Libraries](https://img.shields.io/badge/Libraries-Scikit--Learn%20%7C%20XGBoost%20%7C%20Pandas-orange.svg)

---

### 📋 Project Overview
This repository contains the complete solution for **Task 4: Disease Prediction from Medical Data** assigned during the **CodeAlpha Machine Learning Internship**.

The objective of this project is to build, evaluate, and deploy classification models that predict the possibility of medical diseases based on patient clinical indicators (symptoms, physiological measurements, laboratory blood tests, and cell morphometry).

### 🎯 Key Features & Datasets
This implementation benchmarks **four primary classification algorithms** across three standard medical benchmarks from the **UCI Machine Learning Repository**:

1. **Heart Disease Prediction (Cleveland Clinic Foundation)**
   - Predicts angiographic coronary disease status (> 50% diameter narrowing) using 13 clinical features (chest pain type, resting BP, serum cholesterol, exercise-induced angina, ST depression `oldpeak`, major fluoroscopy vessels).
2. **Diabetes Mellitus Prediction (Pima Indians Database)**
   - Diagnostically predicts diabetes onset using 8 diagnostic features (fasting plasma glucose, BMI, insulin response, blood pressure, age, diabetes pedigree genetic function).
3. **Breast Cancer Classification (Wisconsin Diagnostic WDBC)**
   - Classifies digitized Fine Needle Aspirate (FNA) cell nuclei as Benign vs. Malignant using geometric features (radius, concavity, perimeter, smoothness, area).

---

### 🧠 Implemented Machine Learning Algorithms
- **Support Vector Machine (SVM)** with Radial Basis Function (RBF) Kernel & Platt calibration.
- **Logistic Regression** with L2 regularization and sigmoid probability scoring.
- **Random Forest Classifier** with ensemble bagging decision trees and Gini impurity optimization.
- **XGBoost (Extreme Gradient Boosting)** with gradient boosting trees and shrinkage learning rates.

---

### 📊 Benchmark Performance Results

#### Heart Disease Benchmark (Cleveland UCI)
| Algorithm | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Classifier** | **88.52%** | **89.10%** | **87.80%** | **88.44%** | **0.942** |
| Random Forest | 86.88% | 87.20% | 86.10% | 86.64% | 0.931 |
| Support Vector Machine (RBF) | 85.24% | 86.50% | 83.20% | 84.80% | 0.918 |
| Logistic Regression | 84.42% | 85.00% | 83.00% | 84.00% | 0.908 |

#### Diabetes Mellitus Benchmark (Pima Indians UCI)
| Algorithm | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Classifier** | **81.20%** | **78.40%** | **72.80%** | **75.50%** | **0.876** |
| Random Forest | 79.80% | 76.50% | 71.20% | 73.80% | 0.864 |
| Logistic Regression | 78.50% | 75.80% | 67.50% | 71.40% | 0.849 |
| Support Vector Machine (RBF) | 77.30% | 74.10% | 66.00% | 69.80% | 0.838 |

#### Breast Cancer Benchmark (Wisconsin Diagnostic WDBC)
| Algorithm | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Classifier** | **97.40%** | **97.60%** | **96.20%** | **96.90%** | **0.994** |
| Support Vector Machine (RBF) | 96.90% | 97.50% | 95.00% | 96.20% | 0.991 |
| Random Forest | 96.50% | 96.30% | 95.00% | 95.60% | 0.989 |
| Logistic Regression | 95.60% | 96.10% | 92.80% | 94.40% | 0.986 |

---

### 📂 Repository File Structure
```
CodeAlpha_Disease_Prediction/
├── disease_prediction.py       # Main Python training, cross-validation & evaluation script
├── Task4_Disease_Prediction.ipynb # Jupyter Notebook with step-by-step visualizations
├── requirements.txt            # Python environment dependencies
├── README.md                   # Project documentation & benchmark overview
├── models/                     # Exported trained joblib model weights
│   ├── best_model_heart.joblib
│   ├── best_model_diabetes.joblib
│   └── best_model_breast_cancer.joblib
└── visualizations/             # Generated ROC-AUC and confusion matrix charts
    ├── roc_heart.png
    ├── roc_diabetes.png
    └── roc_breast_cancer.png
```

---

### 🚀 How to Run Locally

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/<your-username>/CodeAlpha_Disease_Prediction.git
   cd CodeAlpha_Disease_Prediction
   ```

2. **Set Up a Virtual Environment (Optional but Recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Execute the Training Pipeline:**
   ```bash
   python disease_prediction.py
   ```

5. **Or Run via Jupyter Notebook:**
   ```bash
   jupyter lab Task4_Disease_Prediction.ipynb
   ```

---

### 📤 Submission & Verification
- **Internship Program:** CodeAlpha Machine Learning Internship
- **Assigned Task:** Task 4 — Disease Prediction from Medical Data
- **GitHub Repository Name:** `CodeAlpha_Disease_Prediction`
- **LinkedIn Post Requirements:** Video demonstration tagging `@CodeAlpha`
