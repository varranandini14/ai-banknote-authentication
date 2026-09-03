# AI-Based Banknote Authentication & Counterfeit Detection System

A production-grade machine learning application for banknote authentication and counterfeit detection using continuous wavelet transform features from the UCI Banknote Authentication Dataset.

The system uses a trained Support Vector Machine (SVM) pipeline and provides an interactive Streamlit web interface for real-time classification.

---

## Dataset

**Source:** [UCI Machine Learning Repository – Banknote Authentication](https://archive.ics.uci.edu/ml/datasets/banknote+authentication)

### Features

| Feature | Description |
| :--- | :--- |
| `variance` | Variance of Wavelet Transformed image data |
| `skewness` | Skewness of Wavelet Transformed image data |
| `curtosis` | Curtosis of Wavelet Transformed image data |
| `entropy` | Entropy of the banknote image data |
| `class` | **Target Label:** `0 = Counterfeit`, `1 = Authentic` |

**Dataset Shape:** 1,372 rows × 5 columns  
**Missing Values:** 0

---

## Project Structure

```text
BANKNOTE-AUTHENTCATION/
│
├── data/
│   └── banknote.csv
│
├── models/
│   └── banknote_model.pkl
│
├── notebooks/
│   └── EDA.ipynb
│
├── results/
│   ├── model_comparison.csv
│   ├── final_model_metrics.csv
│   ├── svm_grid_search_results.csv
│   ├── final_confusion_matrix.png
│   ├── confusion_matrix_logistic_regression.png
│   ├── confusion_matrix_random_forest.png
│   ├── confusion_matrix_support_vector_machine.png
│   └── random_forest_feature_importance.png
│
├── screenshots/
│   ├── home.png
│   ├── authentic_prediction.png
│   └── counterfeit_prediction.png
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
└── .gitignore