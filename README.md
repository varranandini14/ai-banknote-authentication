# AI-Based Banknote Authentication & Counterfeit Detection System

A production-grade machine learning application to authenticate banknotes and detect counterfeits using continuous wavelet transform features derived from the UCI Banknote Authentication Dataset.

## Dataset

**Source:** [UCI Machine Learning Repository – Banknote Authentication](https://archive.ics.uci.edu/ml/datasets/banknote+authentication)

**Features:**
| Feature | Description |
| :--- | :--- |
| `variance` | Variance of Wavelet Transformed image data (continuous) |
| `skewness` | Skewness of Wavelet Transformed image data (continuous) |
| `curtosis` | Curtosis of Wavelet Transformed image data (continuous) |
| `entropy` | Entropy of the banknote image data (continuous) |
| `class` | **Target Label**: `0 = Counterfeit`, `1 = Authentic` |

**Shape:** 1,372 rows × 5 columns (100% complete, 0 missing values)

---

## Project Structure

```
banknote-authentication/
│
├── data/
│   └── banknote.csv                       # Raw UCI Banknote dataset (unmodified)
├── models/
│   └── banknote_model.pkl                 # Serialized final pipeline (StandardScaler + RBF SVM)
├── notebooks/
│   └── EDA.ipynb                          # Complete exploratory analysis notebook
├── results/
│   ├── model_comparison.csv               # Baseline metrics comparison
│   ├── final_model_metrics.csv            # Baseline vs. Tuned SVM comparison
│   ├── svm_grid_search_results.csv        # 20-parameter GridSearchCV results
│   ├── final_confusion_matrix.png         # High-resolution confusion matrix
│   ├── confusion_matrix_logistic_regression.png
│   ├── confusion_matrix_random_forest.png
│   ├── confusion_matrix_support_vector_machine.png
│   └── random_forest_feature_importance.png
├── screenshots/                           # UI application captures
│
├── app.py                                 # Professional Streamlit web interface
├── train_model.py                         # Full training, CV, tuning, and audit pipeline
├── requirements.txt                       # Project dependencies
├── README.md                              # Complete system documentation
└── .gitignore                             # Git ignore rules
```

---

## Machine Learning Pipeline

1. **Preprocessing**: Feature standardization via `StandardScaler`.
2. **Model Architecture**: Support Vector Classifier with Radial Basis Function kernel (`SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)`).
3. **Encapsulation**: Complete scikit-learn `Pipeline` saved to `models/banknote_model.pkl` to prevent data leakage and eliminate manual user scaling.
4. **Validation**: Stratified 5-fold cross-validation with Platt scaling for well-calibrated confidence probabilities.

### Model Performance (Held-Out Test Partition)
- **Accuracy**: 100.00%
- **Precision**: 100.00%
- **Recall**: 100.00%
- **F1-Score**: 100.00%
- **ROC-AUC**: 100.00%

> **Important Evaluation Limitation & Transparency Note:**  
> These metrics are measured on the held-out test partition ($N=275$). The raw UCI dataset contains 24 duplicate rows (1.75% of data). In a standard random stratified split, 9 test samples match identical records in the training set. This overlap can make a random-split evaluation slightly optimistic. These results should **not** be interpreted as guaranteed 100% real-world accuracy on unconstrained physical banknotes in circulation.

---

## Streamlit Application

The project includes an interactive web interface for real-time banknote classification using the trained RBF SVM pipeline.

### Launching the Application

```bash
# Windows
.venv\Scripts\python.exe -m streamlit run app.py

# Generic
streamlit run app.py
```

### Application Features
- **Four Input Fields**: Numeric floating-point inputs for `Variance`, `Skewness`, `Curtosis`, and `Entropy` matching the model's strict feature order.
- **Real-Time Classification**: Instant classification of notes as **AUTHENTIC BANKNOTE** or **COUNTERFEIT BANKNOTE**.
- **Prediction Confidence**: Displays class probability percentage based on calibrated Platt scaling.
- **Preset Dataset Samples**: Pre-configured sample buttons for quick testing and demonstration.
- **Input Validation**: Actively validates against non-numeric, infinite, and NaN inputs.
- **Model Documentation**: Embedded technical tabs covering architecture, performance, wavelet transform feature meanings, and evaluation limitations.

*Note: The model operates on the four statistical wavelet features from the dataset, not directly on raw image uploads.*

---

## Setup & Installation

### Prerequisites
- Python 3.12
- pip

### Virtual Environment Setup

```bash
# Create virtual environment
py -3.12 -m venv .venv

# Activate virtual environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

---

## Dependencies

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `joblib`
- `streamlit`
- `jupyter`

---

## License

This project utilizes the publicly available UCI Banknote Authentication Dataset for educational and research purposes.
