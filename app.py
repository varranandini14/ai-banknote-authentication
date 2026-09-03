"""
app.py
======
AI-Based Banknote Authentication & Counterfeit Detection System
Phase 5 — Professional Streamlit Application

This application provides a production-grade inference interface for classifying
banknote authenticity using statistical wavelet features and a pre-trained RBF SVM pipeline.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==============================================================================
# 1. PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="AI Banknote Authentication",
    page_icon="💵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling for readability, cards, and clean typography
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1D3557;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        font-weight: 400;
        color: #457B9D;
        margin-bottom: 1.2rem;
    }
    .result-card-auth {
        background-color: #EBF8F2;
        border-left: 6px solid #2A9D8F;
        padding: 1.4rem;
        border-radius: 8px;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .result-card-counterfeit {
        background-color: #FDEDEC;
        border-left: 6px solid #E63946;
        padding: 1.4rem;
        border-radius: 8px;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .result-title {
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }
    .confidence-text {
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }
    .explanation-text {
        font-size: 0.95rem;
        color: #2B2D42;
        line-height: 1.5;
    }
    .disclaimer-box {
        font-size: 0.85rem;
        color: #6C757D;
        border-top: 1px solid #E0E0E0;
        margin-top: 0.8rem;
        padding-top: 0.6rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 6px;
        height: 3rem;
        font-size: 1.05rem;
        font-weight: 600;
        background-color: #1D3557;
        color: #FFFFFF;
        border: none;
    }
    .stButton>button:hover {
        background-color: #457B9D;
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. CONSTANTS & MODEL LOADING
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "banknote_model.pkl"
FEATURE_NAMES = ["variance", "skewness", "curtosis", "entropy"]

# Verified Sample Measurements from the Dataset for User Testing
PRESET_SAMPLES: Dict[str, Dict[str, float]] = {
    "Select a pre-configured sample...": {
        "variance": 0.0, "skewness": 0.0, "curtosis": 0.0, "entropy": 0.0
    },
    "Authentic Sample 1 (Note #763)": {
        "variance": -1.3971, "skewness": 3.3191, "curtosis": -1.3927, "entropy": -1.9948
    },
    "Authentic Sample 2 (Note #764)": {
        "variance": 0.39012, "skewness": -0.14279, "curtosis": -0.031994, "entropy": 0.35084
    },
    "Authentic Sample 3 (Note #765)": {
        "variance": -1.6677, "skewness": -7.1535, "curtosis": 7.8929, "entropy": 0.96765
    },
    "Counterfeit Sample 1 (Note #1)": {
        "variance": 3.6216, "skewness": 8.6661, "curtosis": -2.8073, "entropy": -0.44699
    },
    "Counterfeit Sample 2 (Note #2)": {
        "variance": 4.5459, "skewness": 8.1674, "curtosis": -2.4586, "entropy": -1.4621
    },
    "Counterfeit Sample 3 (Note #3)": {
        "variance": 3.8660, "skewness": -2.6383, "curtosis": 1.9242, "entropy": 0.10645
    },
}


@st.cache_resource(show_spinner="Loading trained authentication model...")
def load_trained_pipeline(path: Path) -> Any:
    """
    Safely loads the serialized Pipeline (StandardScaler + RBF SVC).
    Uses Streamlit caching to prevent repeated disk I/O.
    """
    if not path.exists():
        st.error(
            f"❌ Model artifact not found at `{path}`. "
            "Please run `python train_model.py` to train and serialize the model."
        )
        st.stop()

    try:
        pipeline = joblib.load(path)
        return pipeline
    except Exception as exc:
        st.error(f"❌ Failed to load model artifact: {exc}")
        st.stop()


# Load model once
pipeline = load_trained_pipeline(MODEL_PATH)


# ==============================================================================
# 3. SIDEBAR: PROJECT SPECIFICATION & DISCLAIMER
# ==============================================================================
with st.sidebar:
    st.markdown("### 💵 AI Banknote Authentication")
    st.caption("Machine Learning Based Counterfeit Detection")
    st.markdown("---")

    st.markdown("#### Model Specification")
    st.markdown("""
    - **Algorithm:** Support Vector Machine (RBF)
    - **Preprocessing:** Embedded `StandardScaler`
    - **Hyperparameters:** `C=1.0`, `gamma='scale'`
    - **Tuned via:** Stratified 5-Fold GridSearchCV
    - **Input Features:** 4 Continuous Measurements
    - **Target Classes:** Binary (Authentic / Counterfeit)
    """)

    st.markdown("---")
    st.markdown("#### Feature Inputs")
    st.markdown("""
    1. **Variance** (Wavelet Transformed)
    2. **Skewness** (Wavelet Transformed)
    3. **Curtosis** (Wavelet Transformed)
    4. **Entropy** (Image Information)
    """)

    st.markdown("---")
    st.markdown("#### Advisory Notice")
    st.info(
        "This application delivers predictive classifications generated by an empirical machine "
        "learning model. It is designed for analytical and demonstration purposes and is not a "
        "substitute for high-security forensic currency verification equipment."
    )


# ==============================================================================
# 4. MAIN HEADER & INTRODUCTORY OVERVIEW
# ==============================================================================
st.markdown('<div class="main-header">AI Banknote Authentication</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Machine Learning Based Authenticity & Counterfeit Detection</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter the four statistical measurements extracted from a digitized banknote image using Wavelet "
    "Transformation. The trained Support Vector Machine pipeline evaluates the feature pattern to "
    "classify whether the banknote is **Authentic** or **Counterfeit**."
)

st.caption(
    "ℹ️ **Note:** This model operates on the **four numerical wavelet features** from the UCI Banknote "
    "Authentication Dataset, not directly on an uploaded image file."
)

st.markdown("---")

# ==============================================================================
# 5. SAMPLE VALUE SELECTOR & RESET
# ==============================================================================
st.markdown("### 🧪 Try Sample Values")
sample_col1, sample_col2 = st.columns([3, 1])

with sample_col1:
    selected_sample_key = st.selectbox(
        "Select pre-validated benchmark values from the dataset (optional):",
        options=list(PRESET_SAMPLES.keys()),
        index=0,
        help="Choose an authentic or counterfeit banknote sample to automatically populate the input fields."
    )

with sample_col2:
    st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
    reset_clicked = st.button("🔄 Reset Inputs")

# Determine active input values based on selection or reset
if reset_clicked or selected_sample_key == "Select a pre-configured sample...":
    init_variance = 0.0000
    init_skewness = 0.0000
    init_curtosis = 0.0000
    init_entropy = 0.0000
else:
    sample_data = PRESET_SAMPLES[selected_sample_key]
    init_variance = float(sample_data["variance"])
    init_skewness = float(sample_data["skewness"])
    init_curtosis = float(sample_data["curtosis"])
    init_entropy = float(sample_data["entropy"])

# ==============================================================================
# 6. INPUT MEASUREMENTS SECTION
# ==============================================================================
st.markdown("### 📐 Banknote Measurements")
st.caption("Provide the four wavelet transform continuous features (ordered strictly as per model requirements):")

col_v, col_s = st.columns(2)
col_c, col_e = st.columns(2)

with col_v:
    input_variance = st.number_input(
        "1. Variance (Wavelet Transformed)",
        value=init_variance,
        step=0.1,
        format="%.5f",
        help="Variance of the Wavelet Transformed image (typical range: -7.0 to +6.8)."
    )

with col_s:
    input_skewness = st.number_input(
        "2. Skewness (Wavelet Transformed)",
        value=init_skewness,
        step=0.1,
        format="%.5f",
        help="Skewness of the Wavelet Transformed image (typical range: -13.8 to +13.0)."
    )

with col_c:
    input_curtosis = st.number_input(
        "3. Curtosis (Wavelet Transformed)",
        value=init_curtosis,
        step=0.1,
        format="%.5f",
        help="Curtosis of the Wavelet Transformed image (typical range: -5.3 to +17.9)."
    )

with col_e:
    input_entropy = st.number_input(
        "4. Entropy (Image Data)",
        value=init_entropy,
        step=0.1,
        format="%.5f",
        help="Entropy of the banknote image (typical range: -8.5 to +2.5)."
    )

# ==============================================================================
# 7. INFERENCE & RESULT PRESENTATION
# ==============================================================================
st.markdown("<br>", unsafe_allow_html=True)
analyze_btn = st.button("🔍 Analyze Banknote")

if analyze_btn:
    raw_values = [input_variance, input_skewness, input_curtosis, input_entropy]

    # Robust Validation
    is_valid = True
    validation_error = ""

    for name, val in zip(FEATURE_NAMES, raw_values):
        if val is None or np.isnan(val):
            is_valid = False
            validation_error = f"Invalid entry: `{name}` is undefined or NaN."
            break
        if np.isinf(val):
            is_valid = False
            validation_error = f"Invalid entry: `{name}` has an infinite value."
            break

    if not is_valid:
        st.error(f"⚠️ **Validation Error:** {validation_error}")
    else:
        # Create single-row DataFrame with exact training feature names and column order
        input_df = pd.DataFrame([raw_values], columns=FEATURE_NAMES)

        try:
            # Model prediction through the serialized Pipeline (Scales automatically)
            pred_class = int(pipeline.predict(input_df)[0])
            pred_proba = pipeline.predict_proba(input_df)[0]

            # Probabilities: index 0 = Counterfeit (class 0), index 1 = Authentic (class 1)
            prob_counterfeit = float(pred_proba[0]) * 100.0
            prob_authentic = float(pred_proba[1]) * 100.0

            if pred_class == 1:
                confidence = prob_authentic
                card_html = f"""
                <div class="result-card-auth">
                    <div class="result-title" style="color: #2A9D8F;">✅ AUTHENTIC BANKNOTE</div>
                    <div class="confidence-text" style="color: #1D3557;">Prediction Confidence: {confidence:.2f}%</div>
                    <div class="explanation-text">
                        The trained machine learning pipeline classifies the supplied measurements as consistent with the
                        <strong>Authentic</strong> class (probability: {confidence:.2f}%).
                    </div>
                    <div class="disclaimer-box">
                        Disclaimer: This is an empirical ML-based prediction and should not be treated as definitive or
                        legal proof of banknote authenticity.
                    </div>
                </div>
                """
            else:
                confidence = prob_counterfeit
                card_html = f"""
                <div class="result-card-counterfeit">
                    <div class="result-title" style="color: #E63946;">⚠️ COUNTERFEIT BANKNOTE</div>
                    <div class="confidence-text" style="color: #780016;">Prediction Confidence: {confidence:.2f}%</div>
                    <div class="explanation-text">
                        The trained machine learning pipeline classifies the supplied measurements as consistent with the
                        <strong>Counterfeit</strong> class (probability: {confidence:.2f}%).
                    </div>
                    <div class="disclaimer-box">
                        Disclaimer: This is an empirical ML-based prediction and should not be treated as definitive or
                        legal proof of banknote authenticity.
                    </div>
                </div>
                """

            st.markdown(card_html, unsafe_allow_html=True)

            # Input Summary Table
            st.markdown("#### Input Measurements Summary")
            summary_cols = st.columns(4)
            summary_cols[0].metric("Variance", f"{input_variance:.4f}")
            summary_cols[1].metric("Skewness", f"{input_skewness:.4f}")
            summary_cols[2].metric("Curtosis", f"{input_curtosis:.4f}")
            summary_cols[3].metric("Entropy", f"{input_entropy:.4f}")

            # Confidence Breakdown Bar
            st.markdown("#### Class Confidence Distribution")
            bar_col1, bar_col2 = st.columns(2)
            bar_col1.progress(prob_authentic / 100.0, text=f"Authentic: {prob_authentic:.2f}%")
            bar_col2.progress(prob_counterfeit / 100.0, text=f"Counterfeit: {prob_counterfeit:.2f}%")

        except Exception as err:
            st.error(f"❌ An error occurred during model inference: {err}")

st.markdown("---")

# ==============================================================================
# 8. TECHNICAL SPECIFICATIONS & MODEL DOCUMENTATION
# ==============================================================================
tab_model, tab_perf, tab_works, tab_features = st.tabs([
    "ℹ️ About the Model",
    "📊 Model Performance",
    "⚙️ How It Works",
    "🔬 About the Features"
])

with tab_model:
    st.markdown("### Model Architecture & Configuration")
    st.markdown(r"""
    - **Classification Algorithm:** Support Vector Classifier with Radial Basis Function kernel (`SVC(kernel='rbf')`)
    - **Preprocessing Transformer:** Scikit-Learn `StandardScaler` (fits mean and standard deviation)
    - **Encapsulation:** Pure Scikit-Learn `Pipeline([('scaler', StandardScaler()), ('model', SVC(...))])`
    - **Optimal Hyperparameters (GridSearchCV):**
      - Regularization parameter ($C$): `1.0`
      - Kernel coefficient ($\gamma$): `'scale'` ($1 / (n\_features \times X.var())$)
      - Probability Calibration: Platt Scaling Enabled (`probability=True`)
    - **Training Dataset:** UCI Machine Learning Repository — Banknote Authentication Dataset
    - **Sample Count:** 1,372 total records (1,097 training / 275 held-out test)
    - **Input Dimensions:** 4 continuous wavelet variables
    - **Output:** Binary classification (`0 = Counterfeit`, `1 = Authentic`)
    """)

with tab_perf:
    st.markdown("### Verified Test Set Performance (Held-Out Test Partition)")
    metric_cols = st.columns(5)
    metric_cols[0].metric("Test Accuracy", "100.00%")
    metric_cols[1].metric("Test Precision", "100.00%")
    metric_cols[2].metric("Test Recall", "100.00%")
    metric_cols[3].metric("Test F1 Score", "100.00%")
    metric_cols[4].metric("Test ROC-AUC", "100.00%")

    st.markdown("#### Cross-Validation Consistency (Stratified 5-Fold on Training Set)")
    st.markdown("""
    - **Mean CV F1 Score:** `1.0000 ± 0.0000`
    - **Mean CV ROC-AUC:** `1.0000 ± 0.0000`
    """)

    st.warning(
        "⚠️ **Essential Evaluation Limitation & Transparency Note:**\n\n"
        "These performance metrics were measured strictly on the project's held-out test partition ($N=275$). "
        "The raw UCI dataset contains 24 duplicate rows (1.75% of raw data). A standard random stratified split "
        "results in 9 test samples matching identical records present in the training partition. This duplicate overlap "
        "can make a random-split evaluation somewhat optimistic. Therefore, these results should **not** be "
        "interpreted as guaranteed 100% real-world accuracy on unconstrained physical banknotes in circulation."
    )

with tab_works:
    st.markdown("### End-to-End Processing Pipeline")
    st.markdown(r"""
    1. **Feature Input:** The user specifies four continuous statistical values extracted from image wavelet transforms.
    2. **Integrity Validation:** Inputs are checked for numerical validity (rejecting NaN, infinite, or missing inputs).
    3. **Standardization:** The embedded `StandardScaler` centers each feature ($z = (x - \mu) / \sigma$) using statistics fitted strictly during training.
    4. **Non-Linear Decision Boundary:** The RBF Kernel SVM maps normalized feature vectors into infinite-dimensional Hilbert space where the authenticity boundary is maximized.
    5. **Probability Calibration:** The pipeline calculates confidence probabilities using Platt Scaling.
    6. **Decision Rendering:** The application presents the classification card, prediction confidence, and contextual summary.
    """)

with tab_features:
    st.markdown("### Understanding Wavelet Transform Features")
    st.markdown("""
    The dataset features are not raw physical measurements (such as paper weight or note width). Instead, they are
    **mathematical moments derived from 2-Level Discrete Wavelet Transforms (DWT)** of continuous grayscale digitized banknote images:

    - **Variance:** Measures the dispersion and spread of wavelet transform coefficient values. *EDA confirmed variance is the single most powerful linear and non-linear differentiator between authentic and counterfeit notes.*
    - **Skewness:** Measures the degree of asymmetry of the wavelet coefficient distribution around its mean.
    - **Curtosis:** Measures the 'tailedness' and peak sharp-frequency components in the transformed image, reflecting fine printing micro-structures.
    - **Entropy:** Quantifies the global information density and randomness of the banknote surface pattern.
    """)

# ==============================================================================
# 9. FOOTER
# ==============================================================================
st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption(
    "AI-Based Banknote Authentication & Counterfeit Detection System • Phase 5 Production Release • Built with Streamlit & Scikit-Learn"
)
