import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from streamlit_drawable_canvas import st_canvas
from PIL import Image

st.set_page_config(
    page_title="Digit Recognition — SVM",
    page_icon="🔢",
    layout="centered"
)

@st.cache_resource
def load_model():
    model  = joblib.load("model/svm_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    pca    = joblib.load("model/pca.pkl")
    return model, scaler, pca

model, scaler, pca = load_model()

st.title("🔢 Handwritten Digit Recognition")
st.markdown("**Using Support Vector Machine (SVM) | Accuracy: 97.78%**")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🎨 Predict", "📊 Model Info", "📈 EDA & Results"])

with tab1:
    st.subheader("Pick a test digit to classify")

    digits = load_digits()
    X, y   = digits.data, digits.target

    col1, col2 = st.columns([1, 1])

    with col1:
        digit_class = st.selectbox("Select true digit (0–9)", range(10))
        idx_options = np.where(y == digit_class)[0]
        sample_idx  = st.slider("Sample index", 0, len(idx_options)-1, 0)
        chosen_idx  = idx_options[sample_idx]

        img = X[chosen_idx].reshape(8, 8)
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(img, cmap='gray_r', interpolation='nearest')
        ax.axis('off')
        ax.set_title(f"True Label: {digit_class}", fontsize=13, fontweight='bold')
        st.pyplot(fig)

    with col2:
        if st.button("🔍 Predict!", use_container_width=True):
            raw    = X[chosen_idx].reshape(1, -1)
            scaled = scaler.transform(raw)
            pcad   = pca.transform(scaled)
            pred   = model.predict(pcad)[0]
            proba  = model.predict_proba(pcad)[0]

            if pred == digit_class:
                st.success(f"✅ Predicted: **{pred}**")
            else:
                st.error(f"❌ Predicted: **{pred}** (True: {digit_class})")

            st.markdown("#### Confidence per class:")
            fig2, ax2 = plt.subplots(figsize=(5, 3))
            colors = ['#2ecc71' if i == pred else '#95a5a6' for i in range(10)]
            ax2.bar(range(10), proba * 100, color=colors)
            ax2.set_xlabel("Digit"); ax2.set_ylabel("Confidence (%)")
            ax2.set_xticks(range(10))
            ax2.set_title("Prediction Confidence", fontweight='bold')
            st.pyplot(fig2)

with tab2:
    st.subheader("📊 Model Details")
    col1, col2, col3 = st.columns(3)
    col1.metric("Test Accuracy",   "97.78%")
    col2.metric("Algorithm",       "SVM (RBF)")
    col3.metric("Best C",          "1")

    col4, col5, col6 = st.columns(3)
    col4.metric("Best Gamma",      "scale")
    col5.metric("PCA Components",  "40")
    col6.metric("Training Samples","1437")

    st.markdown("---")
    st.markdown("""
    ### 🔧 Pipeline
    1. **Load Data** — sklearn Digits dataset (1797 samples, 8×8 images)
    2. **Train-Test Split** — 80% train, 20% test, stratified
    3. **Standard Scaling** — Zero mean, unit variance
    4. **PCA** — Reduced to 40 components (95% variance retained)
    5. **GridSearchCV** — Tuned C & gamma over 5-fold CV
    6. **SVM (RBF Kernel)** — Final model with best parameters
    """)

with tab3:
    st.subheader("📈 EDA & Evaluation Plots")

    plots = {
        "Class Distribution":       "model/class_distribution.png",
        "Sample Images":            "model/sample_images.png",
        "PCA Variance":             "model/pca_variance.png",
        "Confusion Matrix":         "model/confusion_matrix.png",
        "GridSearch Heatmap":       "model/gridsearch_heatmap.png",
        "Misclassified Examples":   "model/misclassified.png",
    }

    for title, path in plots.items():
        st.markdown(f"#### {title}")
        try:
            st.image(path, use_container_width=True)
        except:
            st.warning(f"Run the notebook first to generate: {path}")

