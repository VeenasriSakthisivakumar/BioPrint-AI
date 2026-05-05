# ==============================
# BioPrint AI - Single File Project
# Fingerprint Blood Group Detection
# No external dataset required
# Built-in synthetic dataset
# Streamlit + OpenCV + Random Forest
# ==============================

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
import random

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="BioPrint AI",
    layout="wide"
)

# ---------------- CUSTOM UI ----------------
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0b0f19, #111827);
}
.title {
    font-size: 48px;
    font-weight: bold;
    color: #ff4b4b;
}
.subtitle {
    font-size: 18px;
    color: #9ca3af;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='title'>BioPrint AI</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Fingerprint-Based Blood Group Detection System</div>",
    unsafe_allow_html=True
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("Fingerprint Pattern")
pattern = st.sidebar.selectbox(
    "Select Pattern Type",
    ["Loop", "Whorl", "Arch", "Tented"]
)

pattern_map = {
    "Loop": 0,
    "Whorl": 1,
    "Arch": 2,
    "Tented": 3
}

# ---------------- SYNTHETIC DATASET ----------------
@st.cache_resource
def train_model():
    blood_groups = ["A+", "B+", "O+", "AB+"]
    data = []

    # Create synthetic biometric dataset
    for _ in range(1200):
        ridge_density = random.uniform(20, 80)
        minutiae_count = random.randint(10, 150)
        pattern_type = random.randint(0, 3)

        # Basic realistic mapping logic
        if ridge_density > 60:
            label = random.choice(["O+", "A+"])
        elif minutiae_count > 90:
            label = random.choice(["B+", "AB+"])
        else:
            label = random.choice(blood_groups)

        data.append([
            ridge_density,
            minutiae_count,
            pattern_type,
            label
        ])

    df = pd.DataFrame(
        data,
        columns=[
            "ridge_density",
            "minutiae_count",
            "pattern_type",
            "blood_group"
        ]
    )

    X = df[["ridge_density", "minutiae_count", "pattern_type"]]
    y = df["blood_group"]

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42
    )

    model.fit(X, y)

    return model


model = train_model()

# ---------------- IMAGE PROCESSING ----------------
def extract_features(image):
    img = np.array(image)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Resize
    gray = cv2.resize(gray, (256, 256))

    # Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold
    _, thresh = cv2.threshold(
        blur,
        127,
        255,
        cv2.THRESH_BINARY_INV
    )

    # Ridge Density
    ridge_density = round(
        np.sum(thresh == 255) / (256 * 256) * 100,
        2
    )

    # Minutiae Approximation
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    minutiae_count = len(contours)

    return ridge_density, minutiae_count, thresh


# ---------------- MAIN LAYOUT ----------------
col1, col2 = st.columns(2)

# -------- LEFT PANEL --------
with col1:
    st.subheader("Upload Fingerprint")

    uploaded_file = st.file_uploader(
        "Choose fingerprint image",
        type=["png", "jpg", "jpeg", "bmp"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Fingerprint",
            use_container_width=True
        )

# -------- RIGHT PANEL --------
with col2:
    st.subheader("Prediction Dashboard")

    if uploaded_file:
        ridge_density, minutiae_count, processed = extract_features(image)

        # Prepare input
        input_data = pd.DataFrame(
            [[
                ridge_density,
                minutiae_count,
                pattern_map[pattern]
            ]],
            columns=[
                "ridge_density",
                "minutiae_count",
                "pattern_type"
            ]
        )

        # Predict
        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(input_data)[0]

        confidence = round(max(probabilities) * 100, 2)

        # Metrics
        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Ridge Density",
            f"{ridge_density}%"
        )

        m2.metric(
            "Minutiae Count",
            minutiae_count
        )

        m3.metric(
            "Confidence",
            f"{confidence}%"
        )

        # Prediction
        st.success(
            f"Predicted Blood Group: {prediction}"
        )

        # Probability Chart
        chart_data = pd.DataFrame({
            "Blood Group": model.classes_,
            "Probability": probabilities
        })

        st.bar_chart(
            chart_data.set_index("Blood Group")
        )

        # Compatibility Info
        compatibility = {
            "O+": "Can donate to O+, A+, B+, AB+",
            "A+": "Can donate to A+, AB+",
            "B+": "Can donate to B+, AB+",
            "AB+": "Can donate to AB+"
        }

        st.warning(
            compatibility.get(
                prediction,
                "Compatibility data limited"
            )
        )

        # Research Disclaimer
        st.info(
            "This project is a research-oriented AI prototype using fingerprint image processing and machine learning for educational purposes."
        )

        # Technical Explanation
        with st.expander("Technical Explanation"):
            st.write("Image Processing: OpenCV")
            st.write("Feature Extraction: Ridge Density + Minutiae Count")
            st.write("Algorithm: Random Forest Classifier")
            st.write("Dataset: Built-in synthetic biometric dataset")
            st.write("Future Scope: CNN + Real Medical Dataset")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption(
    "BioPrint AI | Fingerprint Blood Group Detection "
)