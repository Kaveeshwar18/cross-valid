# app.py

import streamlit as st
import numpy as np
import pickle
import time

st.set_page_config(page_title="Wine Quality Predictor", layout="wide")

# --------- ANIMATED CSS ---------
st.markdown("""
<style>

/* Animated Gradient Background */
.stApp {
    background: linear-gradient(-45deg, #667eea, #764ba2, #ff512f, #dd2476);
    background-size: 400% 400%;
    animation: gradientMove 12s ease infinite;
}

/* Gradient Animation */
@keyframes gradientMove {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Fade-in animation */
.fade-in {
    animation: fadeIn 1.5s ease-in;
}

@keyframes fadeIn {
    from {opacity: 0; transform: translateY(20px);}
    to {opacity: 1; transform: translateY(0);}
}

/* Glass Card */
.card {
    background: rgba(255,255,255,0.15);
    padding: 30px;
    border-radius: 20px;
    backdrop-filter: blur(15px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    color: white;
}

/* Button Animation */
.stButton>button {
    background: linear-gradient(45deg, #ff512f, #dd2476);
    color: white;
    font-size: 18px;
    border-radius: 12px;
    padding: 10px 30px;
    border: none;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.1);
}

h1, h3 {
    color: white;
    text-align: center;
}

label {
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# -------- TITLE --------
st.markdown('<h1 class="fade-in">🍷 Wine Quality Predictor</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="fade-in">Machine Learning Based Prediction System</h3>', unsafe_allow_html=True)

# -------- LOAD MODEL --------
model = pickle.load(open("wine_model.pkl", "rb"))
scaler = pickle.load(open("wine_scaler.pkl", "rb"))

# -------- INPUT SECTION --------
with st.container():
    st.markdown('<div class="card fade-in">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        fixed_acidity = st.number_input("Fixed Acidity", 4.0, 16.0, 7.4)
        volatile_acidity = st.number_input("Volatile Acidity", 0.1, 2.0, 0.7)
        citric_acid = st.number_input("Citric Acid", 0.0, 1.0, 0.0)
        residual_sugar = st.number_input("Residual Sugar", 0.5, 15.0, 1.9)
        chlorides = st.number_input("Chlorides", 0.01, 0.5, 0.07)
        free_sulfur = st.number_input("Free Sulfur Dioxide", 1.0, 80.0, 11.0)

    with col2:
        total_sulfur = st.number_input("Total Sulfur Dioxide", 6.0, 300.0, 34.0)
        density = st.number_input("Density", 0.9900, 1.0050, 0.9978)
        ph = st.number_input("pH", 2.5, 4.5, 3.51)
        sulphates = st.number_input("Sulphates", 0.3, 2.0, 0.56)
        alcohol = st.number_input("Alcohol", 8.0, 15.0, 9.4)

    st.markdown("</div>", unsafe_allow_html=True)

# -------- PREDICTION --------
if st.button("🚀 Predict Wine Quality"):

    with st.spinner("Analyzing Wine Quality..."):
        time.sleep(1.5)

    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid,
                            residual_sugar, chlorides, free_sulfur,
                            total_sulfur, density, ph,
                            sulphates, alcohol]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.markdown('<div class="fade-in">', unsafe_allow_html=True)

    st.metric("Predicted Quality Score", round(prediction, 2))

    if prediction >= 7:
        st.success("🌟 Excellent Quality Wine")
        st.balloons()
    elif prediction >= 5:
        st.warning("🍷 Good Quality Wine")
    else:
        st.error("⚠️ Low Quality Wine")

    st.markdown("</div>", unsafe_allow_html=True)