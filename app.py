
import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import io
import pandas as pd

# -----------------------------
# 🎨 PAGE CONFIG & CSS
# -----------------------------
st.set_page_config(page_title="Energy Predictor", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #f5f7fa;
}
h1, h2, h3 {
    color: #2c3e50;
}
button[kind="primary"] {
    background-color: #4CAF50 !important;
    color: white !important;
    border-radius: 10px !important;
    height: 50px !important;
    width: 100% !important;
    font-size: 16px !important;
}
.stContainer {
    padding: 20px;
}
</style>
""", unsafe_allow_html=True)

st.title("⚡ TEAM TECH-STAR")
st.subheader("🏠 Household Energy Consumption Prediction")

# -----------------------------
# 📦 LOAD MODEL
# -----------------------------
model = joblib.load('linear_energy_model.pkl')
scaler = joblib.load('linear_scaler.pkl')

# -----------------------------
# 📊 INPUT SECTION
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    Global_reactive_power = st.number_input("Reactive Power (kW)", 0.0, 10.0, 0.418)
    Voltage = st.number_input("Voltage (V)", 200.0, 250.0, 234.84)
    Global_intensity = st.number_input("Intensity (A)", 0.0, 50.0, 18.4)
    Sub_metering_1 = st.number_input("Sub Metering 1", 0.0, 30.0, 0.0)
    Sub_metering_2 = st.number_input("Sub Metering 2", 0.0, 30.0, 1.0)
    Sub_metering_3 = st.number_input("Sub Metering 3", 0.0, 30.0, 17.0)

with col2:
    Hour = st.number_input("Hour", 0, 23, 18)
    Day = st.number_input("Day", 1, 31, 24)
    Month = st.number_input("Month", 1, 12, 3)
    DayOfWeek = st.number_input("Day Of Week", 0, 6, 1)
    Lag_1h = st.number_input("Lag 1 Hour", 0.0, 20.0, 1.2)
    Lag_2h = st.number_input("Lag 2 Hour", 0.0, 20.0, 1.1)

# -----------------------------
# 🧮 PREPARE INPUT
# -----------------------------
input_data = pd.DataFrame([[ 
    Global_reactive_power, Voltage, Global_intensity,
    Sub_metering_1, Sub_metering_2, Sub_metering_3,
    Hour, Day, Month, DayOfWeek,
    Lag_1h, Lag_2h
]], columns=[
    "Global_reactive_power", "Voltage", "Global_intensity",
    "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
    "Hour", "Day", "Month", "DayOfWeek",
    "Lag_1h", "Lag_2h"
])

# -----------------------------
# 📄 PDF GENERATOR
# -----------------------------
def generate_pdf(prediction, input_data):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("Energy Prediction Report", styles['Title']))
    content.append(Paragraph(f"Predicted Energy: {prediction:.3f} kW", styles['Normal']))

    for f, v in zip(input_data.columns, input_data.iloc[0]):
        content.append(Paragraph(f"{f}: {v}", styles['Normal']))

    doc.build(content)
    buffer.seek(0)
    return buffer

# -----------------------------
# 🔮 PREDICTION
# -----------------------------
if st.button("🚀 Predict Energy Consumption"):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.success(f"🔋 Predicted Energy: {prediction:.3f} kW")

    # 📥 DOWNLOAD PDF
    pdf = generate_pdf(prediction, input_data)
    st.download_button(
        label="📄 Download Report",
        data=pdf,
        file_name="energy_report.pdf",
        mime="application/pdf"
    )

    # -----------------------------
    # 📊 VISUALIZATIONS
    # -----------------------------
    fig1, ax1 = plt.subplots()
    ax1.barh(input_data.columns, input_data.iloc[0])
    st.subheader("📊 Input Features Overview")
    st.pyplot(fig1)

    # Graph 2: Hour vs Energy
    hours = list(range(24))
    predictions_hour = []
    for h in hours:
        temp = input_data.copy()
        temp["Hour"] = h
        predictions_hour.append(model.predict(scaler.transform(temp))[0])

    fig2, ax2 = plt.subplots()
    ax2.plot(hours, predictions_hour, marker='o')
    ax2.set_xlabel("Hour")
    ax2.set_ylabel("Energy (kW)")
    st.subheader("📉 Energy Consumption by Hour")
    st.pyplot(fig2)

    # Graph 3: Lag vs Prediction
    lag_values = [Lag_2h, Lag_1h, prediction]
    labels = ["Lag 2h", "Lag 1h", "Predicted"]
    fig3, ax3 = plt.subplots()
    ax3.plot(labels, lag_values, marker='o')
    st.subheader("📈 Lag vs Prediction")
    st.pyplot(fig3)
