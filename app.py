import streamlit as st
import matplotlib.pyplot as plt
from soil_crop_model import (
    predict_soil_and_crop,
    get_accuracies,
    get_dataset
)

st.set_page_config(page_title="Soil & Crop Prediction", page_icon="🌱")

st.title("🌱 Soil & Crop Prediction System")

# Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📄 Dataset", "📊 Model Accuracy"])

# ---------------- TAB 1: Prediction ----------------
with tab1:
    st.subheader("Enter Input Values")

    moisture = st.number_input("Soil Moisture (%)", 0.0, 100.0)
    ph = st.number_input("Soil pH", 0.0, 14.0)
    temperature = st.number_input("Temperature (°C)", 0.0, 60.0)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0)

    if st.button("Predict"):
        soil, crop = predict_soil_and_crop(
            moisture, ph, temperature, humidity
        )

        st.success(f"🌍 Soil Type: {soil}")
        st.success(f"🌾 Suggested Crop: {crop}")

# ---------------- TAB 2: Dataset ----------------
with tab2:
    st.subheader("Training Dataset")
    data = get_dataset()
    st.dataframe(data)

# ---------------- TAB 3: Accuracy Chart ----------------
with tab3:
    st.subheader("Model Accuracy")

    soil_acc, crop_acc = get_accuracies()

    st.write(f"**Soil Prediction Accuracy:** {soil_acc * 100:.2f}%")
    st.write(f"**Crop Prediction Accuracy:** {crop_acc * 100:.2f}%")

    # Plot bar chart
    labels = ['Soil Model', 'Crop Model']
    accuracies = [soil_acc * 100, crop_acc * 100]

    fig = plt.figure()
    plt.bar(labels, accuracies)
    plt.ylim(0, 100)
    plt.ylabel("Accuracy (%)")
    plt.title("Model Accuracy Comparison")

    st.pyplot(fig)
