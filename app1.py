import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("heart1.csv")

# Rename if needed
if 'output' in data.columns:
    data.rename(columns={'output': 'target'}, inplace=True)

# 🔥 Add new Indian-based columns (if not present)
new_columns = [
    'oil_intake', 'junk_food_freq', 'sweets_consumption',
    'physical_activity', 'stress_level', 'sleep_hours',
    'yoga', 'meditation', 'ayurveda_usage',
    'region', 'urban_rural',
    'diabetes', 'family_history',
    'tobacco_use', 'alcohol'
]

for col in new_columns:
    if col not in data.columns:
        data[col] = 0   # default value

# Split data
X = data.drop("target", axis=1)
y = data["target"]

# Train model
model = LogisticRegression(max_iter=2000)
model.fit(X, y)

# UI
st.title("🇮🇳 Indian Heart Health Monitoring System")
st.write("ML + Indian Lifestyle + Traditional Methods")

# 🔹 Basic medical inputs
age = st.slider("Age", 20, 80)
sex = st.selectbox("Sex", [0, 1])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.slider("BP", 80, 200)
chol = st.slider("Cholesterol", 100, 400)
fbs = st.selectbox("Fasting Sugar", [0, 1])
restecg = st.selectbox("ECG", [0, 1, 2])
thalach = st.slider("Max Heart Rate", 70, 210)
exang = st.selectbox("Exercise Angina", [0, 1])
oldpeak = st.slider("Oldpeak", 0.0, 6.0)
slope = st.selectbox("Slope", [0, 1, 2])
ca = st.selectbox("Vessels", [0, 1, 2, 3])
thal = st.selectbox("Thal", [0, 1, 2, 3])

# 🔥 Indian features inputs
oil_intake = st.selectbox("Oil Intake", [0,1,2])  # low, medium, high
junk_food_freq = st.slider("Junk Food / week", 0, 7)
sweets_consumption = st.slider("Sweets Intake", 0, 7)

physical_activity = st.selectbox("Physical Activity", [0,1,2])
stress_level = st.selectbox("Stress Level", [0,1,2])
sleep_hours = st.slider("Sleep Hours", 3, 10)

yoga = st.selectbox("Do Yoga?", [0,1])
meditation = st.selectbox("Meditation?", [0,1])
ayurveda_usage = st.selectbox("Use Ayurveda?", [0,1])

region = st.selectbox("Region", [0,1,2,3])
urban_rural = st.selectbox("Urban/Rural", [0,1])

diabetes = st.selectbox("Diabetes", [0,1])
family_history = st.selectbox("Family History", [0,1])
tobacco_use = st.selectbox("Tobacco Use", [0,1])
alcohol = st.selectbox("Alcohol", [0,1])

# Prediction
if st.button("Predict"):

    input_data = np.array([[
        age, sex, cp, trestbps, chol, fbs,
        restecg, thalach, exang, oldpeak,
        slope, ca, thal,
        oil_intake, junk_food_freq, sweets_consumption,
        physical_activity, stress_level, sleep_hours,
        yoga, meditation, ayurveda_usage,
        region, urban_rural,
        diabetes, family_history,
        tobacco_use, alcohol
    ]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ High Risk")

        st.subheader("🌿 Indian Traditional Suggestions")
        st.write("- Do Yoga & Pranayama daily")
        st.write("- Avoid oily & fried foods")
        st.write("- Include turmeric & garlic")
        st.write("- Try Siddha/Ayurveda (doctor advice)")
        st.write("- Walk daily")

    else:
        st.success("✅ Low Risk")
        st.write("Maintain healthy lifestyle 👍")