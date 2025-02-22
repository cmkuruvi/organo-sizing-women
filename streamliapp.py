import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
file_path = "Sizing Spreadsheet - Women_ Height-Weight.csv"
df = pd.read_csv(file_path)

# Remove unwanted columns
df = df.dropna(axis=1, how="all")

# Define Features and Target Variables
X = df[['Weight', 'Height', 'Bust', 'Stomach', 'Hips']]
y = df[['Neck', 'Sleeve Length (F/S)', 'Shoulder Width', 'Torso Length', 'Bicep', 'Wrist',
        'Rise', 'Length (Leg)', 'Waist (Pants)', 'THigh']]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit UI
st.title("👗 AI-Powered Body Measurement Predictor")

st.write("Enter your body measurements below to get AI-generated size recommendations.")

# User Inputs
weight = st.number_input("Enter Weight (kg)", min_value=30.0, max_value=150.0, step=0.1)
height = st.number_input("Enter Height (cm)", min_value=130.0, max_value=220.0, step=0.1)
bust = st.number_input("Enter Bust (inches)", min_value=20.0, max_value=60.0, step=0.1) * 2.54
stomach = st.number_input("Enter Stomach (inches)", min_value=20.0, max_value=60.0, step=0.1) * 2.54
hips = st.number_input("Enter Hips (inches)", min_value=20.0, max_value=60.0, step=0.1) * 2.54

# Prediction button
if st.button("Predict Measurements"):
    new_data = pd.DataFrame({'Weight': [weight], 'Height': [height], 'Bust': [bust], 
                             'Stomach': [stomach], 'Hips': [hips]})
    
    predicted_values = model.predict(new_data)
    
    # Extract predicted measurements
    results = {
        "Neck": round(predicted_values[0][0], 1),
        "Sleeve Length": round(predicted_values[0][1], 1),
        "Shoulder Width": round(predicted_values[0][2], 1),
        "Torso Length": round(predicted_values[0][3], 1),
        "Bicep": round(predicted_values[0][4], 1),
        "Wrist": round(predicted_values[0][5], 1),
        "Rise": round(predicted_values[0][6], 1),
        "Leg Length": round(predicted_values[0][7], 1),
        "Waist (Pants)": round(predicted_values[0][8], 1),
        "Thigh": round(predicted_values[0][9], 1)
    }
    
    st.subheader("📏 Predicted Body Measurements (cm):")
    for key, value in results.items():
        st.write(f"**{key}:** {value} cm")
