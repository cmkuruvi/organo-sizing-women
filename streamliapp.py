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

st.image("2.png", width=200)
st.title("👗 AI-Powered Body Measurement Predictor - WOMEN")

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
    def convert_cm_to_inches(value_cm, reduce_percentage=0):
        """Convert cm to inches and apply reduction percentage if necessary."""
        inches = value_cm / 2.54
        if reduce_percentage:
            inches -= inches * (reduce_percentage / 100)
        return round(inches, 1), round(value_cm, 1)

    # Convert predictions to inches and cm
    neck_in, neck_cm = convert_cm_to_inches(predicted_values[0][0], reduce_percentage=5)
    sleeve_in, sleeve_cm = convert_cm_to_inches(predicted_values[0][1], reduce_percentage=5)
    shoulder_in, shoulder_cm = convert_cm_to_inches(predicted_values[0][2])
    torso_in, torso_cm = convert_cm_to_inches(predicted_values[0][3] - 2.54, reduce_percentage=10)
    bicep_in, bicep_cm = convert_cm_to_inches(predicted_values[0][4])
    wrist_in, wrist_cm = convert_cm_to_inches(predicted_values[0][5])
    
    rise_in, rise_cm = convert_cm_to_inches(predicted_values[0][6])
    leg_length_in, leg_length_cm = convert_cm_to_inches(predicted_values[0][7] - 3.8)
    waist_in, waist_cm = convert_cm_to_inches(predicted_values[0][8])
    thigh_in, thigh_cm = convert_cm_to_inches(predicted_values[0][9])

    shorts_leg_length_in, shorts_leg_length_cm = convert_cm_to_inches(predicted_values[0][7] - 58)
    half_sleeve_in, half_sleeve_cm = convert_cm_to_inches(predicted_values[0][1] / 2.5)

    # Display results
    st.subheader("📏 Predicted Body Measurements:")
    
    st.markdown("### MEASUREMENT FOR FULLSLEEVE SHIRTS")
    st.write(f"**Predicted Neck Measure:** IN: {neck_in} | CM: {neck_cm}")
    st.write(f"**Predicted Sleeve Measure:** IN: {sleeve_in} | CM: {sleeve_cm}")
    st.write(f"**Predicted Shoulder Measure:** IN: {shoulder_in} | CM: {shoulder_cm}")
    st.write(f"**Predicted Torso Length Measure:** IN: {torso_in} | CM: {torso_cm}")
    st.write(f"**Predicted Bicep Measure:** IN: {bicep_in} | CM: {bicep_cm}")
    st.write(f"**Predicted Wrist Measure:** IN: {wrist_in} | CM: {wrist_cm}")

    st.markdown("### MEASUREMENT FOR PANTS")
    st.write(f"**Predicted Rise Measure:** IN: {rise_in} | CM: {rise_cm}")
    st.write(f"**Predicted Leg Length Measure:** IN: {leg_length_in} | CM: {leg_length_cm}")
    st.write(f"**Predicted Waist Measure:** IN: {waist_in} | CM: {waist_cm}")
    st.write(f"**Predicted Thighs Measure:** IN: {thigh_in} | CM: {thigh_cm}")

    st.markdown("### LENGTH MEASUREMENT OF SHORTS")
    st.write(f"**Predicted Shorts Leg Length Measure:** IN: {shorts_leg_length_in} | CM: {shorts_leg_length_cm}")
    st.write(f"**Predicted Half Sleeve Measure:** IN: {half_sleeve_in} | CM: {half_sleeve_cm}")
