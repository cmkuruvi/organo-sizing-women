import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# Password Authentication
st.sidebar.header("🔒 Enter Password to Access Tool")
password = st.sidebar.text_input("Password", type="password")
VALID_PASSWORD = "ohdrog"
if password != VALID_PASSWORD:
    st.sidebar.warning("⚠️ Enter the correct password to proceed!")
    st.stop()

# Load dataset
file_path = "Sizing Spreadsheet - Women_ Height-Weight.csv"
df = pd.read_csv(file_path)
df = df.dropna(axis=1, how="all")

X = df[['Weight', 'Height', 'Bust', 'Stomach', 'Hips']]
y = df[['Neck', 'Sleeve Length (F/S)', 'Shoulder Width', 'Torso Length', 'Bicep', 'Wrist',
        'Rise', 'Length (Leg)', 'Waist (Pants)', 'THigh']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

st.image("Fitsall Logo new 1.png", width=300)
st.title("👗 AI-Powered Body Measurement Predictor - WOMEN")
st.write("Enter your body measurements below to get AI-generated size recommendations.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("📏 Body Measurements")
    weight = st.number_input("Enter Weight (kg)", min_value=40, max_value=120, step=1)
    height = st.number_input("Enter Height (cm)", min_value=140, max_value=220, step=1)
    bust = st.number_input("Enter Bust (inches)", min_value=25, max_value=50, step=1) * 2.54
    stomach = st.number_input("Enter Stomach (inches)", min_value=25, max_value=50, step=1) * 2.54
    hips = st.number_input("Enter Hips (inches)", min_value=25, max_value=50, step=1) * 2.54
with col2:
    st.subheader("👤 Body Type & Fit Preferences")
    body_type = st.selectbox(
        "Select Your Body Type",
        options=["Apple", "Pear", "Hourglass", "Rectangle", "Inverted Triangle"],
        help="Select your body type for more accurate fitting recommendations"
    )
    fit_preference = st.radio(
        "Preferred Fit Style",
        options=["Relaxed Fit", "Regular Fit", "Slim Fit"],
        index=1, help="Choose your preferred fit style for pants"
    )
    THIGH_EASE_INCHES = st.slider(
        "Thigh Ease (inches)", 3.0, 4.5, 3.5, 0.25,
        help="Industry standard for women: 3-4 inches of ease for comfortable fit"
    )
    st.subheader("📐 Optional: Your Known Measurements")
    known_thigh = st.number_input("Your Thigh Measurement (inches) - Optional",
        min_value=0.0, max_value=40.0, value=0.0, step=0.1,
        help="If you know your thigh measurement, enter it for comparison"
    )

BODY_TYPE_ADJUSTMENTS = {
    "Apple": {"waist_adjustment": 1.05, "thigh_adjustment": 1.02, "hip_adjustment": 1.03, "rise_adjustment": 1.04},
    "Pear": {"waist_adjustment": 0.98, "thigh_adjustment": 1.08, "hip_adjustment": 1.06, "rise_adjustment": 1.02},
    "Hourglass": {"waist_adjustment": 0.96, "thigh_adjustment": 1.04, "hip_adjustment": 1.04, "rise_adjustment": 1.01},
    "Rectangle": {"waist_adjustment": 1.02, "thigh_adjustment": 1.01, "hip_adjustment": 1.01, "rise_adjustment": 1.00},
    "Inverted Triangle": {"waist_adjustment": 1.01, "thigh_adjustment": 0.99, "hip_adjustment": 0.98, "rise_adjustment": 1.00}
}

FIT_ADJUSTMENTS = {
    "Relaxed Fit": {"multiplier": 1.08, "ease": 2.0},
    "Regular Fit": {"multiplier": 1.04, "ease": 1.0},
    "Slim Fit": {"multiplier": 1.01, "ease": 0.5}
}

def calculate_garment_measurements(pred_sleeve, pred_bicep, pred_leg_length, body_thigh, pred_rise, fit_preference, body_type):
    """Calculate all derived garment measurements using validated formulas for women"""

    # Base formulas adapted for women (similar ratios as men but adjusted for women's proportions)
    short_sleeve_length = pred_sleeve * 0.42  # Slightly longer ratio for women
    short_sleeve_opening_base = pred_bicep * 1.25  # Slightly more generous for women's fashion

    # Add thigh ease to get garment thigh
    body_thigh_in = body_thigh / 2.54
    garment_thigh_in = body_thigh_in + THIGH_EASE_INCHES
    garment_thigh_cm = garment_thigh_in * 2.54

    # Shorts calculations (adjusted for women's proportions - typically shorter)
    shorts_length = (pred_leg_length - 50) * 0.8  # Women's shorts typically shorter than men's, now scale by 0.8
    shorts_leg_opening_base = garment_thigh_cm * 1.15
    shorts_inseam = shorts_length - (pred_rise / 2)

    # Pant calculations
    pant_leg_opening_base = garment_thigh_cm * 0.72
    pant_inseam = pred_leg_length - (pred_rise / 2)

    # Apply fit adjustments
    fit_multipliers = {
        "Regular Fit": 1.0,
        "Slim Fit": 0.92,
        "Relaxed Fit": 1.08
    }

    body_type_sleeve_multipliers = {
        "Apple": 1.03,
        "Pear": 0.98,
        "Hourglass": 1.00,
        "Rectangle": 1.02,
        "Inverted Triangle": 1.05
    }

    fit_multiplier = fit_multipliers.get(fit_preference, 1.0)
    body_type_sleeve_multiplier = body_type_sleeve_multipliers.get(body_type, 1.0)

    short_sleeve_opening = short_sleeve_opening_base * fit_multiplier * body_type_sleeve_multiplier
    shorts_leg_opening = shorts_leg_opening_base * fit_multiplier
    pant_leg_opening = pant_leg_opening_base * fit_multiplier

    return {
        'short_sleeve_length': short_sleeve_length,
        'short_sleeve_opening': short_sleeve_opening,
        'shorts_length': shorts_length,
        'shorts_leg_opening': shorts_leg_opening,
        'shorts_inseam': shorts_inseam,
        'pant_leg_opening': pant_leg_opening,
        'pant_inseam': pant_inseam,
        'garment_thigh_cm': garment_thigh_cm,
        'body_thigh_in': body_thigh_in,
        'garment_thigh_in': garment_thigh_in
    }

def calculate_leg_opening(thigh_measurement, garment_type="pants"):
    """Calculate leg opening based on thigh measurement and garment type"""
    if garment_type == "pants":
        return thigh_measurement * 0.65
    elif garment_type == "shorts":
        return thigh_measurement * 0.75
    else:
        return thigh_measurement * 0.65

if st.button("Predict Measurements", type="primary"):
    try:
        new_data = pd.DataFrame({'Weight': [weight], 'Height': [height], 'Bust': [bust],
                                 'Stomach': [stomach], 'Hips': [hips]})
        predicted_values = model.predict(new_data)

        adjustments = BODY_TYPE_ADJUSTMENTS[body_type]
        fit_adj = FIT_ADJUSTMENTS[fit_preference]

        def convert_cm_to_inches(value_cm, reduce_percentage=0, body_type_adj=1.0, fit_adj=1.0):
            inches = value_cm / 2.54
            if reduce_percentage:
                inches -= inches * (reduce_percentage / 100)
            inches = inches * body_type_adj * fit_adj
            return round(inches, 1), round(inches * 2.54, 1)

        neck_in, neck_cm = convert_cm_to_inches(predicted_values[0][0], reduce_percentage=5)
        sleeve_in, sleeve_cm = convert_cm_to_inches(predicted_values[0][1], reduce_percentage=5)
        shoulder_in, shoulder_cm = convert_cm_to_inches(predicted_values[0][2])
        torso_in, torso_cm = convert_cm_to_inches(predicted_values[0][3] - 2.54, reduce_percentage=10)
        bicep_in, bicep_cm = convert_cm_to_inches(predicted_values[0][4])
        wrist_in, wrist_cm = convert_cm_to_inches(predicted_values[0][5])

        rise_in, rise_cm = convert_cm_to_inches(
            predicted_values[0][6],
            body_type_adj=adjustments["rise_adjustment"],
            fit_adj=fit_adj["multiplier"]
        )
        leg_length_in, leg_length_cm = convert_cm_to_inches(predicted_values[0][7] - 3.8)
        waist_in, waist_cm = convert_cm_to_inches(
            predicted_values[0][8],
            body_type_adj=adjustments["waist_adjustment"],
            fit_adj=fit_adj["multiplier"]
        )
        thigh_in, thigh_cm = convert_cm_to_inches(
            predicted_values[0][9],
            body_type_adj=adjustments["thigh_adjustment"],
            fit_adj=fit_adj["multiplier"]
        )

        garment_measurements = calculate_garment_measurements(
            predicted_values[0][1], predicted_values[0][4], predicted_values[0][7],
            thigh_cm, predicted_values[0][6], fit_preference, body_type
        )

        st.success("✅ Measurements predicted successfully!")

        # -- rest of your tabbed display code follows as before; blocks kept for brevity --

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.write("Please check your input values and try again.")

# (Footer remains unchanged)
