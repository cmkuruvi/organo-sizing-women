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
        index=1, help="Choose your preferred fit style"
    )
    THIGH_EASE_INCHES = st.slider(
        "Thigh Ease (inches)", 3.0, 4.5, 3.5, 0.25,
        help="Industry standard for women: 3-4 inches of ease for comfortable fit"
    )
    st.subheader("📐 Optional: Your Known Measurements")
    known_thigh = st.number_input("Your Thigh Measurement (inches) - Optional", 
                                  min_value=0.0, max_value=40.0, value=0.0, step=0.1,
                                  help="If you know your thigh measurement, enter it for comparison")

BODY_TYPE_ADJUSTMENTS = {...}  # (same as your existing)

FIT_ADJUSTMENTS = {...}  # (same as your existing)

def calculate_garment_measurements(
    pred_sleeve, pred_bicep, pant_outseam_cm, thigh_cm, rise_cm, fit_preference, body_type,
    shorts_inseam_user=None
):
    short_sleeve_length = pred_sleeve * 0.42
    short_sleeve_opening_base = pred_bicep * 1.25
    body_thigh_in = thigh_cm / 2.54
    garment_thigh_in = body_thigh_in + THIGH_EASE_INCHES
    garment_thigh_cm = garment_thigh_in * 2.54

    pant_inseam_cm = pant_outseam_cm - (rise_cm / 2)
    front_rise_cm = rise_cm * 0.54

    # Shorts inseam (user or calculated, reduce by 20%)
    if shorts_inseam_user is not None:
        shorts_inseam_cm = shorts_inseam_user * 0.8
    else:
        shorts_inseam_cm = (pant_inseam_cm * 0.8)

    shorts_outseam_cm = shorts_inseam_cm + (0.5 * rise_cm)
    shorts_leg_opening_base = garment_thigh_cm * 1.15
    pant_leg_opening_base = garment_thigh_cm * 0.72

    fit_multipliers = {"Regular Fit": 1.0, "Slim Fit": 0.92, "Relaxed Fit": 1.08}
    body_type_sleeve_multipliers = {"Apple": 1.03, "Pear": 0.98, "Hourglass": 1.00, "Rectangle": 1.02, "Inverted Triangle": 1.05}
    fit_multiplier = fit_multipliers.get(fit_preference, 1.0)
    body_type_sleeve_multiplier = body_type_sleeve_multipliers.get(body_type, 1.0)

    short_sleeve_opening = short_sleeve_opening_base * fit_multiplier * body_type_sleeve_multiplier
    shorts_leg_opening = shorts_leg_opening_base * fit_multiplier
    pant_leg_opening = pant_leg_opening_base * fit_multiplier

    return {
        'short_sleeve_length': short_sleeve_length,
        'short_sleeve_opening': short_sleeve_opening,
        'shorts_length': shorts_outseam_cm,
        'shorts_leg_opening': shorts_leg_opening,
        'shorts_inseam': shorts_inseam_cm,
        'pant_leg_opening': pant_leg_opening,
        'pant_inseam': pant_inseam_cm,
        'garment_thigh_cm': garment_thigh_cm,
        'body_thigh_in': body_thigh_in,
        'garment_thigh_in': garment_thigh_in,
        'front_rise_cm': front_rise_cm
    }

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
            predicted_values[0][6], body_type_adj=adjustments["rise_adjustment"], fit_adj=fit_adj["multiplier"])
        leg_length_in_base, leg_length_cm_base = convert_cm_to_inches(predicted_values[0][7] - 3.8)

        # --- Edits here: Increase pant outseam by 10%
        pant_outseam_cm = leg_length_cm_base * 1.1
        pant_outseam_in = pant_outseam_cm / 2.54

        waist_in, waist_cm = convert_cm_to_inches(
            predicted_values[0][8], body_type_adj=adjustments["waist_adjustment"], fit_adj=fit_adj["multiplier"])
        thigh_in, thigh_cm = convert_cm_to_inches(
            predicted_values[0][9], body_type_adj=adjustments["thigh_adjustment"], fit_adj=fit_adj["multiplier"])

        # Front rise
        front_rise_cm = rise_cm * 0.54
        front_rise_in = front_rise_cm / 2.54

        # Pant inseam (based on boosted outseam)
        pant_inseam_cm = pant_outseam_cm - (rise_cm / 2)
        pant_inseam_in = pant_inseam_cm / 2.54

        # Shorts inseam (with 20% reduction), outseam
        shorts_inseam_cm = pant_inseam_cm * 0.8
        shorts_outseam_cm = shorts_inseam_cm + (0.5 * rise_cm)
        shorts_inseam_in = shorts_inseam_cm / 2.54
        shorts_outseam_in = shorts_outseam_cm / 2.54

        # Call revised calculation function for garment-level measurements
        garment_measurements = calculate_garment_measurements(
            predicted_values[0][1], predicted_values[0][4], pant_outseam_cm, thigh_cm, rise_cm, fit_preference, body_type,
            shorts_inseam_user=shorts_inseam_cm)

        st.success("✅ Measurements predicted successfully!")

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "👔 Full-Sleeve Shirts", "🌟 Short-Sleeve Shirts", "👖 Pants", 
            "🩳 Shorts", "📊 Fit Analysis", "🧵 Finished Garment Specs"
        ])

        with tab1:
            st.subheader("📏 Measurements for Full-Sleeve Shirts")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Neck", f"{neck_in}\"", f"{neck_cm} cm")
                st.metric("Sleeve Length", f"{sleeve_in}\"", f"{sleeve_cm} cm")
                st.metric("Shoulder Width", f"{shoulder_in}\"", f"{shoulder_cm} cm")
            with col2:
                st.metric("Torso Length", f"{torso_in}\"", f"{torso_cm} cm")
                st.metric("Bicep", f"{bicep_in}\"", f"{bicep_cm} cm")
                st.metric("Wrist", f"{wrist_in}\"", f"{wrist_cm} cm")

        with tab2:
            st.subheader("🌟 Measurements for Short-Sleeve Shirts")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Short Sleeve Length", 
                        f"{garment_measurements['short_sleeve_length']/2.54:.1f}\"", 
                        f"{garment_measurements['short_sleeve_length']:.1f} cm",
                        help="42% of full sleeve length")
            with col2:
                st.metric("Short Sleeve Opening", 
                        f"{garment_measurements['short_sleeve_opening']/2.54:.1f}\"", 
                        f"{garment_measurements['short_sleeve_opening']:.1f} cm",
                        help=f"Adjusted for {body_type} body type and {fit_preference}")

        with tab3:
            st.subheader("📏 Measurements for Pants")
            st.info(f"🎯 Optimized for {body_type} body type with {fit_preference.lower()}")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total (Back+Front) Rise", f"{rise_in}\"", f"{rise_cm} cm")
                st.metric("Front Rise", f"{front_rise_in:.1f}\"", f"{front_rise_cm:.1f} cm", help="~54% of total rise")
                st.metric("Outseam Length", f"{pant_outseam_in:.1f}\"", f"{pant_outseam_cm:.1f} cm")
                st.metric("Inseam Length", f"{pant_inseam_in:.1f}\"", f"{pant_inseam_cm:.1f} cm")
                st.metric("Waist", f"{waist_in}\"", f"{waist_cm} cm", help=f"Adjusted for {body_type} body type")
            with col2:
                st.metric("Thigh (Body)", f"{garment_measurements['body_thigh_in']:.1f}\"", f"{thigh_cm} cm", help="Natural body measurement")
                st.metric("Thigh (Garment)", f"{garment_measurements['garment_thigh_in']:.1f}\"", f"{garment_measurements['garment_thigh_cm']:.1f} cm", help=f"Body thigh + {THIGH_EASE_INCHES}\" ease")
                st.metric("Pants Leg Opening", f"{garment_measurements['pant_leg_opening']/2.54:.1f}\"", f"{garment_measurements['pant_leg_opening']:.1f} cm")

        with tab4:
            st.subheader("📏 Measurements for Shorts")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Shorts Outseam Length", f"{shorts_outseam_in:.1f}\"", f"{shorts_outseam_cm:.1f} cm", help="Pant inseam + 0.5 × rise")
                st.metric("Shorts Inseam Length", f"{shorts_inseam_in:.1f}\"", f"{shorts_inseam_cm:.1f} cm", help="(Pant inseam × 0.8) reduced by 20% for style")
            with col2:
                st.metric("Shorts Leg Opening", f"{garment_measurements['shorts_leg_opening']/2.54:.1f}\"", f"{garment_measurements['shorts_leg_opening']:.1f} cm")

        # ... (rest of your Fit Analysis and Finished Specs tabs unchanged)

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.write("Please check your input values and try again.")
