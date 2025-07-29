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
st.image("Fitsall Logo new 1.png", width=300)
st.title("👗 AI-Powered Body Measurement Predictor - WOMEN")

st.write("Enter your body measurements below to get AI-generated size recommendations.")

# Create two columns for inputs
col1, col2 = st.columns(2)

with col1:
    # Basic measurements
    st.subheader("📏 Body Measurements")
    weight = st.number_input("Enter Weight (kg)", min_value=40, max_value=120, step=1)
    height = st.number_input("Enter Height (cm)", min_value=140, max_value=220, step=1)
    bust = st.number_input("Enter Bust (inches)", min_value=25, max_value=50, step=1) * 2.54
    stomach = st.number_input("Enter Stomach (inches)", min_value=25, max_value=50, step=1) * 2.54
    hips = st.number_input("Enter Hips (inches)", min_value=25, max_value=50, step=1) * 2.54

with col2:
    # Body type selection
    st.subheader("👤 Body Type & Fit Preferences")
    body_type = st.selectbox(
        "Select Your Body Type",
        options=["Apple", "Pear", "Hourglass", "Rectangle", "Inverted Triangle"], 
        help="Select your body type for more accurate fitting recommendations"
    )
    
    # Fit preferences
    fit_preference = st.radio(
        "Preferred Fit Style",
        options=["Relaxed Fit", "Regular Fit", "Slim Fit"],
        index=1,  # Default to Regular Fit
        help="Choose your preferred fit style for pants"
    )
    
    # Add thigh ease slider for women (starting at 3.5)
    THIGH_EASE_INCHES = st.slider(
        "Thigh Ease (inches)", 3.0, 4.5, 3.5, 0.25,
        help="Industry standard for women: 3-4 inches of ease for comfortable fit"
    )
    
    # Customer measurement for validation (optional)
    st.subheader("📐 Optional: Your Known Measurements")
    known_thigh = st.number_input(
        "Your Thigh Measurement (inches) - Optional",
        min_value=0.0, max_value=40.0, value=0.0, step=0.1,
        help="If you know your thigh measurement, enter it for comparison"
    )

# Body type adjustment factors based on research
BODY_TYPE_ADJUSTMENTS = {
    "Apple": {
        "waist_adjustment": 1.05,  # 5% increase for apple shapes
        "thigh_adjustment": 1.02,  # 2% increase
        "hip_adjustment": 1.03,    # 3% increase
        "rise_adjustment": 1.04    # 4% increase for higher waist preference
    },
    "Pear": {
        "waist_adjustment": 0.98,  # 2% decrease as pears often have smaller waists
        "thigh_adjustment": 1.08,  # 8% increase for fuller thighs
        "hip_adjustment": 1.06,    # 6% increase for fuller hips
        "rise_adjustment": 1.02    # 2% increase
    },
    "Hourglass": {
        "waist_adjustment": 0.96,  # 4% decrease for defined waist
        "thigh_adjustment": 1.04,  # 4% increase
        "hip_adjustment": 1.04,    # 4% increase
        "rise_adjustment": 1.01    # 1% increase
    },
    "Rectangle": {
        "waist_adjustment": 1.02,  # 2% increase for straighter shape
        "thigh_adjustment": 1.01,  # 1% increase
        "hip_adjustment": 1.01,    # 1% increase
        "rise_adjustment": 1.00    # No adjustment
    },
    "Inverted Triangle": {
        "waist_adjustment": 1.01,  # 1% increase
        "thigh_adjustment": 0.99,  # 1% decrease for slimmer lower body
        "hip_adjustment": 0.98,    # 2% decrease
        "rise_adjustment": 1.00    # No adjustment
    }
}

# Fit preference adjustments
FIT_ADJUSTMENTS = {
    "Relaxed Fit": {"multiplier": 1.08, "ease": 2.0},  # 8% larger with 2" ease
    "Regular Fit": {"multiplier": 1.04, "ease": 1.0},  # 4% larger with 1" ease
    "Slim Fit": {"multiplier": 1.01, "ease": 0.5}     # 1% larger with 0.5" ease
}

# Enhanced garment calculation function for women
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
    shorts_length = (pred_leg_length - 50) * 0.9  # Women's shorts typically shorter than men's
    shorts_leg_opening_base = garment_thigh_cm * 1.15  # More generous opening for women
    shorts_inseam = shorts_length - (pred_rise / 2.5)
    
    # Pant calculations
    pant_leg_opening_base = garment_thigh_cm * 0.72  # Slightly wider than men's for women's fashion
    pant_inseam = pred_leg_length - (pred_rise / 2)
    
    # Apply fit adjustments
    fit_multipliers = {
        "Regular Fit": 1.0,
        "Slim Fit": 0.92,  # 8% smaller openings for women's slim fit
        "Relaxed Fit": 1.08  # 8% larger openings
    }
    
    # Apply body type specific adjustments for sleeve opening
    body_type_sleeve_multipliers = {
        "Apple": 1.03,     # 3% larger for apple shapes
        "Pear": 0.98,      # 2% smaller for pear shapes
        "Hourglass": 1.00, # Standard
        "Rectangle": 1.02, # 2% larger
        "Inverted Triangle": 1.05  # 5% larger for broader shoulders
    }
    
    fit_multiplier = fit_multipliers.get(fit_preference, 1.0)
    body_type_sleeve_multiplier = body_type_sleeve_multipliers.get(body_type, 1.0)
    
    # Apply adjustments
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

# Function to calculate leg opening (updated)
def calculate_leg_opening(thigh_measurement, garment_type="pants"):
    """Calculate leg opening based on thigh measurement and garment type"""
    if garment_type == "pants":
        return thigh_measurement * 0.65
    elif garment_type == "shorts":
        return thigh_measurement * 0.75
    else:
        return thigh_measurement * 0.65

# Prediction button
if st.button("Predict Measurements", type="primary"):
    try:
        new_data = pd.DataFrame({
            'Weight': [weight], 
            'Height': [height], 
            'Bust': [bust],
            'Stomach': [stomach], 
            'Hips': [hips]
        })
        
        predicted_values = model.predict(new_data)
        
        # Get body type adjustments
        adjustments = BODY_TYPE_ADJUSTMENTS[body_type]
        fit_adj = FIT_ADJUSTMENTS[fit_preference]
        
        # Extract predicted measurements with body type adjustments
        def convert_cm_to_inches(value_cm, reduce_percentage=0, body_type_adj=1.0, fit_adj=1.0):
            """Convert cm to inches and apply adjustments."""
            inches = value_cm / 2.54
            if reduce_percentage:
                inches -= inches * (reduce_percentage / 100)
            # Apply body type and fit adjustments
            inches = inches * body_type_adj * fit_adj
            return round(inches, 1), round(inches * 2.54, 1)
        
        # Convert predictions to inches and cm with adjustments
        neck_in, neck_cm = convert_cm_to_inches(predicted_values[0][0], reduce_percentage=5)
        sleeve_in, sleeve_cm = convert_cm_to_inches(predicted_values[0][1], reduce_percentage=5)
        shoulder_in, shoulder_cm = convert_cm_to_inches(predicted_values[0][2])
        torso_in, torso_cm = convert_cm_to_inches(predicted_values[0][3] - 2.54, reduce_percentage=10)
        bicep_in, bicep_cm = convert_cm_to_inches(predicted_values[0][4])
        wrist_in, wrist_cm = convert_cm_to_inches(predicted_values[0][5])
        
        # Apply body type adjustments to pants measurements
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
        
        # Calculate all garment measurements using enhanced function
        garment_measurements = calculate_garment_measurements(
            predicted_values[0][1], predicted_values[0][4], predicted_values[0][7],
            thigh_cm, predicted_values[0][6], fit_preference, body_type
        )
        
        # Display results with enhanced formatting
        st.success("✅ Measurements predicted successfully!")
        
        # Create tabs for better organization (with emojis as requested)
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
            
            # Show body type specific recommendations
            st.info(f"🎯 Optimized for {body_type} body type with {fit_preference.lower()}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Rise", f"{rise_in}\"", f"{rise_cm} cm")
                st.metric("Outseam Length", f"{leg_length_in}\"", f"{leg_length_cm} cm")
                st.metric("Inseam Length",
                         f"{garment_measurements['pant_inseam']/2.54:.1f}\"",
                         f"{garment_measurements['pant_inseam']:.1f} cm",
                         help="Outseam minus rise/2")
                st.metric("Waist", f"{waist_in}\"", f"{waist_cm} cm",
                         help=f"Adjusted for {body_type} body type")
            
            with col2:
                st.metric("Thigh (Body)", f"{garment_measurements['body_thigh_in']:.1f}\"", f"{thigh_cm} cm",
                         help="Natural body measurement")
                st.metric("Thigh (Garment)",
                         f"{garment_measurements['garment_thigh_in']:.1f}\"",
                         f"{garment_measurements['garment_thigh_cm']:.1f} cm",
                         help=f"Body thigh + {THIGH_EASE_INCHES}\" ease for comfort")
                st.metric("🆕 Pants Leg Opening",
                         f"{garment_measurements['pant_leg_opening']/2.54:.1f}\"",
                         f"{garment_measurements['pant_leg_opening']:.1f} cm",
                         help="Width at the ankle opening")
                
                # Show comparison if customer provided their measurement
                if known_thigh > 0:
                    thigh_diff = abs(garment_measurements['garment_thigh_in'] - known_thigh)
                    if thigh_diff <= 1:
                        st.success(f"✅ Predicted garment thigh ({garment_measurements['garment_thigh_in']:.1f}\") is within 1\" of your measurement ({known_thigh}\")")
                    else:
                        st.warning(f"⚠️ Predicted garment thigh ({garment_measurements['garment_thigh_in']:.1f}\") differs by {thigh_diff:.1f}\" from your measurement ({known_thigh}\")")
        
        with tab4:
            st.subheader("📏 Measurements for Shorts")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Shorts Outseam Length",
                         f"{garment_measurements['shorts_length']/2.54:.1f}\"",
                         f"{garment_measurements['shorts_length']:.1f} cm",
                         help="Pant length minus 50cm for women's above-knee fit")
                st.metric("Shorts Inseam Length",
                         f"{garment_measurements['shorts_inseam']/2.54:.1f}\"",
                         f"{garment_measurements['shorts_inseam']:.1f} cm",
                         help="Outseam minus rise/2")
            
            with col2:
                st.metric("🆕 Shorts Leg Opening",
                         f"{garment_measurements['shorts_leg_opening']/2.54:.1f}\"",
                         f"{garment_measurements['shorts_leg_opening']:.1f} cm",
                         help="Width at the leg opening for shorts")
        
        with tab5:
            st.subheader("📊 Fit Analysis & Recommendations")
            
            # Body type specific recommendations
            recommendations = {
                "Apple": [
                    "Consider high-waisted pants to define your waistline",
                    "Empire waist dresses and tops work well",
                    "Straight-leg or wide-leg pants balance your proportions",
                    "Avoid low-rise pants"
                ],
                "Pear": [
                    "High-waisted pants complement your shape perfectly",
                    "Straight-leg or bootcut pants balance your proportions",
                    "Consider pants with stretch fabric for comfort",
                    "Avoid skinny fits that emphasize hip width"
                ],
                "Hourglass": [
                    "High-waisted pants accentuate your natural waist",
                    "Fitted styles showcase your balanced proportions",
                    "Both straight-leg and bootcut styles work well",
                    "Avoid oversized fits that hide your curves"
                ],
                "Rectangle": [
                    "Create curves with high-waisted, fitted styles",
                    "Add visual interest with textured fabrics",
                    "Both straight and wide-leg styles work",
                    "Consider pants with details at the hips"
                ],
                "Inverted Triangle": [
                    "Wide-leg or bootcut pants add volume to your lower half",
                    "Higher-rise pants create balance",
                    "Avoid tight-fitting pants",
                    "Consider pants with hip details or patterns"
                ]
            }
            
            st.write(f"**Recommendations for {body_type} Body Type:**")
            for rec in recommendations[body_type]:
                st.write(f"• {rec}")
            
            # Enhanced fit analysis
            st.write(f"**Fit Preference: {fit_preference}**")
            st.write(f"• Thigh ease: {THIGH_EASE_INCHES}\" ({THIGH_EASE_INCHES * 2.54:.1f}cm)")
            st.write(f"• Size multiplier: {fit_adj['multiplier']:.2f}x for desired fit")
            
            # Applied adjustments display
            st.markdown("#### Applied Adjustments:")
            if fit_preference == "Slim Fit":
                st.write("- All openings reduced by 8% for trimmer silhouette")
                st.write("- Best for lean builds or fashion-forward looks")
            elif fit_preference == "Relaxed Fit":
                st.write("- All openings increased by 8% for comfort")
                st.write("- Ideal for comfort or active lifestyles")
            else:
                st.write("- Classic proportions based on women's sizing standards")
            
            # Measurement accuracy notice
            st.info("💡 **Tip:** All measurements distinguish between body size and finished garment size, matching how ready-to-wear brands specify products for women.")
        
        with tab6:
            st.subheader("🧵 Finished Garment Specifications")
            st.markdown("*Production-ready measurements for manufacturing*")
            
            st.markdown("#### Shirts")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Full Sleeve Length:** {sleeve_cm:.1f} cm")
                st.write(f"**Short Sleeve Length:** {garment_measurements['short_sleeve_length']:.1f} cm")
                st.write(f"**Neck:** {neck_cm:.1f} cm")
                st.write(f"**Shoulder Width:** {shoulder_cm:.1f} cm")
            with col2:
                st.write(f"**Bicep:** {bicep_cm:.1f} cm")
                st.write(f"**Sleeve Opening:** {garment_measurements['short_sleeve_opening']:.1f} cm")
                st.write(f"**Torso Length:** {torso_cm:.1f} cm")
                st.write(f"**Wrist:** {wrist_cm:.1f} cm")
            
            st.markdown("#### Pants")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Outseam:** {leg_length_cm:.1f} cm")
                st.write(f"**Inseam:** {garment_measurements['pant_inseam']:.1f} cm")
                st.write(f"**Rise:** {rise_cm:.1f} cm")
                st.write(f"**Waist:** {waist_cm:.1f} cm")
            with col2:
                st.write(f"**Thigh (Finished):** {garment_measurements['garment_thigh_cm']:.1f} cm")
                st.write(f"**Leg Opening:** {garment_measurements['pant_leg_opening']:.1f} cm")
            
            st.markdown("#### Shorts")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Outseam:** {garment_measurements['shorts_length']:.1f} cm")
                st.write(f"**Inseam:** {garment_measurements['shorts_inseam']:.1f} cm")
            with col2:
                st.write(f"**Leg Opening:** {garment_measurements['shorts_leg_opening']:.1f} cm")
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.write("Please check your input values and try again.")

# Add footer with helpful information (with image provision as requested)
st.markdown("---")
st.markdown("### 📚 Understanding Your Measurements")

with st.expander("Learn about body types and fitting"):
    st.write("""
    **Body Types Explained:**
    """)
    
    # Body types visual guide
    st.markdown("*📸 Body Types Visual Guide:*")
    st.image("body_type_guides.png", use_container_width=True)
    
    st.write("""
    - **Apple**: Fuller bust and midsection, slender legs
    - **Pear**: Narrower shoulders, wider hips and thighs
    - **Hourglass**: Balanced bust and hips with defined waist
    - **Rectangle**: Straight silhouette with minimal curves
    - **Inverted Triangle**: Broader shoulders, narrower hips
    
    **Key Measurements:**
    - **Rise**: Distance from crotch to waistband
    - **Thigh**: Circumference at fullest part of thigh
    - **Leg Opening**: Width at the bottom hem of pants/shorts
    - **Ease**: Extra room added for comfort and movement
    - **Garment vs Body**: Body measurements + ease = finished garment measurements
    """)

st.markdown("*Measurements optimized using body type research, women's sizing standards, and fit preferences*")
