import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# Password Authentication
st.sidebar.header("🔒 Enter Password to Access Tool")

password = st.sidebar.text_input("Password", type="password")

# Define your valid password (Change this to your desired password)
VALID_PASSWORD = "ohdrog"

# Check password before allowing access
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
    weight = st.number_input("Enter Weight (kg)", min_value=40, max_value=120, step=1, help="TEST: Need this for estimating body proportion distribution")
    height = st.number_input("Enter Height (cm)", min_value=140, max_value=220, step=1, help="TEST: ")
    bust = st.number_input("Enter Bust (inches)", min_value=25, max_value=50, step=1, help="TEST: ") * 2.54
    stomach = st.number_input("Enter Stomach (inches)", min_value=25, max_value=50, step=1,help="TEST: ") * 2.54
    hips = st.number_input("Enter Hips (inches)", min_value=25, max_value=50, step=1, help="TEST: ") * 2.54

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
    
    # Customer measurement for validation (optional)
    st.subheader("📐 Optional: Your Known Measurements")
    known_thigh = st.number_input("Your Thigh Measurement (inches) - Optional", 
                                  min_value=0.0, max_value=40.0, value=0.0, step=0.1,
                                  help="If you know your thigh measurement, enter it for comparison")

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

# Function to calculate leg opening
def calculate_leg_opening(thigh_measurement, garment_type="pants"):
    """Calculate leg opening based on thigh measurement and garment type"""
    if garment_type == "pants":
        # For pants, leg opening is typically 60-70% of thigh measurement
        return thigh_measurement * 0.65
    elif garment_type == "shorts":
        # For shorts, leg opening is typically 70-80% of thigh measurement
        return thigh_measurement * 0.75
    else:
        return thigh_measurement * 0.65

# Prediction button
if st.button("Predict Measurements", type="primary"):
    new_data = pd.DataFrame({'Weight': [weight], 'Height': [height], 'Bust': [bust], 
                             'Stomach': [stomach], 'Hips': [hips]})
    
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

    # Calculate leg openings
    pants_leg_opening_in = calculate_leg_opening(thigh_in, "pants")
    shorts_leg_opening_in = calculate_leg_opening(thigh_in, "shorts")
    
    shorts_leg_length_in, shorts_leg_length_cm = convert_cm_to_inches(predicted_values[0][7] - 58)
    half_sleeve_in, half_sleeve_cm = convert_cm_to_inches(predicted_values[0][1] / 2.5)

    # Display results with enhanced formatting
    st.success("✅ Measurements predicted successfully!")
    
    # Create tabs for better organization
    tab1, tab2, tab3, tab4 = st.tabs(["👔 Shirts", "👖 Pants", "🩳 Shorts", "📊 Fit Analysis"])
    
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
            st.metric("Half Sleeve", f"{half_sleeve_in}\"", f"{half_sleeve_cm} cm")

    with tab2:
        st.subheader("📏 Measurements for Pants")
        
        # Show body type specific recommendations
        st.info(f"🎯 Optimized for {body_type} body type with {fit_preference.lower()}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Rise", f"{rise_in}\"", f"{rise_cm} cm")
            st.metric("Leg Length", f"{leg_length_in}\"", f"{leg_length_cm} cm")
            st.metric("Waist", f"{waist_in}\"", f"{waist_cm} cm", 
                     help=f"Adjusted for {body_type} body type")
        
        with col2:
            st.metric("Thigh", f"{thigh_in}\"", f"{thigh_cm} cm",
                     help=f"Adjusted for {body_type} body type")
            st.metric("🆕 Pants Leg Opening", f"{pants_leg_opening_in:.1f}\"", 
                     f"{pants_leg_opening_in * 2.54:.1f} cm",
                     help="Width at the ankle opening")
            
            # Show comparison if customer provided their measurement
            if known_thigh > 0:
                difference = abs(thigh_in - known_thigh)
                if difference <= 1:
                    st.success(f"✅ Predicted thigh ({thigh_in}\") is within 1\" of your measurement ({known_thigh}\")")
                else:
                    st.warning(f"⚠️ Predicted thigh ({thigh_in}\") differs by {difference:.1f}\" from your measurement ({known_thigh}\")")

    with tab3:
        st.subheader("📏 Measurements for Shorts")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Shorts Leg Length", f"{shorts_leg_length_in}\"", f"{shorts_leg_length_cm} cm")
        
        with col2:
            st.metric("🆕 Shorts Leg Opening", f"{shorts_leg_opening_in:.1f}\"", 
                     f"{shorts_leg_opening_in * 2.54:.1f} cm",
                     help="Width at the leg opening for shorts")

    with tab4:
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
                "Create curves with high-waised, fitted styles",
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
        
        # Show ease recommendations
        st.write(f"**Fit Preference: {fit_preference}**")
        st.write(f"• Ease allowance: {fit_adj['ease']}\" for comfortable fit")
        st.write(f"• Size multiplier: {fit_adj['multiplier']:.2f}x for desired fit")
        
        # Measurement accuracy notice
        st.info("💡 **Tip:** If measurements seem off, consider taking your measurements while seated for thigh circumference, as this increases when sitting.")

# Add footer with helpful information
st.markdown("---")
st.markdown("### 📚 Understanding Your Measurements")

with st.expander("Learn about body types and fitting"):
    st.write("""
    **Body Types Explained:**
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
    """)

st.markdown("*Measurements optimized using body type research and fit preferences*")

