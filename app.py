from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Initialize Flask app
app = Flask(__name__)

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

# API Route for Prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input
        data = request.json

        # Validate input
        required_fields = ['Weight', 'Height', 'Bust', 'Stomach', 'Hips']
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required fields"}), 400

        # Convert input data into DataFrame
        input_data = pd.DataFrame([data])

        # Make predictions
        predicted_values = model.predict(input_data)

        # Prepare response
        response = {
            "Predicted Measurements (CM)": {
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
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
