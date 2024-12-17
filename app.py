import pandas as pd
from flask import Flask, request, jsonify
from joblib import dump, load
from datetime import datetime
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Load dataset and preprocess
df = pd.read_csv("C:\\Users\\HARSH PATIL\\OneDrive\\New Ml\\Ml_project\\Final Dataset.csv")

# Initialize LabelEncoder for Medicine Names
le = LabelEncoder()
df['Medicine Name Encoded'] = le.fit_transform(df['Medicine Name'])

# Features and target variable
X = df[['Medicine Name Encoded', 'Composition Mg', 'Quantity', 'Remaining Quantity', 'Price']]
y = df['Selling Prices']

# Train the model using RandomForestRegressor
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# Save the model and label encoder
model_path = os.path.join(os.getcwd(), 'model.joblib')
le_path = os.path.join(os.getcwd(), 'label_encoder.joblib')
dump(model, model_path)
dump(le, le_path)

# Function to calculate remaining months from expiry date
def calculate_remaining_months(expiry_date_str):
    expiry_date = datetime.strptime(expiry_date_str, '%Y-%m-%d')
    current_date = datetime.now()
    remaining_months = (expiry_date - current_date).days // 30
    return remaining_months

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    try:
        # Extract the required fields from the request
        medicine_name = data['medicine_name']
        composition_mg = float(data['composition_mg'])
        quantity = int(data['quantity'])
        price = float(data['price'])
        remaining_quantity = int(data['remaining_quantity'])
        expiry_date_str = data['expiry_date']  # Expiry date in format 'dd-mm-yyyy'
    except (KeyError, ValueError):
        return jsonify({'error': 'Invalid input data'}), 400

    # Load the saved label encoder and model
    model = load(model_path)
    le = load(le_path)

    # Convert medicine name to encoded form
    try:
        encoded_medicine = le.transform([medicine_name])[0]
    except ValueError:
        return jsonify({'error': 'Invalid medicine name'}), 400

    # Calculate remaining months based on expiry date
    remaining_months = calculate_remaining_months(expiry_date_str)

    # Prepare the input data for prediction
    input_data = pd.DataFrame({
        'Medicine Name Encoded': [encoded_medicine],
        'Composition Mg': [composition_mg],
        'Quantity': [quantity],
        'Remaining Quantity': [remaining_quantity],
        'Price': [price],
        'Remaining Months': [remaining_months]
    })

    # Make the prediction
    predicted_price = model.predict(input_data)

    # Return the prediction as JSON
    return jsonify({'predicted_price': predicted_price[0]})

# Update the app to listen on all interfaces (for external access)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
