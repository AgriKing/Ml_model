import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Load the saved model and label encoder
model_path = os.path.join(os.getcwd(), 'models', 'model.pkl')
le_path = os.path.join(os.getcwd(), 'models', 'label_encoder.pkl')

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(le_path, 'rb') as le_file:
    le = pickle.load(le_file)

# Function to calculate remaining months from expiry date
def calculate_remaining_months(expiry_date_str):
    expiry_date = datetime.strptime(expiry_date_str, '%Y-%m-%d')
    current_date = datetime.now()
    remaining_months = (expiry_date - current_date).days // 30
    return remaining_months

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    try:
        # Extract required fields from the request
        medicine_name = data['medicine_name']
        composition_mg = float(data['composition_mg'])
        quantity = int(data['quantity'])
        price = float(data['price'])
        remaining_quantity = int(data['remaining_quantity'])
        expiry_date_str = data['expiry_date']  # Expiry date in 'yyyy-mm-dd'
    except (KeyError, ValueError):
        return jsonify({'error': 'Invalid input data'}), 400

    # Convert medicine name to encoded form
    try:
        encoded_medicine = le.transform([medicine_name])[0]
    except ValueError:
        return jsonify({'error': 'Invalid medicine name'}), 400

    # Calculate remaining months based on expiry date
    remaining_months = calculate_remaining_months(expiry_date_str)

    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'Medicine Name Encoded': [encoded_medicine],
        'Composition Mg': [composition_mg],
        'Price': [price],
        'Quantity': [quantity],
        'Remaining Quantity': [remaining_quantity],
        'Remaining Months': [remaining_months]
    })

    # Make the prediction
    predicted_price = model.predict(input_data)

    # Return prediction as JSON
    return jsonify({'predicted_price': predicted_price[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
