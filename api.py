from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

model = joblib.load('model/logistic_regression_model_final.pkl')
scaler = joblib.load('model/scaler_final.pkl')

VALID_WEATHER = ['rain', 'storm', 'clear','fog']
VALID_TRAFFIC = ['light', 'moderate', 'heavy']
VALID_VEHICLE = ['trailer', 'truck', 'container', 'lorry', 'unknown']
VALID_LOCATIONS = ['ahmedabad', 'bangalore', 'chennai', 'delhi', 'hyderabad', 'jaipur', 'kolkata', 'lucknow', 'mumbai', 'pune']

def one_hot_encode(column, value):
    return [1 if loc == value else 0 for loc in column]

@app.route('/predict', methods=['POST'])

def predict_delay():
    try:
        data = request.json
        origin = data.get('origin').lower()
        destination = data.get('destination').lower()
        shipment_date = data.get('shipment_date')
        planned_delivery_date = data.get('planned_delivery_date')
        vehicle_type = data.get('vehicle_type').lower()
        weather_conditions = data.get('weather_conditions').lower()
        traffic_conditions = data.get('traffic_conditions').lower()
        distance = data.get('distance')

        # Validate inputs
        if origin not in VALID_LOCATIONS:
            return jsonify({'error': f'Invalid origin. Valid values: {VALID_LOCATIONS}'}), 400
        if destination not in VALID_LOCATIONS:
            return jsonify({'error': f'Invalid destination. Valid values: {VALID_LOCATIONS}'}), 400
        if vehicle_type not in VALID_VEHICLE:
            return jsonify({'error': f'Invalid vehicle type. Valid values: {VALID_VEHICLE}'}), 400
        if weather_conditions not in VALID_WEATHER:
            return jsonify({'error': f'Invalid weather conditions. Valid values: {VALID_WEATHER}'}), 400
        if traffic_conditions not in VALID_TRAFFIC:
            return jsonify({'error': f'Invalid traffic conditions. Valid values: {VALID_TRAFFIC}'}), 400
        if not isinstance(distance, (int, float)) or distance <= 0:
            return jsonify({'error': 'Invalid distance. It must be a positive number.'}), 400
        distance = int(distance)  # Convert to integer

        # One-hot encode categorical inputs
        input_data = (
            one_hot_encode(VALID_LOCATIONS, origin)
            + one_hot_encode(VALID_LOCATIONS, destination)
            + one_hot_encode(VALID_WEATHER, weather_conditions)
            + one_hot_encode(VALID_TRAFFIC, traffic_conditions)
            + one_hot_encode(VALID_VEHICLE, vehicle_type)
        )


        shipment_date = pd.to_datetime(shipment_date).timestamp()
        planned_delivery_date = pd.to_datetime(planned_delivery_date).timestamp()
        scaled_dates = scaler.transform([[shipment_date, planned_delivery_date]])[0]
        input_data.extend(scaled_dates)

        input_data.append(distance)

        input_data = np.array(input_data).reshape(1, -1)
        prediction = model.predict(input_data)
        prediction_label = 'Yes' if prediction[0] == 1 else 'No'

        return jsonify({'delayed': prediction_label})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
