from flask import Flask, request, jsonify
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load once at startup
scaler = joblib.load('scaler.pkl')
model  = load_model('aqi_lstm_model.h5')

INPUT_STEPS = 6  # must match training

@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects JSON:
      { "data": [ [CO, PM2.5, AQI],  ...  ] }  # length must be 6
    Returns:
      { "forecast": [AQI_t+1, AQI_t+2, AQI_t+3, AQI_t+4] }
    """
    payload = request.get_json(force=True)
    data = np.array(payload.get('data', []), dtype=float)
    if data.shape != (INPUT_STEPS, 3):
        return jsonify(error=f"Expected data shape ({INPUT_STEPS},3), got {data.shape}"), 400

    # Scale & reshape
    scaled = scaler.transform(data)           # (6,3)
    X_new  = scaled.reshape((1, INPUT_STEPS, 3))

    # Predict and inverse-scale AQI channel
    y_scaled = model.predict(X_new).flatten()  # (4,)
    dummy    = np.zeros((4, 3))
    dummy[:,2] = y_scaled
    y_pred = scaler.inverse_transform(dummy)[:,2]

    return jsonify(forecast=y_pred.tolist())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
