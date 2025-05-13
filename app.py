import logging, json, numpy as np, joblib
from flask import Flask, request, jsonify

# Patch LSTM so load_model will ignore time_major kwarg
from tensorflow.keras.layers import LSTM as _LSTM
class LSTM(_LSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None)
        super().__init__(*args, **kwargs)

from tensorflow.keras.models import load_model

# Load once
scaler = joblib.load('scaler.pkl')
model  = load_model('aqi_lstm_model.h5', custom_objects={'LSTM': LSTM})

app = Flask(__name__)
INPUT_STEPS = 6

@app.route('/predict', methods=['POST'])
def predict():
    try:
        body = request.get_json(force=True)
        data = np.array(body['data'], dtype=float)
        if data.shape != (INPUT_STEPS, 3):
            raise ValueError(f"Expected shape ({INPUT_STEPS},3), got {data.shape}")
    except Exception as e:
        return jsonify(error=str(e)), 400

    # Scale & reshape
    scaled = scaler.transform(data)                # (6,3)
    X_new  = scaled.reshape((1, INPUT_STEPS, 3))   # (1,6,3)

    # Predict & inverse-scale only AQI channel
    y_scaled = model.predict(X_new).flatten()      # (4,)
    dummy    = np.zeros((4, 3))
    dummy[:,2] = y_scaled
    y_pred = scaler.inverse_transform(dummy)[:,2]

    return jsonify(forecast=y_pred.tolist())

if __name__ == '__main__':
    # for local testing
    app.run(host='0.0.0.0', port=8000)
