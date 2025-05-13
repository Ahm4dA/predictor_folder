from flask import Flask, request, jsonify
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

scaler = joblib.load('scaler.pkl')
model  = load_model('aqi_lstm_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json(force=True)
    data    = np.array(payload['data'], dtype=float)
    # assume shape (12,3)
    scaled = scaler.transform(data)
    X_new  = scaled.reshape((1,12,3))
    y_scaled = model.predict(X_new).flatten()
    dummy    = np.zeros((len(y_scaled), scaled.shape[1]))
    dummy[:,2] = y_scaled
    y_pred = scaler.inverse_transform(dummy)[:,2]
    return jsonify(forecast=y_pred.tolist())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
