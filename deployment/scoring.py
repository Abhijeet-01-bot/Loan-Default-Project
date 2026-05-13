import json
import joblib
import numpy as np

model = None

def init():
    global model
    model = joblib.load("model.pkl")

def run(raw_data):
    try:
        data = json.loads(raw_data)
        features = np.array(data["input_data"]["data"])

        prediction = model.predict(features)

        return {"prediction": prediction.tolist()}
    
    except Exception as e:
        return {"error": str(e)}