from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# ✅ Load saved objects
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

class InputData(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "Loan Default Prediction API Running"}

@app.post("/predict")
def predict(data: InputData):
    try:
        arr = np.array(data.features).reshape(1, -1)

        # ✅ Apply scaling (IMPORTANT)
        arr = scaler.transform(arr)

        prediction = model.predict(arr)

        return {"prediction": int(prediction[0])}

    except Exception as e:
        return {"error": str(e)}

