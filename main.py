from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def home():
    return {"message": "ML Model API Running Successfully 🚀"}

@app.get("/predict")
def predict(value: float):
    prediction = model.predict(np.array([[value]]))
    return {"prediction": prediction[0]}
