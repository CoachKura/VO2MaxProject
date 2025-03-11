from fastapi import FastAPI
import numpy as np
from sklearn.linear_model import LinearRegression

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Welcome to VO2 Max API"}

@app.post("/predict_vo2max")
def predict_vo2max(pace: float, age: int, weight: float):
    X_train = np.array([[4.5, 25, 70], [5.0, 30, 75], [4.0, 20, 65]])
    y_train = np.array([50, 45, 55])
    model = LinearRegression()
    model.fit(X_train, y_train)
    X_test = np.array([[pace, age, weight]])
    predicted_vo2max = model.predict(X_test)
    return {"vo2_max": round(predicted_vo2max[0], 2)}

