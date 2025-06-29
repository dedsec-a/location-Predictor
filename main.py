# main.py

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import uvicorn

# -------------------------------
# Load models at startup
# -------------------------------

with open("ontime_model.pkl", "rb") as f:
    ontime_bundle = pickle.load(f)

ontime_model = ontime_bundle['model']
ontime_encoders = ontime_bundle['encoders']

with open("store_model.pkl", "rb") as f:
    store_model = pickle.load(f)

# -------------------------------
# Init FastAPI
# -------------------------------

app = FastAPI()

# -------------------------------
# On-Time Prediction Endpoint
# -------------------------------

class OnTimeRequest(BaseModel):
    Warehouse_block: str
    Mode_of_Shipment: str
    Customer_care_calls: int
    Customer_rating: int
    Cost_of_the_Product: int
    Prior_purchases: int
    Product_importance: str
    Gender: str
    Discount_offered: int
    Weight_in_gms: int

@app.post("/predict_on_time")
def predict_on_time(data: OnTimeRequest):
    # Encode
    input_data = [
        ontime_encoders['Warehouse_block'].transform([data.Warehouse_block])[0],
        ontime_encoders['Mode_of_Shipment'].transform([data.Mode_of_Shipment])[0],
        data.Customer_care_calls,
        data.Customer_rating,
        data.Cost_of_the_Product,
        data.Prior_purchases,
        ontime_encoders['Product_importance'].transform([data.Product_importance])[0],
        ontime_encoders['Gender'].transform([data.Gender])[0],
        data.Discount_offered,
        data.Weight_in_gms
    ]
    prediction = ontime_model.predict([input_data])[0]
    result = "On Time" if prediction == 1 else "Late"
    return {"prediction": result}

# -------------------------------
# Store Location Suggestion Endpoint
# -------------------------------

class StoreRequest(BaseModel):
    population: int
    density: float
    land_area: float

@app.post("/suggest_store_location")
def suggest_store_location(data: StoreRequest):
    input_data = np.array([[data.population, data.density, data.land_area]])
    prediction = store_model.predict(input_data)[0]
    result = "High Potential for New Store" if prediction == 1 else "Low Potential"
    return {"recommendation": result}

# -------------------------------
# Run
# -------------------------------

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
