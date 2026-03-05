from fastapi import FastAPI
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
heart_model = joblib.load("../backend/models/heart_model.pkl")
diabetes_model = joblib.load("../backend/models/diabetes_model.pkl")


# -----------------------------
# Hospital Data (Sample)
# -----------------------------
hospital_data = {
    "Delhi": [
        {"name": "AIIMS Delhi", "price": 40000},
        {"name": "Fortis Hospital", "price": 70000},
        {"name": "Apollo Hospital", "price": 65000}
    ],
    "Maharashtra": [
        {"name": "Tata Memorial", "price": 35000},
        {"name": "Kokilaben Hospital", "price": 75000}
    ],
    "Karnataka": [
        {"name": "Manipal Hospital", "price": 60000},
        {"name": "Narayana Health", "price": 50000}
    ]
}


# -----------------------------
# Government Schemes
# -----------------------------
scheme_data = {
    "Delhi": [
        "Ayushman Bharat - ₹5 Lakh Coverage",
        "Delhi Arogya Kosh - Free treatment for poor"
    ],
    "Maharashtra": [
        "Mahatma Jyotiba Phule Scheme",
        "Ayushman Bharat"
    ],
    "Karnataka": [
        "Arogya Karnataka",
        "Ayushman Bharat"
    ]
}


# Preventive measures
def preventive_measures(heart_risk, diabetes_risk):

    measures = []

    if heart_risk > 0.5:
        measures += [
            "Reduce salt intake",
            "Exercise regularly",
            "Monitor blood pressure"
        ]

    if diabetes_risk > 0.5:
        measures += [
            "Reduce sugar consumption",
            "Maintain healthy weight",
            "Regular blood sugar check"
        ]

    if not measures:
        measures.append("Maintain healthy lifestyle")

    return measures


# Diet
def diet_plan(heart_risk, diabetes_risk):

    if heart_risk > 0.5:
        return [
            "Oats and fruits breakfast",
            "Green vegetables",
            "Avoid fried foods"
        ]

    if diabetes_risk > 0.5:
        return [
            "Low sugar diet",
            "Whole wheat foods",
            "Leafy vegetables"
        ]

    return [
        "Balanced diet",
        "Fruits and vegetables",
        "Drink plenty of water"
    ]


@app.get("/")
def home():
    return {"message": "AI Health Predictor API Running"}


@app.post("/predict/all")
def predict(data: dict):

    state = data["state"]

    heart_input = np.array([[
        data["age"], data["sex"], data["cp"], data["trestbps"],
        data["chol"], data["fbs"], data["restecg"], data["thalach"],
        data["exang"], data["oldpeak"], data["slope"], data["ca"], data["thal"]
    ]])

    heart_prob = heart_model.predict_proba(heart_input)[0][1]

    diabetes_input = np.array([[
        data["pregnancies"], data["glucose"], data["bloodpressure"],
        data["skinthickness"], data["insulin"], data["bmi"],
        data["dpf"], data["age"]
    ]])

    diabetes_prob = diabetes_model.predict_proba(diabetes_input)[0][1]

    measures = preventive_measures(heart_prob, diabetes_prob)
    diet = diet_plan(heart_prob, diabetes_prob)

    hospitals = sorted(hospital_data.get(state, []), key=lambda x: x["price"])
    schemes = scheme_data.get(state, [])

    return {
        "heart_risk": round(float(heart_prob * 100), 2),
        "diabetes_risk": round(float(diabetes_prob * 100), 2),
        "preventive_measures": measures,
        "diet_plan": diet,
        "hospitals": hospitals,
        "schemes": schemes
    }
