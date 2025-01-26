from smolmodels import Model

# Create a health risk assessment model with just a description
model = Model(
    intent="Predict the risk of a specific health condition based on user health data",
    input_schema={
        "age": int,
        "bmi": float,
        "smoking_status": str,     # e.g., "non-smoker", "light smoker"
        "physical_activity_level": str,  # e.g., "low", "moderate", "high"
        "family_medical_history": dict   # e.g., {"diabetes": True, "heart_disease": False}
    },
    output_schema={
        "risk_level": str,         # e.g., "low", "medium", "high"
        "recommended_action": str  # e.g., "schedule a check-up", "increase activity"
    }
)

# Build the model with your data
model.build("health.csv")

# Make predictions
risk_assessment = model.predict({
    "age": 45,
    "bmi": 28.5,
    "smoking_status": "light smoker",
    "physical_activity_level": "low",
    "family_medical_history": {"diabetes": True, "heart_disease": False}
})
print(risk_assessment)