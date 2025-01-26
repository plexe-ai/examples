from smolmodels import Model

# Create a fraud detection model with just a description
model = Model(
    intent="Detect fraudulent transactions in real-time",
    input_schema={
        "transaction_id": str,
        "amount": float,
        "merchant_category": str,  # e.g., "electronics", "grocery"
        "timestamp": str,          # ISO-8601 datetime
        "user_behavior_score": float
    },
    output_schema={
        "is_fraud": bool,  # True if fraudulent, False otherwise
        "fraud_score": float
    }
)

# Build the model with your data + optionally generate additional synthetic training data
model.build("transactions.csv", generate_samples=1000)

# Make predictions
fraud_detection = model.predict({
    "transaction_id": "TX12345",
    "amount": 1500.0,
    "merchant_category": "electronics",
    "timestamp": "2025-01-26T10:15:00Z",
    "user_behavior_score": 0.75
})
print(fraud_detection)