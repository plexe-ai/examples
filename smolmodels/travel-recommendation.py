from smolmodels import Model

# Create a travel recommendation model with just a description
model = Model(
    intent="Recommend travel destinations based on user preferences and budget",
    input_schema={
    "preferred_climate": str,   # e.g., "warm", "cold"
    "budget": float,    # maximum budget in USD
    "travel_month": str,    # e.g., "December"
    "travel_type": str  # e.g., "solo", "family", "adventure"
    },
    output_schema={
    "recommended_destinations": list    # e.g., ["Bali", "Swiss Alps"]
    }
)

# Input any that you have
model.build("travel_bookings.csv", "users.csv", "destinations.csv")

# Make predictions
recommendations = model.predict({
    "preferred_climate": "warm",
    "budget": 1500.0,
    "travel_month": "March",
    "travel_type": "adventure"
})
print(recommendations["recommended_destinations"])
