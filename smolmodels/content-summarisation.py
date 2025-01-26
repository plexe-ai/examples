from smolmodels import Model

# Create a content summarization model with just a description
model = Model(
    intent="Summarize long-form content into concise summaries",
    input_schema={
        "content": str,
        "max_summary_length": int  # number of words/characters
    },
    output_schema={
        "summary": str  # summarized content
    }
)

# Build the model with entirely synthetic training data
model.build(generate_samples=1000)

# Make predictions
summary = model.predict({
    "content": "The rapid advancement of artificial intelligence has transformed numerous industries...",
    "max_summary_length": 50
})
print(summary["summary"])