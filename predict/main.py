from fastapi import FastAPI
import joblib
from predict.your_prediction_module import predict_function  # Import your prediction function from your prediction module

app = FastAPI()

# Load your prediction model
testing_model = joblib.load("testing_model.pkl")

# Define your prediction endpoint
@app.post("/predict/")
async def predict():
    # Call your prediction function here, passing the loaded model
    prediction_result = predict_function(testing_model)  # Call your prediction function and pass the model
    return {"prediction": prediction_result}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
