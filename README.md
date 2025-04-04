MLflow Model Deployment
Overview
This project involves deploying a machine learning model that predicts potential side effects based on a given medicine's name, composition, substitutes, and use cases. Two models, Random Forest and Support Vector Machine (SVM), were trained, and the SVM model was chosen for deployment as it performed better.
Installation & Setup
Ensure you have the following installed:
Python 3.11
MLflow
SQLite (for tracking experiments)
Required dependencies for the model
If MLflow is not installed, install it using:
pip install mlflow
Running the MLflow Server & UI
1. Start MLflow Tracking Server
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root file:///Users/jainamdoshi/Desktop/MedAI/mlflow_med_project/mlruns \
  --host 0.0.0.0 \
  --port 5002
This starts the MLflow tracking server, which stores experiment metadata in an SQLite database.
2. Start MLflow UI
mlflow ui -p 5000 --backend-store-uri sqlite:///mlflow.db
This launches the MLflow UI, accessible at http://127.0.0.1:5000, where you can view experiment results.
Serving the Model
3. Serve the Production Model
mlflow models serve -m "models:/MedAI/Production" --port 5003 --env-manager=local
This serves the model registered under the name MedAI in the Production stage at port 5003.
4. Serve a Specific Model Run
mlflow models serve -m "runs:/5077619848874de9839c2f114355b745/medicine_model" -p 5001 --env-manager=local
This serves a specific model version associated with run ID 5077619848874de9839c2f114355b745 at port 5001.
Making Predictions
To send a request to the deployed model and get predictions, use the following curl command:
curl -X POST http://127.0.0.1:5003/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_split": {
      "columns": ["name", "composition", "substitutes", "usecases"],
      "data": [
        ["Aspirin", "Acetylsalicylic acid", ["Ibuprofen", "Paracetamol"], ["Pain relief", "Anti-inflammatory"]]
      ]
    }
  }'
This sends a JSON request with medicine details to the MLflow model, which will return predicted side effects.
Exposing the Model Publicly using ngrok
To make the model accessible over the internet, use ngrok:
ngrok http 5003
This will generate a public URL (e.g., https://abc123.ngrok.io), which can be used to send requests from anywhere.
Conclusion
This setup allows you to:
Train and track ML models using MLflow
Deploy and serve the best-performing model (SVM)
Send requests for real-time predictions
Expose the model endpoint publicly using ngrok
For further improvements, consider deploying the model to a cloud service like AWS, GCP, or Azure.
