import mlflow
import os
from model_wrapper import MedicineSideEffectModel
 
# Paths to model components
model_path = "side_effect_model.joblib"
vectorizer_path = "tfidf_vectorizer.joblib"
mlb_path = "mlb.joblib"
valid_columns_path = "valid_columns.joblib"
 
# Set the tracking URI - using local filesystem
# mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
 
# Start a new MLflow run
with mlflow.start_run(run_name="medicine_side_effects_model") as run:
    
    # Log model
    artifacts = {
        "model": model_path,
        "vectorizer": vectorizer_path,
        "mlb": mlb_path,
        "valid_columns": valid_columns_path
    }
    
    # Log model with its dependencies
    mlflow.pyfunc.log_model(
        artifact_path="medicine_model",
        python_model=MedicineSideEffectModel(),
        artifacts=artifacts,
        conda_env="conda.yaml",
        code_path=["model_wrapper.py"]  # Include wrapper code
    )
    
    # Log additional information
    mlflow.log_param("model_type", "MultiOutputClassifier")
    mlflow.log_param("feature_extraction", "TF-IDF")
    
    print(f"Model logged with run_id: {run.info.run_id}")
    print(f"Model URI: mlflow:/models/medicine_side_effects_model")