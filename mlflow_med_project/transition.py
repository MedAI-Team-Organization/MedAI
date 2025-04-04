from mlflow import MlflowClient
 
# Initialize the client
client = MlflowClient()
 
# Your model details
model_name = "MedAI"  # Replace with your actual model name
model_version = 1  # Your current version number
 
# Transition the model from None stage to Production
client.transition_model_version_stage(
    name=model_name,
    version=model_version,
    stage="Production"
)
 
print(f"Model {model_name} version {model_version} successfully transitioned to Production stage")
 