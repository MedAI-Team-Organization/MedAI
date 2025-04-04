import requests
import pandas as pd
import json

def test_endpoint(endpoint_url):
    """Test the MLflow model endpoint with a sample medicine"""
    
    # Sample input data
    test_data = {
        "columns": ["name", "substitutes", "composition", "usecases"],
        "data": [
            ["Aspirin", "Ibuprofen, Acetaminophen", "Acetylsalicylic acid", "Pain relief, Fever reduction"]
        ]
    }
    
    # Send POST request to the endpoint
    response = requests.post(
        url=endpoint_url, 
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    
    # Check if request was successful
    if response.status_code == 200:
        print("API call successful!")
        result = response.json()
        print("\nPredicted side effects:")
        
        # Parse the predictions
        predictions = result["predictions"][0]["predicted_side_effects"]
        
        if isinstance(predictions, list) and len(predictions) > 0:
            for i, effect in enumerate(predictions, 1):
                print(f"{i}. {effect}")
        else:
            print("No side effects predicted.")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    # Replace with your actual endpoint URL after deployment
    endpoint_url = input("Enter the MLflow model endpoint URL: ")
    test_endpoint(endpoint_url)