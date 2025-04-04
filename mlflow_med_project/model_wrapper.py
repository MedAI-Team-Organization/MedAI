

import pandas as pd
import numpy as np
import joblib
import mlflow.pyfunc
import logging

class MedicineSideEffectModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """Load the model components from the MLflow artifact context"""
        self.model = joblib.load(context.artifacts["model"])
        self.vectorizer = joblib.load(context.artifacts["vectorizer"])
        self.mlb = joblib.load(context.artifacts["mlb"])
        self.valid_columns = joblib.load(context.artifacts["valid_columns"])
    
    def predict(self, context, model_input):
        """
        Make predictions on the input data
        model_input: can be a variety of formats when served through MLflow
        """
        # Add logging to debug the input format
        logging.info(f"Input type: {type(model_input)}")
        
        # Convert input to DataFrame regardless of input format
        try:
            # First check if it might be a DataFrame already
            if hasattr(model_input, 'columns'):
                df = model_input
            # Check for MLflow server input format with 'instances' key
            elif isinstance(model_input, dict) and 'instances' in model_input:
                df = pd.DataFrame(model_input['instances'])
            # Direct pandas.DataFrame conversion for other dict-like structures
            else:
                # The input might be a dict with column names as keys
                df = pd.DataFrame(model_input)
                
                # If conversion failed (dict with different structure), try treating it as a single instance
                if len(df) == 0 or all(isinstance(v, dict) for v in df.values()):
                    df = pd.DataFrame([model_input])
                    
            logging.info(f"Converted DataFrame shape: {df.shape}")
            logging.info(f"DataFrame columns: {df.columns.tolist()}")
            
        except Exception as e:
            logging.error(f"Error converting input to DataFrame: {str(e)}")
            # Last resort: try to process raw data directly
            # This is a fallback for when MLflow serves data in a different format
            try:
                # Try to process instances directly if they exist
                if isinstance(model_input, dict) and 'instances' in model_input:
                    instances = model_input['instances']
                    df = pd.json_normalize(instances)
                else:
                    # Just try to create a single-row dataframe
                    df = pd.DataFrame([model_input])
            except Exception as e2:
                logging.error(f"Failed fallback conversion: {str(e2)}")
                raise ValueError(f"Could not convert input to DataFrame. Original error: {str(e)}, Fallback error: {str(e2)}")

        # Ensure the input has all required columns
        required_cols = [
            'name', 'substitutes', 'usecases', 'Chemical Class', 
            'Habit Forming', 'Therapeutic Class', 'Action Class', 
            'manufacturer_name', 'composition'
        ]
        
        # Fill missing columns with default values
        for col in required_cols:
            if col not in df.columns:
                df[col] = 'unknown'
        
        # Handle list columns by converting them to strings if they're lists
        for col in ['substitutes', 'usecases']:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: ", ".join(x) if isinstance(x, list) else str(x)
                )
        
        # Create text features for each row
        text_features = []
        for _, row in df.iterrows():
            feature = (
                str(row['name']) + ' ' + 
                str(row['substitutes']) + ' ' + 
                str(row['usecases']) + ' ' + 
                str(row['Chemical Class']) + ' ' + 
                str(row['Habit Forming']) + ' ' + 
                str(row['Therapeutic Class']) + ' ' + 
                str(row['Action Class']) + ' ' + 
                str(row['manufacturer_name']) + ' ' + 
                str(row['composition'])
            )
            text_features.append(feature)
        
        # Vectorize
        X_new = self.vectorizer.transform(text_features)
        
        # Predict
        predictions = self.model.predict(X_new)
        
        # Convert predictions to side effect names
        results = []
        for pred in predictions:
            predicted_indices = np.where(pred == 1)[0]
            predicted_side_effects = [self.mlb.classes_[self.valid_columns[idx]] for idx in predicted_indices]
            results.append(predicted_side_effects)
        
        # Return as a dataframe
        return pd.DataFrame({
            'predicted_side_effects': results
        })