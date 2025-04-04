import pandas as pd
import numpy as np
import joblib
import os
import mlflow.pyfunc

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
        model_input: pandas DataFrame with medicine information
        """
        # Ensure the input has all required columns
        required_cols = [
            'name', 'substitutes', 'usecases', 'Chemical Class', 
            'Habit Forming', 'Therapeutic Class', 'Action Class', 
            'manufacturer_name', 'composition'
        ]
        
        # Fill missing columns with default values
        for col in required_cols:
            if col not in model_input.columns:
                model_input[col] = 'unknown'
        
        # Create text features for each row
        text_features = []
        for _, row in model_input.iterrows():
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