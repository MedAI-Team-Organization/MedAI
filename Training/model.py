# import pandas as pd
# import numpy as np
# import joblib
# import os

# def test_side_effect_prediction():
#     """
#     Function to test the saved model for predicting side effects of medicines
#     """
#     # Paths to the saved model components
#     model_path = r"C:\Users\jignesh01\OneDrive - Conestoga College\side_effect_model.joblib"
#     vectorizer_path = r"C:\Users\jignesh01\OneDrive - Conestoga College\tfidf_vectorizer.joblib"
#     mlb_path = r"C:\Users\jignesh01\OneDrive - Conestoga College\mlb.joblib"
    
#     # Check if files exist
#     for path in [model_path, vectorizer_path, mlb_path]:
#         if not os.path.exists(path):
#             print(f"Error: File not found: {path}")
#             return
    
#     # Load the saved model components
#     print("Loading model components...")
#     model = joblib.load(model_path)
#     vectorizer = joblib.load(vectorizer_path)
#     mlb = joblib.load(mlb_path)
#     print("Model components loaded successfully.")
    
#     # Sample medicine data for testing
#     # You can modify this with real data or create multiple test samples
#     sample_medicine = {
#         'name': 'Cetrizine',
#          'usecases': 'Treatment of Bacterial infections',
#         'substitutes': 'Loratadine, Fexofenadine',
#         'composition': 'Phenylephrine (5mg/5ml)'
#     }
    
#     # Create another test caseAn
#     sample_medicine2 = {
#         'name': 'Ibuprofen',
#         'usecases': 'Pain relief, Inflammation, Fever',
#         'substitutes': 'Naproxen, Diclofenac, Acetaminophen',
#         'composition': 'Ibuprofen'
#     }
    
#     # Predict side effects for sample medicines
#     print("\nPredicting side effects for sample medicines...")
    
#     # Test first medicine
#     predict_and_display(sample_medicine, vectorizer, model, mlb)
    
#     # Test second medicine
#     predict_and_display(sample_medicine2, vectorizer, model, mlb)
    
#     # Optional: Add interactive testing capability
#     test_interactive(vectorizer, model, mlb)

# def predict_and_display(medicine_data, vectorizer, model, mlb):
#     """
#     Predict side effects for a given medicine and display results
#     """
#     # Create text features (same as in training)
#     text_features = medicine_data['name'] + ' ' + \
#                    medicine_data['substitutes'] + ' ' + \
#                    medicine_data['usecases'] + ' ' + \

#                    medicine_dat a['composition']
    
#     # Vectorize
#     X_new = vectorizer.transform([text_features])
    
#     # Predict
#     y_pred = model.predict(X_new)
    
#     # Get side effect names
#     predicted_indices = np.where(y_pred[0] == 1)[0]
#     predicted_side_effects = mlb.classes_[predicted_indices]
    
#     # Display results
#     print(f"\nPredicted side effects for {medicine_data['name']}:")
#     if len(predicted_side_effects) > 0:
#         for i, effect in enumerate(predicted_side_effects, 1):
#             print(f"{i}. {effect}")
#     else:
#         print("No side effects predicted.")
    
#     return predicted_side_effects

# def test_interactive(vectorizer, model, mlb):
#     """
#     Allow user to input medicine details interactively and predict side effects
#     """
#     try_again = input("\nWould you like to test with your own medicine data? (yes/no): ").lower()
    
#     if try_again == 'yes' or try_again == 'y':
#         print("\nEnter medicine details:")
        
#         medicine_data = {
#             'name': input("Medicine Name: "),
#             'substitutes': input("Substitutes (comma separated): "),
#             'usecases': input("Use Cases (comma separated): "),
#             'Chemical Class': input("Chemical Class: "),
#             'Habit Forming': input("Habit Forming (Yes/No): "),
#             'Therapeutic Class': input("Therapeutic Class: "),
#             'Action Class': input("Action Class: "),
#             'manufacturer_name': input("Manufacturer Name: "),
#             'composition': input("Composition: ")
#         }
        
#         # Predict and display results
#         predict_and_display(medicine_data, vectorizer, model, mlb)

# if __name__ == "__main__":
#     test_side_effect_prediction()


import pandas as pd
import numpy as np
import joblib
import os

def predict_side_effects_simplified():
    """
    Simplified function to test the saved model for predicting side effects
    Only requires name, substitutes, composition and use cases
    """
    # Paths to the saved model components
    model_path = r"C:\Users\jignesh01\OneDrive - Conestoga College\side_effect_model.joblib"
    vectorizer_path = r"C:\Users\jignesh01\OneDrive - Conestoga College\tfidf_vectorizer.joblib"
    mlb_path = r"C:\Users\jignesh01\OneDrive - Conestoga College\mlb.joblib"
    
    # Check if files exist
    for path in [model_path, vectorizer_path, mlb_path]:
        if not os.path.exists(path):
            print(f"Error: File not found: {path}")
            return
    
    # Load the saved model components
    print("Loading model components...")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    mlb = joblib.load(mlb_path)
    print("Model components loaded successfully.")
    
    # Get user input for the required fields
    print("\nEnter medicine details:")
    name = input("Medicine Name: ")
    substitutes = input("Substitutes (comma separated): ")
    composition = input("Composition: ")
    usecases = input("Use Cases (comma separated): ")
    
    # Create a complete medicine data dictionary with default values for other fields
    medicine_data = {
        'name': name,
        'substitutes': substitutes,
        'composition': composition,
        'usecases': usecases,
        'Chemical Class': 'unknown',  # Default value
        'Habit Forming': 'unknown',   # Default value
        'Therapeutic Class': 'unknown',  # Default value
        'Action Class': 'unknown',    # Default value
        'manufacturer_name': 'unknown'  # Default value
    }
    
    # Predict and display results
    predicted_side_effects = predict_and_display(medicine_data, vectorizer, model, mlb)
    
    # Ask if user wants to try another medicine
    try_another = input("\nWould you like to test another medicine? (yes/no): ").lower()
    if try_another == 'yes' or try_another == 'y':
        predict_side_effects_simplified()

def predict_and_display(medicine_data, vectorizer, model, mlb):
    """
    Predict side effects for a given medicine and display results
    """
    # Create text features (same as in training)
    text_features = medicine_data['name'] + ' ' + \
                   medicine_data['substitutes'] + ' ' + \
                   medicine_data['usecases'] + ' ' + \
                   medicine_data['Chemical Class'] + ' ' + \
                   medicine_data['Habit Forming'] + ' ' + \
                   medicine_data['Therapeutic Class'] + ' ' + \
                   medicine_data['Action Class'] + ' ' + \
                   medicine_data['manufacturer_name'] + ' ' + \
                   medicine_data['composition']
    
    # Vectorize
    X_new = vectorizer.transform([text_features])
    
    # Predict
    y_pred = model.predict(X_new)
    
    # Get side effect names
    predicted_indices = np.where(y_pred[0] == 1)[0]
    predicted_side_effects = mlb.classes_[predicted_indices]
    
    # Display results
    print(f"\nPredicted side effects for {medicine_data['name']}:")
    if len(predicted_side_effects) > 0:
        for i, effect in enumerate(predicted_side_effects, 1):
            print(f"{i}. {effect}")
    else:
        print("No side effects predicted.")
    
    return predicted_side_effects

if __name__ == "__main__":
    predict_side_effects_simplified()