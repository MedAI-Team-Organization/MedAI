import pandas as pd
import numpy as np
 
# File path (you should modify this to the actual path of your CSV file)
file_path = "/Users/jainamdoshi/Desktop/MedAI/MedAI/Data/combinedDataset_categorized.csv"
 
try:
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Store the original column names and their data types
    original_dtypes = df.dtypes.to_dict()
    
    # Replace missing or empty values with "N/A"
    # This will replace:
    # - NaN values
    # - Empty strings ""
    # - Strings that only contain whitespace
    
    # First, replace NaN values with "N/A"
    df = df.fillna("N/A")
    
    # Next, replace empty strings or strings with only whitespace
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].apply(lambda x: "N/A" if isinstance(x, str) and x.strip() == "" else x)
    
    # Save the modified data back to the same file
    df.to_csv(file_path, index=False)
    
    print(f"Successfully processed {file_path}")
    print(f"All missing or empty values have been replaced with 'N/A'")
    
except FileNotFoundError:
    print(f"Error: The file {file_path} was not found.")
except pd.errors.EmptyDataError:
    print(f"Error: The file {file_path} is empty.")
except Exception as e:
    print(f"An error occurred: {str(e)}")