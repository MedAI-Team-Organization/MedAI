import pandas as pd
import numpy as np

def analyze_critical_side_effects(csv_filepath):
    """
    Analyzes a CSV file to identify rows with critical side effects and
    updates the 'active' column in the same file.
    
    Parameters:
    csv_filepath (str): Path to the input CSV file
    
    Returns:
    pd.DataFrame: DataFrame with added criticalSideEffects and active columns
    """
    critical_side_effects = {
        "Cardiovascular": [
            "Heart attack", "Cardiac arrest", "Myocardial infarction", "Brain hemorrhage",
            "Cardiac failure", "Heart failure", "Pulmonary hypertension", "Ventricular arrhythmia",
            "Hypertensive crisis", "Blood clots"
        ],
        "Central Nervous System": [
            "Seizure", "Convulsion", "Coma", "Tonic-clonic seizures",
            "Generalized tonic-clonic seizure", "Suicidal behaviors",
            "Increased intracranial pressure", "Brain swelling", "Psychosis", 
            "Severe depression", "CNS toxicity"
        ],
        "Respiratory": [
            "Respiratory arrest", "Respiratory depression", "Pulmonary hemorrhage",
            "Bronchospasm", "Shortness of breath", "Hypoxia", "Pneumothorax"
        ],
        "Allergic and Immunological": [
            "Anaphylactic reaction", "Severe allergic reaction", "Cytokine release syndrome",
            "Serum sickness"
        ],
        "Hematological": [
            "Agranulocytosis", "Bone marrow failure", "Bone marrow suppression",
            "Aplastic anemia", "Febrile neutropenia", "Thrombotic thrombocytopenic purpura",
            "Hemorrhage", "Sepsis"
        ],
        "Dermatological": [
            "Stevens-Johnson syndrome", "Toxic epidermal necrolysis", 
            "Drug rash with eosinophilia and systemic symptoms"
        ],
        "Urinary and Renal": [
            "Acute renal failure", "Kidney damage", "Renal injury"
        ],
        "Musculoskeletal": [
            "Bone fracture"
        ],
        "Gastrointestinal": [
            "Gastrointestinal perforation", "Gastrointestinal bleeding", "Hemorrhagic colitis"
        ],
        "Endocrine and Metabolic": [
            "Adrenal insufficiency", "Lactic acidosis"
        ]
    }
    
    try:
        df = pd.read_csv(csv_filepath)
        print(f"Successfully loaded {csv_filepath}")
    except Exception as e:
        print(f"Error loading the CSV file: {e}")
        return None
    
    df['criticalSideEffects'] = 0
    
    df['active'] = 0
    
    available_categories = [col for col in critical_side_effects.keys() if col in df.columns]
    
    if not available_categories:
        print("No matching category columns found in the CSV file.")
        return df
    
    row_count = len(df)
    critical_count = 0
    
    for idx, row in df.iterrows():
        has_critical = False
        
        for category in available_categories:
            if pd.isna(row[category]) or not isinstance(row[category], str):
                continue
            
            side_effects_in_cell = row[category].split(',')
            for side_effect in side_effects_in_cell:
                side_effect = side_effect.strip()
                if side_effect in critical_side_effects[category]:
                    has_critical = True
                    break
            
            if has_critical:
                break
        
        if has_critical:
            df.at[idx, 'criticalSideEffects'] = 1
            df.at[idx, 'active'] = 1 
            critical_count += 1
    
    print(f"Analysis complete. Found {critical_count} out of {row_count} rows with critical side effects.")
    
    df.to_csv(csv_filepath, index=False)
    print(f"Results saved to the same file: {csv_filepath}")
    
    return df

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        analyze_critical_side_effects(input_file)
    else:
        print("Usage: python script.py input_file.csv")