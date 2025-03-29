import pandas as pd
import os

def extract_unique_values(csv_file_path):
    df = pd.read_csv(csv_file_path)

    side_effect_columns = [f"sideEffect{i}" for i in range(42) if f"sideEffect{i}" in df.columns]

    unique_values = set()
    for col in side_effect_columns:
        unique_values.update(df[col].dropna().unique())

    unique_df = pd.DataFrame(sorted(unique_values), columns=["UniqueSideEffects"])

    output_file_path = os.path.join(os.path.dirname(csv_file_path), "unique_side_effects.csv")
    unique_df.to_csv(output_file_path, index=False)
    print(f"Unique side effects saved to: {output_file_path}")


csv_file_path = "MedAI/Data/combinedDataset.csv"  
extract_unique_values(csv_file_path)
