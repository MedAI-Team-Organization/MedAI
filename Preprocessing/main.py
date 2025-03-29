import pandas as pd
import sys

def merge_csv(file1_path, file2_path, output_path):
    df1 = pd.read_csv(file1_path)
    
    df2 = pd.read_csv(file2_path, usecols=["manufacturer_name", "short_composition1", "short_composition2"])
    
    merged_df = pd.concat([df1, df2], axis=1)
    merged_df.to_csv(output_path, index=False)
    
    print(f"Merged file saved as: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python merge_csv.py <file1_path> <file2_path> <output_path>")
        sys.exit(1)
    
    file1_path = sys.argv[1]
    file2_path = sys.argv[2]
    output_path = sys.argv[3]
    
    merge_csv(file1_path, file2_path, output_path)
