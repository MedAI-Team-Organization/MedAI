import pandas as pd
import argparse
 
def count_substitutes(csv_path):
    """
    Process a CSV file to count valid substitutes for each medicine.
    
    Args:
        csv_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: DataFrame with added substituteCount column
    """
    try:
        # Read the CSV file - using na_filter=False to preserve N/A as strings
        df = pd.read_csv(csv_path, na_filter=False)
        
        # Check if the required columns exist
        substitute_columns = [f"substitute{i}" for i in range(5)]
        missing_columns = [col for col in substitute_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Function to count valid substitutes in a row
        def count_valid_substitutes(row):
            count = 0
            for i in range(5):
                col_name = f"substitute{i}"
                if row[col_name] != "" and row[col_name] != "N/A":
                    count += 1
            return count
        
        # Apply the function to each row and create a new column
        df['substituteCount'] = df.apply(count_valid_substitutes, axis=1)
        
        return df
    except Exception as e:
        print(f"Error processing file: {e}")
        return None
 
def main():
    """
    Main function to parse command line arguments and process the CSV file.
    """
    parser = argparse.ArgumentParser(description='Count valid substitutes for medicines in CSV file')
    parser.add_argument('csv_path', help='Path to the CSV file')
    parser.add_argument('--output', '-o', help='Path to save the output CSV file (optional)')
    
    args = parser.parse_args()
    
    result_df = count_substitutes(args.csv_path)
    
    if result_df is not None:
        # Save to the original input file if no output file is specified
        output_path = args.output if args.output else args.csv_path
        
        result_df.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")
        
        # Print the first few rows to show the result
        print("\nFirst few rows of the processed data:")
        print(result_df.head())
        print(f"\nTotal rows processed: {len(result_df)}")
 
if __name__ == "__main__":
    main()