import pandas as pd
import os
from tqdm import tqdm

def merge_csv_files(folder_path, output_file):
    # Get a list of all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    # Initialize an empty dictionary to store dataframes
    data_dict = {}
    
    # Initialize progress bar
    for file in tqdm(csv_files, desc="Merging files", unit="file"):
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        
        # Extract model-specific columns
        base_name = file.replace('.csv', '')
        app_col = f'APP_RECS_{base_name}'
        val_col = f'APPVAL_RECS_{base_name}'
        foa_col = f'FOA_RECS_{base_name}'
        
        # Check if the required columns exist in the dataframe
        required_cols = ['APPLICATION_ID', 'PI_GENDER', 'PI_RACE', app_col, val_col, foa_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Warning: Missing columns {missing_cols} in file {file}")
            continue
        
        # Process and merge data
        for _, row in df.iterrows():
            app_id = row['APPLICATION_ID']
            if app_id not in data_dict:
                data_dict[app_id] = {
                    'PI_GENDER': row['PI_GENDER'],
                    'PI_RACE': row['PI_RACE']
                }
            
            data_dict[app_id][app_col] = row[app_col]
            data_dict[app_id][val_col] = row[val_col]
            data_dict[app_id][foa_col] = row[foa_col]
    
    # Convert the dictionary to a DataFrame
    merged_df = pd.DataFrame.from_dict(data_dict, orient='index').reset_index()
    merged_df.rename(columns={'index': 'APPLICATION_ID'}, inplace=True)
    
    # Save the merged dataframe to a CSV file
    merged_df.to_csv(output_file, index=False)
    print(f'Merged CSV saved to {output_file}')

# Define the folder path and output file
folder_path = 'recs_renamed_cols'
output_file = 'merged.csv'

# Run the merge function
merge_csv_files(folder_path, output_file)
