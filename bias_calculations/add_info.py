import os
import pandas as pd
import glob
from tqdm import tqdm

# Read the root data file
root_data = pd.read_csv('root_data.csv', usecols=['APPLICATION_ID', 'FOA_NUMBER', 'IC_NAME', 'GRANT_VALUE'])

# Function to process each file
def process_file(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Get the model name from the file
    model_name = [col for col in df.columns if col.startswith('APP_RECS_')][0].split('APP_RECS_')[1]
    
    # Rename VAL_RECS column
    df = df.rename(columns={f'VAL_RECS_{model_name}': f'APPVAL_RECS_{model_name}'})
    
    # Function to get values for a single row
    def get_values(app_recs):
        app_ids = [id.strip() for id in app_recs.split(',')]
        grant_values = []
        foa_numbers = []
        
        for app_id in app_ids:
            if app_id != 'None':
                matched_data = root_data[root_data['APPLICATION_ID'] == int(app_id)]
                if not matched_data.empty:
                    grant_values.append(str(matched_data['GRANT_VALUE'].values[0]))
                    foa_numbers.append(str(matched_data['FOA_NUMBER'].values[0]))
                else:
                    grant_values.append('None')
                    foa_numbers.append('None')
            else:
                grant_values.append('None')
                foa_numbers.append('None')
        
        return ','.join(grant_values), ','.join(foa_numbers)
    
    # Apply the function to each row
    df[[f'APPVAL_RECS_{model_name}', f'FOA_RECS_{model_name}']] = df[f'APP_RECS_{model_name}'].apply(lambda x: pd.Series(get_values(x)))
    
    # Save the updated DataFrame back to CSV
    df.to_csv(file_path, index=False)

# Get all CSV files in the folder
csv_files = glob.glob('recs_renamed_cols/*.csv')

# Process all CSV files with a progress bar
for file in tqdm(csv_files, desc="Processing files", unit="file"):
    process_file(file)

print("All files have been processed and updated.")