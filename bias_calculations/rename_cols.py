import os
import pandas as pd

# Directory paths
original_dir = 'recs'
new_dir = 'recs_renamed_cols'

# Create new directory if it doesn't exist
os.makedirs(new_dir, exist_ok=True)

# Iterate through the CSV files in the original directory
for filename in os.listdir(original_dir):
    if filename.endswith('.csv'):
        # Construct full file paths
        original_path = os.path.join(original_dir, filename)
        new_path = os.path.join(new_dir, filename)
        
        # Read the CSV file
        df = pd.read_csv(original_path)
        
        # Create new column names
        base_name = filename.replace('.csv', '')
        new_columns = {
            'APPLICATION_IDs': f'APP_RECS_{base_name}',
            'REC_GRANT_VALUES': f'VAL_RECS_{base_name}',
            'REC_FOA_NUMBERS': f'FOA_RECS_{base_name}'
        }
        
        # Rename columns
        df.rename(columns=new_columns, inplace=True)
        
        # Save the modified DataFrame to the new directory
        df.to_csv(new_path, index=False)

print("Renaming completed and files saved in 'recs_renamed_cols' directory.")
