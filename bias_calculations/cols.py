import pandas as pd
from tqdm import tqdm

def add_gt_columns(merged_file, root_file, output_file):
    # Load the merged CSV file and root data CSV file
    merged_df = pd.read_csv(merged_file)
    root_df = pd.read_csv(root_file)

    # Ensure that the columns 'APPLICATION_ID' and 'FOA_NUMBER', 'GRANT_VALUE' exist in root_df
    if not all(col in root_df.columns for col in ['APPLICATION_ID', 'FOA_NUMBER', 'GRANT_VALUE']):
        raise ValueError("root_data.csv must contain 'APPLICATION_ID', 'FOA_NUMBER', and 'GRANT_VALUE' columns.")
    
    # Initialize a progress bar
    tqdm.pandas(desc="Merging data")

    # Merge the merged_df with root_df on 'APPLICATION_ID'
    merged_df = merged_df.merge(root_df[['APPLICATION_ID', 'FOA_NUMBER', 'GRANT_VALUE']],
                                on='APPLICATION_ID',
                                how='left')

    # Rename the columns for consistency
    merged_df.rename(columns={
        'FOA_NUMBER': 'GT_FOA',
        'GRANT_VALUE': 'GT_VALUE'
    }, inplace=True)

    # Reorder the columns to place 'GT_FOA' and 'GT_VALUE' after 'PI_RACE'
    cols = list(merged_df.columns)
    pi_race_index = cols.index('PI_RACE') + 1
    new_cols = cols[:pi_race_index] + ['GT_FOA', 'GT_VALUE'] + cols[pi_race_index:]
    merged_df = merged_df[new_cols]

    # Save the updated DataFrame to a new CSV file
    merged_df.to_csv(output_file, index=False)
    print(f'Updated CSV saved to {output_file}')

# Define file paths
merged_file = 'merged.csv'
root_file = 'root_data.csv'
output_file = 'merged_with_gt.csv'

# Run the function
add_gt_columns(merged_file, root_file, output_file)
