# read combined_file_with_titles.csv and grant_numbers_complete.csv
# both have a column named 'Grant Number'

# take the intersection of the two columns and print the grant numbers occurring in both files to 
# a new file named 'grant_numbers_all_sections_and_titles.csv'

import pandas as pd

# Define input file paths
combined_file = 'combined_file_with_titles.csv'
grant_numbers_file = 'grant_numbers_complete.csv'
output_file = 'grant_numbers_all_sections_and_titles.csv'

# Read the input files
combined_df = pd.read_csv(combined_file)
grant_numbers_df = pd.read_csv(grant_numbers_file)

# Find the intersection of the two columns
grant_numbers_intersection = pd.merge(combined_df, grant_numbers_df, on='Grant Number')['Grant Number']

# Write the grant numbers to a new file
grant_numbers_intersection.to_csv(output_file, index=False)

print(f"Grant numbers occurring in both files saved as '{output_file}'.")