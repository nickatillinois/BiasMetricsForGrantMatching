# read processed_combined_aligned_name_split.csv
#take APPLICATION_ID and FOA_NUMBER column
# save as foa_list.csv

import pandas as pd

df = pd.read_csv('processed_combined_aligned_name_split.csv')

df = df['FOA_NUMBER']

# only keep unique values

df = df.drop_duplicates()

df.to_csv('foa_list.csv', index=False)

print('done')