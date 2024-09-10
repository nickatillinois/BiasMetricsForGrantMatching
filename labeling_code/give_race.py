from  ethnicolr2 import pred_fl_last_name, pred_fl_full_name
# https://github.com/appeler/ethnicolr?tab=readme-ov-file
import pandas as pd
import os
import time
from tqdm import tqdm


# Read the CSV file into a DataFrame and select specific columns
df = pd.read_csv('genderized1.csv', usecols=['APPLICATION_ID', 'PI_FIRST_NAME', 'PI_LAST_NAME'])

df = df.iloc[:100, :]




# Remove rows with missing values in the selected columns
print("Original DataFrame contains:", df.shape[0], "rows")
df = df.dropna(subset=['APPLICATION_ID', 'PI_FIRST_NAME', 'PI_LAST_NAME'])
# add new column named PI_ETHNICITY which is empty
df['PI_ETHNICITY'] = ""


total = df.shape[0]
# Apply pred_fl_full_name to the entire DataFrame
for idx, row in (df.iterrows()):
    # if idx not zero, print the previous prediction
    # if idx != 0:
    #    print(df.head())
    # if idx % 1000 == 0:
    #     print("=====================================")
    #     print(f"Processing row {idx} of {total}")
    #     print("=====================================")
    try:
        #print("Index:", idx)
        #print("Row:", row)
        #print("df at index:", df.loc[idx])
        first_name, last_name = str(row['PI_FIRST_NAME']), str(row['PI_LAST_NAME'])
        # put to lower case
        first_name, last_name = first_name.lower(), last_name.lower()
        tempdf = pd.DataFrame({'First': [str(row['PI_FIRST_NAME'])], 'Last': [str(row['PI_LAST_NAME'])]})
        tempdf = pred_fl_full_name(tempdf, lname_col="Last", fname_col="First")
        tempdf['PI_ETHNICITY'] = tempdf['probs'].apply(lambda x: max(x, key=x.get))
        #print(tempdf)
        df.loc[idx, 'PI_ETHNICITY'] = tempdf['preds'].values[0]
        #os.system('clear')
        #print(df.head())
    except Exception as e:
        print(f"Skipping error at index {idx}: {e}")
    # if multiple of 1000, print progress
    # if idx % 1000 == 0:
    #     os.system('clear')
    #     print("=====================================")
    #     print(f"Processing row {idx} of {total}")
    #     print("=====================================")
    #     time.sleep(5)

# Clear the console
os.system('clear')

# Print the head of the new DataFrame
print("Head of DataFrame after applying pred_fl_full_name:")
print(df.head())

# Map values in 'preds' column to match 'Race' column
preds_mapping = {'nh_white': 'white', 'nh_black': 'black'}
df['PI_ETHNICITY'] = df['PI_ETHNICITY'].map(preds_mapping).fillna(df['PI_ETHNICITY'])

# Print the head of the new DataFrame
print("Head of DataFrame after mapping 'preds' column:")
print(df.head())

# Save the new DataFrame to a CSV file
df.to_csv('gave_races.csv', index=False)

"""
# only give the first row
df = df.iloc[:2, :]
df = pred_fl_full_name(df, lname_col = "PI_LAST_NAME", fname_col = "PI_FIRST_NAME")
print(df)
# in df probs, take the highest probability and put it in preds
# in column probs, there is a dictionary with the probabilities of each, take the highest probability and put it in preds
df['preds'] = df['probs'].apply(lambda x: max(x, key=x.get))
print(df)
# print dictionary in probs key - value pair by pair
for idx, row in df.iterrows():
    print("Index:", idx)
    for key, value in row['probs'].items():
        print(f"{key}: {value}")
    print("=====================================")
"""