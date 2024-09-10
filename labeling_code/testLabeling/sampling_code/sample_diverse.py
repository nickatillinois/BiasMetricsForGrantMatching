import pandas as pd
import numpy as np

# Load the CSV file into a DataFrame
df = pd.read_csv('labeled_by_model.csv')

# Extract the highest probability ethnicity for each row
def get_highest_ethnicity(probs):
    probs_dict = eval(probs)
    return max(probs_dict, key=probs_dict.get)

df['highest_ethnicity'] = df['PI_ETHNICITY_PROBS'].apply(get_highest_ethnicity)

# Filter the rows for each ethnicity
asian_df = df[df['highest_ethnicity'] == 'asian']
black_df = df[df['highest_ethnicity'] == 'black']
hispanic_df = df[df['highest_ethnicity'] == 'hispanic']

# Randomly sample 20 rows from each ethnicity
asian_sample = asian_df.sample(n=20, random_state=1)
black_sample = black_df.sample(n=20, random_state=1)
hispanic_sample = hispanic_df.sample(n=20, random_state=1)

# Combine the samples into one DataFrame
sampled_df = pd.concat([asian_sample, black_sample, hispanic_sample])

# Save the sampled DataFrame to a new CSV file
sampled_df.to_csv('sampled_rows.csv', index=False)
