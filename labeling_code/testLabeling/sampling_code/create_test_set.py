# read gave_races2py
# has columns APPLICATION_ID,PI_LAST_NAME,PI_FIRST_NAME,PI_ETHNICITY_PROBS

import pandas as pd
import numpy as np

df = pd.read_csv('gave_races2.csv', usecols=["APPLICATION_ID","PI_LAST_NAME","PI_FIRST_NAME","PI_ETHNICITY_PROBS"])



df['PI_ETHNICITY_PROBS'] = [eval(prob_str) for prob_str in df['PI_ETHNICITY_PROBS']]
df['PI_ETHNICITY_PROBS'] = [list(prob_dict.values()) for prob_dict in df['PI_ETHNICITY_PROBS']]
df['PI_ETHNICITY_PROBS'] = np.array(df['PI_ETHNICITY_PROBS'])
print(df.head())

# select a random sample of 60 rows
random_sample = df.sample(n=60, random_state=42)

# select rows of which probability of being asian is greater than 0.5 (i.e. the second element in PI_ETHNICITY_PROBS)
asian_sample = df[df['PI_ETHNICITY_PROBS'].apply(lambda x: x[0] > 0.5)]
black_sample = df[df['PI_ETHNICITY_PROBS'].apply(lambda x: x[1] > 0.5)]
hispanic_sample = df[df['PI_ETHNICITY_PROBS'].apply(lambda x: x[2] > 0.5)]
white_sample = df[df['PI_ETHNICITY_PROBS'].apply(lambda x: x[3] > 0.5)]

# now of each, select a random sample of 20 rows
asian_sample = asian_sample.sample(n=20, random_state=42)
black_sample = black_sample.sample(n=20, random_state=42)
hispanic_sample = hispanic_sample.sample(n=20, random_state=42)
white_sample = white_sample.sample(n=20, random_state=42)

# combine the samples
combined_sample = pd.concat([random_sample, asian_sample, black_sample, hispanic_sample, white_sample])
print(combined_sample.head())
combined_sample["GT_GENDER"] = ""
combined_sample["GT_RACE"] = ""
# drop PI_ETHNICITY_PROBS
combined_sample = combined_sample.drop(columns=["PI_ETHNICITY_PROBS"])
combined_sample.to_csv('test_set.csv', index=False)