# read labeled_by_model.csv into pandas df

import pandas as pd
import numpy as np

# read gt.csv into pandas df
df = pd.read_csv('labeled_by_model.csv', encoding='latin1')
#columns=['APPLICATION_ID','PI_LAST_NAME','PI_FIRST_NAME','PI_ETHNICITY_PROBS']
df = df[['APPLICATION_ID','PI_LAST_NAME','PI_FIRST_NAME','PI_ETHNICITY_PROBS']]
                 
print(df.head())

# take 300 random records, save them to a new csv file called sample.csv
sample = df.sample(n=60)
# drop PI_ETHNICITY_PROBS
sample = sample.drop('PI_ETHNICITY_PROBS', axis=1)
# do PI_LAST_NAME and PI_FIRST_NAME remove spaces and make them lowercase
sample['PI_LAST_NAME'] = sample['PI_LAST_NAME'].str.lower()
sample['PI_LAST_NAME'] = sample['PI_LAST_NAME'].str.replace(' ', '')
sample['PI_FIRST_NAME'] = sample['PI_FIRST_NAME'].str.lower()
sample['PI_FIRST_NAME'] = sample['PI_FIRST_NAME'].str.replace(' ', '')

# make new columnw GT_GENDER,GT_RACE which have 'filler' value
sample['GT_GENDER'] = 'filler'
sample['GT_RACE'] = 'filler'
sample.to_csv('sample.csv', index=False)