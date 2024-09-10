import pyethnicity
import pandas as pd
import os
df = pd.read_csv('genderized1_corrected.csv', usecols=['APPLICATION_ID', 'PI_FIRST_NAME', 'PI_LAST_NAME'])
# add a new column and set it to empty dictionary
df['PI_ETHNICITY_PROBS'] = ""



# Create an empty DataFrame to store non-matching records
non_matching_records = []

for index, row in df.iterrows():
    if index % 100 == 0:
        print(f"Processing row {index} of {df.shape[0]}")
    first_name = row['PI_FIRST_NAME']
    last_name = row['PI_LAST_NAME']
    # put to lower case
    first_name, last_name = first_name.lower(), last_name.lower()
    
    # Call pyethnicity.predict_race_fl() for each record
    predicted_race = pyethnicity.predict_race_fl(first_name, last_name)
    #predicted_race_max_prob = predicted_race[['asian', 'black', 'hispanic', 'white']].idxmax(axis=1).iloc[0]
    #print("the maxprob is: ", predicted_race[[predicted_race_max_prob]])
    # make a dictionary of the probabilities with 'asian', 'black', 'hispanic', 'white' as keys
    prob_asian = predicted_race[['asian']].values[0][0]
    prob_black = predicted_race[['black']].values[0][0]
    prob_hispanic = predicted_race[['hispanic']].values[0][0]
    prob_white = predicted_race[['white']].values[0][0]
    prob_dict = {'asian': prob_asian, 'black': prob_black, 'hispanic': prob_hispanic, 'white': prob_white}
    row['PI_ETHNICITY_PROBS'] = prob_dict
    #prob = predicted_race[[predicted_race_max_prob]].values[0][0]
    #print("---------------------------------")
    #print("the probability of this race is", prob)
    #print("---------------------------------")

   

    #if prob >= 0.95:
     #   row['PI_ETHNICITY'] = predicted_race_max_prob
    #else:
    #    row['PI_ETHNICITY'] = "unknown"
    df.loc[index] = row

    
    # Add the predicted race as a column in the DataFrame
    #row['pyethnicity'] = predicted_race_max_prob

df.to_csv('gave_races2.csv', index=False)