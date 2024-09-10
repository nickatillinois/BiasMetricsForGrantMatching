# read ner_output.csv into a pandas dataframe
# this has columns Grant Number,Content,cash,years,cash_formatted
# only keep rows where Grant Number also appears in combined_file_with_titles.csv and grant_numbers_all_sections_and_titels.csv

# then, join with combined_file_all_sections_and_titles.csv on Grant Number
# and save the resulting dataframe to a csv file named "processed_grants.csv"

import pandas as pd

# read the ner_output.csv file into a pandas dataframe

ner_output = pd.read_csv("ner_output.csv")

# print the number of rows 

print(ner_output.shape[0])

# print the number of rows where the column cash_formatted is not empty, not null

print(ner_output[ner_output["cash_formatted"].notnull()].shape[0])

# only take rows where the column cash_formatted is not empty, not null

ner_output = ner_output[ner_output["cash_formatted"].notnull()]

# rename column "Content" to "sectionii"

ner_output = ner_output.rename(columns={"Content": "sectionii"})

# read the combined_file_with_titles.csv file into a pandas dataframe

combined_file_with_titles = pd.read_csv("combined_file_with_titles.csv")
print(combined_file_with_titles.head())

# read the grant_numbers_all_sections_and_titles.csv file into a pandas dataframe

grant_numbers_all_sections_and_titles = pd.read_csv("grant_numbers_all_sections_and_titles.csv")
print(grant_numbers_all_sections_and_titles.head())

# keep only rows where Grant Number also appears in combined_file_with_titles.csv and grant_numbers_all_sections_and_titels.csv

ner_output = ner_output[ner_output["Grant Number"].isin(combined_file_with_titles["Grant Number"]) & ner_output["Grant Number"].isin(grant_numbers_all_sections_and_titles["Grant Number"])]
print(ner_output.head())

# join with combined_file_all_sections_and_titles.csv on Grant Number

processed_grants = pd.merge(ner_output, combined_file_with_titles, on="Grant Number")
sample_records = processed_grants.sample(5, random_state=1)  # Random sample for demonstration

for index, row in sample_records.iterrows():
    print("Grant Number:", ' '.join(str(row["Grant Number"]).split()[:5]))
    
    # Convert row["sectionii"] to string and split
    print("sectionii:", ' '.join(str(row["sectionii"]).split()[:5]))
    
    # Convert row["cash_formatted"] to string and split
    print("cash_formatted:", ' '.join(str(row["cash_formatted"]).split()[:5]))
    
    # Convert row["years"] to string and split
    print("years:", ' '.join(str(row["years"]).split()[:5]))
    
    # Convert row["Title"] to string and split
    print("Title:", ' '.join(str(row["Title"]).split()[:5]))
    
    # Convert row["table"] to string and split
    print("table:", ' '.join(str(row["table"]).split()[:5]))
    
    # Convert row["sectioniii"] to string and split
    print("sectioniii:", ' '.join(str(row["sectioniii"]).split()[:5]))
    
    print("\n")

# save the resulting dataframe to a csv file named "processed_grants.csv"

processed_grants.to_csv("processed_grants.csv", index=False)

