import pandas as pd


genders = pd.read_csv('genders.csv', usecols=["APPLICATION_ID","ACTIVITY","ADMINISTERING_IC","APPLICATION_TYPE","ARRA_FUNDED","AWARD_NOTICE_DATE","BUDGET_START","BUDGET_END","CFDA_CODE","CORE_PROJECT_NUM","ED_INST_TYPE","FOA_NUMBER","FULL_PROJECT_NUM","SUBPROJECT_ID","FUNDING_ICs","FY","IC_NAME","NIH_SPENDING_CATS","ORG_CITY","ORG_COUNTRY","ORG_DEPT","ORG_DISTRICT","ORG_DUNS","ORG_FIPS","ORG_NAME","ORG_STATE","ORG_ZIPCODE","PHR","PI_IDS","PI_NAMEs","PROGRAM_OFFICER_NAME","PROJECT_START","PROJECT_END","PROJECT_TERMS","PROJECT_TITLE","SERIAL_NUMBER","STUDY_SECTION","STUDY_SECTION_NAME","SUFFIX","SUPPORT_YEAR","TOTAL_COST","TOTAL_COST_SUB_PROJECT","ABSTRACT_TEXT","PI_LAST_NAME","PI_FIRST_NAME","PI_GENDER"])

print(genders.head())

races = pd.read_csv('gave_races2.csv', usecols=["APPLICATION_ID","PI_LAST_NAME","PI_FIRST_NAME","PI_ETHNICITY_PROBS"])

print(races.head())

# for every APPLICATION_ID in races, add a new column named "PI_GENDER"
# and set the value to the corresponding value in column 'PI_GENDER' in genders
# if there is no corresponding value, print "no value in genders'
import pandas as pd

# Read the CSV files
genders = pd.read_csv('genders.csv', usecols=["APPLICATION_ID","ACTIVITY","ADMINISTERING_IC","APPLICATION_TYPE","ARRA_FUNDED","AWARD_NOTICE_DATE","BUDGET_START","BUDGET_END","CFDA_CODE","CORE_PROJECT_NUM","ED_INST_TYPE","FOA_NUMBER","FULL_PROJECT_NUM","SUBPROJECT_ID","FUNDING_ICs","FY","IC_NAME","NIH_SPENDING_CATS","ORG_CITY","ORG_COUNTRY","ORG_DEPT","ORG_DISTRICT","ORG_DUNS","ORG_FIPS","ORG_NAME","ORG_STATE","ORG_ZIPCODE","PHR","PI_IDS","PI_NAMEs","PROGRAM_OFFICER_NAME","PROJECT_START","PROJECT_END","PROJECT_TERMS","PROJECT_TITLE","SERIAL_NUMBER","STUDY_SECTION","STUDY_SECTION_NAME","SUFFIX","SUPPORT_YEAR","TOTAL_COST","TOTAL_COST_SUB_PROJECT","ABSTRACT_TEXT","PI_LAST_NAME","PI_FIRST_NAME","PI_GENDER"])

print("Genders columns:", genders.columns)
print(genders.head())

races = pd.read_csv('gave_races2.csv', usecols=["APPLICATION_ID","PI_LAST_NAME","PI_FIRST_NAME","PI_ETHNICITY_PROBS"])

print("Races columns:", races.columns)
print(races.head())

# Merge the two DataFrames on the APPLICATION_ID column
merged_df = races.merge(genders[['APPLICATION_ID', 'PI_GENDER']], on='APPLICATION_ID', how='left')

# only keep the columns we need: APPLICATION_ID, PI_LAST_NAME, PI_FIRST_NAME, PI_ETHNICITY_PROBS, PI_GENDER
merged_df = merged_df[['APPLICATION_ID', 'PI_LAST_NAME', 'PI_FIRST_NAME', 'PI_ETHNICITY_PROBS', 'PI_GENDER']]

# Count the number of NaN values in the PI_GENDER column
nan_count = merged_df['PI_GENDER'].isna().sum()
print(f"Number of NaN values in PI_GENDER: {nan_count}")

# Display the first few rows of the merged DataFrame
print(merged_df.head())
