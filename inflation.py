# custom inflation rates for each year from https://www.usinflationcalculator.com/
inflation_rates = {
    1985: 2.92, 1986: 2.87, 1987: 2.77, 1988: 2.66, 1989: 2.53,
    1990: 2.40, 1991: 2.31, 1992: 2.24, 1993: 2.17, 1994: 2.12,
    1995: 2.06, 1996: 2.00, 1997: 1.96, 1998: 1.93, 1999: 1.89,
    2000: 1.82, 2001: 1.78, 2002: 1.75, 2003: 1.71, 2004: 1.66,
    2005: 1.61, 2006: 1.56, 2007: 1.52, 2008: 1.46, 2009: 1.46,
    2010: 1.44, 2011: 1.40, 2012: 1.37, 2013: 1.35, 2014: 1.33,
    2015: 1.33, 2016: 1.31, 2017: 1.25, 2018: 1.25, 2019: 1.23,
    2020: 1.23, 2021: 1.16, 2022: 1.07, 2023: 1.03
}
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV files
df = pd.read_csv('processed_grants.csv')

df_year = pd.read_csv('processed_projects_labelled.csv')

# Read the CSV file

df = pd.read_csv('processed_grants.csv')
df_year = pd.read_csv('processed_projects_labelled.csv')
# print size of df
print(df_year.shape[0])
# merge df_year on 'Grant Number' in df and 'FOA_NUMBER' in df_year, only keep rows where the Grant Number is in both df and df_year
# Perform the merge
merged_df = pd.merge(df, df_year, left_on='Grant Number', right_on='FOA_NUMBER')

# Print size of merged_df
print(f'Size of merged_df after merge: {merged_df.shape[0]}')

merged_df['GRANT_VALUE'] = merged_df.apply(lambda x: x['cash_formatted'] * inflation_rates[x['FY']], axis=1)
# remove column 'Grant Number' from merged_df
merged_df.drop('Grant Number', axis=1, inplace=True)
merged_df.drop('cash', axis=1, inplace=True)
merged_df.drop('years', axis=1, inplace=True)
# print average of 'cash_formatted' in merged_df
print(merged_df['cash_formatted'].mean())
print(merged_df['GRANT_VALUE'].mean())
merged_df.drop('cash_formatted', axis=1, inplace=True)
# rename column "sectionii" to "GRANT_SECTIONII"
merged_df.rename(columns={'sectionii': 'GRANT_SECTIONII'}, inplace=True)
merged_df.rename(columns={'sectioniii': 'GRANT_SECTIONIII'}, inplace=True)
merged_df.rename(columns={'table': 'GRANT_OVERVIEW'}, inplace=True)
merged_df.rename(columns={'Title': 'GRANT_TITLE'}, inplace=True)

desired_order = [
    'APPLICATION_ID', 'ACTIVITY', 'ADMINISTERING_IC', 'APPLICATION_TYPE', 'ARRA_FUNDED', 'AWARD_NOTICE_DATE',
    'BUDGET_START', 'BUDGET_END', 'CFDA_CODE', 'CORE_PROJECT_NUM', 'ED_INST_TYPE', 'FOA_NUMBER', 'FULL_PROJECT_NUM',
    'SUBPROJECT_ID', 'FUNDING_ICs', 'FY', 'IC_NAME', 'NIH_SPENDING_CATS', 'ORG_CITY', 'ORG_COUNTRY', 'ORG_DEPT',
    'ORG_DISTRICT', 'ORG_DUNS', 'ORG_FIPS', 'ORG_NAME', 'ORG_STATE', 'ORG_ZIPCODE', 'PHR', 'PI_IDS', 'PI_NAMEs',
    'PROGRAM_OFFICER_NAME', 'PROJECT_START', 'PROJECT_END', 'PROJECT_TERMS', 'PROJECT_TITLE', 'SERIAL_NUMBER',
    'STUDY_SECTION', 'STUDY_SECTION_NAME', 'SUFFIX', 'SUPPORT_YEAR', 'TOTAL_COST', 'TOTAL_COST_SUB_PROJECT',
    'ABSTRACT_TEXT', 'PI_LAST_NAME', 'PI_FIRST_NAME', 'PI_RACE', 'PI_GENDER', 'GRANT_OVERVIEW', 'GRANT_TITLE',
    'GRANT_SECTIONII', 'GRANT_SECTIONIII', 'GRANT_VALUE'
]

# Reorder the columns in df
merged_df = merged_df[desired_order]
merged_df = merged_df[merged_df['PI_RACE'].isin(['w','b','a','h'])]
merged_df = merged_df[merged_df['PI_GENDER'].isin(['m','f'])]
# save as a new csv file: 'processed_grants_labelled_inflation_adjusted.csv'
merged_df.to_csv('processed_projects_labelled_inflation_adjusted.csv', index=False)
print(merged_df.shape[0])


