import pandas as pd

# Load the datasets
df_with = pd.read_csv('perfectly_sampled/with_without/with/with.csv')
df_without = pd.read_csv('perfectly_sampled/with_without/without/without.csv')

# Add a new id column for each df called "GRANT_ID"
df_with['GRANT_ID'] = range(1, len(df_with) + 1)
df_without['GRANT_ID'] = range(1, len(df_without) + 1)

# Create dictionaries of true matches
true_matches_with = {row['APPLICATION_ID']: row['GRANT_ID'] for index, row in df_with.iterrows()}
true_matches_without = {row['APPLICATION_ID']: row['GRANT_ID'] for index, row in df_without.iterrows()}

# Function to create false matches
def create_false_matches(df):
    df_shuffled = df.sample(frac=1).reset_index(drop=True)
    false_matches = []
    for index, row in df.iterrows():
        shuffled_row = df_shuffled.loc[index]
        while row['APPLICATION_ID'] == shuffled_row['APPLICATION_ID']:
            df_shuffled = df_shuffled.sample(frac=1).reset_index(drop=True)
            shuffled_row = df_shuffled.loc[index]
        false_matches.append((row['APPLICATION_ID'], shuffled_row['GRANT_ID']))
    return false_matches

# Create false matches
false_matches_with = create_false_matches(df_with)
false_matches_without = create_false_matches(df_without)

# Create dataframes for true and false matches
true_matches_with_df = df_with[['APPLICATION_ID', 'GRANT_ID']]
true_matches_with_df['label'] = 1
true_matches_without_df = df_without[['APPLICATION_ID', 'GRANT_ID']]
true_matches_without_df['label'] = 1

false_matches_with_df = pd.DataFrame(false_matches_with, columns=['APPLICATION_ID', 'GRANT_ID'])
false_matches_with_df['label'] = 0
false_matches_without_df = pd.DataFrame(false_matches_without, columns=['APPLICATION_ID', 'GRANT_ID'])
false_matches_without_df['label'] = 0

# Combine true and false matches
combined_with_df = pd.concat([true_matches_with_df, false_matches_with_df]).sample(frac=1).reset_index(drop=True)
combined_without_df = pd.concat([true_matches_without_df, false_matches_without_df]).sample(frac=1).reset_index(drop=True)

# Merge with descriptions to replace IDs with descriptions
combined_with_df = combined_with_df.merge(df_with[['APPLICATION_ID', 'PROJECT_DESCRIPTION']], on='APPLICATION_ID')
combined_with_df = combined_with_df.merge(df_with[['GRANT_ID', 'GRANT_DESCRIPTION']], on='GRANT_ID')
#combined_with_df = combined_with_df[['PROJECT_DESCRIPTION', 'GRANT_DESCRIPTION', 'label']]

combined_without_df = combined_without_df.merge(df_without[['APPLICATION_ID', 'PROJECT_DESCRIPTION']], on='APPLICATION_ID')
combined_without_df = combined_without_df.merge(df_without[['GRANT_ID', 'GRANT_DESCRIPTION']], on='GRANT_ID')
#combined_without_df = combined_without_df[['PROJECT_DESCRIPTION', 'GRANT_DESCRIPTION', 'label']]

# Save to CSV files
combined_with_df.to_csv('perfectly_sampled/with_without/with/combined_with_descriptions.csv', index=False)
combined_without_df.to_csv('perfectly_sampled/with_without/without/combined_without_descriptions.csv', index=False)

print("Saved combined datasets with descriptions and labels.")
