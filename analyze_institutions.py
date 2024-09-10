import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv('data.csv')

# Map race and gender to full names if they're not already
race_map = {'a': 'Asian', 'b': 'Black', 'h': 'Hispanic', 'w': 'White'}
gender_map = {'f': 'Female', 'm': 'Male'}

df['PI_RACE'] = df['PI_RACE'].map(race_map).fillna(df['PI_RACE'])
df['PI_GENDER'] = df['PI_GENDER'].map(gender_map).fillna(df['PI_GENDER'])

# Calculate the overall percentages for each gender and race
overall_gender_percentage = (df['PI_GENDER'].value_counts(normalize=True) * 100).to_dict()
overall_race_percentage = (df['PI_RACE'].value_counts(normalize=True) * 100).to_dict()

# Calculate the average grant value and percentage of records per IC_NAME, race, and gender
institution_stats = []

for institution in df['IC_NAME'].unique():
    inst_data = df[df['IC_NAME'] == institution]
    avg_grant_value = inst_data['GRANT_VALUE'].mean()
    
    total_records = len(inst_data)
    race_percentage = (inst_data['PI_RACE'].value_counts(normalize=True) * 100).to_dict()
    gender_percentage = (inst_data['PI_GENDER'].value_counts(normalize=True) * 100).to_dict()
    
    # Create a dictionary for each institution with the calculated statistics
    inst_stats = {
        'Institution': institution.upper(),  # Capitalize institution names
        'Average_Grant_Value': avg_grant_value,
        'Total_Records': total_records
    }
    
    for race in ['Asian', 'Black', 'Hispanic', 'White']:
        inst_stats[f'Race_{race}_Diff'] = race_percentage.get(race, 0) - overall_race_percentage.get(race, 0)
    
    for gender in ['Female', 'Male']:
        inst_stats[f'Gender_{gender}_Diff'] = gender_percentage.get(gender, 0) - overall_gender_percentage.get(gender, 0)
    
    institution_stats.append(inst_stats)

# Convert the list of dictionaries to a DataFrame for better visualization
institution_stats_df = pd.DataFrame(institution_stats)

# Prepare data for heatmaps
gender_diff_columns = [col for col in institution_stats_df.columns if col.startswith('Gender_')]
race_diff_columns = [col for col in institution_stats_df.columns if col.startswith('Race_')]

# Include 'Average_Grant_Value' in the heatmap data
gender_diff_columns.append('Average_Grant_Value')
race_diff_columns.append('Average_Grant_Value')

# Order by Average_Grant_Value
institution_stats_df = institution_stats_df.sort_values('Average_Grant_Value', ascending=False)

# Heatmap for gender differences
plt.figure(figsize=(20, 12))  # Increased figure size
gender_heatmap_data = institution_stats_df.set_index('Institution')[gender_diff_columns]
sns.heatmap(gender_heatmap_data, annot=True, fmt='.1f', cmap='coolwarm', center=0, linewidths=.5, annot_kws={"size": 10})
plt.xticks(ticks=range(len(gender_diff_columns)), labels=['Female', 'Male', 'Av. Grant Value'], rotation=45, fontsize=28)
plt.yticks(fontsize=12)
plt.ylabel('Institution', fontsize=28)
plt.tight_layout()
plt.savefig('gender_diff_and_avg_grant_value_per_institution.eps',bbox_inches='tight')
plt.show()

# Heatmap for race differences
plt.figure(figsize=(20, 12))  # Increased figure size
race_heatmap_data = institution_stats_df.set_index('Institution')[race_diff_columns]
sns.heatmap(race_heatmap_data, annot=True, fmt='.1f', cmap='coolwarm', center=0, linewidths=.5, annot_kws={"size": 10})
plt.xticks(ticks=range(len(race_diff_columns)), labels=['Asian', 'Black', 'Hispanic', 'White', 'Av. Grant Value'], rotation=45, fontsize=28)
plt.yticks(fontsize=12)
plt.ylabel('Institution', fontsize=28)
plt.tight_layout()
plt.savefig('race_diff_and_avg_grant_value_per_institution.eps', bbox_inches='tight')
plt.show()

print("All plots have been generated and saved.")
