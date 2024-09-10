import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import numpy as np
from scipy import stats

# Read the CSV file
df = pd.read_csv('data/unique_1.csv')
print(df.head())

# Calculate the 33rd and 66th percentiles
small_threshold = df['GRANT_VALUE'].quantile(0.33)
medium_threshold = df['GRANT_VALUE'].quantile(0.66)
# Define the categories based on these thresholds
def categorize_grant(value):
    if value <= small_threshold:
        return 'Small'
    elif value <= medium_threshold:
        return 'Medium'
    else:
        return 'Large'

# Apply the categorization
df['GRANT_CATEGORY'] = df['GRANT_VALUE'].apply(categorize_grant)

# Print the thresholds for reference
print(f"Small grants: <= {small_threshold}")
print(f"Medium grants: > {small_threshold} and <= {medium_threshold}")
print(f"Large grants: > {medium_threshold}")



#df = df[np.abs(stats.zscore(df['GRANT_VALUE'])) < 3]


# Ensure GRANT_VALUE is numeric and remove any non-numeric values
df['GRANT_VALUE'] = pd.to_numeric(df['GRANT_VALUE'], errors='coerce')


# Map race and gender to full names
race_map = {'a': 'Asian', 'b': 'Black', 'h': 'Hispanic', 'w': 'White'}
gender_map = {'f': 'Female', 'm': 'Male'}

df['PI_RACE'] = df['PI_RACE'].map(race_map)
df['PI_GENDER'] = df['PI_GENDER'].map(gender_map)

# Define the order for race and gender
race_order = ['Asian', 'Black', 'Hispanic', 'White']
gender_order = ['Female', 'Male']

# Set a minimum sample size for inclusion in plots
MIN_SAMPLE_SIZE = 30  # Adjust this value as needed

# Filter out categories with small sample sizes
df_filtered = df.groupby(['PI_RACE', 'PI_GENDER']).filter(lambda x: len(x) >= MIN_SAMPLE_SIZE)

# Set up the plotting style using Seaborn
sns.set_theme(style="whitegrid")
sns.set_palette("deep")
df_grouped = df_filtered.groupby(['PI_RACE', 'PI_GENDER'])
df_mean = df_grouped['GRANT_VALUE'].mean().unstack()
df_sem = df_grouped['GRANT_VALUE'].sem().unstack()

# Create a figure with two subplots
# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# 1. Bar plot with error bars
df_mean.plot(kind='bar', yerr=df_sem, capsize=12, ax=ax1)
ax1.set_xlabel('PI-Race', fontsize=28)
ax1.set_ylabel('Mean Grant Value', fontsize=28)
ax1.legend(title='PI-Gender', fontsize=22, title_fontsize=28)

# Increase font size of the bar plot labels
ax1.tick_params(axis='x', labelsize=22)
ax1.tick_params(axis='y', labelsize=22)

# Add sample sizes to the plot with increased font size
for i, race in enumerate(race_order):
    for j, gender in enumerate(gender_order):
        n = len(df_filtered[(df_filtered['PI_RACE'] == race) & (df_filtered['PI_GENDER'] == gender)])
        ax1.text(i + j * 0.2, ax1.get_ylim()[1] * 1.01, f'n={n}', horizontalalignment='center', fontsize=22, color='black', weight='semibold', rotation=45)

# Format y-axis to show absolute amounts
ax1.get_yaxis().set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))

# 2. Heatmap with sample sizes
df_median = df_filtered.groupby(['PI_RACE', 'PI_GENDER'])['GRANT_VALUE'].median().unstack()
df_median = df_median.reindex(index=race_order, columns=gender_order)
sns.heatmap(df_median, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax2, annot_kws={"size": 22})

# Increase font size of the heatmap labels
ax2.set_xticklabels(ax2.get_xticklabels(), fontsize=22)
ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=22)

# Add sample sizes to the heatmap with increased font size
for i, race in enumerate(race_order):
    for j, gender in enumerate(gender_order):
        n = len(df_filtered[(df_filtered['PI_RACE'] == race) & (df_filtered['PI_GENDER'] == gender)])
        if n >= MIN_SAMPLE_SIZE:
            ax2.text(j + 0.5, i + 0.85, f'n={n}', ha='center', va='center', color='black', fontsize=22)

ax2.set_xlabel('PI-Gender', fontsize=28)
ax2.set_ylabel('PI-Race', fontsize=28)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('combined_grant_value_plots_RSset.png')
plt.close()



print("All plots have been generated and saved.")

# Function to calculate mode (most frequent value)
def calculate_mode(x):
    mode_result = stats.mode(x)
    if hasattr(mode_result, 'mode'):
        # For older scipy versions
        if isinstance(mode_result.mode, np.ndarray):
            return mode_result.mode[0] if len(mode_result.mode) > 0 else np.nan
        else:
            return mode_result.mode
    else:
        # For newer scipy versions
        return mode_result[0][0] if len(mode_result[0]) > 0 else np.nan

# Calculate descriptive statistics for each race
race_stats = df.groupby('PI_RACE')['GRANT_VALUE'].agg([
    ('count', 'count'),
    ('median', 'median'),
    ('mean', 'mean'),
    ('min', 'min'),
    ('max', 'max'),
    ('mode', calculate_mode)
])

# Calculate descriptive statistics for each gender
gender_stats = df.groupby('PI_GENDER')['GRANT_VALUE'].agg([
    ('count', 'count'),
    ('median', 'median'),
    ('mean', 'mean'),
    ('min', 'min'),
    ('max', 'max'),
    ('mode', calculate_mode)
])

# Calculate descriptive statistics for each gender within race
gender_race_stats = df.groupby(['PI_RACE', 'PI_GENDER'])['GRANT_VALUE'].agg([
    ('count', 'count'),
    ('median', 'median'),
    ('mean', 'mean'),
    ('min', 'min'),
    ('max', 'max'),
    ('mode', calculate_mode)
])

# Print the results
print("\nDescriptive Statistics by Race:")
print(race_stats)

print("\nDescriptive Statistics by Gender:")
print(gender_stats)

print("\nDescriptive Statistics by Race and Gender:")
print(gender_race_stats)

# Calculate mean grant value by PI race and gender
df_mean = df_filtered.groupby(['PI_RACE', 'PI_GENDER'])['GRANT_VALUE'].mean().unstack()
df_mean = df_mean.reindex(index=race_order, columns=gender_order)

# Calculate mean grant value for white males
mean_white_male = df_mean.loc['White', 'Male']

# Calculate the factor relative to white males
factor_relative_to_white_male = df_mean / mean_white_male

# Print the factors
print("Factors of Grant Value Relative to White Males by PI-Race and PI-Gender:")
print(factor_relative_to_white_male)

# print the smallest and largest GRANT_VALUE
print(df['GRANT_VALUE'].min())
print(df['GRANT_VALUE'].max())