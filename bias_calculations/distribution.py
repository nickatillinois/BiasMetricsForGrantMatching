import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Load the dataset
df = pd.read_csv("merged_with_gt.csv")

# Define mappings
gender_mapping = {'m': 'Male', 'f': 'Female'}
race_mapping = {'a': 'Asian', 'b': 'Black', 'h': 'Hispanic', 'w': 'White'}

# Replace abbreviations with full names
df['PI_GENDER'] = df['PI_GENDER'].map(gender_mapping)
df['PI_RACE'] = df['PI_RACE'].map(race_mapping)

# Define the order for categories
gender_order = ['Female', 'Male']
race_order = ['Asian', 'Black', 'Hispanic', 'White']

# Convert columns to categorical with specified order
df['PI_GENDER'] = pd.Categorical(df['PI_GENDER'], categories=gender_order, ordered=True)
df['PI_RACE'] = pd.Categorical(df['PI_RACE'], categories=race_order, ordered=True)

# Filter the DataFrame
df_filtered = df[(df['PI_GENDER'].isin(gender_order)) & (df['PI_RACE'].isin(race_order))]

# Distribution of PI_GENDER (Female/Male)
gender_distribution = df_filtered['PI_GENDER'].value_counts().reindex(gender_order)
# Distribution of PI_RACE (Asian/Black/Hispanic/White)
race_distribution = df_filtered['PI_RACE'].value_counts().reindex(race_order)
# Distribution of PI_GENDER within PI_RACE (relative)
gender_in_race = df_filtered.groupby('PI_RACE')['PI_GENDER'].value_counts(normalize=True).unstack().reindex(columns=gender_order)

# Set the style and context for the plots
sns.set(style="whitegrid", context="talk")

# Create subplots with gridspec to handle layout
fig = plt.figure(figsize=(16, 20))
gs = GridSpec(3, 2, width_ratios=[5, 1])  # Adjust width ratios to fit the legend

# Plot for PI Gender Distribution (Bar Chart)
ax1 = fig.add_subplot(gs[0, 0])
gender_distribution.plot(kind="bar", color=['lightcoral', 'skyblue'], ax=ax1)
ax1.set_title("Distribution of PI Gender")
ax1.set_xlabel("PI Gender")
ax1.set_ylabel("Count")
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
for p in ax1.patches:
    ax1.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='baseline', fontsize=12, color='black', xytext=(0, 5), textcoords='offset points')
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Plot for PI Race Distribution (Horizontal Bar Chart)
ax2 = fig.add_subplot(gs[1, 0])
race_distribution.plot(kind="barh", color=['gold', 'lightgreen', 'lightcoral', 'royalblue'], ax=ax2)
ax2.set_title("Distribution of PI Race")
ax2.set_xlabel("Count")
ax2.set_ylabel("PI Race")
for p in ax2.patches:
    ax2.annotate(f'{p.get_width()}', (p.get_width(), p.get_y() + p.get_height() / 2.), ha='left', va='center', fontsize=12, color='black', xytext=(5, 0), textcoords='offset points')
ax2.grid(axis='x', linestyle='--', alpha=0.7)


# Plot for PI Gender within PI Race (Stacked Bar Chart)
ax3 = fig.add_subplot(gs[2, 0])
gender_in_race.plot(kind="bar", stacked=True, color=['lightcoral', 'skyblue'], ax=ax3, legend=False)  # Disable the legend here
ax3.set_title("Relative Distribution of PI Gender within PI Race")
ax3.set_xlabel("PI Race")
ax3.set_ylabel("Proportion")
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)
for container in ax3.containers:
    labels = [f'{v.get_height():.2f}' if v.get_height() > 0 else '' for v in container]
    ax3.bar_label(container, labels=labels, label_type='center', fontsize=12, color='white')
ax3.grid(axis='y', linestyle='--', alpha=0.7)

# Create a new axis for the legend
ax_legend = fig.add_subplot(gs[:, 1], frame_on=False)  # Use the entire right column for the legend
handles, labels = ax3.get_legend_handles_labels()
ax_legend.legend(handles, labels, title="PI Gender", loc='center left', fontsize=12)
ax_legend.axis('off')  # Hide the legend axis

# Adjust layout and save the figure
plt.tight_layout(pad=4.0)  # Increase padding between plots
plt.savefig('combined_distribution_plots2.png', dpi=300)
plt.show()


# Distribution of PI_GENDER (Female/Male)
gender_distribution = df_filtered['PI_GENDER'].value_counts().reindex(gender_order)
# Distribution of PI_RACE (Asian/Black/Hispanic/White)
race_distribution = df_filtered['PI_RACE'].value_counts().reindex(race_order)

# Calculate the overall percentage of each gender within the entire population
total_count = len(df_filtered)
female_percentage = (gender_distribution['Female'] / total_count) * 100
male_percentage = (gender_distribution['Male'] / total_count) * 100

# Calculate the percentage of female and male PIs within each race as part of the entire population
female_race_distribution = df_filtered[df_filtered['PI_GENDER'] == 'Female']['PI_RACE'].value_counts().reindex(race_order)
male_race_distribution = df_filtered[df_filtered['PI_GENDER'] == 'Male']['PI_RACE'].value_counts().reindex(race_order)

female_race_percentages = (female_race_distribution / total_count) * 100
male_race_percentages = (male_race_distribution / total_count) * 100

# Print the percentages
print(f"Overall percentage of Female PIs: {female_percentage:.2f}%")
print(f"Overall percentage of Male PIs: {male_percentage:.2f}%")
print("Percentage of Female PIs by Race as part of the entire population:")
print(female_race_percentages)
print("Percentage of Male PIs by Race as part of the entire population:")
print(male_race_percentages)