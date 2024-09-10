
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr
# Read the CSV file
df = pd.read_csv('intrinsic_bias.csv')
df_mean_new = pd.read_csv('/home/nisse/Documents/vakken_l/Thesis/nih/post_poster_code/july/data/final_files/updated_projects_labelled_inflation_adjusted2.csv')
df_mean_new['GT_VALUE'] = pd.to_numeric(df_mean_new['GRANT_VALUE'], errors='coerce')


model_order = ["bert-base", "roberta", "xlm-roberta", "distilbert", "albert", "spanbert",
               "deberta", "electra", "biobert", "scibert", "bluebert", "biomedbert",
               "bert-xxxxx", "bert-xxxx", "bert-xxx", "bert-xx", "bert-x", 
               "bert-s", "bert-multi"]

model_name_mapping = {
    "bert-base": "BERT-base",
    "roberta": "RoBERTa",
    "xlm-roberta": "XLM-RoBERTa",
    "distilbert": "DistilBERT",
    "albert": "ALBERT",
    "spanbert": "SpanBERT",
    "deberta": "DeBERTa",
    "electra": "ELECTRA",
    "biobert": "BioBERT",
    "scibert": "SciBERT",
    "bluebert": "BlueBERT",
    "biomedbert": "PubMedBERT",
    "bert-xxxxx": "BERT-5XS",
    "bert-xxxx": "BERT-4XS",
    "bert-xxx": "BERT-3XS",
    "bert-xx": "BERT-2XS",
    "bert-x": "BERT-XS",
    "bert-s": "BERT-S",
    "bert-multi": "Multilingual BERT"
}
# Calculate mean and median for each specified group
mean_asian = df_mean_new[df_mean_new['PI_RACE'] == 'a']['GT_VALUE'].mean()
mean_black = df_mean_new[df_mean_new['PI_RACE'] == 'b']['GT_VALUE'].mean()
mean_hispanic = df_mean_new[df_mean_new['PI_RACE'] == 'h']['GT_VALUE'].mean()
mean_white = df_mean_new[df_mean_new['PI_RACE'] == 'w']['GT_VALUE'].mean()
mean_female = df_mean_new[df_mean_new['PI_GENDER'] == 'f']['GT_VALUE'].mean()
mean_male = df_mean_new[df_mean_new['PI_GENDER'] == 'm']['GT_VALUE'].mean()
mean_female_black = df_mean_new[(df_mean_new['PI_GENDER'] == 'f') & (df_mean_new['PI_RACE'] == 'b')]['GT_VALUE'].mean()
mean_female_white = df_mean_new[(df_mean_new['PI_GENDER'] == 'f') & (df_mean_new['PI_RACE'] == 'w')]['GT_VALUE'].mean()
gt_factor_mean_asian = mean_asian / mean_white
print(gt_factor_mean_asian)
# Black
gt_factor_mean_black = mean_black / mean_white
print(gt_factor_mean_black)
# Hispanic
gt_factor_mean_hispanic = mean_hispanic / mean_white
print(gt_factor_mean_hispanic)
# Female
gt_factor_mean_female = mean_female / mean_male


# Function to adjust the DataFrame
def adjust_bias(df, factor_dict):
    for col in df.columns:
        if '_f_mean_norm' in col:
            df[col] -= factor_dict['f']
        elif '_a_mean_norm' in col:
            df[col] -= factor_dict['a']
        elif '_b_mean_norm' in col:
            df[col] -= factor_dict['b']
        elif '_h_mean_norm' in col:
            df[col] -= factor_dict['h']
    return df

# Apply adjustments
factors = {
    'f': gt_factor_mean_female,
    'a': gt_factor_mean_asian,
    'b': gt_factor_mean_black,
    'h': gt_factor_mean_hispanic,
}

#df = adjust_bias(df, factors)
# Define the possible values for each variable
chosen_data_options = ['o','r','p']
names_included_options = ['w','wi']
gender_or_race_options = ['f', 'm', 'a', 'b', 'h', 'w']
average_or_median_options = ['mean']
bias_type_options = ['norm']

# Generate all combinations
combinations = list(product(chosen_data_options, names_included_options, gender_or_race_options, average_or_median_options, bias_type_options))

# Store the column names and differences
column_differences = {}
column_averages = {}
column_meds = {}

# Mappings for readability
data_mapping = {
    'r': 'Realistic',
    'p': 'Balanced',
    'o': 'Original'
}

names_included_mapping = {
    'wi': 'No Names',
    'w': 'Names'
}

bias_mapping = {
    'b': 'Black/White',
    'h': 'Hispanic/White',
    'a': 'Asian/White',
    'f': 'Female/Male',
    'm': 'Male',
    'w': 'White'
}

# Function to parse and map column names
def parse_column_name(col_name):
    parts = col_name.split('_')
    data = data_mapping.get(parts[0], parts[0])
    names_included = names_included_mapping.get(parts[1], parts[1])
    bias = bias_mapping.get(parts[3], parts[3])
    average = parts[4].capitalize()
    bias_type = parts[5].capitalize()
    readable_name = f"{data} - {names_included} - {bias}"
    return readable_name

# Iterate over each combination and construct the column name
for combination in combinations:
    chosen_data, names_included, gender_or_race, average_or_median, bias_type = combination
    extrinsic_metric = f'{chosen_data}_{names_included}_bias_{gender_or_race}_{average_or_median}_{bias_type}'
    
    # Check if the column exists in the DataFrame
    if extrinsic_metric in df.columns:
        # Calculate the row differences (e.g., standard deviation as a measure of variation)
        row_diff = df[extrinsic_metric].std()
        av = df[extrinsic_metric].mean()
        med = df[extrinsic_metric].median()
        column_differences[extrinsic_metric] = row_diff
        column_averages[extrinsic_metric] = av
        column_meds[extrinsic_metric] = med

# Sort the columns by the row differences
sorted_columns = sorted(column_differences.items(), key=lambda x: x[1], reverse=True)
sorted_av = sorted(column_averages.items(), key=lambda x: x[1], reverse=True)
sorted_med = sorted(column_meds.items(), key=lambda x: x[1], reverse=True)

# Generate readable labels
readable_column_names = [parse_column_name(col) for col, _ in sorted_columns]
"""

# Create a DataFrame with readable column names, standard deviation, and average
plot_df = pd.DataFrame({'Column': readable_column_names,
                        'Standard Deviation': [diff for _, diff in sorted_columns],
                        'Average': [av for _, av in sorted_av]})

plt.figure(figsize=(10, 8))  # Adjust figure size for compactness

# Combined bar and line plot with error bars (optional)
plt.barh(plot_df['Column'], plot_df['Standard Deviation'], label='SD', alpha=0.7, color='skyblue')
plt.errorbar(plot_df['Average'], plot_df.index, yerr=None, fmt='o', label='Avg', color='red', markersize=6)  # Adjust marker size
plt.tick_params(axis='x', which='both', labelsize=20)  # Adjust as needed
plt.xlabel('NormDiff@1', fontsize=24)  # Reduce font size
plt.ylabel('Fine-Tuning Settings & Social Group', fontsize=24)
#plt.title('SD & Avg by Column', fontsize=12)  # Shortened title
plt.legend(loc='upper left', bbox_to_anchor=(0.5, 1), fontsize=18)  # Smaller legend font
plt.gca().invert_yaxis()  # Invert y-axis

# Grid with adjustments
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, which='both', axis='y')
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, which='both', axis='x') 
  # Customize grid appearance

plt.tight_layout()
plt.savefig("stdev_av.eps")
plt.show()

"""





#disco_asian,disco_black,disco_hispanic,lpbs_mean,lpbs_std,SEAT_gender,SEAT_EAvsAA,SEAT_ABW,tan_SEAT_gender,tan_SEAT_EAvsAA,tan_SEAT_ABW,lau_SEAT_gender,lau_SEAT_EAvsAA,lau_SEAT_ABW,
# Define the column name to be plotted
column_name = 'r_wi_bias_b_mean_norm'
intrinsic_bias = 'disco_black'

# Check if the column exists
if column_name not in df.columns:
    print(f'The column "{column_name}" does not exist in the DataFrame.')
else:
    # Extract the bias values from the specified column
    bias_values = df[column_name]
    
    # Model name mapping
    model_name_mapping = {
        "bert-base": "BERT-base",
        "roberta": "RoBERTa",
        "xlm-roberta": "XLM-RoBERTa",
        "distilbert": "DistilBERT",
        "albert": "ALBERT",
        "spanbert": "SpanBERT",
        "deberta": "DeBERTa",
        "electra": "ELECTRA",
        "biobert": "BioBERT",
        "scibert": "SciBERT",
        "bluebert": "BlueBERT",
        "biomedbert": "PubMedBERT",
        "bert-xxxxx": "BERT-5XS",
        "bert-xxxx": "BERT-4XS",
        "bert-xxx": "BERT-3XS",
        "bert-xx": "BERT-2XS",
        "bert-x": "BERT-XS",
        "bert-s": "BERT-S",
        "bert-multi": "Multilingual BERT"
    }
    
    # Create a list of model names based on the order of the mapping
    model_names = list(model_name_mapping.values())
    
    # Plotting
    plt.figure(figsize=(12, 8))
    plt.barh(model_names, bias_values[:len(model_names)], color='skyblue')
    plt.xlabel('NormDiff@1')
    #plt.title(f'Bias Scores for {column_name} across Different Models')
    plt.tight_layout()
    plt.show()

















# Modify the bias_mapping to ensure we have all types
bias_mapping = {
    'b': 'Black/White',
    'h': 'Hispanic/White',
    'a': 'Asian/White',
    'f': 'Female/Male'
}
# Filter out only the relevant columns from the DataFrame
columns_of_interest = []
for combination in combinations:
    chosen_data, names_included, gender_or_race, average_or_median, bias_type = combination
    extrinsic_metric = f'{chosen_data}_{names_included}_bias_{gender_or_race}_{average_or_median}_{bias_type}'
    if extrinsic_metric in df.columns:
        columns_of_interest.append(extrinsic_metric)

# Subset the DataFrame to include only the relevant columns
df_subset = df[columns_of_interest]

# Melt the DataFrame to have a long-form structure
df_melted = pd.melt(df_subset)

# Parse the melted column names to extract the group identifiers
df_melted['Data'] = df_melted['variable'].apply(lambda x: x.split('_')[0])
df_melted['Names'] = df_melted['variable'].apply(lambda x: x.split('_')[1])
df_melted['Bias'] = df_melted['variable'].apply(lambda x: x.split('_')[3])

# Apply the mappings
df_melted['Data'] = df_melted['Data'].map(data_mapping)
df_melted['Names'] = df_melted['Names'].map(names_included_mapping)
df_melted['Bias'] = df_melted['Bias'].map(bias_mapping)

# First, let's check the data
print(df_melted.head())
print(df_melted['Bias'].unique())
print(df_melted['Data'].unique())
print(df_melted['Names'].unique())
print(plt.style.available)

plt.style.use('seaborn-v0_8-whitegrid')

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 22), sharex=True)

#fig.subplots_adjust(bottom=0.2, top=0.93, wspace=0, hspace=1.5)

# Color palette
colors = sns.color_palette("deep", 3)

# Plot 1: Comparing different data types
sns.barplot(x='Bias', y='value', hue='Data', data=df_melted, ax=ax1, palette=colors)
ax1.set_ylabel('NormDiff@1', fontsize=28)
ax1.legend(title='Data Type', title_fontsize=24, fontsize=20, loc='upper right')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.tick_params(axis='y', labelsize=24)
 
#ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')  # Rotate and align labels
ax1.set_xticklabels([])
#ax1.spines['right'].set_visible(False)
#ax1.spines['top'].set_visible(False)
# Annotate each bias type outside the plot area
#bias_types = df_melted['Bias'].unique()
#bias_positions = df_melted['Bias'].value_counts().index
##for i, bias in enumerate(bias_types):
 ##   y_position = ax1.get_ylim()[1] + 0.5  # Position above the plot area
 #   ax1.text(i, y_position, bias, ha='center', fontsize=12, va='bottom')

# Plot 2: Comparing With and Without PI Names
sns.barplot(x='Bias', y='value', hue='Names', data=df_melted, ax=ax2, palette=colors[:2])
ax2.set_ylabel('NormDiff@1', fontsize=28)
ax2.legend(title='PI Names', title_fontsize=24, fontsize=20, loc='upper right')
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.tick_params(axis='y', labelsize=24)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')  # Rotate and align labels
ax2.set_xlabel('')
#ax2.spines['right'].set_visible(False)
#ax2.spines['top'].set_visible(False)
# Annotate each bias type outside the plot area
bias_types = df_melted['Bias'].unique()
x_positions = range(len(bias_types))
ax1.set_position([0.1, 0.55, 0.8, 0.4])  # [left, bottom, width, height]
ax2.set_position([0.1, 0.05, 0.8, 0.4])
# Positioning the labels midway between the two plots
bias_types = df_melted['Bias'].unique()
x_ticks = ax2.get_xticks()
for i, bias in enumerate(bias_types):
    x_position = (x_ticks[i] - x_ticks[0]) / (x_ticks[-1] - x_ticks[0])
    fig.text(0.1 + x_position * 0.8, 0.5, bias, ha='center', fontsize=28, va='center')

# Add subplot titles
fig.text(0.5, 0.96, '(a) Fine-Tuning with Different Social Distributions', ha='center', fontsize=24, fontweight='bold')
fig.text(0.5, 0.46, '(b) Fine-Tuning With & Without PI Names', ha='center', fontsize=24, fontweight='bold')
# Adjust layout and add main title
#plt.tight_layout()


plt.savefig("overall_data_name_effect_norm.svg")
plt.show()
