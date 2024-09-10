import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
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
#print("adjusted")
# Define the possible values for each variable
chosen_data_options = ['o', 'r', 'p']
names_included_options = ['w', 'wi']
gender_or_race_options = ['f', 'm', 'a', 'b', 'h', 'w']
average_or_median_options = ['mean']
bias_type_options = ['norm']

# Generate all combinations
combinations = list(product(chosen_data_options, names_included_options, gender_or_race_options, average_or_median_options, bias_type_options))


# Function to parse and map column names (improved clarity)
def parse_column_name(col_name):
    parts = col_name.split('_')
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
    data = data_mapping.get(parts[0], parts[0])
    names_included = names_included_mapping.get(parts[1], parts[1])
    bias = bias_mapping.get(parts[3], parts[3])
    average = parts[4].capitalize()
    bias_type = parts[5].capitalize()
    readable_name = f"{data} - {names_included} - {bias}"
    return readable_name

# Calculate boxplot data (improved organization)
boxplot_data = {}  # Dictionary to store boxplot data by label
for combination in combinations:
    chosen_data, names_included, gender_or_race, average_or_median, bias_type = combination
    extrinsic_metric = f'{chosen_data}_{names_included}_bias_{gender_or_race}_{average_or_median}_{bias_type}'
    

    if extrinsic_metric in df.columns:
        print(extrinsic_metric)
        # Calculate standard deviation (assuming this is the desired measure for boxplots)
        data = df[extrinsic_metric]
        boxplot_data.setdefault(parse_column_name(extrinsic_metric), []).extend(data.tolist())

# Create the boxplot with the specified figure layout
plt.figure(figsize=(10, 8))  # Adjust figure size for compactness

# Horizontal boxplots (consistent with prompt)
boxplot = plt.boxplot(
    [data for label, data in boxplot_data.items()],
    labels=[label for label in boxplot_data.keys()],
    vert=False,
    patch_artist=True
)
# Customize boxplot appearance (optional)
for patch in boxplot['boxes']:
    patch.set_facecolor('skyblue')  # Adjust box color
    patch.set_alpha(0.7)

for median_line in boxplot['medians']:
    median_line.set_linewidth(3)  # Increase line width for median line
    median_line.set_color('darkred') 
# Customize boxplot appearance (optional)
for patch in boxplot['boxes']:
    patch.set_facecolor('skyblue')  # Adjust box color
    patch.set_alpha(0.7)
plt.axvline(x=1, color='orange', linestyle='--', linewidth=2)  # Customize color, linestyle, and linewidth

# X-axis label and adjustments (combined standard deviation and average)
plt.xlabel('NormDiff@1', fontsize=24)
plt.ylabel('Fine-Tuning Settings & Social Group', fontsize=24)
plt.tick_params(axis='x', which='both', labelsize=20)
# Inverted y-axis and grid
plt.gca().invert_yaxis()  # Invert y-axis for better readability
plt.grid(True, linestyle='--', linewidth=1, alpha=0.7, which='both', axis='y')
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, which='both', axis='x')   # Grid for both axes
# Draw horizontal lines every 4 entries
num_entries = len(boxplot_data)
for i in range(5, num_entries, 4):
    plt.axhline(y=i-0.5, color='red', linestyle='-', linewidth=2.5)  # Customize color, linestyle, and linewidth
for i in range(9, num_entries, 8):
    plt.axhline(y=i-0.5, color='black', linestyle='-', linewidth=6)  # Customize color, linestyle, and linewidth
# Tight layout
plt.subplots_adjust(left=0.267, bottom=0.01, right=0.867, top=0.98)
plt.tight_layout()

# Save the figure (optional)
plt.savefig("stdev_boxplot_norm.svg")

# Display the plot
plt.show()
