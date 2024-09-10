import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import numpy as np
from scipy import stats
df_mean_new = pd.read_csv('/home/nisse/Documents/vakken_l/Thesis/nih/originals/july/data/final_files/updated_projects_labelled_inflation_adjusted2.csv')
df_mean_new['GT_VALUE'] = pd.to_numeric(df_mean_new['GRANT_VALUE'], errors='coerce')

# Read the CSV file
df = pd.read_csv('merged_with_gt.csv')

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
def get_topn_recs(df, model_name, dataset, version, topn):
    if topn < 1 or topn > 5:
        raise ValueError("topn must be between 1 and 5")

    appval_col = f"APPVAL_RECS_{model_name}_{dataset}_{version}"
    foa_col = f"FOA_RECS_{model_name}_{dataset}_{version}"

    if appval_col not in df.columns or foa_col not in df.columns:
        raise ValueError(f"Columns for {model_name}_{dataset}_{version} not found in dataframe")

    def get_topn(series):
        return series.apply(lambda x: ','.join(x.split(',')[:topn]))

    topn_appval = get_topn(df[appval_col])
    topn_foa = get_topn(df[foa_col])

    return topn_appval, topn_foa

def get_top_1_grant(df, model_name, dataset, version):
    # Get the top 1 recommendation
    topn_appval, _ = get_topn_recs(df, model_name, dataset, version, 1)
    # Convert topn_appval to numeric, replacing any non-numeric values with NaN
    topn_appval_numeric = pd.to_numeric(topn_appval, errors='coerce')
    return topn_appval_numeric
def get_top1grant_val_for_all_models(dataset, version, gender, race):
    if gender == 'all' and race == 'all':
        filtered_df = df
    elif gender == 'all':
        filtered_df = df[df['PI_RACE'] == race]
    elif race == 'all':
        filtered_df = df[df['PI_GENDER'] == gender]
    else:
        filtered_df = df[(df['PI_GENDER'] == gender) & (df['PI_RACE'] == race)]   
    top1grantval_dict = {}
    for model in model_order:
        top1s_allPIs = get_top_1_grant(filtered_df,model, dataset, version)
        top1grantval_dict[model] = top1s_allPIs
    return top1grantval_dict
#df = df[np.abs(stats.zscore(df['GT_VALUEALUE'])) < 3]

def average_grant_top1_val(dataset, version, gender, race):
    all_vals_dict = get_top1grant_val_for_all_models(dataset, version, gender, race)
    
    # Find the maximum length of the grant values list
    max_length = max(len(values) for values in all_vals_dict.values())
    
    # Initialize a list to store the sums
    sum_vals = np.zeros(max_length)
    count_vals = np.zeros(max_length)
    
    # Sum the values at each position
    for values in all_vals_dict.values():
        for i, val in enumerate(values):
            sum_vals[i] += val
            count_vals[i] += 1
    
    # Compute the average values
    avg_vals = sum_vals / count_vals
    
    return avg_vals

# Map race and gender to full names
race_map = {'a': 'Asian', 'b': 'Black', 'h': 'Hispanic', 'w': 'White'}
gender_map = {'f': 'Female', 'm': 'Male'}

df['PI_RACE'] = df['PI_RACE'].map(race_map)
df['PI_GENDER'] = df['PI_GENDER'].map(gender_map)

# Define the order for race and gender
race_order = ['Asian', 'Black', 'Hispanic', 'White']
gender_order = ['Female', 'Male']


# Ensure GT_VALUEALUE is numeric and remove any non-numeric values
df['GT_VALUE'] = pd.to_numeric(df['GT_VALUE'], errors='coerce')
avg_grant_values_wi = average_grant_top1_val('o', 'wi','all', 'all')
avg_grant_values_w = average_grant_top1_val('o', 'w','all', 'all')

df['AVERAGE_GRANT_VALUES_WI'] = avg_grant_values_wi
df['AVERAGE_GRANT_VALUES_WI'] = pd.to_numeric(df['AVERAGE_GRANT_VALUES_WI'], errors='coerce')
df['AVERAGE_GRANT_VALUES_W'] = avg_grant_values_w
df['AVERAGE_GRANT_VALUES_W'] = pd.to_numeric(df['AVERAGE_GRANT_VALUES_W'], errors='coerce')
# Set a minimum sample size for inclusion in plots
MIN_SAMPLE_SIZE = 30  # Adjust this value as needed

# Filter out categories with small sample sizes
df_filtered = df.groupby(['PI_RACE', 'PI_GENDER']).filter(lambda x: len(x) >= MIN_SAMPLE_SIZE)

# Set up the plotting style using Seaborn
sns.set_theme(style="whitegrid")
sns.set_palette("deep")
df_grouped = df_filtered.groupby(['PI_RACE', 'PI_GENDER'])
df_mean_wi = df_grouped['AVERAGE_GRANT_VALUES_WI'].mean().unstack()
df_sem_wi = df_grouped['AVERAGE_GRANT_VALUES_WI'].sem().unstack()
df_mean_w = df_grouped['AVERAGE_GRANT_VALUES_W'].mean().unstack()
df_sem_w = df_grouped['AVERAGE_GRANT_VALUES_W'].sem().unstack()

# Create a figure with two subplots
# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# 1. Bar plot with error bars
df_mean_wi.plot(kind='bar', yerr=df_sem_wi, capsize=12, ax=ax1)
ax1.set_xlabel('PI Race', fontsize=28)
ax1.set_ylabel('Mean Grant Value of Average Top-1', fontsize=24)
ax1.legend(title='PI Gender', fontsize=22, title_fontsize=24)

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
# Update df_mean with new data
df_mean = df_mean_new.groupby(['PI_RACE', 'PI_GENDER'])['GT_VALUE'].mean().unstack()
df_sem = df_mean_new.groupby(['PI_RACE', 'PI_GENDER'])['GT_VALUE'].sem().unstack()

#df_mean = df_grouped['GT_VALUE'].mean().unstack()
#df_sem = df_grouped['GT_VALUE'].sem().unstack()
df_mean.plot(kind='bar', yerr=df_sem, capsize=12, ax=ax2)
ax2.set_xlabel('PI Race', fontsize=28)
ax2.set_ylabel('Mean Grant Value in Dataset', fontsize=24)
ax2.legend(title='PI Gender', fontsize=22, title_fontsize=24)

# Increase font size of the bar plot labels
ax2.tick_params(axis='x', labelsize=22)
ax2.tick_params(axis='y', labelsize=22)

# Add sample sizes to the plot with increased font size
for i, race in enumerate(race_order):
    for j, gender in enumerate(gender_order):
        n = len(df_filtered[(df_filtered['PI_RACE'] == race) & (df_filtered['PI_GENDER'] == gender)])
        ax2.text(i + j * 0.2, ax2.get_ylim()[1] * 1.01, f'n={n}', horizontalalignment='center', fontsize=22, color='black', weight='semibold', rotation=45)

# Format y-axis to show absolute amounts
ax2.get_yaxis().set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))


# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('combined_grant_value_plots_o.png')
plt.close()


# Set up the plotting style using Seaborn
sns.set_theme(style="whitegrid")
sns.set_palette("deep")

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 10))

# Set width of bars
barWidth = 0.35
df_mean_pred_w = df_grouped['AVERAGE_GRANT_VALUES_W'].mean().unstack()
# Set positions of the bars on X axis
r = np.arange(len(df_mean_pred_w))

# Define hatching patterns
hatch_patterns = ['', '//']

# Function to create bars for each subplot
def create_bars(ax, df_mean_pred, df_sem_pred, df_mean_gt, include_legend=False):
    for i, gender in enumerate(gender_order):
        # Ground Truth bars (wider, with hatching)
        ax.bar(r + i*barWidth, df_mean_gt[gender], width=barWidth, 
               label=f'{gender} (Ground Truth)', color=f'C{i}', hatch=hatch_patterns[0])
        
        # Predicted bars (narrower, with different hatching and error bars)
        ax.bar(r + i*barWidth, df_mean_pred[gender], width=barWidth*0.6, yerr=df_sem_pred[gender], 
               capsize=5, label=f'{gender} (Predicted)', color=f'C{i}', hatch=hatch_patterns[1])

    # Add labels
    ax.set_xlabel('PI Race', fontsize=28)
    ax.set_ylabel('Mean Grant Value', fontsize=24)

    # Add xticks on the middle of the group bars
    ax.set_xticks(r + barWidth/2)
    ax.set_xticklabels(race_order)

    # Increase font size of the plot labels
    ax.tick_params(axis='x', labelsize=22)
    ax.tick_params(axis='y', labelsize=22)

    # Create legend only for the first subplot
    if include_legend:
        ax.legend(fontsize=18, title='PI Gender and Type', title_fontsize=20)

    # Format y-axis to show absolute amounts
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))

# Create bars for the first subplot (AVERAGE_GRANT_VALUES_W)
df_grouped = df_filtered.groupby(['PI_RACE', 'PI_GENDER'])

df_mean_gt = df_grouped['GT_VALUE'].mean().unstack()

# Create bars for the second subplot (AVERAGE_GRANT_VALUES_WI)
df_mean_pred_wi = df_grouped['AVERAGE_GRANT_VALUES_WI'].mean().unstack()
df_sem_pred_wi = df_grouped['AVERAGE_GRANT_VALUES_WI'].sem().unstack()
create_bars(ax1, df_mean_pred_wi, df_sem_pred_wi, df_mean_gt, include_legend=True)

# Create bars for the first subplot (AVERAGE_GRANT_VALUES_W)
df_mean_pred_w = df_grouped['AVERAGE_GRANT_VALUES_W'].mean().unstack()
df_sem_pred_w = df_grouped['AVERAGE_GRANT_VALUES_W'].sem().unstack()

create_bars(ax2, df_mean_pred_w, df_sem_pred_w, df_mean_gt, include_legend=False)

# Ensure y-axis has the same scaling on both plots
max_y_value = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
ax1.set_ylim(0, max_y_value)
ax2.set_ylim(0, max_y_value)

# Add sample sizes to the plot above the axes
for ax in [ax1, ax2]:
    for i, race in enumerate(race_order):
        for j, gender in enumerate(gender_order):
            n = len(df_filtered[(df_filtered['PI_RACE'] == race) & (df_filtered['PI_GENDER'] == gender)])
            ax.text(i + j * barWidth, max_y_value * 1.02, f'n={n}', 
                    horizontalalignment='center', fontsize=18, color='black', weight='semibold', rotation=45)

# Add titles below the plots
ax1.text(0.5, -0.15, '(a) Models fine-tuned without PI names', transform=ax1.transAxes, 
         ha='center', va='center', fontsize=28)
ax2.text(0.5, -0.15, '(b) Models fine-tuned with PI names', transform=ax2.transAxes, 
         ha='center', va='center', fontsize=28)

# Adjust layout and save the figure
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)  # Make room for the titles below the plots
plt.savefig('comparison_grant_value_plots_o.eps', bbox_inches='tight')
plt.show()

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
race_stats = df.groupby('PI_RACE')['AVERAGE_GRANT_VALUES_WI'].agg([
    ('count', 'count'),
    ('median', 'median'),
    ('mean', 'mean'),
    ('min', 'min'),
    ('max', 'max'),
    ('mode', calculate_mode)
])

# Calculate descriptive statistics for each gender
gender_stats = df.groupby('PI_GENDER')['AVERAGE_GRANT_VALUES_WI'].agg([
    ('count', 'count'),
    ('median', 'median'),
    ('mean', 'mean'),
    ('min', 'min'),
    ('max', 'max'),
    ('mode', calculate_mode)
])

# Calculate descriptive statistics for each gender within race
gender_race_stats = df.groupby(['PI_RACE', 'PI_GENDER'])['AVERAGE_GRANT_VALUES_WI'].agg([
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
df_mean = df_filtered.groupby(['PI_RACE', 'PI_GENDER'])['AVERAGE_GRANT_VALUES_WI'].mean().unstack()
df_mean = df_mean.reindex(index=race_order, columns=gender_order)

# Calculate mean grant value for white males
mean_white_male = df_mean.loc['White', 'Male']

# Calculate the factor relative to white males
factor_relative_to_white_male = df_mean / mean_white_male

# Print the factors
print("Factors of Grant Value Relative to White Males by PI Race and Gender:")
print(factor_relative_to_white_male)

# print the smallest and largest AVERAGE_GRANT_VALUESALUE
print(df['AVERAGE_GRANT_VALUES_WI'].min())
print(df['AVERAGE_GRANT_VALUES_WI'].max())












