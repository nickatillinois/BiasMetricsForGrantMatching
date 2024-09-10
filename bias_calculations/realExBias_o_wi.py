import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import numpy as np
from scipy import stats
import statistics

# Read the CSV file
df = pd.read_csv('merged_with_gt.csv')
chosen_data = 'p'
names_included = 'w'
df['GT_VALUE'] = pd.to_numeric(df['GT_VALUE'], errors='coerce')
df_mean_new = pd.read_csv('/home/nisse/Documents/vakken_l/Thesis/nih/originals/july/data/final_files/updated_projects_labelled_inflation_adjusted2.csv')
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

# Map race and gender to full names
race_map = {'a': 'Asian', 'b': 'Black', 'h': 'Hispanic', 'w': 'White'}
gender_map = {'f': 'Female', 'm': 'Male'}

df['PI_RACE'] = df['PI_RACE'].map(race_map)
df['PI_GENDER'] = df['PI_GENDER'].map(gender_map)

# Define the order for race and gender
race_order = ['Asian', 'Black', 'Hispanic', 'White']
gender_order = ['Female', 'Male']

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
    #max_length = max(len(values) for values in all_vals_dict.values())
    
    # Initialize a list to store the sums
    #sum_vals = np.zeros(max_length)
    #count_vals = np.zeros(max_length)
    
    # Sum the values at each position
    #for values in all_vals_dict.values():
    #    for i, val in enumerate(values):
    #        sum_vals[i] += val
    #        count_vals[i] += 1
    
    # Compute the average values
    #avg_vals = sum_vals / count_vals
    return_list_mean = []
    return_list_median = []
    #return avg_vals
    for values in all_vals_dict.values():
        return_list_mean.append(values.mean())
    for values in all_vals_dict.values():
        return_list_median.append(values.mean())
    #print(return_list)
    #exit(1)
    return return_list_mean, return_list_median

def update_column_content(df, chosen_data, names_included, bias_values):
    # Define column prefix based on chosen_data
    prefix = f'{chosen_data}_{names_included}_'

    # Define the new columns and their corresponding bias values
    columns_to_update = {
        'bias_a_mean_desc': f'{prefix}bias_a_mean_desc',
        'bias_a_median_desc': f'{prefix}bias_a_median_desc',
        'bias_b_mean_desc': f'{prefix}bias_b_mean_desc',
        'bias_b_median_desc': f'{prefix}bias_b_median_desc',
        'bias_h_mean_desc': f'{prefix}bias_h_mean_desc',
        'bias_h_median_desc': f'{prefix}bias_h_median_desc',
        'bias_f_mean_desc': f'{prefix}bias_f_mean_desc',
        'bias_f_median_desc': f'{prefix}bias_f_median_desc',
        'bias_fb_mean_desc': f'{prefix}bias_fb_mean_desc',
        'bias_fb_median_desc': f'{prefix}bias_fb_median_desc',
        'bias_a_mean_norm': f'{prefix}bias_a_mean_norm',
        'bias_a_median_norm': f'{prefix}bias_a_median_norm',
        'bias_b_mean_norm': f'{prefix}bias_b_mean_norm',
        'bias_b_median_norm': f'{prefix}bias_b_median_norm',
        'bias_h_mean_norm': f'{prefix}bias_h_mean_norm',
        'bias_h_median_norm': f'{prefix}bias_h_median_norm',
        'bias_f_mean_norm': f'{prefix}bias_f_mean_norm',
        'bias_f_median_norm': f'{prefix}bias_f_median_norm',
        'bias_fb_mean_norm': f'{prefix}bias_fb_mean_norm',
        'bias_fb_median_norm': f'{prefix}bias_fb_median_norm'
    }
    
    for bias_name, column_name in columns_to_update.items():
        if column_name in df.columns:
            df[column_name] = bias_values[bias_name]
        else:
            print(f"Column {column_name} does not exist in the DataFrame.")
    
    return df

# Calculate mean and median for each specified group
mean_asian = df_mean_new[df_mean_new['PI_RACE'] == 'a']['GT_VALUE'].mean()
median_asian = df_mean_new[df_mean_new['PI_RACE'] == 'a']['GT_VALUE'].median()

mean_black = df_mean_new[df_mean_new['PI_RACE'] == 'b']['GT_VALUE'].mean()
median_black = df_mean_new[df_mean_new['PI_RACE'] == 'b']['GT_VALUE'].median()

mean_hispanic = df_mean_new[df_mean_new['PI_RACE'] == 'h']['GT_VALUE'].mean()
median_hispanic = df_mean_new[df_mean_new['PI_RACE'] == 'h']['GT_VALUE'].median()

mean_white = df_mean_new[df_mean_new['PI_RACE'] == 'w']['GT_VALUE'].mean()
median_white = df_mean_new[df_mean_new['PI_RACE'] == 'w']['GT_VALUE'].median()

mean_female = df_mean_new[df_mean_new['PI_GENDER'] == 'f']['GT_VALUE'].mean()
median_female = df_mean_new[df_mean_new['PI_GENDER'] == 'f']['GT_VALUE'].median()

mean_male = df_mean_new[df_mean_new['PI_GENDER'] == 'm']['GT_VALUE'].mean()
median_male = df_mean_new[df_mean_new['PI_GENDER'] == 'm']['GT_VALUE'].median()

mean_female_black = df_mean_new[(df_mean_new['PI_GENDER'] == 'f') & (df_mean_new['PI_RACE'] == 'b')]['GT_VALUE'].mean()
median_female_black = df_mean_new[(df_mean_new['PI_GENDER'] == 'f') & (df_mean_new['PI_RACE'] == 'b')]['GT_VALUE'].median()

mean_female_white = df_mean_new[(df_mean_new['PI_GENDER'] == 'f') & (df_mean_new['PI_RACE'] == 'w')]['GT_VALUE'].mean()
median_female_white = df_mean_new[(df_mean_new['PI_GENDER'] == 'f') & (df_mean_new['PI_RACE'] == 'w')]['GT_VALUE'].median()

gt_factor_mean_asian = mean_asian / mean_white
gt_factor_median_asian = median_asian / median_white
# Black
gt_factor_mean_black = mean_black / mean_white
gt_factor_median_black = median_black / median_white

# Hispanic
gt_factor_mean_hispanic = mean_hispanic / mean_white
gt_factor_median_hispanic = median_hispanic / median_white

# Female
gt_factor_mean_female = mean_female / mean_male
gt_factor_median_female = median_female / median_male

# Female and Black
gt_factor_mean_female_black = mean_female_black / mean_female_white
gt_factor_median_female_black = median_female_black / median_female_white


mean_grant_values_wi_all_asian,median_grant_values_wi_all_asian  = average_grant_top1_val(chosen_data, names_included,'all', 'Asian')
mean_grant_values_wi_all_black,median_grant_values_wi_all_black = average_grant_top1_val(chosen_data, names_included,'all', 'Black')
mean_grant_values_wi_all_hispanic,median_grant_values_wi_all_hispanic = average_grant_top1_val(chosen_data, names_included,'all', 'Hispanic')
mean_grant_values_wi_all_white,median_grant_values_wi_all_white = average_grant_top1_val(chosen_data, names_included,'all', 'White')
mean_grant_values_wi_female_all,median_grant_values_wi_female_all = average_grant_top1_val(chosen_data, names_included,'Female', 'all')
mean_grant_values_wi_male_all, median_grant_values_wi_male_all = average_grant_top1_val(chosen_data, names_included,'Male', 'all')
mean_grant_values_wi_female_black,median_grant_values_wi_female_black = average_grant_top1_val(chosen_data, names_included,'Female', 'Black')
mean_grant_values_wi_female_white, median_grant_values_wi_female_white = average_grant_top1_val(chosen_data, names_included,'Female', 'White')

pr_factor_mean_asian = [a / b for a, b in zip(mean_grant_values_wi_all_asian, mean_grant_values_wi_all_white)]
pr_factor_median_asian = [a / b for a, b in zip(median_grant_values_wi_all_asian,median_grant_values_wi_all_white)]


pr_factor_mean_black = [a / b for a, b in zip(mean_grant_values_wi_all_black,mean_grant_values_wi_all_white)]
pr_factor_median_black = [a / b for a, b in zip(median_grant_values_wi_all_black,median_grant_values_wi_all_white)]

pr_factor_mean_hispanic = [a / b for a, b in zip(mean_grant_values_wi_all_hispanic,mean_grant_values_wi_all_white)]
pr_factor_median_hispanic = [a / b for a, b in zip(median_grant_values_wi_all_hispanic,median_grant_values_wi_all_white)]

pr_factor_mean_female = [a / b for a, b in zip(mean_grant_values_wi_female_all,mean_grant_values_wi_male_all)]
pr_factor_median_female = [a / b for a, b in zip(median_grant_values_wi_female_all,median_grant_values_wi_male_all)]

pr_factor_mean_female_black = [a / b for a, b in zip(mean_grant_values_wi_female_black,mean_grant_values_wi_female_white)]
pr_factor_median_female_black = [a / b for a, b in zip(median_grant_values_wi_female_black,median_grant_values_wi_female_white)]


bias_a_mean_desc = abs(pr_factor_mean_asian - gt_factor_mean_asian)
print(pr_factor_mean_asian)
print('-------------')
print(gt_factor_mean_asian)
bias_a_median_desc = abs(pr_factor_median_asian - gt_factor_median_asian)
bias_b_mean_desc = abs(pr_factor_mean_black - gt_factor_mean_black)
bias_b_median_desc = abs(pr_factor_median_black - gt_factor_median_black)
bias_h_mean_desc = abs(pr_factor_mean_hispanic - gt_factor_mean_hispanic)
bias_h_median_desc = abs(pr_factor_median_hispanic - gt_factor_median_hispanic)
bias_f_mean_desc = abs(pr_factor_mean_female - gt_factor_mean_female)
bias_f_median_desc = abs(pr_factor_median_female - gt_factor_median_female)
bias_fb_mean_desc = abs(pr_factor_mean_female_black - gt_factor_mean_female_black)
bias_fb_median_desc = abs(pr_factor_median_female_black - gt_factor_median_female_black)

bias_a_mean_norm = pr_factor_mean_asian
bias_a_median_norm = pr_factor_median_asian
bias_b_mean_norm = pr_factor_mean_black
bias_b_median_norm = pr_factor_median_black
bias_h_mean_norm = pr_factor_mean_hispanic
bias_h_median_norm = pr_factor_median_hispanic
bias_f_mean_norm = pr_factor_mean_female
bias_f_median_norm = pr_factor_median_female
bias_fb_mean_norm = pr_factor_mean_female_black
bias_fb_median_norm = pr_factor_median_female_black


filename = 'intrinsic_bias.csv'
df_input = pd.read_csv(filename)
bias_values = {
    'bias_a_mean_desc': bias_a_mean_desc,
    'bias_a_median_desc': bias_a_median_desc,
    'bias_b_mean_desc': bias_b_mean_desc,
    'bias_b_median_desc': bias_b_median_desc,
    'bias_h_mean_desc': bias_h_mean_desc,
    'bias_h_median_desc': bias_h_median_desc,
    'bias_f_mean_desc': bias_f_mean_desc,
    'bias_f_median_desc': bias_f_median_desc,
    'bias_fb_mean_desc': bias_fb_mean_desc,
    'bias_fb_median_desc': bias_fb_median_desc,
    'bias_a_mean_norm': bias_a_mean_norm,
    'bias_a_median_norm': bias_a_median_norm,
    'bias_b_mean_norm': bias_b_mean_norm,
    'bias_b_median_norm': bias_b_median_norm,
    'bias_h_mean_norm': bias_h_mean_norm,
    'bias_h_median_norm': bias_h_median_norm,
    'bias_f_mean_norm': bias_f_mean_norm,
    'bias_f_median_norm': bias_f_median_norm,
    'bias_fb_mean_norm': bias_fb_mean_norm,
    'bias_fb_median_norm': bias_fb_median_norm
}
df_output = update_column_content(df_input, chosen_data, names_included, bias_values)

# Save the updated DataFrame back to CSV
#df_output.to_csv(filename, index=False)