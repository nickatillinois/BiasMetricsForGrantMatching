import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import numpy as np

# Load the data
data = pd.read_csv('merged_with_gt.csv')
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
#topn_appval, topn_foa = get_topn_recs('roberta', 'r', 'w', 1)


# now make a functio that has parameters model_name, dataset, version
# calls get_topn_recs(model_name, dataset, version, 1),
# for output topn_appval (which is 1 number here), calculates the Mean Absolute Percentage Error (MAPE) between GT_VALUE and topn_appval
def calculate_percentage_errors(df, model_name, dataset, version):
    # Get the top 1 recommendation
    topn_appval, _ = get_topn_recs(df, model_name, dataset, version, 1)
    
    # Convert topn_appval to numeric, replacing any non-numeric values with NaN
    topn_appval_numeric = pd.to_numeric(topn_appval, errors='coerce')
    
    # Calculate the percentage error for each row
    percentage_error = (topn_appval_numeric - df['GT_VALUE']) / df['GT_VALUE']
    # Calculate the absolute percentage error for each row
    absolute_percentage_error = np.abs(percentage_error)
    
    # Calculate the mean percentage error (MPE)
    mpe = percentage_error.mean(skipna=True)
    
    # Calculate the mean absolute percentage error (MAPE)
    mape = absolute_percentage_error.mean(skipna=True)
    #print(mpe)
    #print(mape)
    
    return mpe, mape

#mape = calculate_mape('roberta', 'r', 'w')
#print(f"MAPE for RoBERTa: {mape:.4f}")


def get_mape_for_all_models(dataset, version, gender, race):
    if gender == 'all' and race == 'all':
        filtered_df = data
    elif gender == 'all':
        filtered_df = data[data['PI_RACE'] == race]
    elif race == 'all':
        filtered_df = data[data['PI_GENDER'] == gender]
    else:
        filtered_df = data[(data['PI_GENDER'] == gender) & (data['PI_RACE'] == race)]   
    mape_dict = {}
    mpe_dict = {}
    for model in model_order:
        mpe,mape = calculate_percentage_errors(filtered_df,model, dataset, version)
        mape_dict[model] = mape
        mpe_dict[model] = mpe
    
    return mpe_dict,mape_dict

def calculate_P_FoA(df, model_name, dataset, version):
    # Get the top 1 recommendation
    _, topn_foa = get_topn_recs(df, model_name, dataset, version, 1)
    df.loc[:, 'GT_FOA'] = df['GT_FOA'].astype(str)
    topn_foa = topn_foa.astype(str)
    # Calculate the percentage error for each row
    # Calculate the percentage error as a binary vector
    percentage_error = (df['GT_FOA'] == topn_foa).astype(int)
    proportion_correct = percentage_error.sum() / len(percentage_error)
    return proportion_correct


def get_precision_foa_top1_for_all_models(dataset, version, gender, race):
    # Filter the dataframe based on PI_GENDER and PI_RACE
    if gender == 'all' and race == 'all':
        filtered_df = data
    elif gender == 'all':
        filtered_df = data[data['PI_RACE'] == race]
    elif race == 'all':
        filtered_df = data[data['PI_GENDER'] == gender]
    else:
        filtered_df = data[(data['PI_GENDER'] == gender) & (data['PI_RACE'] == race)]
    print(f"Length of filtered dataframe: {len(filtered_df)}")
    proportion_dict = {}
    for model in model_order:
        proportion_correct = calculate_P_FoA(filtered_df, model, dataset, version)
        proportion_dict[model] = proportion_correct  
    return proportion_dict

def extract_institution_code(grant_code):
    # Use a regular expression to find the text between the first and second hyphens
    match = re.search(r'-(.*?)-', grant_code)
    if match:
        return match.group(1).lower()
    print("no match")
    return None

def calculate_P_Inst(df, model_name, dataset, version):
    # Get the top 1 recommendation
    _, topn_foa = get_topn_recs(df, model_name, dataset, version, 1)
    df.loc[:, 'GT_FOA'] = df['GT_FOA'].astype(str)
    topn_foa = topn_foa.astype(str)
    topn_foa = topn_foa.apply(extract_institution_code)
    # Calculate the percentage error for each row
    # Calculate the percentage error as a binary vector
    percentage_error = (df['GT_FOA'].apply(extract_institution_code) == topn_foa)
    proportion_correct = percentage_error.sum() / len(percentage_error)
    return proportion_correct


def get_precision_inst_top1_for_all_models(dataset, version, gender, race):
    # Filter the dataframe based on PI_GENDER and PI_RACE
    if gender == 'all' and race == 'all':
        filtered_df = data
    elif gender == 'all':
        filtered_df = data[data['PI_RACE'] == race]
    elif race == 'all':
        filtered_df = data[data['PI_GENDER'] == gender]
    else:
        filtered_df = data[(data['PI_GENDER'] == gender) & (data['PI_RACE'] == race)]
    print(f"Length of filtered dataframe: {len(filtered_df)}")
    proportion_dict = {}
    for model in model_order:
        proportion_correct = calculate_P_Inst(filtered_df, model, dataset, version)
        proportion_dict[model] = proportion_correct  
    return proportion_dict

# Calculate changes for each dictionary
def calculate_changes(dict_wi, dict_w):
    changes = {}
    for model in model_order:
        if model in dict_wi and model in dict_w:
            if dict_wi[model] <= dict_w[model]:
                changes[model] = abs(dict_wi[model] - dict_w[model])
            else:
                changes[model] = -abs(dict_wi[model] - dict_w[model])
        else:
            changes[model] = None  # Or some default value if model is missing
    return changes
foa_top1_precition_dict_a_a_o_wi = get_precision_foa_top1_for_all_models('o', 'wi', 'all', 'all')
foa_top1_precition_dict_a_a_o_w = get_precision_foa_top1_for_all_models('o', 'w', 'all', 'all')
inst_top1_precition_dict_a_a_o_wi = get_precision_inst_top1_for_all_models('o', 'wi', 'all', 'all')
inst_top1_precition_dict_a_a_o_w = get_precision_inst_top1_for_all_models('o', 'w', 'all', 'all')
mpe_dict_a_a_o_wi, _ = get_mape_for_all_models('o', 'wi', 'all', 'all')
mpe_dict_a_a_o_w, _ = get_mape_for_all_models('o', 'w', 'all', 'all')



mpe_changes = calculate_changes(mpe_dict_a_a_o_wi, mpe_dict_a_a_o_w)

inst_top1_precition_changes = calculate_changes(inst_top1_precition_dict_a_a_o_wi, inst_top1_precition_dict_a_a_o_w)
foa_top1_precition_changes = calculate_changes(foa_top1_precition_dict_a_a_o_wi, foa_top1_precition_dict_a_a_o_w)
# Prepare data
metrics = ['Type@1', 'Inst@1', 'MPE']
wi_data = {
    'Type@1': [foa_top1_precition_dict_a_a_o_wi[model] for model in model_order],
    'Inst@1': [inst_top1_precition_dict_a_a_o_wi[model] for model in model_order],
    'MPE': [mpe_dict_a_a_o_wi[model] for model in model_order]
}
change_data = {
    'Type@1': [foa_top1_precition_changes[model] for model in model_order],
    'Inst@1': [inst_top1_precition_changes[model] for model in model_order],
    'MPE': [mpe_changes[model] for model in model_order]
}


fig, ax = plt.subplots(figsize=(28, 20))  # Increased height to 20
x = np.arange(len(model_order))
width = 0.15
multiplier = 0

# Plot each metric
for metric in metrics:
    offset = width * multiplier
    rects1 = ax.bar(x + offset, wi_data[metric], width, label=f'{metric} fine-tuned without PI-names')
    rects2 = ax.bar(x + offset + width, change_data[metric], width, label=f'$\Delta$ in {metric} fine-tuned with PI-names')
    multiplier += 2

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Percentages', fontsize=32)
ax.set_xticks(x + width * (len(metrics) - 0.5))
ax.set_xticklabels([model_name_mapping[model] for model in model_order], rotation=45, ha='right', fontsize=32)
ax.tick_params(axis='y', labelsize=28)
ax.tick_params(axis='x', length=10, width=2)
ax.tick_params(axis='y', length=10, width=2)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=28)  # Adjusted bbox_to_anchor

# Add grid to the plot
ax.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)

# Add value labels on the bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', rotation=90, fontsize=15)

for metric in metrics:
    autolabel(ax.containers[metrics.index(metric) * 2])
    autolabel(ax.containers[metrics.index(metric) * 2 + 1])

# Adjust layout
fig.tight_layout()
plt.subplots_adjust(bottom=0.25)  # Increase bottom margin

plt.savefig("performance_original_top1.eps", bbox_inches='tight')
plt.show()
