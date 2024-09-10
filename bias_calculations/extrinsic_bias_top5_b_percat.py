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
#topn_appval, topn_foa = get_topn_recs('roberta', 'p', 'w', 1)


# now make a functio that has parameters model_name, dataset, version
# calls get_topn_recs(model_name, dataset, version, 1),
# for output topn_appval (which is 1 number here), calculates the Mean Absolute Percentage Error (MAPE) between GT_VALUE and topn_appval
def calculate_percentage_errors(df, model_name, dataset, version):
    # Get the top 1 recommendation
    topn_appval, _ = get_topn_recs(df, model_name, dataset, version, 5)
    topn_appval = topn_appval.apply(lambda x: sum(map(float, x.split(','))) / len(x.split(',')))
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
    
    return mpe, mape

#mape = calculate_mape('roberta', 'p', 'w')
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

def calculate_uap(recommended_column, true_column):
    """
    Calculate the Unweighted Average Precision (UAP) for the given columns.

    Parameters:
    recommended_column (pd.Series): A pandas Series with 5 recommended items per row (as comma-separated strings).
    true_column (pd.Series): A pandas Series with the true item for each row.

    Returns:
    float: The mean UAP across all rows.
    """
    def precision_at_k(recommended, true_code, k):
        # Split the recommended items into a list
        recommended_codes = recommended
        recommended_at_k = recommended_codes[:k]
        num_matching_codes = recommended_at_k.count(true_code)
        return num_matching_codes / k
    def uap_for_row(recs,true_code):
        recommended_at_k = recs.split(',')
        r = recommended_at_k.count(true_code)
        if r == 0:
            return 0
        uap = 0
        for i in range(1,6):
            current = recommended_at_k[i - 1]
            if current == true_code:
                uap += precision_at_k(recommended_at_k,true_code,i)
        uap = uap / r
        return uap
    # Apply UAP calculation to each row
    uap_values = []
    for i in range(len(recommended_column)):
        recs = recommended_column.iloc[i]
        true_code = true_column.iloc[i]
        uap = uap_for_row(recs, true_code)
        uap_values.append(uap)
    
    # Return the mean UAP across all rows
    return sum(uap_values) / len(uap_values)

def calculate_P_FoA(df, model_name, dataset, version):
    # Get the top 1 recommendation
    _, topn_foa = get_topn_recs(df, model_name, dataset, version, 5)
    df.loc[:, 'GT_FOA'] = df['GT_FOA'].astype(str)
    topn_foa = topn_foa.astype(str)
    mean_uap = calculate_uap(topn_foa, df['GT_FOA'])
    # Calculate the percentage error for each row
    # Calculate the percentage error as a binary vector
    #percentage_error = (df['GT_FOA'] == topn_foa).astype(int)
    #print(percentage_error)
    #proportion_correct = percentage_error.sum() / len(percentage_error)
    return mean_uap


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

def calculate_uap_inst(recommended_column, true_column):
    """
    Calculate the Unweighted Average Precision (UAP) for the given columns.

    Parameters:
    recommended_column (pd.Series): A pandas Series with 5 recommended items per row (as comma-separated strings).
    true_column (pd.Series): A pandas Series with the true item for each row.

    Returns:
    float: The mean UAP across all rows.
    """
    def precision_at_k(recommended, true_code, k):
        # Split the recommended items into a list
        recommended_codes = recommended
        recommended_at_k = recommended_codes[:k]
        num_matching_codes = recommended_at_k.count(true_code)
        return num_matching_codes / k
    def uap_for_row(recs,true_code):
        
        r = recs.count(true_code)
        if r == 0:
            return 0
        uap = 0
        for i in range(1,6):
            current = recs[i - 1]
            if current == true_code:
                uap += precision_at_k(recs,true_code,i)
        uap = uap / r
        return uap
    # Apply UAP calculation to each row
    uap_values = []
    for i in range(len(recommended_column)):
        recs = recommended_column.iloc[i]
        recs = recs.split(',')
        recs_insts = []
        for x in recs:
           recs_insts.append(extract_institution_code(x))
        true_code = extract_institution_code(true_column.iloc[i])
        uap = uap_for_row(recs_insts, true_code)
        uap_values.append(uap)
    
    # Return the mean UAP across all rows
    return sum(uap_values) / len(uap_values)

def calculate_P_Inst(df, model_name, dataset, version):
    # Get the top 1 recommendation
    _, topn_foa = get_topn_recs(df, model_name, dataset, version, 5)
    df.loc[:, 'GT_FOA'] = df['GT_FOA'].astype(str)
    topn_foa = topn_foa.astype(str)
    mean_uap = calculate_uap_inst(topn_foa, df['GT_FOA'])

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
mpe_dict_f_a_o_wi,_ = get_mape_for_all_models('p', 'wi', 'f', 'a')
mpe_dict_m_a_o_wi,_ = get_mape_for_all_models('p', 'wi', 'm', 'a')
mpe_dict_f_b_o_wi,_ = get_mape_for_all_models('p', 'wi', 'f', 'b')
mpe_dict_m_b_o_wi,_ = get_mape_for_all_models('p', 'wi', 'm', 'b')
mpe_dict_f_h_o_wi,_ = get_mape_for_all_models('p', 'wi', 'f', 'h')
mpe_dict_m_h_o_wi,_ = get_mape_for_all_models('p', 'wi', 'm', 'h')
mpe_dict_f_w_o_wi,_ = get_mape_for_all_models('p', 'wi', 'f', 'w')
mpe_dict_m_w_o_wi,_ = get_mape_for_all_models('p', 'wi', 'm', 'w')

mpe_dict_f_a_o_w,_ = get_mape_for_all_models('p', 'w', 'f', 'a')
mpe_dict_m_a_o_w,_ = get_mape_for_all_models('p', 'w', 'm', 'a')
mpe_dict_f_b_o_w,_ = get_mape_for_all_models('p', 'w', 'f', 'b')
mpe_dict_m_b_o_w,_ = get_mape_for_all_models('p', 'w', 'm', 'b')
mpe_dict_f_h_o_w,_ = get_mape_for_all_models('p', 'w', 'f', 'h')
mpe_dict_m_h_o_w,_ = get_mape_for_all_models('p', 'w', 'm', 'h')
mpe_dict_f_w_o_w,_ = get_mape_for_all_models('p', 'w', 'f', 'w')
mpe_dict_m_w_o_w,_ = get_mape_for_all_models('p', 'w', 'm', 'w')

mpe_changes_f_a_o_w = calculate_changes(mpe_dict_f_a_o_wi,mpe_dict_f_a_o_w)
mpe_changes_m_a_o_w = calculate_changes(mpe_dict_m_a_o_wi,mpe_dict_m_a_o_w)
mpe_changes_f_b_o_w = calculate_changes(mpe_dict_f_b_o_wi,mpe_dict_f_b_o_w)
mpe_changes_m_b_o_w = calculate_changes(mpe_dict_m_b_o_wi,mpe_dict_m_b_o_w)
mpe_changes_f_h_o_w = calculate_changes(mpe_dict_f_h_o_wi,mpe_dict_f_h_o_w)
mpe_changes_m_h_o_w = calculate_changes(mpe_dict_m_h_o_wi,mpe_dict_m_h_o_w)
mpe_changes_f_w_o_w = calculate_changes(mpe_dict_f_w_o_wi,mpe_dict_f_w_o_w)
mpe_changes_m_w_o_w = calculate_changes(mpe_dict_m_w_o_wi,mpe_dict_m_w_o_w)

print(mpe_changes_m_w_o_w)
# Prepare data for plotting
wi_data = {
    'f_a_o_wi': [mpe_dict_f_a_o_wi[model] for model in model_order],
    'm_a_o_wi': [mpe_dict_m_a_o_wi[model] for model in model_order],
    'f_b_o_wi': [mpe_dict_f_b_o_wi[model] for model in model_order],
    'm_b_o_wi': [mpe_dict_m_b_o_wi[model] for model in model_order],
    'f_h_o_wi': [mpe_dict_f_h_o_wi[model] for model in model_order],
    'm_h_o_wi': [mpe_dict_m_h_o_wi[model] for model in model_order],
    'f_w_o_wi': [mpe_dict_f_w_o_wi[model] for model in model_order],
    'm_w_o_wi': [mpe_dict_m_w_o_wi[model] for model in model_order],
}

change_data = {
    'f_a_o_w': [mpe_changes_f_a_o_w[model] for model in model_order],
    'm_a_o_w': [mpe_changes_m_a_o_w[model] for model in model_order],
    'f_b_o_w': [mpe_changes_f_b_o_w[model] for model in model_order],
    'm_b_o_w': [mpe_changes_m_b_o_w[model] for model in model_order],
    'f_h_o_w': [mpe_changes_f_h_o_w[model] for model in model_order],
    'm_h_o_w': [mpe_changes_m_h_o_w[model] for model in model_order],
    'f_w_o_w': [mpe_changes_f_w_o_w[model] for model in model_order],
    'm_w_o_w': [mpe_changes_m_w_o_w[model] for model in model_order],
}
# Define label names
label_names = {
    'f_a_o_wi': 'Female Asian',
    'm_a_o_wi': 'Male Asian',
    'f_b_o_wi': 'Female Black',
    'm_b_o_wi': 'Male Black',
    'f_h_o_wi': 'Female Hispanic',
    'm_h_o_wi': 'Male Hispanic',
    'f_w_o_wi': 'Female Wite',
    'm_w_o_wi': 'Male White',
    'f_a_o_w': 'Female Asian',
    'm_a_o_w': 'Male Asian',
    'f_b_o_w': 'Female Black',
    'm_b_o_w': 'Male Black',
    'f_h_o_w': 'Female Hispanic',
    'm_h_o_w': 'Male Hispanic',
    'f_w_o_w': 'Female Wite',
    'm_w_o_w': 'Male White',
}

# Set up the plot
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(28, 36))

# Set parameters for bars
x = np.arange(len(model_order))
width = 0.1
labels = ['f_a_o_wi', 'm_a_o_wi', 'f_b_o_wi', 'm_b_o_wi', 'f_h_o_wi', 'm_h_o_wi', 'f_w_o_wi', 'm_w_o_wi']

# Plot wi_data
for i, label in enumerate(labels):
    offset = width * i
    axs[0].bar(x + offset, wi_data[label], width, label=label_names[label])

# Plot change_data
labels_w = ['f_a_o_w', 'm_a_o_w', 'f_b_o_w', 'm_b_o_w', 'f_h_o_w', 'm_h_o_w', 'f_w_o_w', 'm_w_o_w']
for i, label in enumerate(labels_w):
    offset = width * i
    axs[1].bar(x + offset, change_data[label], width, label=label_names[label])

# Customize axes
for ax in axs:
    ax.set_xticks(x + width * (len(labels) // 2))
    ax.set_xticklabels([model_name_mapping[model] for model in model_order], rotation=45, ha='right', fontsize=32)
    ax.tick_params(axis='y', labelsize=28)  # Set y-axis tick labels font size
    ax.tick_params(axis='x', length=10, width=2)  # Set x-axis tick length and width
    ax.tick_params(axis='y', length=10, width=2)  # Set y-axis tick length and width
    ax.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)
    # Add value labels on the bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=90, fontsize=15)
    # Apply autolabel function
    #for rects in ax.containers:
    #    autolabel(rects)
axs[0].set_ylabel('MPE fine-tuned without PI names', fontsize=32)
axs[1].set_ylabel('$\Delta$ in MPE fine-tuned with PI names', fontsize=32)
# Add a single legend for both plots
handles, labels = axs[0].get_legend_handles_labels()
handles_w, labels_w = axs[1].get_legend_handles_labels()
handles.extend(handles_w)
labels.extend(labels_w)
# Use the same labels for both plots
unique_labels = list(dict.fromkeys(labels))  # Remove duplicates while preserving order
axs[1].legend(handles, unique_labels, loc='lower left', bbox_to_anchor=(0.02, 0.02), ncol=4, fontsize=25)


# Adjust layout
fig.tight_layout()
plt.savefig("MPE_p_with_without_percat_top5.eps")
plt.show()