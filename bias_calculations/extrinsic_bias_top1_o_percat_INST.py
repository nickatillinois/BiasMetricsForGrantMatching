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


inst_dict_f_a_o_wi = get_precision_inst_top1_for_all_models('o', 'wi', 'f', 'a')
inst_dict_m_a_o_wi = get_precision_inst_top1_for_all_models('o', 'wi', 'm', 'a')
inst_dict_f_b_o_wi = get_precision_inst_top1_for_all_models('o', 'wi', 'f', 'b')
inst_dict_m_b_o_wi = get_precision_inst_top1_for_all_models('o', 'wi', 'm', 'b')
inst_dict_f_h_o_wi = get_precision_inst_top1_for_all_models('o', 'wi', 'f', 'h')
inst_dict_m_h_o_wi = get_precision_inst_top1_for_all_models('o', 'wi', 'm', 'h')
inst_dict_f_w_o_wi = get_precision_inst_top1_for_all_models('o', 'wi', 'f', 'w')
inst_dict_m_w_o_wi = get_precision_inst_top1_for_all_models('o', 'wi', 'm', 'w')

inst_dict_f_a_o_w = get_precision_inst_top1_for_all_models('o', 'w', 'f', 'a')
inst_dict_m_a_o_w = get_precision_inst_top1_for_all_models('o', 'w', 'm', 'a')
inst_dict_f_b_o_w = get_precision_inst_top1_for_all_models('o', 'w', 'f', 'b')
inst_dict_m_b_o_w = get_precision_inst_top1_for_all_models('o', 'w', 'm', 'b')
inst_dict_f_h_o_w = get_precision_inst_top1_for_all_models('o', 'w', 'f', 'h')
inst_dict_m_h_o_w = get_precision_inst_top1_for_all_models('o', 'w', 'm', 'h')
inst_dict_f_w_o_w = get_precision_inst_top1_for_all_models('o', 'w', 'f', 'w')
inst_dict_m_w_o_w = get_precision_inst_top1_for_all_models('o', 'w', 'm', 'w')

inst_changes_f_a_o_w = calculate_changes(inst_dict_f_a_o_wi,inst_dict_f_a_o_w)
inst_changes_m_a_o_w = calculate_changes(inst_dict_m_a_o_wi,inst_dict_m_a_o_w)
inst_changes_f_b_o_w = calculate_changes(inst_dict_f_b_o_wi,inst_dict_f_b_o_w)
inst_changes_m_b_o_w = calculate_changes(inst_dict_m_b_o_wi,inst_dict_m_b_o_w)
inst_changes_f_h_o_w = calculate_changes(inst_dict_f_h_o_wi,inst_dict_f_h_o_w)
inst_changes_m_h_o_w = calculate_changes(inst_dict_m_h_o_wi,inst_dict_m_h_o_w)
inst_changes_f_w_o_w = calculate_changes(inst_dict_f_w_o_wi,inst_dict_f_w_o_w)
inst_changes_m_w_o_w = calculate_changes(inst_dict_m_w_o_wi,inst_dict_m_w_o_w)
precision_dicts = [
    inst_dict_f_a_o_wi,
    inst_dict_m_a_o_wi,
    inst_dict_f_b_o_wi,
    inst_dict_m_b_o_wi,
    inst_dict_f_h_o_wi,
    inst_dict_m_h_o_wi,
    inst_dict_f_w_o_wi,
    inst_dict_m_w_o_wi
]
def apply_model_name_mapping(dicts, mapping):
    mapped_dicts = []
    for d in dicts:
        mapped_dict = {mapping.get(k, k): v for k, v in d.items()}
        mapped_dicts.append(mapped_dict)
    return mapped_dicts

# Define the model name mapping
model_name_mapping = {
    "bert-base": "BERT-base",
    "roberta": "RoBERTa",
    "xlm-roberta": "XLM-R",
    "distilbert": "DistilBERT",
    "albert": "ALBERT",
    "spanbert": "SpanBERT",
    "deberta": "DeBERTa",
    "electra": "ELECTRA",
    "biobert": "BioBERT",
    "scibert": "SciBERT",
    "bluebert": "BlueBERT",
    "biomedbert": "PubMed.",
    "bert-xxxxx": "BERT-5X",
    "bert-xxxx": "BERT-4X",
    "bert-xxx": "BERT-3X",
    "bert-xx": "BERT-2X",
    "bert-x": "BERT-X",
    "bert-s": "BERT-S",
    "bert-multi": "Multi-BERT"
}


# Dictionaries with changes
changes_dicts = [
    inst_dict_f_a_o_wi,
    inst_changes_f_a_o_w,
    inst_dict_m_a_o_wi,
    inst_changes_m_a_o_w,
    inst_dict_f_b_o_wi,
    inst_changes_f_b_o_w,
    inst_dict_m_b_o_wi,
    inst_changes_m_b_o_w,
    inst_dict_f_h_o_wi,
    inst_changes_f_h_o_w,
    inst_dict_m_h_o_wi,
    inst_changes_m_h_o_w,
    inst_dict_f_w_o_wi,
    inst_changes_f_w_o_w,
    inst_dict_m_w_o_wi,
    inst_changes_m_w_o_w
]
changes_dicts = apply_model_name_mapping(changes_dicts, model_name_mapping)

# Print mapped dictionaries to verify
for i, mapped_dict in enumerate(changes_dicts):
    print(f"Mapped Dictionary {i+1}: {mapped_dict}")

# Combine both lists
all_dicts = changes_dicts
list_of_dicts = [
    inst_changes_f_a_o_w,
    inst_changes_m_a_o_w,
    inst_changes_f_b_o_w,
    inst_changes_m_b_o_w,
    inst_changes_f_h_o_w,
    inst_changes_m_h_o_w,
    inst_changes_f_w_o_w,
    inst_changes_m_w_o_w
]

# Function to calculate average of dictionary values
def average_dict_values(d):
    return sum(d.values()) / len(d) if d else 0

# Calculate and print averages
for idx, d in enumerate(changes_dicts):
    avg = average_dict_values(d)
    print(f"Average for dictionary {idx + 1}: {avg:.2f}")
def generate_latex_table(dicts, models):
    # Ensure that all dictionaries have the same keys
    model_set = set(models)
    for idx, d in enumerate(dicts):
        dict_keys = set(d.keys())
        if dict_keys != model_set:
            print(f"Dictionary {idx} has different keys: {dict_keys}")
            print(f"Expected keys: {model_set}")
            print()
    
    if not all(set(d.keys()) == model_set for d in dicts):
        raise ValueError("Not all dictionaries have the same keys as models.")
    
    latex_string = ""
    
    for i, model in enumerate(models):
        row = f"\\textbf{{{model}}}"
        for d in dicts:
            value = float(d[model])  # Convert string to float
            row += f" & {value:.2f}"
        row += " \\\\"
        latex_string += row + "\n"
    
    return latex_string

models = ["bert-base", "roberta", "xlm-roberta", "distilbert", "albert", "spanbert",
               "deberta", "electra", "biobert", "scibert", "bluebert", "biomedbert",
               "bert-xxxxx", "bert-xxxx", "bert-xxx", "bert-xx", "bert-x", 
               "bert-s", "bert-multi"]

latex_code = generate_latex_table(all_dicts, model_name_mapping.values())
print(latex_code)
#list_models = [inst_dict_f_a_o_wi[model] for model in model_order]

# Format the values to 2 decimal places as strings
formatted_dict = {k: f"{v:.2f}" for k, v in inst_changes_m_w_o_w .items()}

# Print the formatted dictionary
#print(formatted_dict)
# Prepare data for plotting
wi_data = {
    'f_a_o_wi': [inst_dict_f_a_o_wi[model] for model in model_order],
    'm_a_o_wi': [inst_dict_m_a_o_wi[model] for model in model_order],
    'f_b_o_wi': [inst_dict_f_b_o_wi[model] for model in model_order],
    'm_b_o_wi': [inst_dict_m_b_o_wi[model] for model in model_order],
    'f_h_o_wi': [inst_dict_f_h_o_wi[model] for model in model_order],
    'm_h_o_wi': [inst_dict_m_h_o_wi[model] for model in model_order],
    'f_w_o_wi': [inst_dict_f_w_o_wi[model] for model in model_order],
    'm_w_o_wi': [inst_dict_m_w_o_wi[model] for model in model_order],
}

change_data = {
    'f_a_o_w': [inst_changes_f_a_o_w[model] for model in model_order],
    'm_a_o_w': [inst_changes_m_a_o_w[model] for model in model_order],
    'f_b_o_w': [inst_changes_f_b_o_w[model] for model in model_order],
    'm_b_o_w': [inst_changes_m_b_o_w[model] for model in model_order],
    'f_h_o_w': [inst_changes_f_h_o_w[model] for model in model_order],
    'm_h_o_w': [inst_changes_m_h_o_w[model] for model in model_order],
    'f_w_o_w': [inst_changes_f_w_o_w[model] for model in model_order],
    'm_w_o_w': [inst_changes_m_w_o_w[model] for model in model_order],
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
axs[0].set_ylabel('Inst@1 fine-tuned without PI names', fontsize=32)
axs[1].set_ylabel('$\Delta$ in Inst@1 fine-tuned with PI names', fontsize=32)
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
plt.savefig("inst_o_with_without_percat.eps")
plt.show()