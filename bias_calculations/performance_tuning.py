import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('evaluation.csv')

# Extract Model and Dataset Information
df['MODEL_NAME'] = df['MODEL'].apply(lambda x: '_'.join(x.split('_')[:-2]))
df['DATASET'] = df['MODEL'].apply(lambda x: x.split('_')[-2])
df['VERSION'] = df['MODEL'].apply(lambda x: x.split('_')[-1])

# Define the order of models
model_order = ["bert-base", "roberta", "xlm-roberta", "distilbert", "albert", "spanbert",
               "deberta", "electra", "biobert", "scibert", "bluebert", "biomedbert",
               "bert-xxxxx", "bert-xxxx", "bert-xxx", "bert-xx", "bert-x", 
               "bert-s", "bert-multi"] # -"bert-base(para)",

# Create a mapping dictionary for display names
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

# Apply the mapping to the MODEL_NAME column
df['MODEL_NAME'] = df['MODEL_NAME'].replace(model_name_mapping)

# Set the order for the MODEL_NAME column
df['MODEL_NAME'] = pd.Categorical(df['MODEL_NAME'], categories=[model_name_mapping[m] for m in model_order], ordered=True)

# Set the theme
sns.set_theme(style="whitegrid")

# Melt the dataframe for easier plotting
df_melted = pd.melt(df, id_vars=["MODEL_NAME", "DATASET", "VERSION"], 
                    value_vars=["ORIGINAL_PERF", "TUNED_PERF"], 
                    var_name="PERFORMANCE_TYPE", value_name="PERFORMANCE")

# Plot the performance comparison
plt.figure(figsize=(24, 12))  # Increased figure size
ax = sns.barplot(x="MODEL_NAME", y="PERFORMANCE", hue="PERFORMANCE_TYPE", data=df_melted, errorbar=None, palette="Set2")
plt.xticks(rotation=90, fontsize=20)  # Decreased font size for tick labels
plt.yticks(fontsize=20)  # Decreased font size for tick labels
plt.xlabel("Model Name", fontsize=24)  # Increased font size for x-label
plt.ylabel("Accuracy", fontsize=24)  # Increased font size for y-label

# Modify the legend
legend = plt.legend(title="Model", fontsize=14, title_fontsize=16)
legend.set_title("Model")  # Change the title of the legend
for text in legend.get_texts():
    text.set_text(text.get_text().replace("ORIGINAL_PERF", "Base Model"))
    text.set_text(text.get_text().replace("TUNED_PERF", "Fine-tuned Model"))   # Replace old labels with new ones

plt.subplots_adjust(bottom=0.4)  # Adjusted bottom margin to give more space for names
plt.savefig("original_vs_tuned.eps")
plt.show()

# Create a figure with two subplots
fig, axes = plt.subplots(2, 1, figsize=(24, 30))  # 2 rows, 1 column with increased figure size

# Plot Threshold with highest accuracy
sns.barplot(ax=axes[0], x="MODEL_NAME", y="THRES", hue="DATASET", data=df, palette="Set1")
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=90, fontsize=20)
axes[0].set_yticklabels(axes[0].get_yticklabels(), fontsize=20)
axes[0].set_xlabel("Model", fontsize=20)
axes[0].set_ylabel("Selected Threshold", fontsize=20)
legend = axes[0].legend(title="Dataset", fontsize=16, title_fontsize=18)
for text in legend.get_texts():
    text.set_text(text.get_text().replace("p", "Balanced"))
    text.set_text(text.get_text().replace("r", "Reality"))
    text.set_text(text.get_text().replace("o", "Original"))
axes[0].set_title("Threshold with Highest Accuracy", fontsize=24)

# Plot F1 score for weighted average
sns.barplot(ax=axes[1], x="MODEL_NAME", y="F1_THRES", hue="DATASET", data=df, palette="Set1")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=90, fontsize=20)
axes[1].set_yticklabels(axes[1].get_yticklabels(), fontsize=20)
axes[1].set_xlabel("Model", fontsize=20)
axes[1].set_ylabel("F1 Score", fontsize=20)
legend = axes[1].legend(title="Dataset", fontsize=16, title_fontsize=18)
for text in legend.get_texts():
    text.set_text(text.get_text().replace("p", "Balanced"))
    text.set_text(text.get_text().replace("r", "Reality"))
    text.set_text(text.get_text().replace("o", "Original"))
axes[1].set_title("F1 Score for Weighted Average", fontsize=24)

# Adjust the vertical space between the plots
plt.subplots_adjust(hspace=0.3, bottom=0.3)  # Increased 'hspace' for more vertical space

# Save and show the combined figure
plt.savefig("combined_plots.eps")
plt.show()