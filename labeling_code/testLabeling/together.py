import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve

def plot_race_metrics(df, race, ax):
    # Your existing code for data preparation goes here
    # Make sure to replace 'a' with the appropriate race code, and 'asian' with the appropriate race name
    
    true_labels = df['GT_RACE'].apply(lambda x: 1 if x == race[0].lower() else 0)
    # Assuming `df` has a column `PI_ETHNICITY_PROBS` which contains string representations of dictionaries
    race_probs = df['PI_ETHNICITY_PROBS']
    prob_dicts = []
    for prob_str in race_probs:
        prob_dict = eval(prob_str)  # Convert string to dictionary
        prob_dicts.append(prob_dict)

    # Step 2: Extract the values from each dictionary
    prob_values = []
    for prob_dict in prob_dicts:
        values = list(prob_dict.values())  # Extract values
        prob_values.append(values)

    # Step 3: Convert the list of lists into a numpy array
    model_probs = prob_values
    model_probs = np.array(model_probs)
    df_probs = pd.DataFrame(model_probs, columns=list(prob_dicts[0].keys()))
    df_probs['true_labels'] = true_labels

    # Initialize lists to store percentages
    false_positives_percentage = []
    false_negatives_percentage = []
    true_positives_percentage = []
    true_negatives_percentage = []
    accuracies_percentage = []
    f1_scores_percentage = []
    precision_percentage = []
    recall_percentage = []
    model_black_percentage = []

    # Number of samples
    num_samples = len(df_probs)
    class_index = df_probs.columns.get_loc(race.lower())
    # Calculate the ground truth percentage where the label is 'black'
    ground_truth_black_percentage = (df_probs['true_labels'].sum() / num_samples) * 100

    #Diagnostic Efficiency = Sensitivity * Specificity
    #Where...
    #Sensitivity = TP / (TP + FN)
    #Specificity = TN / (TN + FP)

    sensitivity = []
    specificity = []
    fp_list = []
    fn_list = []
    beta = 0.5
    fbeta_score = []

    # Generate thresholds from 0 to 1 with step size 0.01
    thresholds = np.arange(0, 1.01, 0.01)

    for threshold in thresholds:
        # Get predictions based on the threshold
        predicted_labels = (df_probs.iloc[:, class_index] > threshold).astype(int)
        true_binary = (df_probs['true_labels'] == 1).astype(int)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(true_binary, predicted_labels, labels=[0, 1]).ravel()
        
        # Calculate percentages
        not_divide_by_zero = 0e-6
        fp_percentage = (fp / num_samples) * 100
        fn_percentage = (fn / num_samples) * 100
        tp_percentage = (tp / num_samples) * 100
        tn_percentage = (tn / num_samples) * 100
        accuracy_percentage = ((tp + tn) / num_samples) * 100
        recall_p = (tp / (tp + fn)) * 100
        precision_p = (tp / (tp + fp)) * 100
        f_beta = (1 + beta**2) * (precision_p * recall_p) / (beta**2 * precision_p + recall_p)

        sensitivity.append(tp / (tp + fn))
        specificity.append(tn / (tn + fp))
        precision_percentage.append(precision_p)
        recall_percentage.append(recall_p)
        fbeta_score.append(f_beta)
        fn_list.append(fn)
        fp_list.append(fp)

        
        # Calculate F1 score
        if (tp + fp) > 0 and (tp + fn) > 0:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1_score_value = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score_value = 0.0
        
        f1_score_percentage = f1_score_value * 100

        
        # Calculate model percentage of 'black' predictions
        black_model_percentage = (df_probs.iloc[:, class_index] > threshold).mean() * 100
        
        # Store results
        false_positives_percentage.append(fp_percentage)
        false_negatives_percentage.append(fn_percentage)
        true_positives_percentage.append(tp_percentage)
        true_negatives_percentage.append(tn_percentage)
        accuracies_percentage.append(accuracy_percentage)
        f1_scores_percentage.append(f1_score_percentage)
        model_black_percentage.append(black_model_percentage)
    
    # Your existing code for calculating metrics goes here
    
    # Plot the results
    lines = []
    lines.append(ax.plot(thresholds, false_positives_percentage, color='red')[0])
    lines.append(ax.plot(thresholds, false_negatives_percentage, color='blue')[0])
    lines.append(ax.plot(thresholds, true_positives_percentage, color='orange')[0])
    lines.append(ax.plot(thresholds, true_negatives_percentage, color='cyan')[0])
    lines.append(ax.plot(thresholds, precision_percentage, color='magenta')[0])
    lines.append(ax.plot(thresholds, recall_percentage, color='yellow')[0])
    lines.append(ax.plot(thresholds, fbeta_score, color='black')[0])
    lines.append(ax.plot(thresholds, accuracies_percentage, color='green')[0])
    lines.append(ax.plot(thresholds, f1_scores_percentage, color='purple')[0])
    lines.append(ax.plot(thresholds, model_black_percentage, color='brown')[0])
    lines.append(ax.plot(thresholds, model_black_percentage, color='brown')[0])
     #Correct the axhline append
    ground_truth_line = ax.axhline(y=ground_truth_black_percentage, color='black', linestyle='--')
    lines.append(ground_truth_line)
    threshold_asian = 0.98
    threshold_black = 0.88
    threshold_hispanic = 0.72
    threshold_white = 0.58
    if (race == 'Asian'):
        threshold_truth_line = ax.axvline(threshold_asian, color='black', linestyle='--')
    elif (race == 'Black'):
        threshold_truth_line = ax.axvline(threshold_black, color='black', linestyle='--')
    elif (race == 'Hispanic'):
        threshold_truth_line = ax.axvline(threshold_hispanic, color='black', linestyle='--')
    elif (race == 'White'):
        threshold_truth_line = ax.axvline(threshold_white, color='black', linestyle='--')
    lines.append(threshold_truth_line)
    
    
    ax.set_title(race, fontsize=40)
    ax.set_xlabel('Threshold', fontsize=35)
    ax.set_ylabel('Percentage', fontsize=35)
    ax.tick_params(axis='both', which='major', labelsize=28)
    ax.grid(True)
    return lines

# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(24, 20))
#fig.suptitle('Race Metrics Comparison', fontsize=24)

# Flatten the axs array for easier indexing
axs = axs.flatten()

# List of races to plot
races = ['Asian', 'Black', 'Hispanic', 'White']
# Plot each race in its respective subplot
#df = pd.read_csv('test_set_cleaned.csv')
#df = pd.read_csv('testLabeling/train_set.csv')
df = pd.read_csv('testLabeling/test_set_cleaned.csv')

for i, race in enumerate(races):
    plot_race_metrics(df, race, axs[i])

# Create a single legend with color meanings
legend_elements = [
    plt.Line2D([0], [0], color='red', label='False Positives %'),
    plt.Line2D([0], [0], color='blue', label='False Negatives %'),
    plt.Line2D([0], [0], color='orange', label='True Positives %'),
    plt.Line2D([0], [0], color='cyan', label='True Negatives %'),
    plt.Line2D([0], [0], color='green', label='Accuracy %'),
    plt.Line2D([0], [0], color='magenta', label='Precision %'),
    plt.Line2D([0], [0], color='yellow', label='Recall %'),
    plt.Line2D([0], [0], color='black', label='F0.5 Score %'),
    plt.Line2D([0], [0], color='purple', label='F1 Score %'),
    plt.Line2D([0], [0], color='brown', label='Model % Predictions'),
    plt.Line2D([0], [0], color='black', linestyle='--', label='Ground Truth %')
]

# Add the legend to the figure
fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.1, 0.5), fontsize=28)
# Adjust the layout to prevent overlapping
plt.tight_layout()
plt.subplots_adjust(top=0.93, right=0.85, hspace=0.20, wspace=0.30)

# Save the figure
plt.savefig('combined_race_metrics_test.eps', bbox_inches='tight')

# Show the plot
plt.show()