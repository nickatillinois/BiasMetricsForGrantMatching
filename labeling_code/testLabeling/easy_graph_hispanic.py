import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve

# Load your CSV data
df = pd.read_csv('train_set.csv')

# Simulate true labels
true_labels = df['GT_RACE'].apply(lambda x: 1 if x == 'h' else 0)

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
model_hispanic_percentage = []

# Number of samples
num_samples = len(df_probs)

# Define the class of interest
class_index = df_probs.columns.get_loc('hispanic')  # Assuming 'hispanic' is one of the columns

# Calculate the ground truth percentage where the label is 'hispanic'
ground_truth_hispanic_percentage = (df_probs['true_labels'].sum() / num_samples) * 100

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

    
    # Calculate model percentage of 'hispanic' predictions
    hispanic_model_percentage = (df_probs.iloc[:, class_index] > threshold).mean() * 100
    
    # Store results
    false_positives_percentage.append(fp_percentage)
    false_negatives_percentage.append(fn_percentage)
    true_positives_percentage.append(tp_percentage)
    true_negatives_percentage.append(tn_percentage)
    accuracies_percentage.append(accuracy_percentage)
    f1_scores_percentage.append(f1_score_percentage)
    model_hispanic_percentage.append(hispanic_model_percentage)

# Plot the results
plt.figure(figsize=(14, 10))

# Plot False Positives Percentage
plt.plot(thresholds, false_positives_percentage, label='False Positives (%)', color='red')

# Plot False Negatives Percentage
plt.plot(thresholds, false_negatives_percentage, label='False Negatives (%)', color='blue')

# Plot True Positives Percentage
plt.plot(thresholds, true_positives_percentage, label='True Positives (%)', color='orange')

# Plot True Negatives Percentage
plt.plot(thresholds, true_negatives_percentage, label='True Negatives (%)', color='cyan')

# Plot Precision Percentage
plt.plot(thresholds, precision_percentage, label='Precision (%)', color='magenta')

# Plot Recall Percentage
plt.plot(thresholds, recall_percentage, label='Recall (%)', color='yellow')

# Plot Fbeta Score
plt.plot(thresholds, fbeta_score, label='Fbeta Score (%)', color='black')

# Plot Accuracy Percentage
plt.plot(thresholds, accuracies_percentage, label='Accuracy (%)', color='green')

# Plot F1 Score Percentage
plt.plot(thresholds, f1_scores_percentage, label='F1 Score (%)', color='purple')

# Plot Model's Predicted Percentage of 'hispanic'
plt.plot(thresholds, model_hispanic_percentage, label='Model % hispanic Predictions (%)', color='brown')


# Add a horizontal line for the ground truth percentage where label is 'hispanic'
plt.axhline(y=ground_truth_hispanic_percentage, color='black', linestyle='--', label='Ground Truth % (hispanic)')

# Add titles and labels
#plt.title('Percentage of False Positives, False Negatives, True Positives, True Negatives, Accuracy, F1 Score, and Model % hispanic Predictions vs. Threshold')
plt.title('Hispanic')
plt.xlabel('Threshold')
plt.ylabel('Percentage')
#plt.legend()
plt.grid(True)
# remove the nan values from the lists and replace by 0
sensitivity = [0 if np.isnan(x) else x for x in sensitivity]
specificity = [0 if np.isnan(x) else x for x in specificity]
fp_list = [0 if np.isnan(x) else x for x in fp_list]
fn_list = [0 if np.isnan(x) else x for x in fn_list]
accuracies_percentage = [0 if np.isnan(x) else x for x in accuracies_percentage]
f1_scores_percentage = [0 if np.isnan(x) else x for x in f1_scores_percentage]
precision_percentage = [0 if np.isnan(x) else x for x in precision_percentage]
recall_percentage = [0 if np.isnan(x) else x for x in recall_percentage]
model_hispanic_percentage = [0 if np.isnan(x) else x for x in model_hispanic_percentage]


# Calculate the number of 'hispanic' records
num_hispanic_records = df_probs['true_labels'].sum()
perfect_threshold_f1 = thresholds[np.argmax(f1_scores_percentage)]
perfect_threshold_accuracy = thresholds[np.argmax(accuracies_percentage)]
perfect_threshold_sensitivity = thresholds[np.argmax(sensitivity)]
perfect_threshold_specificity = thresholds[np.argmax(specificity)]
try:
    diagnostic_efficiency = [sensitivity[i] * specificity[i] for i in range(len(sensitivity))]
except:
    diagnostic_efficiency = [0]
    print("Error in calculating diagnostic efficiency")
perfect_threshold_diagnostic_efficiency = thresholds[np.argmax(diagnostic_efficiency)]
# point where false positives equal to false negatives
equal_error_rate = thresholds[np.argmin(np.abs(np.array(fp_list) - np.array(fn_list)))]
print(f"Perfect threshold for F1: {perfect_threshold_f1}")
print(f"Perfect threshold for Accuracy: {perfect_threshold_accuracy}")
print(f"Perfect threshold for Sensitivity: {perfect_threshold_sensitivity}")
print(f"Perfect threshold for Specificity: {perfect_threshold_specificity}")
print(f"Perfect threshold for Diagnostic Efficiency: {perfect_threshold_diagnostic_efficiency}")
print(f"Equal Error Rate: {equal_error_rate}")
threshold_max_precision = thresholds[np.argmax(precision_percentage)]
print(f"Perfect threshold for Precision: {threshold_max_precision}")
# print maximum of precision_percentage
print(f"Max Precision: {max(precision_percentage)}")
print(precision_percentage)
# print the index of precision_percentage where it is maximum
print(f"Index of Max Precision: {np.argmax(precision_percentage)}")
print(thresholds)

# fbeta_score = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)
beta = 0.5
fbeta_score = []
for i in range(len(precision_percentage)):
    fbeta_score.append((1 + beta**2) * (precision_percentage[i] * recall_percentage[i]) / (beta**2 * precision_percentage[i] + recall_percentage[i]))
# remove the nan values from the lists and replace by 0
fbeta_score = [0 if np.isnan(x) else x for x in fbeta_score]
max_fbeta_score = max(fbeta_score)
max_threshold_fbeta = np.argmax(fbeta_score)
# print maximum fbeta_score and the index where it is maximum
print(f"Max Fbeta Score: {max_fbeta_score}")
print(f"Index of Max Fbeta Score: {max_threshold_fbeta/100}")


# Display total number of records and number of 'hispanic' records below the plot
#plt.figtext(0.5, 0.001, f'Total number of records: {num_samples}, Number of true hispanic records: {num_hispanic_records}, '
#                        f'F1-T: {round(perfect_threshold_f1, 2)}, Accuracy-T: {round(perfect_threshold_accuracy, 2)}, '
##                        f'Diagnostic-T: {round(perfect_threshold_diagnostic_efficiency, 2)}, Balance-T: {round(equal_error_rate, 2)}, Precision-T: {round(threshold_max_precision, 2)}, '
 #                       f' Fbeta-T: {round(max_threshold_fbeta/100, 2)}',
 #                       ha='center', fontsize=12)

plt.savefig('easy_graph_hispanic.png')
# Show the plot
plt.show()

# Plot Precision-Recall curve
plt.figure(figsize=(10, 6))

# Calculate precision and recall for the model's probabilities
precision, recall, thresholds_pr = precision_recall_curve(true_labels, df_probs.iloc[:, class_index])

# Plot precision-recall curve
plt.plot(recall, precision, label='Precision-Recall Curve', color='blue')

# Add titles and labels
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.grid(True)

# Display the plot
plt.savefig('precision_recall_curve.png')
plt.show()