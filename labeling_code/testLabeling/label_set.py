import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def analyze_race_classification(df, thresholds):
    race_probs = df['PI_ETHNICITY_PROBS']
    prob_dicts = [eval(prob_str) for prob_str in race_probs]
    prob_values = [list(prob_dict.values()) for prob_dict in prob_dicts]
    model_probs = np.array(prob_values)
    df_probs = pd.DataFrame(model_probs, columns=list(prob_dicts[0].keys()))
    df_probs['model_labels'] = ""
    df_probs['APPLICATION_ID'] = df['APPLICATION_ID']
    t_a = thresholds[0]
    t_b = thresholds[1]
    t_h = thresholds[2]
    t_w = thresholds[3]
    for i in range(len(df_probs)):
        label = ""
        asian_prob = df_probs['asian'][i]
        black_prob = df_probs['black'][i]
        hispanic_prob = df_probs['hispanic'][i]
        white_prob = df_probs['white'][i]
        is_asian = asian_prob > t_a
        is_black = black_prob > t_b
        is_hispanic = hispanic_prob > t_h
        is_white = white_prob > t_w
        if is_black:
            label = 'b'
        elif is_white:
            label = 'w'
        elif is_hispanic:
            label = 'h'
        elif is_asian:
            label = 'a'
        if not is_black and not is_white and not is_hispanic and not is_asian:
            label = 'u'
        df_probs.loc[i, 'model_labels'] = label
    unknown = df_probs[df_probs['model_labels'] == 'u'].shape[0]
    total = df_probs.shape[0]
    unknown_percentage = unknown / total
    # remove unknowns
    df_probs = df_probs[df_probs['model_labels'] != 'u']
    print('Unknown: ', unknown)
    print('Unknown Percentage: ', unknown_percentage)
    return df_probs

# Load your CSV data
current = pd.read_csv('test_set_cleaned.csv')
size_before = current['APPLICATION_ID'].shape[0]
threshold_asian_fbeta = 0.98
threshold_black_fbeta = 0.88
threshold_hispanic_fbeta = 0.72
threshold_white_fbeta = 0.58
thresholds_fbeta = [threshold_asian_fbeta, threshold_black_fbeta, threshold_hispanic_fbeta, threshold_white_fbeta]

# Analyze race classification
df = analyze_race_classification(current, thresholds_fbeta)
print(df.head())
size_after = df['APPLICATION_ID'].shape[0]
print('Size before: ', size_before)
print('Size after: ', size_after)
print('Size difference: ', size_before - size_after)
print('Size percentage: ', (size_before - size_after) / size_before)

# Get the value counts of each label in the model_labels column
labels_counts = df['model_labels'].value_counts()
print(labels_counts)

# Plot the pie chart of the model_labels column
labels = labels_counts.index
sizes = labels_counts.values

plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Distribution of Model Labels')
plt.show()
