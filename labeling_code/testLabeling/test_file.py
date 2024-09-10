import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def analyze_race_classification(df, thresholds):
    true_labels = df['GT_RACE']
    race_probs = df['PI_ETHNICITY_PROBS']
    prob_dicts = [eval(prob_str) for prob_str in race_probs]
    prob_values = [list(prob_dict.values()) for prob_dict in prob_dicts]
    model_probs = np.array(prob_values)
    df_probs = pd.DataFrame(model_probs, columns=list(prob_dicts[0].keys()))
    df_probs['true_labels'] = true_labels
    df_probs['model_labels'] = ""
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
    misclassified = df_probs[df_probs['model_labels'] != df_probs['true_labels']].shape[0]
    accuracy = 1 - misclassified / total
    accuracy_asian = df_probs[(df_probs['model_labels'] == 'a') & (df_probs['true_labels'] == 'a')].shape[0] / df_probs[df_probs['true_labels'] == 'a'].shape[0]
    accuracy_black = df_probs[(df_probs['model_labels'] == 'b') & (df_probs['true_labels'] == 'b')].shape[0] / df_probs[df_probs['true_labels'] == 'b'].shape[0]
    accuracy_hispanic = df_probs[(df_probs['model_labels'] == 'h') & (df_probs['true_labels'] == 'h')].shape[0] / df_probs[df_probs['true_labels'] == 'h'].shape[0]
    accuracy_white = df_probs[(df_probs['model_labels'] == 'w') & (df_probs['true_labels'] == 'w')].shape[0] / df_probs[df_probs['true_labels'] == 'w'].shape[0]
    cm = confusion_matrix(df_probs['true_labels'], df_probs['model_labels'])
    #print('Confusion Matrix: ', cm)
    class_accuracy = {df_probs.columns[i]: cm[i][i] / np.sum(cm[i]) for i in range(len(cm))}
    # round class accuracy to 2 decimal places
    class_accuracy = {k: round(v, 2) for k, v in class_accuracy.items()}
    print('Class Accuracy: ', class_accuracy)
    precision = {df_probs.columns[i]: cm[i][i] / np.sum(cm[:, i]) for i in range(len(cm))}
    recall = {df_probs.columns[i]: cm[i][i] / np.sum(cm[i]) for i in range(len(cm))}
    
    # f2 score: precision twice as important as recall
    f2_score = {df_probs.columns[i]: (5 * precision[df_probs.columns[i]] * recall[df_probs.columns[i]]) / (4 * precision[df_probs.columns[i]] + recall[df_probs.columns[i]]) for i in range(len(cm))}
    # round f2 score to 2 decimal places
    f2_score = {k: round(v, 2) for k, v in f2_score.items()}
    # round precision to 2 decimal places
    precision = {k: round(v, 2) for k, v in precision.items()}
    # round recall to 2 decimal places
    recall = {k: round(v, 2) for k, v in recall.items()}
    print('Precision: ', precision)
    print('Recall: ', recall)

    print('F2 Score: ', f2_score)
    print('Misclassified: ', misclassified)
    print('Unknown: ', unknown)


    
    print('Unknown Percentage: ', unknown_percentage)
    return df_probs




trainset = pd.read_csv('train_set.csv')
testset = pd.read_csv('test_set_cleaned.csv')
current = testset
threshold_asian_balance = 0.45
threshold_black_balance = 0.59
threshold_hispanic_balance = 0.24
threshold_white_balance = 0.39
thresholds_balance = [threshold_asian_balance, threshold_black_balance, threshold_hispanic_balance, threshold_white_balance]


threshold_asian_fbeta = 0.98
threshold_black_fbeta = 0.88
threshold_hispanic_fbeta = 0.72
threshold_white_fbeta = 0.58
thresholds_fbeta = [threshold_asian_fbeta, threshold_black_fbeta, threshold_hispanic_fbeta, threshold_white_fbeta]





# make bar chart of the distribution of the true labels
true_labels_before = current['GT_RACE']
size_before = true_labels_before.value_counts()

df = analyze_race_classification(current, thresholds_fbeta)
true_labels_after = df['true_labels']
model_labels_after = df['model_labels']
# make bar chart showing the distribution of the true labels before and after removing unknowns
fig, ax = plt.subplots(1, 2)
true_labels_before.value_counts().plot(kind='bar', ax=ax[0], title='Before')
true_labels_after.value_counts().plot(kind='bar', ax=ax[1], title='After')
size_after = true_labels_after.value_counts()
#print('Before: ', size_before)
print('total before: ', size_before.sum())
#print('After: ', size_after)
print('total after: ', size_after.sum())
print('unknowns per group that are now lost as percentage: ', (size_before - size_after) / size_before)
print('unknowns per group that are now lost: ', size_before - size_after)
print('lost total: ', (size_before - size_after).sum())


plt.show()