import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def analyze_race_classification(white_factor):
    # Load your CSV data
    df = pd.read_csv('test_set.csv')

    true_labels = df['GT_RACE']
    race_probs = df['PI_ETHNICITY_PROBS']
    prob_dicts = [eval(prob_str) for prob_str in race_probs]

    prob_values = [list(prob_dict.values()) for prob_dict in prob_dicts]
    model_probs = np.array(prob_values)
    df_probs = pd.DataFrame(model_probs, columns=list(prob_dicts[0].keys()))
    df_probs['true_labels'] = true_labels
    df_probs['model_labels'] = ""

    threshold_white = 0.393939393939394
    threshold_black = 0.595959595959596
    threshold_hispanic = 0.24242424242424243
    threshold_asian = 0.4545454545454546

    for i in range(len(df_probs)):
        df_probs.loc[i, 'white'] *= white_factor
        is_black = df_probs.loc[i, 'black'] > threshold_black
        is_white = df_probs.loc[i, 'white'] > threshold_white
        is_hispanic = df_probs.loc[i, 'hispanic'] > threshold_hispanic
        is_asian = df_probs.loc[i, 'asian'] > threshold_asian

        is_certainly_black = is_black & ~(is_white | is_hispanic | is_asian)
        is_certainly_white = is_white & ~(is_black | is_hispanic | is_asian)
        is_certainly_hispanic = is_hispanic & ~(is_black | is_white | is_asian)
        is_certainly_asian = is_asian & ~(is_black | is_white | is_hispanic)

        if not any([is_certainly_black, is_certainly_white, is_certainly_hispanic, is_certainly_asian]):
            max_prob = max(df_probs.loc[i, ['black', 'white', 'hispanic', 'asian']])
            label = df_probs.columns[df_probs.iloc[i] == max_prob][0]
            label = 'unknown'
        else:
            label = next(l for l, c in zip(['black', 'white', 'hispanic', 'asian'], 
                                           [is_certainly_black, is_certainly_white, is_certainly_hispanic, is_certainly_asian]) if c)
        
        df_probs.loc[i, 'model_labels'] = label[0]

    misclassified = df_probs[df_probs['model_labels'] != df_probs['true_labels']].shape[0]
    unknown = df_probs[df_probs['model_labels'] == 'u'].shape[0]
    total = df_probs.shape[0]
    unknown_percentage = unknown / total
    unknown_black = df_probs[(df_probs['model_labels'] == 'u') & (df_probs['true_labels'] == 'b')].shape[0]
    black_before_removal = df_probs[df_probs['true_labels'] == 'b'].shape[0]
    white_as_black_before = df_probs[(df_probs['true_labels'] == 'w') & (df_probs['model_labels'] == 'b')].shape[0]
    

    
    df_probs = df_probs[df_probs['model_labels'] != 'u']
    misclassified_after_removal = df_probs[df_probs['model_labels'] != df_probs['true_labels']].shape[0]
    unknown_after_removal = df_probs[df_probs['model_labels'] == 'u'].shape[0]
    black_after_removal = df_probs[df_probs['true_labels'] == 'b'].shape[0]

    cm = confusion_matrix(df_probs['true_labels'], df_probs['model_labels'])
    accuracy = np.trace(cm) / np.sum(cm)

    class_accuracy = {df_probs.columns[i]: cm[i][i] / np.sum(cm[i]) for i in range(len(cm))}
    #f1 score for each class
    f1_score = {df_probs.columns[i]: 2 * cm[i][i] / (np.sum(cm[i]) + np.sum(cm[:, i])) for i in range(len(cm))}


    black_misclassified = df_probs[(df_probs['true_labels'] == 'b') & (df_probs['model_labels'] != 'b')].shape[0]
    white_as_black = df_probs[(df_probs['true_labels'] == 'w') & (df_probs['model_labels'] == 'b')].shape[0]

    return {
        'df_probs': df_probs,
        'total': total,
        'unknown_percentage': unknown_percentage,
        'misclassified': misclassified,
        'black_before_removal': black_before_removal,
        'unknown': unknown,
        'unknown_black': unknown_black,
        'white_as_black_before': white_as_black_before,
        'misclassified_after_removal': misclassified_after_removal,
        'unknown_after_removal': unknown_after_removal,
        'black_after_removal': black_after_removal,
        'accuracy': accuracy,
        'class_accuracy': class_accuracy,
        'black_misclassified': black_misclassified,
        'white_as_black': white_as_black,
        'f1_black': f1_score['black'],
        'f1_white': f1_score['white'],
        'f1_hispanic': f1_score['hispanic'],
        'f1_asian': f1_score['asian']
    }


df = pd.read_csv('test_set.csv')

true_labels = df['GT_RACE']
race_probs = df['PI_ETHNICITY_PROBS']
# make pie chart of the distribution of the true labels
true_labels.value_counts().plot.pie(autopct='%1.1f%%')
plt.show()

# run results with a factor between 0 and 1 with steps 0.01
i_array = []
results_array = []
for i in range(101):
    i_array.append(i/100)
    results = analyze_race_classification(i/100)
    results_array.append(results)


# max_black_after_removal = 0
max_f1_black = 0
best_results = None
# find the best results: the factor that gives the best f1 score for the black class while keeping the f1 score for the other classes above 0.6
for i in range(len(results_array)):
    results = results_array[i]
    f1_scores = [results['f1_black'], results['f1_white'], results['f1_hispanic'], results['f1_asian']]
    class_accuracy = results['class_accuracy']
    if all(f1_score > 0.6 for f1_score in f1_scores) and all(acc > 0.6 for acc in class_accuracy.values()):
        f1_black = results['f1_black']
        if f1_black > max_f1_black:
            max_f1_black = f1_black
            best_results = results
            print(f'Best results so far: {best_results}')
            print(i)

df = best_results['df_probs']
true_labels = df['true_labels']
# pie chart of the distribution of the true labels
true_labels.value_counts().plot.pie(autopct='%1.1f%%')
plt.show()

