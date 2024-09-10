import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('test_set.csv')
# drop rows where gender_pred is 'unknown'
df = df[df['PI_GENDER'] != 'unknown']
gender_true = df['GT_GENDER']
race_true = df['GT_RACE']
gender_pred = df['PI_GENDER']

# calculate the number of rows where gender_true is equal to gender_pred
correct_gender_predictions = (gender_true == gender_pred).sum()
# Calculate the number of rows where gender_true != gender_pred and gender_pred is 'unknown'
num_mismatched_unknown = ((gender_true != gender_pred) & (gender_pred == 'unknown')).sum()

# Calculate the number of rows where gender_true != gender_pred and gender_pred is not 'unknown'
num_mismatched_not_unknown = ((gender_true != gender_pred) & (gender_pred != 'unknown')).sum()

# Calculate the percentages
percentage_mismatched_unknown = (num_mismatched_unknown / len(df)) * 100
percentage_mismatched_not_unknown = (num_mismatched_not_unknown / len(df)) * 100

# Print the results
print(f'Percentage of rows where gender_true != gender_pred and gender_pred is unknown: {percentage_mismatched_unknown:.2f}%')
print(f'Percentage of rows where gender_true != gender_pred and gender_pred is not unknown: {percentage_mismatched_not_unknown:.2f}%')

print(f'Number of correct gender predictions: {correct_gender_predictions}')
print(f'Total number of predictions: {len(df)}')
print(f'Accuracy: {correct_gender_predictions / len(df) * 100:.2f}%')


