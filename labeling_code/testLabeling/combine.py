import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# Read the CSV files
trainset = pd.read_csv('train_set.csv')
testset = pd.read_csv('test_set_cleaned.csv')

# Merge the datasets
df = pd.concat([trainset, testset], axis=0)
print("total", len(df))

# Calculate total number and percentage of 'unknown' PI_GENDER
total_unknown = df[df['PI_GENDER'] == 'unknown'].shape[0]
total_records = df.shape[0]
total_unknown_percentage = (total_unknown / total_records) * 100

# Calculate number and percentage of 'unknown' PI_GENDER per GT_RACE category
unknown_per_race = df[df['PI_GENDER'] == 'unknown'].groupby('GT_RACE').size()
total_per_race = df.groupby('GT_RACE').size()
percentage_per_race = (unknown_per_race / total_per_race) * 100

# Print results for GT_RACE
print(f"Total number of records with 'unknown' PI_GENDER: {total_unknown}")
print(f"Percentage of records with 'unknown' PI_GENDER: {total_unknown_percentage:.2f}%")

print("\nNumber and percentage of 'unknown' PI_GENDER per GT_RACE category:")
for race in percentage_per_race.index:
    print(f"GT_RACE = {race}:")
    print(f"  Number of 'unknown': {unknown_per_race[race]}")
    print(f"  Percentage of 'unknown': {percentage_per_race[race]:.2f}%")

# Calculate number and percentage of 'unknown' PI_GENDER per GT_GENDER category
unknown_per_gender = df[df['PI_GENDER'] == 'unknown'].groupby('GT_GENDER').size()
total_per_gender = df.groupby('GT_GENDER').size()
percentage_per_gender = (unknown_per_gender / total_per_gender) * 100

# Print results for GT_GENDER
print("\nNumber and percentage of 'unknown' PI_GENDER per GT_GENDER category:")
for gender in percentage_per_gender.index:
    print(f"GT_GENDER = {gender}:")
    print(f"  Number of 'unknown': {unknown_per_gender[gender]}")
    print(f"  Percentage of 'unknown': {percentage_per_gender[gender]:.2f}%")

# Exclude records where PI_GENDER is 'unknown'
df_filtered = df[df['PI_GENDER'] != 'unknown']

# Total number of records after filtering
total_records_filtered = df_filtered.shape[0]

# Count the number of records for each PI_GENDER category and their percentage of total records
gender_counts = df_filtered['PI_GENDER'].value_counts()
gender_percentages = (gender_counts / total_records_filtered) * 100

# Print counts and percentages
print("\nNumber and percentage of records for each PI_GENDER category:")
for gender, count in gender_counts.items():
    percentage = gender_percentages[gender]
    print(f"{gender}: {count} records ({percentage:.2f}%)")

# Calculate precision, recall, and F1-score for the filtered dataset
true_gender = df_filtered['GT_GENDER']
pred_gender = df_filtered['PI_GENDER']

# Calculate metrics
precision = precision_score(true_gender, pred_gender, average='binary', pos_label='f', zero_division=0)
recall = recall_score(true_gender, pred_gender, average='binary', pos_label='f', zero_division=0)
f1 = f1_score(true_gender, pred_gender, average='binary', pos_label='f', zero_division=0)

print("\nOverall Metrics:")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
