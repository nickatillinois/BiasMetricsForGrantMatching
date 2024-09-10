import pandas as pd

# Load datasets
trainset = pd.read_csv('train_set.csv')
testset = pd.read_csv('test_set_cleaned.csv')

# Merge the datasets
df = pd.concat([trainset, testset], axis=0)
print("Total number of records:", len(df))

# Count total records per GT_RACE category
total_per_race = df.groupby('PI_RACE').size()

# Calculate percentage of total records per GT_RACE category
percentage_per_race = (total_per_race / len(df)) * 100

# Print results
print("\nNumber and percentage of records per GT_RACE category:")
for race in total_per_race.index:
    num_records = total_per_race[race]
    percent_records = percentage_per_race[race]
    print(f"PI_RACE = {race}:")
    print(f"  Number of records: {num_records}")
    print(f"  Percentage of total records: {percent_records:.2f}%")
