import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('test_set_cleaned.csv')

# Print the first few rows of the DataFrame to understand its structure
print(df.head())
print("Total number of records:", df.shape[0])

# Distribution of GT_RACE
race_counts = df['GT_RACE'].value_counts()
print("Absolute number of records for each category in GT_RACE:")
print(race_counts)

plt.figure(figsize=(8, 8))
plt.pie(race_counts, labels=race_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of GT_RACE')
plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
plt.show()

# Distribution of GT_GENDER
gender_counts = df['GT_GENDER'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of GT_GENDER')
plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
plt.show()
