import csv

# Define input file paths
sectioniii_file = 'sectioniii.csv'
tables_file = 'tables.csv'
output_file = 'combined_file.csv'

# Dictionary to store content based on Grant Number
grant_content_dict = {}

# Function to replace consecutive white spaces longer than 1 with a single white space
def replace_multiple_spaces(text):
    return ' '.join(text.split())

# Read sectioniii.csv
with open(sectioniii_file, 'r', encoding='latin-1') as file:
    # Create a CSV reader object
    csv_reader = csv.reader(file)

    # Skip the header
    header = next(csv_reader)

    # Iterate through rows in the input file
    for row in csv_reader:
        grant_number = row[0]
        content = replace_multiple_spaces(row[1].replace('\n', ' '))
        grant_content_dict.setdefault(grant_number, {'sectioniii': content})

# Read tables.csv and update the dictionary
with open(tables_file, 'r', encoding='latin-1') as file:
    # Create a CSV reader object
    csv_reader = csv.reader(file)

    # Skip the header
    header = next(csv_reader)

    # Iterate through rows in the input file
    for row in csv_reader:
        grant_number = row[0]
        content = replace_multiple_spaces(row[1].replace('\n', ' '))
        grant_content_dict.setdefault(grant_number, {}).update({'table': content})

# Write the combined content to a new CSV file
with open(output_file, 'w', encoding='utf-8', newline='') as output_file:
    # Create a CSV writer object
    csv_writer = csv.writer(output_file)

    # Write the header
    csv_writer.writerow(['Grant Number', 'table', 'sectioniii'])

    # Write the combined content
    for grant_number, content_dict in grant_content_dict.items():
        sectioniii_content = content_dict.get('sectioniii', '')
        table_content = content_dict.get('table', '')
        csv_writer.writerow([grant_number, table_content, sectioniii_content])

print(f"Files successfully combined, saved as '{output_file}', and consecutive white spaces replaced.")
