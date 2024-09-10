import pandas as pd

# Read the CSV file into a DataFrame
file_path = 'combined_file.csv'
df = pd.read_csv(file_path)

# Initialize an empty list to store results
result_list = []

# Keywords indicating the end of the title
end_keywords = ['activity code', 'announcement', 'announcements', 'name', 'principal investigators']

# Counter for records with "NOTFOUND"
not_found_count = 0

# Iterate through the rows of the DataFrame
for index, row in df.iterrows():
    grant_number = row['Grant Number']
    table_data = str(row['table']).lower()

    # Check for the presence of 'title' or 'funding opportunity title' and identify the title
    title_start = table_data.find('title:')
    title_start = title_start if title_start != -1 else table_data.find('title')
    
    if title_start != -1:
        # Identify the end of the title based on keywords, excluding -1 values
        title_end_candidates = [table_data.find(keyword, title_start) for keyword in end_keywords if keyword in table_data and table_data.find(keyword, title_start) != -1]
        title_end = min(title_end_candidates) if title_end_candidates else len(table_data)

        # Extract the title and exclude the keyword "title"
        title = table_data[title_start + 5:title_end].strip() if 'title:' in table_data else table_data[title_start:title_end].strip()
        
        # Remove characters until the first letter
        title = title.lstrip(' .,:;()[]{}!@#$%^&*-_+=<>?/\\|`~\'"')
        
        # Remove characters between 'title' and the first word afterwards
        title_parts = title.split()
        if 'title' in title_parts:
            title_parts = title_parts[title_parts.index('title') + 1:]
        title = ' '.join(title_parts)
        
        # Remove leading and trailing quotation marks if present
        title = title.strip('"')
        
        result_list.append({'Grant Number': grant_number, 'Title': title})
    else:
        # If 'title' or 'funding opportunity title' not found, write "NOTFOUND" and increment the counter
        not_found_count += 1
        #result_list.append({'Grant Number': grant_number, 'Title': 'NOTFOUND'})

# Print the count of records with "NOTFOUND"
print(f"Number of records with 'NOTFOUND': {not_found_count}")

# Create a new DataFrame with the results
result_df = pd.DataFrame(result_list)

# Save the new DataFrame to a CSV file
result_df.to_csv('titles.csv', index=False, columns=['Grant Number', 'Title'])
