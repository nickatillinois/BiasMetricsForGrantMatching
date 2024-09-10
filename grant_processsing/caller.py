import pandas as pd
import csv
import re
from getContent import extract_webpage_sections

def clean_abstract_text(text):
    text = re.sub(r'\s\s+', ' ', text)
    text = text.strip()  # Remove leading and trailing whitespace
    return text

def add_quotes_if_needed(text):
    if not text.startswith('"'):
        text = f'"{text}"'
    return text

def combine_multiline_records_and_clean_abstract(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='latin1') as infile:
            reader = csv.reader(infile)
            rows = list(reader)

        if not rows:
            print(f"Skipping empty file: {input_file}")
            return

        headers = rows[0]
        abstract_text_index = headers.index('Content')

        with open(output_file, 'w', newline='', encoding='latin1') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(headers)
            
            for row in rows[1:]:
                # Combine multiline fields into a single line
                combined_row = [' '.join(field.splitlines()) for field in row]
                
                # Clean the 'ABSTRACT_TEXT' field
                abstract_text = clean_abstract_text(combined_row[abstract_text_index])
                
                # Add quotes if needed
                combined_row[abstract_text_index] = add_quotes_if_needed(abstract_text)
                
                writer.writerow(combined_row)
        
        print(f"Processed file: {input_file}")  # Print message after processing the file
        
    except Exception as e:
        print(f"Error processing file {input_file}: {e}")


# Function to write data to a file
def write_to_file(filename, data):
    df = pd.DataFrame(data, columns=['Grant Number', 'Content'])
    df.to_csv(filename, index=False)

# Read URLs from overlap.csv
urls_df = pd.read_csv('workingRFAs.csv', header=None)

urls = urls_df[0].tolist()

# for each record, add https://grants.nih.gov/grants/guide/rfa-files/ before and .html after
#urls = ['https://grants.nih.gov/grants/guide/rfa-files/' + url + '.html' for url in urls]

nb_urls = len(urls)
# Lists to store data
tables = []
sectioniiis = []
sectionivs = []
section_iis = []
counter = 0
# Extract strings for each URL
for url in urls:
    print(url)
    counter += 1
    print(f'Processing URL {counter}/{nb_urls}: {url}')
    try:
        table, section_iii, section_iv, section_ii = extract_webpage_sections(url)
    except Exception as e:
        print(f'Error occurred for URL: {url}')
        continue

    grant_number = url.split('/')[-1].split('.')[0]  # Extract grant number from URL

    # Check if any string is empty
    #if not nb_year or not cash:
    #    print(f'One or more strings are empty for URL: {url}')

    # Add data to lists
    tables.append([grant_number, table])
    sectioniiis.append([grant_number, section_iii])
    sectionivs.append([grant_number, section_iv])
    section_iis.append([grant_number, section_ii])
# Write data to files
write_to_file('tables.csv', tables)
write_to_file('sectioniii.csv', sectioniiis)
write_to_file('sectioniv.csv', sectionivs)
write_to_file('sectionii.csv', section_iis)
combine_multiline_records_and_clean_abstract('tables.csv', 'tables.csv')
combine_multiline_records_and_clean_abstract('sectioniii.csv', 'sectioniii.csv')
combine_multiline_records_and_clean_abstract('sectioniv.csv', 'sectioniv.csv')
combine_multiline_records_and_clean_abstract('sectionii.csv', 'sectionii.csv')


