import csv

# Read titles.csv
with open('titles.csv', 'r') as titles_file:
    titles_reader = csv.DictReader(titles_file)
    titles_dict = {row['Grant Number']: row['Title'] for row in titles_reader}

# Read combined_file.csv and create combined_file_with_titles.csv
with open('combined_file.csv', 'r') as combined_file, open('combined_file_with_titles.csv', 'w', newline='') as output_file:
    combined_reader = csv.reader(combined_file)
    output_writer = csv.writer(output_file)

    # Write headers to the output file
    output_writer.writerow(['Grant Number', 'Title', 'table', 'sectioniii'])

    # Skip headers in the combined_file.csv
    next(combined_reader, None)

    for row in combined_reader:
        grant_number = row[0]
        table = row[1]
        sectioniii = row[2]

        # Check if there is a valid title for the current grant_number
        title = titles_dict.get(grant_number)

        if title:
            # Check if quotation marks are present and format accordingly
            title = f'"{title}"' if '"' not in title else f"'{title}'"
            table = f'"{table}"' if '"' not in table else f"'{table}'"
            sectioniii = f'"{sectioniii}"' if '"' not in sectioniii else f"'{sectioniii}'"

            # Write the row to the output file
            output_writer.writerow([grant_number, title, table, sectioniii])

print("Task completed successfully!")
