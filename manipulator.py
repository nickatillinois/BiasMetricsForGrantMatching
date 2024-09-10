import os
import pandas as pd
import re
from collections import Counter
import warnings
import csv

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
        abstract_text_index = headers.index('ABSTRACT_TEXT')

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

def process_all_csv_files_in_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            input_file = os.path.join(directory, filename)
            output_file = input_file  # Overwrite the original file
            combine_multiline_records_and_clean_abstract(input_file, output_file)


# Function to rename files in a directory to the year in the file name
def rename_files_to_year(directory_path):
    try:
        # List all files in the directory
        files = os.listdir(directory_path)

        # Regular expression to match 4 consecutive digits
        year_pattern = re.compile(r'\d{4}')

        for file_name in files:
            # Search for the year pattern in the file name
            match = year_pattern.search(file_name)

            if match:
                # Extract the matched year
                year = match.group()

                # Construct the new file name with the year
                new_name = os.path.join(directory_path, f"{year}{os.path.splitext(file_name)[1]}")

                # Rename the file
                os.rename(os.path.join(directory_path, file_name), new_name)
                print(f"Renamed '{file_name}' to '{new_name}'.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Function to replace the column name "OPPORTUNITY NUMBER" with "FOA_NUMBER" in all CSV files in a directory
def replace_column_name(csv_directory='.'):
    try:
        # List all CSV files in the directory
        csv_files = [file for file in os.listdir(csv_directory) if file.endswith('.csv')]

        for csv_file in csv_files:
            # Read the CSV file into a DataFrame with Latin-1 encoding
            try:
                df = pd.read_csv(os.path.join(csv_directory, csv_file), encoding='latin-1')
            except pd.errors.ParserError as pe:
                print(f"Error parsing '{csv_file}': {str(pe)}")
                continue

            # Check if "OPPORTUNITY NUMBER" column exists
            if 'OPPORTUNITY NUMBER' in df.columns:
                # Replace the column name
                df.rename(columns={'OPPORTUNITY NUMBER': 'FOA_NUMBER'}, inplace=True)

                # Save the updated DataFrame back to the same CSV file
                df.to_csv(os.path.join(csv_directory, csv_file), index=False, encoding='latin-1')

                print(f"Replaced 'OPPORTUNITY NUMBER' with 'FOA_NUMBER' in '{csv_file}'.")
            else:
                print(f"'OPPORTUNITY NUMBER' column not found in '{csv_file}'.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


def check_column_names(csv_directory='.'):
    try:
        l = os.listdir(csv_directory)
        # List all CSV files in the directory
        csv_files = [file for file in os.listdir(csv_directory) if file.endswith('.csv')]

        for csv_file in csv_files:
            # Read the CSV file into a DataFrame
            df = pd.read_csv(os.path.join(csv_directory, csv_file), encoding='latin-1', on_bad_lines='skip')

            # Check if "OPPORTUNITY NUMBER" and "FOA_NUMBER" columns exist
            has_opportunity_number = 'OPPORTUNITY NUMBER' in df.columns
            has_foa_number = 'FOA_NUMBER' in df.columns

            # Print the results
            print("lo`")
            print(f"File: '{csv_file}'")
            print(f"  - 'OPPORTUNITY NUMBER' column: {'Exists' if has_opportunity_number else 'Not Found'}")
            print(f"  - 'FOA_NUMBER' column: {'Exists' if has_foa_number else 'Not Found'}")
            #print(f"  - 'Both' column: {'Found' if not has_foa_number and not has_opportunity_number else 'Not Found'}")
            print()
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def find_duplicate_application_id(csv_directory='.'):
    try:
        # List all CSV files in the directory and sort them by year
        csv_files = sorted([file for file in os.listdir(csv_directory) if file.endswith('.csv')],
                           key=lambda x: int(re.search(r'\d{4}', x).group()))

        for csv_file in csv_files:
            # Read the CSV file into a DataFrame with Latin-1 encoding
            try:
                df = pd.read_csv(os.path.join(csv_directory, csv_file), encoding='latin-1')
            except pd.errors.ParserError as pe:
                print(f"Error parsing '{csv_file}': {str(pe)}")
                continue

            # Check for duplicate APPLICATION_ID
            duplicate_ids = [item for item, count in Counter(df['APPLICATION_ID']).items() if count > 1]

            if duplicate_ids:
                print(f"In '{csv_file}', found at least one 'APPLICATION_ID' that occurs at least twice:")
                print(duplicate_ids[0])
                print()

            # Call the new function to delete duplicate records in-place
            delete_duplicate_records(df, 'APPLICATION_ID', inplace=True)

            # Save the updated DataFrame back to the same CSV file
            df.to_csv(os.path.join(csv_directory, csv_file), index=False, encoding='latin-1')
            print(f"Deleted duplicate records in '{csv_file}'.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


def delete_duplicate_records(df, column_name, inplace=False):
    """
    Delete duplicate records based on the specified column in a DataFrame.

    Parameters:
    - df: DataFrame
    - column_name: str, Name of the column to check for duplicates
    - inplace: bool, Whether to modify the DataFrame in-place or return a new DataFrame

    Returns:
    - DataFrame if inplace is False, None otherwise
    """
    if inplace:
        df.drop_duplicates(subset=column_name, keep='first', inplace=True)
    else:
        return df.drop_duplicates(subset=column_name, keep='first')
    

def keep_columns_in_place(columns_to_keep, csv_directory='.'):
    try:
        # List all CSV files in the directory
        csv_files = [file for file in os.listdir(csv_directory) if file.endswith('.csv')]

        for csv_file in csv_files:
            # Read the CSV file into a DataFrame with Latin-1 encoding
            try:
                df = pd.read_csv(os.path.join(csv_directory, csv_file), encoding='latin-1')
            except pd.errors.ParserError as pe:
                print(f"Error parsing '{csv_file}': {str(pe)}")
                continue

            # Delete columns that are not in the specified list to keep
            columns_to_delete = [col for col in df.columns if col not in columns_to_keep]
            df.drop(columns=columns_to_delete, inplace=True, errors='ignore')

            # Save the updated DataFrame back to the same CSV file
            df.to_csv(os.path.join(csv_directory, csv_file), index=False, encoding='latin-1')

            print(f"Kept columns {columns_to_keep} in file '{csv_file}'.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

def find_filenames_with_47_records(csv_directory='.'):
    try:
        # List all CSV files in the directory
        csv_files = [file for file in os.listdir(csv_directory) if file.endswith('.csv')]

        for csv_file in csv_files:
            # Read the CSV file into a DataFrame with Latin-1 encoding
            try:
                df = pd.read_csv(os.path.join(csv_directory, csv_file), encoding='latin-1')
            except pd.errors.ParserError as pe:
                print(f"Error parsing '{csv_file}': {str(pe)}")
                
                # If ParserError is due to the mismatch in the number of fields, correct the lines
                if "Expected 46 fields" in str(pe):
                    with open(os.path.join(csv_directory, csv_file), 'r', encoding='latin-1') as file:
                        lines = file.readlines()

                    # Rewrite the file excluding the last field for problematic lines
                    with open(os.path.join(csv_directory, csv_file), 'w', encoding='latin-1') as file:
                        for line in lines:
                            fields = line.split(',')
                            if len(fields) > 46:
                                corrected_line = ','.join(fields[:-1]) + '\n'
                                file.write(corrected_line)
                            else:
                                file.write(line)

                continue

            # Check the number of records in the DataFrame
            num_records = len(df)

            # Print the filename if it has 47 records
            if num_records == 47:
                print(f"File '{csv_file}' has 47 records.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


def delete_records_with_empty_pi_names(csv_directory='.'):
    try:
        # List all CSV files in the directory
        csv_files = [file for file in os.listdir(csv_directory) if file.endswith('.csv')]

        for csv_file in csv_files:
            # Read the CSV file into a DataFrame with Latin-1 encoding
            try:
                df = pd.read_csv(os.path.join(csv_directory, csv_file), encoding='latin-1')
            except pd.errors.ParserError as pe:
                print(f"Error parsing '{csv_file}': {str(pe)}")
                continue

            # Count the number of records before deletion
            num_records_before = len(df)

            # Delete records where 'PI_NAMEs' is empty
            df = df.dropna(subset=['PI_NAMEs'])

            # Count the number of deleted records
            num_deleted_records = num_records_before - len(df)

            # Print the number of deleted records per file
            print(f"Deleted {num_deleted_records} records with empty 'PI_NAMEs' in file '{csv_file}'.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


def remove_exact_duplicates(file_path, encoding='latin-1'):
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path, encoding=encoding)

        # Check for exact duplicates and keep only the first occurrence
        df.drop_duplicates(keep='first', inplace=True)

        # Save the updated DataFrame back to the same CSV file
        df.to_csv(file_path, index=False, encoding=encoding)

        print(f"Removed exact duplicates in file '{file_path}'.")

    except Exception as e:
        print(f"Error processing file '{file_path}': {e}")

# Function to remove exact duplicates for each file in the directory
def remove_duplicates_in_directory(directory):
    # List all files in the specified directory
    files = os.listdir(directory)

    # Iterate through each file
    for file_name in files:
        file_path = os.path.join(directory, file_name)

        # Check if the file is a regular file
        if os.path.isfile(file_path):
            # Call the function to remove exact duplicates for each file
            remove_exact_duplicates(file_path)


def remove_records_without_foa(file_path, encoding='latin-1'):
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path, encoding=encoding)

        # Remove records without FOA_NUMBER
        df.dropna(subset=['FOA_NUMBER'], inplace=True)

        # Save the updated DataFrame back to the same CSV file
        df.to_csv(file_path, index=False, encoding=encoding)

        print(f"Removed records without FOA_NUMBER in file '{file_path}'.")

    except Exception as e:
        print(f"Error processing file '{file_path}': {e}")

# Function to remove records without FOA_NUMBER for each file in the directory
def remove_records_without_foa_in_directory(directory):
    # List all files in the specified directory
    files = os.listdir(directory)

    # Iterate through each file
    for file_name in files:
        file_path = os.path.join(directory, file_name)

        # Check if the file is a regular file
        if os.path.isfile(file_path):
            # Call the function to remove records without FOA_NUMBER for each file
            remove_records_without_foa(file_path)


def merge_files(year, abs_folder, proj_folder, output_folder):
    abs_file = os.path.join(abs_folder, f'{year}.csv')
    proj_file = os.path.join(proj_folder, f'{year}.csv')
    
    # Check if both files exist
    if os.path.exists(abs_file) and os.path.exists(proj_file):
        # Read abstract and project files with Latin-1 encoding
        abs_df = pd.read_csv(abs_file, encoding='latin-1')
        proj_df = pd.read_csv(proj_file, encoding='latin-1')
        
        # Ensure APPLICATION_ID columns are of the same type (convert to string)
        abs_df['APPLICATION_ID'] = abs_df['APPLICATION_ID'].astype(str)
        proj_df['APPLICATION_ID'] = proj_df['APPLICATION_ID'].astype(str)
        
        # Select only APPLICATION_ID and ABSTRACT_TEXT columns from abs_df
        abs_df = abs_df[['APPLICATION_ID', 'ABSTRACT_TEXT']]
        
        # Enclose ABSTRACT_TEXT values in abs_df with double quotes
        abs_df['ABSTRACT_TEXT'] = abs_df['ABSTRACT_TEXT'].apply(lambda x: f'"{x}"' if pd.notnull(x) else x)
        
        # Merge on APPLICATION_ID, keeping all rows from proj_df
        merged_df = pd.merge(proj_df, abs_df, on='APPLICATION_ID', how='left')
        
        # Save merged file
        output_file = os.path.join(output_folder, f'{year}withabs.csv')
        merged_df.to_csv(output_file, index=False)
        print(f'Merged {year} files successfully!')
    else:
        print(f'Files for year {year} not found.')

# function that sums the number of rows in all files in a directory and prints the total
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

def count_rows_in_directory(directory):
    # List all files in the specified directory
    files = os.listdir(directory)

    # Initialize a counter for the total number of rows
    total_rows = 0

    # Iterate through each file
    for file_name in files:
        file_path = os.path.join(directory, file_name)

        # Check if the file is a regular file
        if os.path.isfile(file_path) and file_path.endswith('.csv'):
            # Read the CSV file into a DataFrame
            try:
                df = pd.read_csv(file_path, encoding='latin-1')
            except pd.errors.ParserError as pe:
                print(f"Error parsing '{file_path}': {str(pe)}")
                continue
            # Increment the total number of rows
            total_rows += len(df)

    # Print the total number of rows
    print(f"Total number of rows in directory '{directory}': {total_rows}")


# function that changes the name of the column 'FUNDING_Ics' to 'FUNDING_ICs' in the 2012 file
def rename_column_in_2012(file_path):
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path, encoding='latin-1')

        # Rename the column 'FUNDING_Ics' to 'FUNDING_ICs'
        df.rename(columns={'FUNDING_Ics': 'FUNDING_ICs'}, inplace=True)

        # Save the updated DataFrame back to the same CSV file
        df.to_csv(file_path, index=False, encoding='latin-1')

        print(f"Renamed 'FUNDING_Ics' to 'FUNDING_ICs' in file '{file_path}'.")

    except Exception as e:
        print(f"Error processing file '{file_path}': {e}")

# function that deletes deletes columns that are not in the list of columns to keep, this is not a parameter
def delete_columns_not_in_list_of_columns(file_path, list_of_columns, encoding='latin-1'):
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path, encoding=encoding)

        # Delete columns that are not in the specified list to keep
        columns_to_delete = [col for col in df.columns if col not in list_of_columns]
        df.drop(columns=columns_to_delete, inplace=True, errors='ignore')

        # Save the updated DataFrame back to the same CSV file
        df.to_csv(file_path, index=False, encoding=encoding)

        print(f"Deleted columns not in the list of columns in file '{file_path}'.")

    except Exception as e:
        print(f"Error processing file '{file_path}': {e}")

# function that applies the function delete_columns_not_in_list_of_columns to all files in a directory
def delete_columns_not_in_list_of_columns_in_directory(directory, list_of_columns):
    # List all files in the specified directory
    files = os.listdir(directory)

    # Iterate through each file
    for file_name in files:
        file_path = os.path.join(directory, file_name)

        # Check if the file is a regular file
        if os.path.isfile(file_path):
            # Call the function to delete columns not in the list of columns for each file
            delete_columns_not_in_list_of_columns(file_path, list_of_columns)

def add_updated_funding_info(updated_directory, project_directory):
    # List all files in the specified directory
    rename_files_to_year(updated_directory)
    
    for year in range(1985, 2000):
        # Paths for the CSV files
        left = os.path.join(project_directory, f'{year}withabs.csv')
        right = os.path.join(updated_directory, f'{year}.csv')

        # Read the CSV files
        left_df = pd.read_csv(left, encoding='latin-1')
        right_df = pd.read_csv(right, encoding='latin-1')
        print("left_df has columns: ", left_df.columns)
        print("left_df has rows: ", len(left_df))

        # Add '_right' to the column names in the right_df except for 'APPLICATION_ID'
        right_df.columns = [f'{col}_right' if col != 'APPLICATION_ID' else col for col in right_df.columns]

        # Perform a left join
        merged_df = pd.merge(left_df, right_df, left_on='APPLICATION_ID', right_on='APPLICATION_ID', how='left')

        # Update the left DataFrame with values from the right DataFrame
        for col in right_df.columns:
            if col != 'APPLICATION_ID':  # Skip 'APPLICATION_ID'
                without_right = col[:-6]  # Remove '_right' suffix
                if without_right == 'FUNDING_ICS':
                    without_right = 'FUNDING_ICs'
                # If the column is empty in the left DataFrame, fill it with the value from the right DataFrame
                merged_df[without_right] = merged_df[without_right].combine_first(merged_df[col])
                # If both columns are not empty, overwrite the left DataFrame with the right DataFrame
                merged_df[without_right] = merged_df.apply(lambda row: row[col] if pd.notnull(row[col]) else row[without_right], axis=1)

        # Remove the columns from the right DataFrame
        merged_df.drop(columns=[col for col in merged_df.columns if col.endswith('_right')], inplace=True)
        print("merged_df has columns: ", merged_df.columns)
        print("merged_df has rows: ", len(merged_df))
        # Save the merged DataFrame to a new file
        output_file = os.path.join('/home/nisse/Documents/vakken_l/Thesis/nih/originals/july/data/withFunding', f'{year}withFunding.csv')
        merged_df.to_csv(output_file, index=False)
        print(f'Merged {year} files successfully!')



if __name__ == "__main__":
    directory_path = 'data/projs/'
    abs_folder = 'data/proj_abs/'
    output_folder = 'data/with_abs/'
    
    #start: 2791430 records
    #rename_files_to_year(directory_path)
    #find_filenames_with_47_records(directory_path)
    #count_rows_in_directory(directory_path)
    #rename_column_in_2012('/home/nisse/Documents/vakken_l/Thesis/nih/originals/july/data/projs/2012.csv')
    #count_rows_in_directory(directory_path)
    #replace_column_name(directory_path)
    #count_rows_in_directory(directory_path)
    #check_column_names(directory_path)
    #count_rows_in_directory(directory_path)
    #process_all_csv_files_in_directory(abs_folder)
    #rename_files_to_year(abs_folder)
    #count_rows_in_directory(directory_path)
    #for year in range(1985, 2024):
    #    merge_files(year, abs_folder, directory_path, output_folder)
    #count_rows_in_directory(directory_path)
    #list_of_cols = ['APPLICATION_ID', 'ACTIVITY', 'ADMINISTERING_IC', 'APPLICATION_TYPE', 'ARRA_FUNDED', 'AWARD_NOTICE_DATE', 'BUDGET_START', 'BUDGET_END', 'CFDA_CODE', 'CORE_PROJECT_NUM', 'ED_INST_TYPE', 'FOA_NUMBER', 'FULL_PROJECT_NUM', 'SUBPROJECT_ID', 'FUNDING_ICs', 'FY', 'IC_NAME', 'NIH_SPENDING_CATS', 'ORG_CITY', 'ORG_COUNTRY', 'ORG_DEPT', 'ORG_DISTRICT', 'ORG_DUNS', 'ORG_FIPS', 'ORG_NAME', 'ORG_STATE', 'ORG_ZIPCODE', 'PHR', 'PI_IDS', 'PI_NAMEs', 'PROGRAM_OFFICER_NAME', 'PROJECT_START', 'PROJECT_END', 'PROJECT_TERMS', 'PROJECT_TITLE', 'SERIAL_NUMBER', 'STUDY_SECTION', 'STUDY_SECTION_NAME', 'SUFFIX', 'SUPPORT_YEAR', 'TOTAL_COST', 'TOTAL_COST_SUB_PROJECT', 'ABSTRACT_TEXT']
    #delete_columns_not_in_list_of_columns_in_directory(output_folder, list_of_cols)
    #count_rows_in_directory(directory_path)
    #add_updated_funding_info('/home/nisse/Documents/vakken_l/Thesis/nih/originals/july/data/RePORTER_PRJFUNDING_C_FY1985_FY1999', output_folder)
    

    #now, take the union of the two groups of files: 1985-1999 and 2000-2023 (saved in withFunding and with_abs)
    #remember to delete the files in with_abs that are now in withFunding (1985-1999), we have put these in folder with_abs_before2000
    #the code for taking the union is in the respective folder
    #now we have part1.csv and part2.csv in the folder named 2files
    #before we join these, we have to align the files

    #then, align the columns of the two files so the order of the columns etc is the same, see 2files
    #then, take the union of the two files and save it as one file: combined_aligned.csv
    #then, we need to filter the important columns and split the PI name, see filter.py in the 2files folder
    #result is processed_combined_aligned_name_split.csv

    #now we have to deal with the grants, this is discussed in the grant_processsing folder


    