# read sectionii.csv, sectioniii.csv, tables.csv
# these have two columns: Grant Number, Content

# make a new csv file that contains only grant numbers appearing in all three: sectionii.csv, sectioniii.csv, tables.csv

import pandas as pd
import csv

def gatherRFAsallsectionsOK():
    # Read sectionii.csv, sectioniii.csv, tables.csv as latin-1
    sectionii = pd.read_csv('sectionii.csv', encoding='latin-1')
    sectioniii = pd.read_csv('sectioniii.csv', encoding='latin-1')
    tables = pd.read_csv('tables.csv', encoding='latin-1')

    # these have two columns: Grant Number, Content

    # make a new csv file that contains only grant numbers appearing in all three: sectionii.csv, sectioniii.csv, tables.csv

    # get the intersection of the three sets of grant numbers
    grant_numbers = set(sectionii['Grant Number']) & set(sectioniii['Grant Number']) & set(tables['Grant Number'])

    # write the grant numbers to a new csv file
    with open('grant_numbers_complete.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Grant Number'])
        for grant_number in grant_numbers:
            writer.writerow([grant_number])


if __name__ == '__main__':
    gatherRFAsallsectionsOK()