import pandas as pd





df = pd.read_csv('data.csv')
print("the number of records in df",len(df))
df.loc[df['GRANT_VALUE'] < 30, 'GRANT_VALUE'] *= 1_000_000
#df.loc[df['GRANT_VALUE'] < 1000, 'GRANT_VALUE'] *= 1_000
inflation_rates = {
    1985: 2.92, 1986: 2.87, 1987: 2.77, 1988: 2.66, 1989: 2.53,
    1990: 2.40, 1991: 2.31, 1992: 2.24, 1993: 2.17, 1994: 2.12,
    1995: 2.06, 1996: 2.00, 1997: 1.96, 1998: 1.93, 1999: 1.89,
    2000: 1.82, 2001: 1.78, 2002: 1.75, 2003: 1.71, 2004: 1.66,
    2005: 1.61, 2006: 1.56, 2007: 1.52, 2008: 1.46, 2009: 1.46,
    2010: 1.44, 2011: 1.40, 2012: 1.37, 2013: 1.35, 2014: 1.33,
    2015: 1.33, 2016: 1.31, 2017: 1.25, 2018: 1.25, 2019: 1.23,
    2020: 1.23, 2021: 1.16, 2022: 1.07, 2023: 1.03
}

# Update GRANT_VALUE for FOA_NUMBER 'RFA-ES-14-004' based on the inflation rate of the year in the FY column
df.loc[df['FOA_NUMBER'] == 'RFA-ES-14-004', 'GRANT_VALUE'] = 100_000 * df['FY'].map(inflation_rates)
df.loc[df['FOA_NUMBER'] == 'RFA-ES-19-010', 'GRANT_VALUE'] = 100_000 * df['FY'].map(inflation_rates)

df.loc[df['FOA_NUMBER'] == 'RFA-MD-13-002', 'GRANT_VALUE'] = 250_000 * df['FY'].map(inflation_rates)
df.loc[df['FOA_NUMBER'] == 'RFA-AA-12-007', 'GRANT_VALUE'] = 2_000_000 * df['FY'].map(inflation_rates)
df.loc[df['FOA_NUMBER'] == 'RFA-HD-09-004', 'GRANT_VALUE'] = 750_000 * df['FY'].map(inflation_rates)
df.loc[df['FOA_NUMBER'] == 'RFA-NS-13-004', 'GRANT_VALUE'] = 650_000 * df['FY'].map(inflation_rates)
df.loc[df['FOA_NUMBER'] == 'RFA-DK-11-015', 'GRANT_VALUE'] = 1_000_000 * df['FY'].map(inflation_rates)
df.loc[df['FOA_NUMBER'] == 'RFA-DK-22-012', 'GRANT_VALUE'] = 96_000 * df['FY'].map(inflation_rates)
df.loc[df['FOA_NUMBER'] == 'RFA-DE-17-008', 'GRANT_VALUE'] = 150_007.5 * df['FY'].map(inflation_rates)
df.loc[df['FOA_NUMBER'] == 'RFA-FD-23-010', 'GRANT_VALUE'] = 500_000 * df['FY'].map(inflation_rates)
df.loc[df['FOA_NUMBER'] == 'RFA-HD-23-012', 'GRANT_VALUE'] = 988_000 * df['FY'].map(inflation_rates)
df.loc[df['FOA_NUMBER'] == 'RFA-HD-12-189', 'GRANT_VALUE'] = 400_000 * df['FY'].map(inflation_rates)
df.loc[df['FOA_NUMBER'] == 'RFA-DK-12-003', 'GRANT_VALUE'] = 200_000 * df['FY'].map(inflation_rates)
df.loc[df['FOA_NUMBER'] == 'RFA-DE-23-001', 'GRANT_VALUE'] = 250_000 * df['FY'].map(inflation_rates)
df.loc[df['FOA_NUMBER'] == 'RFA-HG-22-004', 'GRANT_VALUE'] = 350_000 * df['FY'].map(inflation_rates)
df.loc[df['FOA_NUMBER'] == 'RFA-DK-22-006', 'GRANT_VALUE'] = 750_000 * df['FY'].map(inflation_rates)
df.loc[df['FOA_NUMBER'] == 'RFA-HD-10-007', 'GRANT_VALUE'] = 400_000 * df['FY'].map(inflation_rates)
df.loc[df['FOA_NUMBER'] == 'RFA-HD-23-010', 'GRANT_VALUE'] = 425_000 * df['FY'].map(inflation_rates)
df.loc[df['FOA_NUMBER'] == 'RFA-DK-22-004', 'GRANT_VALUE'] = 135_000 * df['FY'].map(inflation_rates)

df.loc[df['FOA_NUMBER'] == 'RFA-DK-10-009', 'GRANT_VALUE'] = 200_000 * df['FY'].map(inflation_rates)
df.loc[df['FOA_NUMBER'] == 'RFA-RM-06-006', 'GRANT_VALUE'] = 325_000 * df['FY'].map(inflation_rates)
df.loc[df['FOA_NUMBER'] == 'RFA-FD-23-010', 'GRANT_VALUE'] = 500_000 * df['FY'].map(inflation_rates)
df.loc[df['FOA_NUMBER'] == 'RFA-DK-22-012', 'GRANT_VALUE'] = 96_000 * df['FY'].map(inflation_rates)
df.loc[df['FOA_NUMBER'] == 'RFA-DE-17-008', 'GRANT_VALUE'] = 150_000 * df['FY'].map(inflation_rates)
df.loc[df['FOA_NUMBER'] == 'RFA-AA-12-007', 'GRANT_VALUE'] = 2_000_000 * df['FY'].map(inflation_rates)




# Print FOA_NUMBER and GRANT_VALUE if GRANT_VALUE is less than 100,000
#print(df.loc[df['GRANT_VALUE'] < 100_000, ['FOA_NUMBER']])

# Save the updated DataFrame to a new CSV file
#df.to_csv('updated_projects_labelled_inflation_adjusted.csv', index=False)
