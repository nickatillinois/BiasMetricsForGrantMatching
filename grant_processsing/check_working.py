import pandas as pd
import requests

rfa_list = pd.read_csv('foa_list.csv')
base = 'https://grants.nih.gov/grants/guide/rfa-files/'
# take the number of rows in the list
length = len(rfa_list)
# create an empty list
workingRFAs = []
for i in range(0, length):
    print(i/length*100, "%")
    url = base + rfa_list.iloc[i,0] + '.html'
    response = requests.get(url)
    if response.status_code == 200:
        #append the url to the list
        workingRFAs.append(url)
    else:
        #print('Failed to retrieve the webpage. Status code:', response.status_code)
        pass
print("From the list of", length, "RFAs, the number of working RFAs is", len(workingRFAs), "and the number of non-working RFAs is", length - len(workingRFAs), ".")
# save the list to a csv file
workingRFAs = pd.DataFrame(workingRFAs)
workingRFAs.to_csv('workingRFAs.csv', index=False)