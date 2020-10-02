"""Script for obtaining the EC 3.1.1.* sequences from UniProt.
"""

import requests

BASE = 'http://www.uniprot.org'
KB_ENDPOINT = '/uniprot/'
TOOL_ENDPOINT = '/uploadlists/'

DATA_PATH = "ec311_raw.csv"

with open(DATA_PATH, 'w+') as data_file:
    data_file.write("id,entry_name,ec,sequence\n")

payload = {'query': f"ec:3.1.1.*",
           'format': 'tab',
           'columns': 'id,entry_name,ec,sequence'}

result = requests.get(BASE + KB_ENDPOINT, params=payload)

if not result.ok:
    exit()

print("Writing Data")

with open(DATA_PATH, 'a') as data_file:
    response = '\n'.join(result.text.split_dataset("\n")[1:-1])
    response = response.replace("\t", ",")
    data_file.write(response)
