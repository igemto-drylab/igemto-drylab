"""Script for cleaning the EC 3.1.1.* data.
"""

import csv

DATA_PATH = "ec_3_1_1_seqs_raw.csv"
CLEANED_PATH = "ec_3_1_1_seqs_cleaned.txt"

cleaned_data = []

with open(DATA_PATH, 'r') as data_file:
    reader = csv.DictReader(data_file)

    for row in reader:
        if 100 <= len(row['sequence']) <= 300:
            cleaned_data.append(row)

with open(CLEANED_PATH, 'w', newline='') as cleaned_file:
    for row in cleaned_data:
        cleaned_file.write(row['sequence'] + "\n")
