"""Script for cleaning the EC 3.1.1.* data.
"""

import csv

DATA_PATHS = ["ec_3_1_1_seqs_raw.csv", "petases.csv"]
CLEANED_PATH = "ec_3_1_1_seqs_cleaned.txt"

cleaned_data = set()

for path in DATA_PATHS:
    with open(path, 'r', encoding='utf-8') as data_file:
        reader = csv.DictReader(data_file)

        for row in reader:
            seq = row['sequence']
            if (100 <= len(seq) <= 300) and ('X' not in seq):
                cleaned_data.add(seq)

print(len(cleaned_data), "Cleaned Sequences")
exit()

with open(CLEANED_PATH, 'w', newline='') as cleaned_file:
    for row in cleaned_data:
        cleaned_file.write(row + "\n")
