"""Script for cleaning the EC 3.1.1.* data.
"""

import csv
import os
import random
import matplotlib.pyplot as plt

SEED = 777
SPLIT_RATIO = [0.8, 0.1, 0.1]
DATA_PATHS = ["ec_3_1_1.csv", "petases.csv"]

random.seed(SEED)  # lucky 7 : )

for path in DATA_PATHS:

    train_path = os.path.splitext(path)[0] + "_cleaned_train.txt"
    val_path = os.path.splitext(path)[0] + "_cleaned_val.txt"
    test_path = os.path.splitext(path)[0] + "_cleaned_test.txt"

    cleaned_data = set()

    with open(path, 'r', encoding='utf-8') as data_file:
        reader = csv.DictReader(data_file)

        for row in reader:
            seq = row['sequence']
            if (100 <= len(seq) <= 300) and ('X' not in seq):
                cleaned_data.add(seq)

    cleaned_data = list(cleaned_data)
    random.shuffle(cleaned_data)

    train_val_idx = int(SPLIT_RATIO[0] * len(cleaned_data))
    val_test_idx = train_val_idx + int(SPLIT_RATIO[1] * len(cleaned_data))

    train_data = cleaned_data[:train_val_idx]
    val_data = cleaned_data[train_val_idx:val_test_idx]
    test_data = cleaned_data[val_test_idx:]

    # histogram of lengths
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    ax1.hist([len(s) for s in train_data], bins=50)
    ax2.hist([len(s) for s in val_data], bins=50)
    ax3.hist([len(s) for s in test_data], bins=50)
    plt.show()

    with open(train_path, 'w', newline='') as file:
        for row in train_data:
            file.write(row + "\n")

    with open(val_path, 'w', newline='') as file:
        for row in val_data:
            file.write(row + "\n")

    with open(test_path, 'w', newline='') as file:
        for row in test_data:
            file.write(row + "\n")
