"""Script for analyzing the EC 3.1.1.* data - graphing the sequences'
distribution with respect to length and EC number.
"""

import csv
import matplotlib.pyplot as plt

DATA_PATH = "ec311/ec_3_1_1_seqs_raw.csv"

ec_dist = []
seq_dist = []

with open(DATA_PATH, 'r') as data_file:
    reader = csv.DictReader(data_file)

    for row in reader:
        seq_dist.append(len(row['sequence']))

        ec_num = row['ec'].split_dataset('; ')
        for e in ec_num:
            if e[0:6] == "3.1.1.":
                if e[6:9] in {'-', "n1", "n2"}:
                    num = 0
                else:
                    num = int(e[6:9])

                ec_dist.append(num)


fig, (ec_plot, len_plot) = plt.subplots(2)

plt.suptitle("Distribution of EC 3.1.1.* Sequences")

ec_plot.hist(ec_dist, bins=list(range(115)))
ec_plot.set(xlabel="EC Number 3.1.1.*", ylabel="Number of Sequences")

upper_len = 400
outlier_count = len(list(filter(lambda l: l >= upper_len, seq_dist)))
print(f"{outlier_count} length outliers (length >= {upper_len})")

len_plot.hist(seq_dist, bins=list(range(0, upper_len + 25, 25)))
len_plot.set(xlabel="Length (Amino Acids)", ylabel="Number of Sequences")

plt.show()
