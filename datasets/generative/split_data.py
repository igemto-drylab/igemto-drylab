import pathlib
import random

import pandas as pd


def keep_seq(s):
    return isinstance(s, str) and (200 <= len(s) <= 320) and ('X' not in s)


def add_reverse_seqs(seqs):
    reverse_seqs = [s[::-1] for s in seqs]
    seqs.extend(reverse_seqs)
    seqs.sort()  # for consistency


def split_dataset(seqs, split_ratio, data_dir):
    seqs = list(sorted(seqs))

    train_size = int(split_ratio[0] * len(seqs))
    val_size = int(split_ratio[1] * len(seqs))

    train_set = seqs[:train_size]
    val_set = seqs[:train_size + val_size]
    test_set = seqs[train_size + val_size:]

    add_reverse_seqs(train_set)
    add_reverse_seqs(val_set)
    add_reverse_seqs(test_set)

    with open(data_dir / 'train.txt', 'w+') as f:
        f.write('\n'.join(train_set))
    with open(data_dir / 'val.txt', 'w+') as f:
        f.write('\n'.join(val_set))
    with open(data_dir / 'test.txt', 'w+') as f:
        f.write('\n'.join(test_set))


def main():
    split_ratio = [0.8, 0.1, 0.1]
    data_dir = pathlib.Path(__file__).parent

    ec311 = pd.read_csv(data_dir / 'ec311' / 'ec311_raw.csv')
    ec311 = ec311[ec311['sequence'].map(keep_seq)]
    ec311 = set(ec311['sequence'].unique())

    petases = pd.read_csv(data_dir / 'petases' / 'petases.csv')
    petases = petases[petases['sequence'].map(keep_seq)]
    petases = set(petases['sequence'].unique())

    ec311 -= petases

    print(f"EC311 - {len(ec311)} Sequences")
    print(f"Petases -> {len(petases)} Sequences")

    split_dataset(ec311, split_ratio, data_dir / 'ec311')
    split_dataset(petases, split_ratio, data_dir / 'petases')


if __name__ == '__main__':

    seed = 777  # lucky 7s : )
    random.seed(seed)

    main()
