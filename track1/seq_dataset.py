import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

VOCAB = ['<pad>', '<s>', '</s>',
         'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
AA_TO_IDX = {c: i for i, c in enumerate(VOCAB)}
IDX_TO_AA = {i: c for c, i in AA_TO_IDX.items()}


class SeqDataset(Dataset):

    def __init__(self, file_path):
        with open(file_path, 'r') as seq_file:
            contents = seq_file.readlines()

            self.sequences = list(map(int_encode_and_process, contents))
            self.sequences.sort(key=len)

            self.max_len = max(len(s) for s in self.sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, item):
        return self.sequences[item]


# Helpers

def int_encode_and_process(seq):
    seq = seq.rstrip()

    int_encoded = [AA_TO_IDX[char] for char in seq]
    int_encoded.insert(AA_TO_IDX['<s>'], 0)
    int_encoded.append(AA_TO_IDX['</s>'])
    return torch.tensor(int_encoded)


def pad_batch(batch):
    return pad_sequence(batch, True, AA_TO_IDX['<pad>'])
