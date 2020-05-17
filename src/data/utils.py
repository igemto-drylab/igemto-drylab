"""Some useful data manipulation methods and variables, such as the amino acid
alphabet and conversion methods between string, integer encoded, one-hot
encoded protein representations.

Note that '_' represents an empty space in our representations, making a total
of 21 characters in the protein sequence alphabet.
"""

from typing import List, Tuple

import torch
from torch import Tensor

amino_acids = {'G', 'A', 'V', 'L', 'I', 'P', 'F', 'Y', 'W', 'S',
               'T', 'C', 'M', 'N', 'Q', 'D', 'E', 'K', 'R', 'H'}

# Building the amino acid to index code for encoding
# Here, '_' represents an empty space for padding
aa_to_index = {a: i for i, a in enumerate(sorted(amino_acids | {'_'}))}
index_to_aa = {i: a for a, i in aa_to_index.items()}


def seq_to_int_encoded(seq: str) -> Tuple[int, ...]:
    """Converts a protein string sequence to a integer encoded tuple, in
    accordance with <aa_to_index>.
    """
    int_encoded = [aa_to_index[a] for a in seq]
    return tuple(int_encoded)


def seq_to_one_hot(seq: str) -> Tensor:
    """Converts a protein string sequence to a one-hot encoded Tensor, in
    accordance with <aa_to_index>.
    """
    int_encoded = seq_to_int_encoded(seq)
    one_hot = [[float(idx == i) for i in range(21)]
               for idx in int_encoded]
    return torch.tensor(one_hot)


def one_hot_to_seq(one_hot: Tensor) -> List[str]:
    """Converts a one-hot encoded protein vector into its lettered string
     representation. Recall that <one_hot> is expected to be a Tensor of size
     (batch, seq_len, 21). This function will take the maximum element along
     the second dimension to be index of the amino acid.
    """
    int_encoded = one_hot.argmax(dim=2)

    sequences = []
    for protein in int_encoded:
        seq = "".join(map(lambda i: index_to_aa[i], protein.tolist()))
        seq = seq.replace('_', '')  # remove empty spaces
        sequences.append(seq)

    return sequences
