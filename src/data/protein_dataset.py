from csv import DictReader
from typing import List

from torch import Tensor
from torch.utils.data import Dataset

from src.data.utils import seq_to_one_hot


class ProTextDataset(Dataset):
    """A indexed data set of protein sequences, derived from a csv text file.

    Attributes
        seq_len: the fixed length of the protein sequences.
    """
    seq_len: int
    data_set: List[str]

    def __init__(self, file_path: str, column: str = 'sequence') -> None:
        """Initializes this data set of protein sequences from the data stored
        at <file_path>. The file is expected to be a csv file with a column
        named <column>, which holds the protein sequences.
        """
        with open(file_path, 'r') as data_file:
            proteins = list(map(lambda x: x[column], DictReader(data_file)))

        # get maximum sequence length
        self.seq_len = max(len(s) for s in proteins)

        # build padded data set
        self.data_set = []
        for p in proteins:
            padded = p + '_' * (self.seq_len - len(p))
            self.data_set.append(padded)

        print("--> Dataset initialized")

    def __len__(self) -> int:
        """Returns the length of this data set.
        """
        return len(self.data_set)

    def __getitem__(self, idx: int) -> Tensor:
        """Returns the protein sequence indexed at <idx> in the form of
        a one-hot encoded Tensor of size (seq_len, 21).
        """
        seq = self.data_set[idx]
        return seq_to_one_hot(seq)  # encoded upon request to conserve memory
