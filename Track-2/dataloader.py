import torch
from torch.utils.data import Dataset
import pickle
import os
import numpy as np

class MeltomeUnirepDataset(Dataset):
    '''Creates a dataloader, with the output being a dictionary with the attributes of vector ["vec"] (3*1900) (avg_hidden, final_hidden, final_cell) 
        as a numpy array
        along with another attribute of melting temperature ["meltingTemp"]

    '''
    def __init__(self, URdir, dictPath):
        '''
        Args:
           URdir (string): path to the unirep vector directory
           dictPath (string): path to the protein ID to melting temp dictionary
        '''
        self.URdir = URdir

        self.unirepVecs = []
        for _, _, files in os.walk(URdir):
            for name in files:
                self.unirepVecs.append(name)

        self.protID2MT = pickle.load(open(dictPath, "rb"))

    def __len__(self):
        return len(self.unirepVecs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # accessing the .npy file that is at the ith index
        vec = np.load(os.path.join(self.URdir, self.unirepVecs[idx]))
        # getting the melting temperature from a protein ID to melting temp dictionary
        meltingTemp = self.protID2MT[self.unirepVecs[idx].replace(".npy", "")]
        output = {"vec": torch.tensor(vec), "meltingTemp": torch.tensor(meltingTemp)}

        return output