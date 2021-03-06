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
    def __init__(self, URdir, dictPath, data="train", fusion=True):
        '''
        Args:
           URdir (string): path to the unirep vector directory
           dictPath (string): path to the protein ID to melting temp dictionary
        '''
        if data == "train":
            mod = "train"
        elif data == "val":
            mod = "val"
        elif data == "test":
            mod = "test"
        else:
            print("invalid data! you dun goofed")

        self.fusion = fusion

        self.URdir = os.path.join(URdir, mod)

        self.unirepVecs = []
        for _, _, files in os.walk(self.URdir):
            for name in files:
                self.unirepVecs.append(name)

        self.protID2MT = pickle.load(open(dictPath, "rb"))

    def __len__(self):
        return len(self.unirepVecs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # accessing the .npy file that is at the ith index
        vec = np.load(os.path.join(self.URdir, self.unirepVecs[idx]), allow_pickle=True)
        # getting the melting temperature from a protein ID to melting temp dictionary
        if not self.fusion:
            vec = vec[0]
        meltingTemp = self.protID2MT[self.unirepVecs[idx].replace(".npy", "")]
        output = {"vec": torch.tensor(vec), "meltingTemp": torch.tensor(meltingTemp)}

        return output
    
    def normalize(self, lbound, ubound):
        melting_temps = [self.protID2MT[encoding.replace(".npy", "")] for encoding in self.unirepVecs]
        max_val = max(melting_temps)
        min_val = min(melting_temps)
        range_val = max_val - min_val
        norm_range = ubound - lbound

        for encoding in self.unirepVecs:
            curr_val = self.protID2MT[encoding.replace(".npy", "")]
            self.protID2MT[encoding.replace(".npy", "")] = norm_range * (curr_val - min_val) / (range_val) + lbound
        
        return max_val, min_val