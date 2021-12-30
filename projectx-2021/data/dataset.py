import torch
from torch.utils.data import Dataset

# TODO: change to match proper formatting once data prep is finished

class MethylationDataset(Dataset):
    """Torch Dataset for handling methylation data."""
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cur = self.data[idx]
        return {"data": cur[0], "label": cur[1]}
