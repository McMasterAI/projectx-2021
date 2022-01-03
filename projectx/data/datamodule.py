import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch_geometric.nn as tgnn
import pytorch_lightning as pl
from dataset import MethylationDataset

class MethylationDataModule(pl.LightningDataModule):
    """Handling the methylation data in batches"""
    def __init__(self, data_dir, feat_dim):
        super().__init__()
        self.data_dir = data_dir
        self.feat_dim = feat_dim

    def setup(self, data=None):
        # Load in the data
        # Methylation data structured as feature vectors padded to size N

        # Generate random training data
        if not data:
            data = []

            for i in range(100):
                cur = []
                for j in range(len(vocab.keys())):
                    cur.append((random.random(), random.randint(0,1))) # random label 
                data.append(cur)
        

        train, test, val = np.split(data, [int(len(data)*0.8), int(len(data)*0.9)])

        self.train_dataset = MethylationDataset(train)
        self.test_dataset = MethylationDataset(test)
        self.val_dataset = MethylationDataset(val)

    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset)

    def val_dataloader(self):
        return DataLoader(self.val_dataset)
    
    

    
