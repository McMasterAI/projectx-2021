import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import torch_geometric.nn as tgnn
import pytorch_lightning as pl

class PSPred(pl.LightningModule):
    def __init__(self, num_classes, num_nodes, graph):
        super().__init__(self)
        self.save_hyperparameters()

        self.mlp = nn.Sequential(
            tgnn.CGConv(self.hparams.num_nodes + graph.x.shape[1], 128),
            nn.ReLU(inplace=True),
            tgnn.CGConv(128, self.hparams.num_classes),
        )
    
    def forward(self, methyl, idx):
        gnn = torch.cat([self.graph.x, methyl])
        gnn = self.mlp(gnn)
        return gnn

    def training_step(self, batch, idx):
        methyl, labels = batch
        preds = self.forward(methyl, idx)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(preds, labels)

        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, idx):
        methyl, labels = batch
        preds = self.forward(methyl, idx)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(preds, labels)

        self.log("val_loss", loss)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=1, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1)