import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, GCNConv, PNAConv
from torch_geometric.nn.glob.glob import global_mean_pool
import pytorch_lightning as pl

class PSPred(pl.LightningModule):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, graph, drop_prob=0.0):
        super().__init__()
        # self.save_hyperparameters()

        assert in_channels > 0
        assert hidden_channels > 0
        assert num_layers > 0
        assert out_channels > 0

        layers = [in_channels] + [hidden_channels for _ in range(num_layers)]
        self.convs = nn.ModuleList(
            [GCNConv(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        )
        self.fc = nn.Linear(hidden_channels, out_channels)
        self.drop_prob = drop_prob
        self.graph = graph
    
    def forward(self, methyl, idx):
        x = torch.cat([self.graph.x, methyl])
        edge_index = self.graph.edge_index
        
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_prob, training=self.training)

        x - self.convs[-1](x, edge_index)
        x = global_mean_pool(x, self.graph.batch)
        x = self.fc(x)
        return x

    def training_step(self, data, idx):
        methyl, labels = data["methyl"], data["labels"]
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

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.001)