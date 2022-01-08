import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import ChebConv, GCNConv, PNAConv
from torch_geometric.nn.glob.glob import global_mean_pool


class SimpleGCN(nn.Module):
    def __init__(
        self, in_channels, hidden_channels, num_layers, out_channels, drop_prob=0.0
    ):
        super().__init__()
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

    def forward(self, x, edge_index, batch):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_prob, training=self.training)

        x = self.convs[-1](x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x


def train(model, train_loader, loss_fn, optimizer, device):
    model.train()
    running_loss, n_total = 0, 0

    for data in train_loader:
        # send data to device
        data = data.to(device)
        batch_size = data.y.shape[0]

        # feed forward
        out = model(data.x, data.edge_index, data.batch)

        # compute loss
        loss = loss_fn(out, data.y.view(-1))
        running_loss += loss.item() * batch_size
        n_total += batch_size

        # backprop and update step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return running_loss / n_total


@torch.no_grad()
def test(model, test_loader, loss_fn, device):
    model.eval()
    running_loss, n_total, n_correct = 0, 0, 0

    for data in test_loader:
        # send data to device
        data = data.to(device)
        batch_size = data.y.shape[0]

        # feed forward
        out = model(data.x, data.edge_index, data.batch)

        # compute loss
        loss = loss_fn(out, data.y.view(-1))
        running_loss += loss.item() * batch_size
        n_total += batch_size

        # get predictions
        preds = torch.argmax(out, axis=-1)

        # compute number of correct predictions
        n_correct += torch.sum(preds == data.y.view(-1)).item()

    return running_loss / n_total, n_correct / n_total


def main():
    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load dataset
    dataset = TUDataset("data", name="PROTEINS")

    # train test split
    dataset = dataset.shuffle()

    # test_size_ratio = 0.2
    # split_index = int(len(dataset)*test_size_ratio)
    # train_dataset = dataset[split_index:]
    # val_dataset = dataset[:split_index]

    n = (len(dataset) + 9) // 10  # idk why the example splits like this lol
    test_dataset = dataset[:n]
    val_dataset = dataset[n : 2 * n]
    train_dataset = dataset[2 * n :]
    test_loader = DataLoader(test_dataset, batch_size=20)
    val_loader = DataLoader(val_dataset, batch_size=20)
    train_loader = DataLoader(train_dataset, batch_size=20)

    # model
    model = SimpleGCN(
        in_channels=dataset.num_features,
        hidden_channels=128,
        num_layers=4,
        out_channels=dataset.num_classes,
        drop_prob=0.0,
    ).to(device)

    # loss function
    loss_fn = nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # loaders
    val_loader = DataLoader(val_dataset, batch_size=16)
    train_loader = DataLoader(train_dataset, batch_size=16)

    # train
    test_acc = 0
    best_val_loss = float("inf")
    patience = start_patience = 50
    for epoch in range(1, 100):
        train_loss = train(model, train_loader, loss_fn, optimizer, device)
        _, train_acc = test(model, train_loader, loss_fn, device)
        val_loss, val_acc = test(model, val_loader, loss_fn, device)
        if val_loss < best_val_loss:
            test_loss, test_acc = test(model, test_loader, loss_fn, device)
            patience = start_patience
        else:
            patience -= 1
            if patience == 0:
                break
        print(
            f"Epoch: {epoch:03d}, Train Loss: {train_loss:.3f}, "
            f"Train Acc: {train_acc:.3f}, Val Loss: {val_loss:.3f}, "
            f"Val Acc: {val_acc:.3f}, Test Loss: {test_loss:.3f}, "
            f"Test Acc: {test_acc:.3f}"
        )


if __name__ == "__main__":
    main()
