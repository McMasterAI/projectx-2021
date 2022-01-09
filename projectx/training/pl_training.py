import torch
from pytorch_lightning import Trainer 
from projectx.data.datamodule import MethylationDataModule
from projectx.graph.graph import geneGraph
from projectx.model.model import PSPred

if __name__ == "__main__":
    model_params = dict(
        in_channels=1,
        hidden_channels=128,
        num_layers=4,
        out_channels=2, #labels
        drop_prob=0.01
    )

    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create graph
    graph = geneGraph("./Corpuses/string_db.json")
    vocab = graph.vocab # mapping of gene ID to node number
    graph = graph.data
    model_params["graph"] = graph

    # Load in data
    methylation_data = MethylationDataModule("")
    methylation_data.setup()

    # load model
    model = PSPred(**model_params)

    # train
    trainer = Trainer()
    trainer.fit(model, train_dataloaders=methylation_data)


