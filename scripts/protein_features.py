import sys
import json
import torch
import numpy as np
from tqdm import tqdm
from bioservices import UniProt
from projectx.graph.inference import ProtBertModule

# TODO: Pad proteins for equal vectors

def chunks(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)] 

def load_data(pth):
    data = json.load(open(pth))
    string_db_ids = set()
    for k, v in data.items():
        if k: # Empty string
            string_db_ids.add(k)
            string_db_ids.add(v["protein2"])
    
    print("StringDB ids loaded.")
    return list(string_db_ids)

def poll_uniprot(node):
    u = UniProt(verbose=False)
    o = u.search(node, frmt="fasta").split("\n")
    full = u.search(node).split("\n")
    name = full[1].split("\t")[0]
    return (name, "".join(o[1:]))

def featurize(ids, device, batch_size, out_dir):
    print("Loading in ProtBert...")
    protbert = ProtBertModule(device=device)
    batches = chunks(ids, batch_size)

    print("Creating feature vectors...")
    for batch in tqdm(batches):

        # Get data for the batch
        names, fastas = [], []
        for x in batch:
            try:
                name, fasta = poll_uniprot(x)
                names.append(name)
                fastas.append(fasta)
            except:
                print(f"Missing query for {name}")
        
        fastas = protbert.encode(fastas)
        feats = protbert(fastas).hidden_states[-1][:, :, 0] # Output of the last layer of model, at cls token
        feats = torch.squeeze(feats).cpu().detach().numpy()

        # Write to file
        for name, feat in zip(names, feats):
            np.save(open(f"{out_dir}{name}.npy", "wb"), feat)

if __name__ == "__main__":
    device = torch.device("cuda")
    out_dir = "Corpuses/protein_features/"
    batch_size = 2
    ids = load_data("Corpuses/string_db.json")

    featurize(ids, device, batch_size, out_dir)
    print("Done.")
    pass