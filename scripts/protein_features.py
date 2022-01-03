import sys
import json
import torch
import numpy as np
from tqdm import tqdm
from bioservices import UniProt
from projectx.graph.inference import ProtBertModule

# TODO: Pad proteins for equal vectors

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

def featurize(ids, device, out_dir):
    print("Loading in ProtBert...")
    protbert = ProtBertModule(device=device)
    
    output = open(out_dir, "w")

    print("Creating feature vectors...")
    for idx in tqdm(ids):
        name, fasta = poll_uniprot(idx)
        fasta = protbert.encode(fasta)
        feats = protbert(fasta).logits
        feats = torch.squeeze(feats).cpu().detach().numpy()

        # Write to file
        output.write(f"{name}\t{feats.flatten().tolist()}\n")
        print(feats.shape)


if __name__ == "__main__":
    device = torch.device("cuda")
    out_dir = "Corpuses/features.tsv"
    ids = load_data("Corpuses/string_db.json")

    featurize(ids, device, out_dir)
    pass