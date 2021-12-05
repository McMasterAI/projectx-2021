# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.3
#   kernelspec:
#     display_name: 'Python 3.9.7 64-bit (''gnn'': conda)'
#     language: python
#     name: python3
# ---

# %%
import torch
from torch_geometric.data import Data
from tqdm import tqdm

# %% [markdown]
# # Creating Edges
# Index string_db, and connect nodes with a combined score over 500 (arbitrary)

# %%
import json

stringdb = json.load(open("../Corpuses/string_db.json"))

# %% [markdown]
# Mapping all gene names to a vocab, then creating an edge connection tensor

# %%
edge_index = [[], []] # starts, ends
vocab = {}
idx = 0
for k, v in tqdm(stringdb.items()):
    if 'combined_score' in v and int(v['combined_score']) > 500:
        if k not in vocab:
            vocab[k] = idx
            idx += 1
        if v['protein2'] not in vocab:
            vocab[v['protein2']] = idx
            idx += 1
        
        edge_index[0].extend([vocab[k], vocab[v['protein2']]])
        edge_index[1].extend([vocab[v['protein2']], vocab[k]])
edge_index = torch.tensor(edge_index, dtype=torch.long)
print(edge_index.shape)

# %% [markdown]
# # Creating Node Feature Vectors

# %%
import sys
import torch
from bioservices import UniProt
sys.path.append("../")
from protbert.inference import ProtBertModule

protbert = ProtBertModule(torch.device('cpu'))
u = UniProt(verbose=False)

# %%
from pprint import pprint
from tqdm import tqdm

feat_vecs = [] # attach in protbert inference here

for node in tqdm(vocab.keys()):
    # Finding associated protein for each gene
    o = u.search(node, frmt="fasta").split("\n")
    fasta = "".join(o[1:])
    
    # inference
    fasta = protbert.encode(fasta)
    preds = protbert(fasta).logits
    preds = torch.squeeze(preds).cpu().detach().numpy()
    feat_vecs.append(preds)
    del preds
    del fasta

# %%
data = Data(x=feat_vecs, edge_index=edge_index)
