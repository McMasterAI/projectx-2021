import json
import torch
from bioservices import UniProt
from tqdm import tqdm
from torch_geometric.data import Data

class geneGraph:
    def __init__(self, stringdb_dir, batches=20, feat_vecs=None):
        self.device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stringdb = json.load(open(stringdb_dir))
        self.feat_vecs = feat_vecs
        self.batches = batches
        # Create the edges
        self.__create_edges()

        # graph
        self.data = Data(x=self.feat_vecs, edge_index=self.edge_index, batch=self.batches)
    
    def __create_edges(self):
        self.edge_index = [[], []] # starts, ends
        self.vocab = {}
        idx = 0
        for k, v in tqdm(self.stringdb.items()):
            if 'combined_score' in v and int(v['combined_score']) > 500:
                if k not in self.vocab:
                    self.vocab[k] = idx
                    idx += 1
                if v['protein2'] not in self.vocab:
                    self.vocab[v['protein2']] = idx
                    idx += 1
                
                self.edge_index[0].extend([self.vocab[k], self.vocab[v['protein2']]])
                self.edge_index[1].extend([self.vocab[v['protein2']], self.vocab[k]])
        self.edge_index = torch.tensor(self.edge_index, dtype=torch.long)