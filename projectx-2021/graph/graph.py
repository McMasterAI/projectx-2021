import json
import torch
from bioservices import UniProt
from tqdm import tqdm
from inference import ProtBertModule
from torch_geometric.data import Data

class geneGraph:
    def __init__(self, stringdb_dir, device="cpu"):
        self.device = torch.device(device)
        self.stringdb = json.load(open(stringdb_dir))
        # Create the edges
        self.__create_edges()
        # Create node features
        self.__create_features()
        # graph
        self.data = Data(x=self.feat_vecs, edge_index=self.edge_index)
    
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
    
    def __create_features(self):
        print("Loading ProtBert...")
        protbert = ProtBertModule(self.device)

        # Uniprot Lookup for each node
        u = UniProt(verbose=False)

        self.feat_vecs = []

        for node in tqdm(self.vocab.keys()):
            o = u.search(node, frmt="fasta").split("\n")
            fasta = "".join(o[1:])
            
            # inference
            fasta = protbert.encode(fasta)
            preds = protbert(fasta).logits
            preds = torch.squeeze(preds).cpu().detach().numpy()
            self.feat_vecs.append(preds)
            del preds
            del fasta