import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM


class ProtBertModule(nn.Module):
    def __init__(self, device):
        super(ProtBertModule, self).__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert")
        self.model = AutoModelForMaskedLM.from_pretrained("Rostlab/prot_bert")
        self.model.to(device)
        self.model.eval()

    def encode(self, x):
        x = " ".join(x)
        return self.tokenizer(x, return_tensors='pt').to(self.device)
    
    def decode(self, x):
        x = self.tokenizer.decode(x)
        x = "".join(x.split(" "))
        return x

    def forward(self, x):
        return self.model(**x)