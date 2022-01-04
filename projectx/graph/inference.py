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
        for i in range(len(x)):
            x[i] = " ".join(x[i])
        return self.tokenizer(x, return_tensors='pt', max_length=512, truncation=True, padding="max_length").to(self.device)
    
    def decode(self, x):
        x = self.tokenizer.decode(x)
        x = "".join(x.split(" "))
        return x

    def forward(self, x):
        return self.model(**x, output_hidden_states=True)