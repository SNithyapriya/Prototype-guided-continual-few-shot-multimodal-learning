import torch
import torch.nn as nn

class ProgressionForecaster(nn.Module):
    def __init__(self,input_dim=256,hidden_dim=128):
        super().__init__()
        self.gru=nn.GRU(input_dim,hidden_dim,batch_first=True)
        self.head=nn.Linear(hidden_dim,1)
    def forward(self,x):
        _,h=self.gru(x)
        return self.head(h[-1])
