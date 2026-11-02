import torch
import torch.nn as nn

class MultimodalFusion(nn.Module):
    def __init__(self,dim=256):
        super().__init__()
        self.fusion=nn.Sequential(
            nn.Linear(dim*4,dim),nn.LayerNorm(dim),nn.ReLU())
    def forward(self,v,s,t,tau):
        return self.fusion(torch.cat([v,s,t,tau],dim=-1))
