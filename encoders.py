import torch.nn as nn
import torchvision.models as models

class VisualEncoder(nn.Module):
    def __init__(self,embed_dim=256):
        super().__init__()
        m=models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        m.fc=nn.Identity()
        self.backbone=m
        self.projection=nn.Linear(512,embed_dim)
    def forward(self,x):
        return self.projection(self.backbone(x))

class TemporalEncoder(nn.Module):
    def __init__(self,input_dim=256,hidden_dim=256):
        super().__init__()
        self.gru=nn.GRU(input_dim,hidden_dim,batch_first=True)
    def forward(self,x):
        _,h=self.gru(x)
        return h[-1]

class TextEncoder(nn.Module):
    def __init__(self,vocab_size=10000,embed_dim=256):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,embed_dim)
    def forward(self,tokens):
        return self.embedding(tokens).mean(dim=1)

class SpectralEncoder(nn.Module):
    def __init__(self,bands=16,embed_dim=256):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(bands,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(64,embed_dim))
    def forward(self,x):
        return self.net(x)
