import torch
import torch.nn.functional as F

def prototype_language_loss(v,t):
    v=F.normalize(v,dim=-1); t=F.normalize(t,dim=-1)
    return 1-(v*t).sum(dim=-1).mean()

def stability_loss(old,new):
    x=[F.mse_loss(new[k],old[k]) for k in old if k in new]
    return torch.stack(x).mean() if x else torch.tensor(0.0)

def mse_loss(pred,target):
    return F.mse_loss(pred,target)
