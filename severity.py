import torch

def severity_stage(score):
    if score<33.3:return "Low"
    if score<66.7:return "Moderate"
    return "High"

def severity_metrics(pred,target):
    p=torch.tensor(pred,dtype=torch.float32); t=torch.tensor(target,dtype=torch.float32)
    mae=torch.mean(torch.abs(p-t)).item()
    rmse=torch.sqrt(torch.mean((p-t)**2)).item()
    r2=(1-torch.sum((t-p)**2)/(torch.sum((t-t.mean())**2)+1e-8)).item()
    return {"MAE":mae,"RMSE":rmse,"R2":r2}
