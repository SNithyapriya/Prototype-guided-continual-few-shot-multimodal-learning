import torch
import torch.nn.functional as F

def create_k_shot_episode(features,labels,k):
    support=[]; query=[]
    for c in torch.unique(labels):
        idx=torch.where(labels==c)[0]
        idx=idx[torch.randperm(len(idx))]
        if len(idx)>=k+1:
            support.extend(idx[:k].tolist()); query.extend(idx[k:].tolist())
    return features[support],labels[support],features[query],labels[query]

def build_prototypes(features,labels):
    return {int(c):features[labels==c].mean(0) for c in torch.unique(labels)}

def classify_queries(queries,prototypes):
    out=[]
    for q in queries:
        scores={c:F.cosine_similarity(q[None,:],p[None,:]).item()
                for c,p in prototypes.items()}
        out.append(max(scores,key=scores.get))
    return torch.tensor(out)
