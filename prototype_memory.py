import torch
import torch.nn.functional as F

class PrototypeMemory:
    def __init__(self):
        self.memory={}
    def add_prototype(self,class_id,prototype):
        self.memory[int(class_id)]=prototype.detach().clone()
    def prototypes(self):
        return self.memory
    def classify(self,query):
        q=F.normalize(query.unsqueeze(0),dim=-1)
        scores={}
        for cid,p in self.memory.items():
            scores[cid]=F.cosine_similarity(q,F.normalize(p.unsqueeze(0),dim=-1)).item()
        return max(scores,key=scores.get),scores

def compute_prototype(features):
    return features.mean(dim=0)

def compute_reliable_prototype(features,reliability):
    w=reliability/(reliability.sum()+1e-8)
    return torch.sum(w.unsqueeze(1)*features,dim=0)
