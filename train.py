import torch
import torch.nn as nn
from config import DEVICE,EMBED_DIM,LR,EPOCHS
from models.encoders import VisualEncoder
from models.fusion import MultimodalFusion

def main():
    visual=VisualEncoder(EMBED_DIM).to(DEVICE)
    fusion=MultimodalFusion(EMBED_DIM).to(DEVICE)
    optimizer=torch.optim.AdamW(list(visual.parameters())+list(fusion.parameters()),lr=LR)
    print("Device:",DEVICE)
    print("Prototype-guided multimodal training scaffold initialized.")
    print("Connect verified dataset loaders before reporting experiments.")

if __name__=="__main__": main()
