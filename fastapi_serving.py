import torch.nn as nn
import torch

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

class Item(BaseModel):
    data: List[float]

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('mps')
        self.model = torch.load("model.pth").to(self.device)
        self.model.eval() 

    def forward(self, x):
        return self.model(x)