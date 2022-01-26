import torch

__all__ = ['identity']

class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pixels = torch.nn.Identity()
        
    def forward(self, x):
        return self.pixels(x)
    
def identity(**kwargs):
    return Identity()