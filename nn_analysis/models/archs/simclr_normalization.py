import torch
import torch.nn as nn

class SimclrNormalization(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()
        self.mean = mean
        self.std = std
        
    def forward(self, x):
        dtype = x.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=x.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=x.device)
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        x = x * std + mean # undo imagenet normalization
        return x
    
def add_simclr_normalization(model):
    model.preprocess = SimclrNormalization()
    def hook(module, inp):
        return model.preprocess(inp[0]) # inp is a tuple
    model.register_forward_pre_hook(hook)
    return model