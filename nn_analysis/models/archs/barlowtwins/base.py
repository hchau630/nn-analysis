from argparse import Namespace

import torch
import torch.nn as nn
import torchvision.models as models

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class BarlowTwins(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = models.resnet50(zero_init_residual=True)
        self.backbone.fc = nn.Identity()

        # projector
        sizes = [2048] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)
        self.projector_sizes = sizes

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)
        
    def forward(self, x):
        return self.bn(self.projector(self.backbone(x)))

    def loss_forward(self, y1, y2):
        z1 = self.projector(self.backbone(y1))
        z2 = self.projector(self.backbone(y2))

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)

        # use --scale-loss to multiply the loss by a constant factor
        # see the Issues section of the readme
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().mul(self.args.scale_loss)
        off_diag = off_diagonal(c).pow_(2).sum().mul(self.args.scale_loss)
        loss = on_diag + self.args.lambd * off_diag
        return loss
    
def barlowtwins(projector='8192-8192-8192', batch_size=1024, scale_loss=0.024, lambd=0.0051, **kwargs):
    args = Namespace(projector=projector, batch_size=batch_size, scale_loss=scale_loss, lambd=lambd, **kwargs)
    return BarlowTwins(args)