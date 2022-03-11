from argparse import Namespace

import torch
import torch.nn as nn

from .base import BarlowTwins

class Model(BarlowTwins):
    def __init__(self, args):
        super().__init__(args)
        
        sizes = {'bn': torch.Size([self.projector_sizes[-1]])}
        
        # create embedding matrix
        self.register_buffer("embedding", nn.init.orthogonal_(torch.empty(7, *sizes['bn'])))
        
        self.pos_weight = args.pos_weight
        self.scale_weight = args.scale_weight
        self.color_weight = args.color_weight
        
    def forward(self, x):
        return self.bn(self.projector(self.backbone(x)))
        
    def loss_forward(self, y1, y2, **kwargs):
        # args and kwargs should be on gpu
        delta_pos = torch.cat([kwargs[f'{key}_1'] - kwargs[f'{key}_0'] for key in ['cam_pos_x', 'cam_pos_y']], dim=1)*self.pos_weight
        delta_scale = torch.cat([kwargs[f'{key}_1'] - kwargs[f'{key}_0'] for key in ['cam_scale']], dim=1)*self.scale_weight
        delta_color = torch.cat([kwargs[f'{key}_1'] - kwargs[f'{key}_0'] for key in ['brightness', 'contrast', 'saturation', 'hue']], dim=1)*self.color_weight
        is_not_bw = ((1.0-kwargs['applied_RandomGrayscale_0'])*(1.0-kwargs['applied_RandomGrayscale_1'])).squeeze()
        is_color_jittered = (kwargs['applied_ColorJitter_0']*kwargs['applied_ColorJitter_1']).squeeze()
        delta_color = torch.einsum('b,bm->bm',is_not_bw*is_color_jittered,delta_color)
        
        delta_vec = torch.cat([delta_pos, delta_scale, delta_color], dim=1).matmul(self.embedding)
        z1 = self.bn(self.projector(self.backbone(y1)))
        z1 = z1 + delta_vec
        z2 = self.bn(self.projector(self.backbone(y2)))
        
        # empirical cross-correlation matrix
        c = z1.T @ z2

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)

        # use --scale-loss to multiply the loss by a constant factor
        # see the Issues section of the readme
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().mul(self.args.scale_loss)
        off_diag = off_diagonal(c).pow_(2).sum().mul(self.args.scale_loss)
        loss = on_diag + self.args.lambd * off_diag
        return loss
    
    def get_encoder(self):
        return self.backbone
    
### Utilities ###

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def equivariant_all_bn(projector='8192-8192-8192', batch_size=1024, scale_loss=0.024, lambd=0.0051, pos_weight=0.3, scale_weight=200.0, color_weight=150.0, **kwargs):
    args = Namespace(projector=projector, batch_size=batch_size, scale_loss=scale_loss, lambd=lambd, pos_weight=pos_weight, scale_weight=scale_weight, color_weight=color_weight, **kwargs)
    return Model(args)