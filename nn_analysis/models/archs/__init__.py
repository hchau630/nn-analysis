from torchvision.models import *
from .wide_resnets_simclr_v1 import *
from .big_resnets_infomin import *  # IMPORTANT: NETWORKS IMPORTED FROM THIS LINE DO NOT HAVE FINAL FC LAYERS
from .bw_resnets import *
from .simclr_normalization import add_simclr_normalization as _add_simclr_normalization
from .identity import *

def resnet50_simclr(*args, **kwargs):
    return _add_simclr_normalization(resnet50(*args, **kwargs))

def resnet50_1x_simclr(*args, **kwargs):
    return _add_simclr_normalization(resnet50_1x(*args, **kwargs))

def resnet50_2x_simclr(*args, **kwargs):
    return _add_simclr_normalization(resnet50_2x(*args, **kwargs))