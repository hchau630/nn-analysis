import torch

from nn_analysis.models import archs
from nn_analysis import utils
from nn_analysis import exceptions
from nn_analysis.constants import ENV_CONFIG_PATH

env_config = utils.load_config(ENV_CONFIG_PATH)

def get_custom_model(arch, path=None, extract_method=None, epoch=None, model_kwargs={}, device='cpu'):
    archs_dict = {k: v for k, v in archs.__dict__.items() if not k.startswith("__") and callable(v) and k.islower()}
    model = archs_dict[arch](**model_kwargs)
    
    if arch == 'identity':
        model.to(device)
        return model
    
    if path is None:
        raise exceptions.ConfigurationError("Model configuration 'path' is not set. 'path' must be set when arch is not 'identity'.")
    
    path = f"{env_config['model_base_path']}/{path}"
    if epoch is not None:
        path = f'{path}/{epoch:04d}.pth.tar'

    for name, param in model.named_parameters():
        param.requires_grad = False
    
    with open(path, 'rb') as f:
        state_dict = torch.load(f, map_location="cpu")['state_dict']

    if extract_method == 'moco':
        new_state_dict = {}

        encoder_module = 'module.encoder_q'
        old_fc_module = 'module.encoder_q.fc'

        for k, v in state_dict.items():
            if k.startswith(encoder_module) and not k.startswith(old_fc_module):
                new_state_dict[k[len(encoder_module)+1:]] = v

        state_dict = new_state_dict

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")
    
    model.to(device)
    
    return model