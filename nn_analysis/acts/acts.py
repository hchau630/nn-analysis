import torch

from nn_analysis import utils
from nn_analysis import datasets as ds
from nn_analysis import models as md
from nn_analysis import acts as ac
from nn_analysis.constants import ENV_CONFIG_PATH, MODEL_CONFIGS_PATH, ARCH_CONFIGS_PATH

model_configs = utils.load_config(MODEL_CONFIGS_PATH)
arch_configs = utils.load_config(ARCH_CONFIGS_PATH)
env_config = utils.load_config(ENV_CONFIG_PATH)

def transform_data(data_list, transform):
    if transform == 'identity':
        assert len(data_list) == 1
        data = data_list[0]
    elif transform == 'diff':
        assert len(data_list) == 2
        data = data_list[1] - data_list[0]
    else:
        raise NotImplementedError(f"The transform {transform} is not implemented.")
    return data

class Acts:
    def __init__(self, model_name, epoch, dataset_configs):
        self.model_name = model_name
        self.epoch = epoch
        self.dataset_configs = dataset_configs
        self.dataset_names = list(dataset_configs.keys())
        
    def save_tmp_acts(self, tmp_acts_file, layer_names, n_samples=10000, batch_size=256, verbose=False):
        # Configure model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = md.get_custom_model(epoch=self.epoch, **model_configs[self.model_name], device=device)
        
        # Configure datasets
        datasets = [ds.get_custom_dataset(**self.dataset_configs[dataset_name]) for dataset_name in self.dataset_names]
        seeds = [self.dataset_configs[dataset_name]['seed'] for dataset_name in self.dataset_names]
        
        # Configure postprocessor
        postprocessor = ac.core.Sampler(
            n_samples,
            set_seed=True, # if set_seed=True, then the neurons sampled from each layer of a model will be consistent across trials
        )
        
        # Save model activations temporarily
        ac.core.save(
            tmp_acts_file,
            model,
            self.model_name,
            self.epoch,
            layer_names,
            datasets,
            self.dataset_names,
            seeds=seeds,
            device=device,
            batch_size=batch_size,
            postprocessor=postprocessor,
            dtype='float32',
            log=verbose,
        )
        
    def get_x(self, tmp_acts_file, transform='identity'):
        _xs = []
        for dataset_name in self.dataset_names:
            _x = ac.core.load_x(tmp_acts_file, self.model_name, self.epoch, dataset_name) # (images, targets)
            _xs.append(_x)
        x = transform_data(_xs, transform)
        return {'x': x}
    
    def get_y(self, tmp_acts_file, layer_name, transform='identity', n_pcs=None, svd_solver='auto'):
        _ys = []
        for dataset_name in self.dataset_names:
            _y = ac.core.load(tmp_acts_file, self.model_name, self.epoch, dataset_name, layer_name) # (images, neurons)
            _ys.append(_y)
        y = transform_data(_ys, transform)

        # Compute PCs
        if n_pcs is not None:
            y, evr = utils.get_pcs(y, n_pcs, svd_solver=svd_solver)
            return {'y': y, 'evr': evr}
        return {'y': y}
