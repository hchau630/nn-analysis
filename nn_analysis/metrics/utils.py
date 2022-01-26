import pathlib
import os

import numpy as np

from nn_analysis import utils
from nn_analysis import exceptions
from nn_analysis.constants import ENV_CONFIG_PATH

env_config = utils.load_config(ENV_CONFIG_PATH)

def _get_data_path(model_name, epoch, layer_name, metric_name, version):
    if epoch is not None:
        path = f"{env_config['save_results_path']}/{metric_name}/{version:02d}/{model_name}/{epoch:04d}/{layer_name}.pkl"
    else:
        path = f"{env_config['save_results_path']}/{metric_name}/{version:02d}/{model_name}/{layer_name}.pkl"
    
    return pathlib.Path(path)

def data_exists(*args, **kwargs):
    return _get_data_path(*args, **kwargs).is_file()

def save_data(model_name, epoch, layer_name, metric_name, version, data_dict, overwrite=False):
    if epoch is not None:
        path = f"{env_config['save_results_path']}/{metric_name}/{version:02d}/{model_name}/{epoch:04d}/{layer_name}.pkl"
    else:
        path = f"{env_config['save_results_path']}/{metric_name}/{version:02d}/{model_name}/{layer_name}.pkl"
    utils.save_data(path, data_dict, overwrite=overwrite)
    
def load_data(*args, **kwargs):
    return utils.load_data(_get_data_path(*args, **kwargs))

def load_neural_data(base_dataset_name, region, standardize=True):
    data = np.load(
        os.path.join(env_config['neural_data_path'], base_dataset_name, f'{region}_data.npy'),
        allow_pickle=True,
        encoding='latin1',
    )
    
    if region == 'V4' and base_dataset_name == 'hvm':
        data = data[:,:,:17]
    
    if base_dataset_name == 'rust':
        data = np.transpose(data, axes=(1,0,2))
#     print(data.shape)
    if standardize and data.ndim == 3:
        # Standardize s.t. trial-averaged firing rate has zero mean and unit variance across images.
        data = data - data.mean(axis=(0, 2), keepdims=True)
        data = data / data.mean(axis=2, keepdims=True).std(axis=0, keepdims=True)
    elif data.ndim == 2:
        # Standardize s.t. voxel values have zero mean and unit variance across images.
        data = data - data.mean(axis=0, keepdims=True)
        data = data / data.std(axis=0, keepdims=True)
    else:
        raise RuntimeError("data should have either 2 or 3 dimensions.")
    return data