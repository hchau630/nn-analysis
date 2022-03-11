import pathlib
import os

import numpy as np

from nn_analysis import utils, exceptions
from nn_analysis.constants import ENV_CONFIG_PATH, ACTS_CONFIGS_PATH

env_config = utils.load_config(ENV_CONFIG_PATH)
acts_configs = utils.load_config(ACTS_CONFIGS_PATH)

DEFAULT_SAVE_LOC = 'save_acts_path' # If you change this, make sure to also move the acts you previously saved at this location to the location that you change to.

def _get_data_path(model_name, epoch, acts_name, version, layer_name=None, data_type='y'):
    # print(acts_configs[acts_name])
    if "save_loc" in acts_configs[acts_name][f"{version:02d}"]["kwargs"]:
        load_loc = acts_configs[acts_name][f"{version:02d}"]["kwargs"]["save_loc"]
    else:
        load_loc = DEFAULT_SAVE_LOC
    if epoch is not None:
        path = f"{env_config[load_loc]}/{acts_name}/{version:02d}/{model_name}/{epoch:04d}"
    else:
        path = f"{env_config[load_loc]}/{acts_name}/{version:02d}/{model_name}"
    
    if data_type == 'y' or data_type == 'evr':
        if layer_name is None:
            raise ValueError("layer_name must be provieded if data_type is 'y' or 'evr'")
        return pathlib.Path(f"{path}/{layer_name}/{data_type}.pkl")
    elif data_type == 'x':
        return pathlib.Path(f"{path}/x.pkl")
    else:
        raise NotImplementedError(f"data_type {data_type} not implemented.")

def data_exists(*args, **kwargs):
    return _get_data_path(*args, **kwargs).is_file()

def save_data(model_name, epoch, acts_name, version, data_dict, layer_name=None, save_loc=DEFAULT_SAVE_LOC, overwrite=False):
    if epoch is not None:
        path = f"{env_config[save_loc]}/{acts_name}/{version:02d}/{model_name}/{epoch:04d}"
    else:
        path = f"{env_config[save_loc]}/{acts_name}/{version:02d}/{model_name}"
        
    with utils.SimulFileHandler(*[f"{path}/{layer_name}/{k}.pkl" for k in data_dict.keys() if k in ['y', 'evr']]):
        for k, v in data_dict.items():
            if k == 'x':
                path_x = pathlib.Path(f"{path}/x.pkl")
                if not overwrite and path_x.is_file():
                    existing_x = utils.load_data(path_x)
                    if np.allclose(existing_x, v):
                        return
                    print(existing_x)
                    print(v)
                    raise exceptions.PathAlreadyExists(f"The data 'x' already exists for {model_name} {epoch} {acts_name} v{version} and is different from the provided data 'x'. If you want override this error, set overwrite=True.")

                utils.save_data(f"{path}/x.pkl", v, overwrite=overwrite)

            elif k == 'y' or k == 'evr':
                if layer_name is None:
                    raise ValueError("layer_name must be provided if the data_dict to be saved contains keys 'y' or 'evr'.")
                utils.save_data(f"{path}/{layer_name}/{k}.pkl", v, overwrite=overwrite)
            else:
                raise NotImplementedError(f"data_dict key {k} not implemented.")
    
def load_data(*args, **kwargs):
    return utils.load_data(_get_data_path(*args, **kwargs))

def assert_consistent_x(acts_name, version):
    if "save_loc" in acts_configs[acts_name][f"{version:02d}"]["kwargs"]:
        load_loc = acts_configs[acts_name][f"{version:02d}"]["kwargs"]["save_loc"]
    else:
        load_loc = DEFAULT_SAVE_LOC
    path = f"{env_config[load_loc]}/{acts_name}/{version:02d}"
    xs = []
    for cur_path, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename == 'x.pkl':
                xs.append(utils.load_data(os.path.join(cur_path, filename)))
    if len(xs) > 0:
        for x in xs[1:]:
            assert np.allclose(x, xs[0])
    print(f"All x.pkl files under {acts_name} v{version} are consistent: checked {len(xs)} files.")
