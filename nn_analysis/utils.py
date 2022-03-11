from functools import wraps
import contextlib
import random
import pathlib
import pickle
import json
import os
import sys
import signal

import numpy as np
import torch
from sklearn.decomposition import PCA

from nn_analysis import exceptions
from nn_analysis.constants import ARCH_CONFIGS_PATH, MODEL_CONFIGS_PATH

### IO utils ### 

class SimulFileHandler:
    """
    A context manager that ensures all the files asscoiated with filenames 
    must coexist. If any one file is missing upon exiting the with block
    (either because of an exception, a SIGTERM, or normal exiting),
    all the files will be deleted.
    """
    def __init__(self, *filenames):
        self.filenames = filenames
        
    def cleanup(self):
        if all([os.path.isfile(filename) for filename in self.filenames]):
            print("All files exist. No cleanup needed.")
            return
        print("Cleaning up...")
        for filename in self.filenames:
            if os.path.isfile(filename):
                os.remove(filename)
        print("Finished cleanup.")
        
    def handler(self, signum, frame):
        print(f"Received signal {signal.strsignal(signum)}.")
        sys.exit(0) # This throws the exception SystemExit, which is then caught by __exit__ and triggers self.cleanup()
    
    def __enter__(self):
        self.old_sigterm = signal.signal(signal.SIGTERM, self.handler)
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            print(f"Caught exception {exc_type}: {exc_val}")
        self.cleanup()
        signal.signal(signal.SIGTERM, self.old_sigterm)


class TmpFileHandler:
    """
    A context manager that removes all temporary files with names tmp_filenames
    upon encountering an exception or a termination signal SIGTERM.
    """
    def __init__(self, *tmp_filenames):
        self.tmp_filenames = tmp_filenames
        
    def cleanup(self):
        print("Cleaning up...")
        for tmp_filename in self.tmp_filenames:
            if os.path.isfile(tmp_filename):
                os.remove(tmp_filename)
        print("Finished cleanup.")
        
    def handler(self, signum, frame):
        print(f"Received signal {signal.strsignal(signum)}.")
        sys.exit(0) # This throws the exception SystemExit, which is then caught by __exit__ and triggers self.cleanup()
    
    def __enter__(self):
        self.old_sigterm = signal.signal(signal.SIGTERM, self.handler)
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            print(f"Caught exception {exc_type}: {exc_val}")
        self.cleanup()
        signal.signal(signal.SIGTERM, self.old_sigterm)

def assign_dict(d, keys, value):
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value

def flatten_dict(d, depth=-1):
    # depth=0 returns the dictionary unchanged, depth=-1 returns a fully flattened dictionary
    # depth=n means the first n layers of the dictionary is flattened, so a depth k dict becomes a depth k-1 dict.
    # for example: let d = {'a': {'b': {'c': d}}}
    # then flatten_dict(d, depth=1) = {('a','b'): {'c': d}}
    flattened_dict = {}
    for k, v in d.items():
        if depth == 0 or not isinstance(v, dict) or v == {}: # if depth == 0 or v is leaf
            flattened_dict[k] = v
        else:
            for new_k, new_v in flatten_dict(v, depth=depth-1).items():
                flattened_dict[(k, *new_k)] = new_v
    return flattened_dict

def save(path, data, extension, depth=-1, overwrite=False):
    # Saves data at path.
    # If path has a suffix, this must match the extension, and the data 
    # is stored as a single file with the specified extension at path.
    # The depth parameter is ignored in this case.
    # If path does not have suffix, then path is the directory in which
    # the data is stored. The data must be a dictionary.
    # The depth parameter controls the depth of the directory.
    # If depth=0, then path will become a depth 0 directory, and
    # if depth=-1, then path will be a directory as deep as the data dictionary.
    
    path = pathlib.Path(path)
    
    if extension == 'pkl':
        def save_func(filename, dat):
            with open(filename, 'wb') as f:
                pickle.dump(dat, f)
    elif extension == 'json':
        def save_func(filename, dat):
            with open(filename, 'w') as f:
                json.dump(dat, f, indent=4)
    else:
        raise NotImplementedError(f"Extension {extension} is not yet implemented")
    
    if path.suffix != '': # path is a filename
        suffix = path.suffix[1:]
        if suffix != extension:
            raise ValueError(f"path suffix must match extension if suffix is present. suffix: {suffix}, extension: {extension}.")
        if not overwrite and path.is_file():
            raise exceptions.PathAlreadyExists(f"The file {str(path)} already exists.")
        path.parent.mkdir(parents=True, exist_ok=True)
        save_func(path, data)
        
    else: # path is a directory
        for key, val in flatten_dict(data, depth=depth).items():
            filename = path / f"{'/'.join(key)}.{extension}"
            filename.parent.mkdir(parents=True, exist_ok=True)
            if not overwrite and filename.is_file():
                raise exceptions.PathAlreadyExists(f"The file {str(filename)} already exists.")
            save_func(filename, val)
        
def save_data(path, data_dict, **kwargs):
    save(path, data_dict, 'pkl', **kwargs)
    
def save_config(path, config, **kwargs):
    save(path, config, 'json', **kwargs)
    
def load(path, extension):
    path = pathlib.Path(path)
    if not path.exists():
        raise exceptions.PathNotFound(f"The path {path} does not exist.")
        
    if extension == 'pkl':
        def load_func(filename):
            with open(filename, 'rb') as f:
                return pickle.load(f)
    elif extension == 'json':
        def load_func(filename):
            with open(filename, 'r') as f:
                return json.load(f)
    else:
        raise NotImplementedError(f"Extension {extension} is not yet implemented")
        
    if path.is_file():
        data = load_func(path)
    elif path.is_dir():
        data = {}
        for cur_path, dirnames, filenames in os.walk(path):
            if '.ipynb_checkpoints' not in cur_path:
                for filename in filenames:
                    filename = pathlib.Path(filename)
                    if filename.suffix == f'.{extension}':
                        # with open(os.path.join(cur_path,filename), 'r') as f:
                        #     print(f.read())
                        # print(f"Done reading file {os.path.join(cur_path,filename)}")
                        cur_path_rel = pathlib.Path(cur_path).relative_to(path)
                        assign_dict(data, [*cur_path_rel.parts,filename.stem], load_func(os.path.join(cur_path,filename)))
    else:
        raise IOError("Path {path} is neither file nor directory")
        
    return data
        
def load_data(path):
    return load(path, 'pkl')

def load_config(path):
    return load(path, 'json')

### random utils ### 

@contextlib.contextmanager
def set_seed(seed):
    python_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        
### math utils ### 

def prod(size):
    return torch.Tensor(list(size)).prod().int()

def numpy_to_torch(func):
    """
    Converts all numpy arugments to torch.
    In current implementation, if there is a mix of torch and numpy arguments,
    the torch arguments must be on CPU.
    """
    @wraps(func)
    def decorated_func(*args, **kwargs):
        args = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, np.ndarray):
                args[i] = torch.from_numpy(arg).float()
        args = tuple(args)
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                kwargs[k] = torch.from_numpy(v).float()
        return func(*args, **kwargs)
    return decorated_func

def get_pcs(X, n_pcs, **kwargs):
    """
    Assumes X has shape (...,n_features)
    """
    shape = X.shape
    X = X.reshape((-1,shape[-1]))
    max_n_pcs = min(X.shape) # min(n_data_points, n_features)
    if n_pcs == -1:
        n_pcs = max_n_pcs
    assert n_pcs <= max_n_pcs # Can't have more than max_n_pcs
    pca = PCA(n_components=n_pcs, **kwargs)
    X = pca.fit_transform(X)
    X = X.reshape((*shape[:-1],n_pcs))

    return X, np.cumsum(pca.explained_variance_ratio_) # Return PCA'd X and a list of summed explained variance ratios

### config utils ###

def get_layer_names(model_name, layers):
    arch_configs = load_config(ARCH_CONFIGS_PATH)
    model_configs = load_config(MODEL_CONFIGS_PATH)
    return arch_configs[model_configs[model_name]['arch']]['layer_names'][layers]

def get_acts_config(acts_name, version):
    acts_configs = load_config(ACTS_CONFIGS_PATH)
    return acts_configs[acts_name][f'{version:02d}']