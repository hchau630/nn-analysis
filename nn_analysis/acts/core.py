from abc import ABC, abstractmethod
import random
import hashlib
import os
import traceback
import pathlib

import h5py
import torch
import numpy as np

from nn_analysis import utils

def attach_hooks(model, layer_names, get_hook):
    handles = []
    for layer_name, module in model.named_modules():
        if layer_name in layer_names:
            hook = get_hook(layer_name)
            handle = module.register_forward_hook(hook)
            handles.append(handle)
    return handles

def remove_hooks(handles):
    for handle in handles:
        handle.remove()

def compute_sizes(model, layer_names, dataset, device='cpu'):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
    images = next(iter(dataloader))[0].to(device)
    sizes = {}
    
    def get_hook(layer_name):
        def hook(_0, _1, out):
            sizes[layer_name] = utils.prod(out.size()[1:])
        return hook
    
    try:
        handles = attach_hooks(model, layer_names, get_hook)
        model.eval()
        with torch.no_grad():
            model(images)
    finally:
        remove_hooks(handles)
        
    targets = next(iter(dataloader))[1]
    assert targets.ndim <= 2
    if targets.ndim == 1:
        sizes['target'] = 1
    else:
        sizes['target'] = targets.size()[1]
    sizes['dataset'] = dataset.shape
    return sizes

def create_group_datasets(grp, model, layer_names, sizes, meta_dicts=None, dtype='float32'):
    for layer_name, module in model.named_modules():
        if layer_name in layer_names:
            layer_grp = grp.create_group(layer_name) # Must be new, cannot overwrite
            if meta_dicts is not None:
                for k, v in meta_dicts[0].items():
                    if layer_name in v.keys():
                        layer_grp.attrs[k] = v[layer_name]
                for k, v in meta_dicts[1].items():
                    if layer_name in v.keys():
                        layer_grp[k] = v[layer_name] # Too large to fit in as attribute
            layer_grp.create_dataset('y', shape=(*sizes['dataset'],sizes[layer_name]), dtype=dtype) # Must be new, cannot overwrite
    grp.create_dataset('x', shape=(*sizes['dataset'],sizes['target']), dtype=dtype) # Must be new, cannot overwrite

def save_dataset(filename, path, model, layer_names, dataset, device='cpu', batch_size=128, postprocessor=None, dtype='float32', log=False):
    sizes = compute_sizes(model, layer_names, dataset, device=device)
    if postprocessor is None:
        postprocess = lambda y, *args, **kwargs: y
    else:
        sizes = postprocessor.configure(sizes)
        postprocess = postprocessor.process
    meta_dicts = postprocessor.meta_dicts if postprocessor is not None else None

    with h5py.File(filename, 'a') as f:
        grp = f[path]
        create_group_datasets(grp, model, layer_names, sizes, meta_dicts=meta_dicts, dtype=dtype)
        
    model.eval()
    
    def get_hook(layer_name):
        def hook(_0, _1, out):
            y = out.detach()
            y = y.reshape(y.size(0),-1)
            activations = postprocess(y,layer_name,device=device,dtype=dtype).cpu()
            with h5py.File(filename, 'a') as f:
                # print(f"Activations size: {activations.size()}")
                # print(f"file size: {os.path.getsize(filename)}")
                try:
                    f[path][layer_name]['y'][indices] = activations
                except TypeError as err:
                    # Fancy indexing cannot handle multi-dimensional individual elements inexing
                    for j, index in enumerate(zip(*indices)):
                        f[path][layer_name]['y'][index] = activations[j]
        return hook
    
    try:
        handles = attach_hooks(model, layer_names, get_hook)
        dl = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=False)
        print_freq = len(dl)//10 if len(dl) > 10 else 1
        for i, (images, targets, indices) in enumerate(dl):
            if i % print_freq == 0:
                print(f"Processing batch {i}/{len(dl)}")
            images = images.to(device)
            if indices.ndim == 1:
                indices = indices.view(-1,1)
            indices = tuple(indices.t().long().numpy())
            if targets.ndim == 1:
                targets = targets.view(-1,1)
            with h5py.File(filename, 'a') as f:
                try:
                    f[path]['x'][indices] = targets
                except TypeError as err:
                    # Fancy indexing cannot handle multi-dimensional individual elements inexing
                    for j, index in enumerate(zip(*indices)):
                        f[path]['x'][index] = targets[j]
            with torch.no_grad():
                model(images)
    finally:
        remove_hooks(handles)
            
def save(filename, model, model_name, epoch, layer_names, datasets, dataset_names, seeds=None, device='cpu', batch_size=256, postprocessor=None, dtype='float32', log=False):
    assert len(dataset_names) == len(seeds)
    
    pathlib.Path(filename).parent.mkdir(parents=True, exist_ok=True)
    
    if epoch is None:
        model_version = '0000'
    else:
        model_version = f'{epoch:04d}'
    
    with h5py.File(filename, 'a') as f:
        model_grp = f.require_group(model_name)
        model_version_grp = model_grp.require_group(model_version)
        for i, dataset in enumerate(datasets):
            print(f"Processing dataset {i}: {dataset_names[i]}")
            dataset_grp = model_version_grp.require_group(dataset_names[i])
            if seeds is not None:
                dataset_grp.attrs['seed'] = seeds[i]
                
    for i, dataset in enumerate(datasets):
        with h5py.File(filename, 'a') as f:
            path = f[model_name][model_version][dataset_names[i]].name
        if seeds is not None:
            with utils.set_seed(seeds[i]):
                save_dataset(filename, path, model, layer_names, dataset, device=device, batch_size=batch_size, postprocessor=postprocessor, dtype=dtype, log=log)
        else:
            save_dataset(filename, path, model, layer_names, dataset, device=device, batch_size=batch_size, postprocessor=postprocessor, dtype=dtype, log=log)
            
def load(filename, model_name, epoch, dataset_name, layer_name):
    if epoch is None:
        model_version = '0000'
    else:
        model_version = f'{epoch:04d}'
    with h5py.File(filename, 'r') as f:
        grp = f[model_name][model_version][dataset_name]
        y = grp[layer_name]['y'][...]
        return y
    
def load_x(filename, model_name, epoch, dataset_name):
    if epoch is None:
        model_version = '0000'
    else:
        model_version = f'{epoch:04d}'
    with h5py.File(filename, 'r') as f:
        grp = f[model_name][model_version][dataset_name]
        x = grp['x'][...]
        return x
    
class Processor(ABC):
    @property
    @abstractmethod
    def meta_dicts(self):
        # List of two dicts, the first one containing meta attributes and the second one containing meta datasets
        pass
    
    @abstractmethod
    def configure(self, layer_sizes):
        pass
    
    @abstractmethod
    def process(self, tensor, layer_name, **kwargs):
        pass

class Compose(Processor):
    def __init__(self, processors):
        self.processors = processors
        
    @property
    def meta_dicts(self):
        out = [{},{}]
        for processor in self.processor:
            out[0].update(processor.meta_dicts[0])
            out[1].update(processor.meta_dicts[1])
        return out
    
    def configure(self, layer_sizes):
        for processor in self.processors:
            layer_sizes = processor.configure(layer_sizes)
        return layer_sizes
            
    def process(self, tensor, layer_name, **kwargs):
        for processor in self.processors:
            tensor = processor.process(tensor, layer_name, **kwargs)
        return tensor
    
class Sampler(Processor):
    def __init__(self, n_samples, set_seed=True):
        self.n_samples = n_samples
        self.indices = {}
        self.configured = False
        if set_seed:
            self.seeds = {}
        self.set_seed = set_seed
        
    @property
    def meta_dicts(self):
        if self.set_seed:
            return [{'seed': self.seeds}, {'indices': self.indices}]
        return [{}, {'indices': self.indices}]
        
    def configure(self, sizes):
        layer_sizes = {k: v for k, v in sizes.items() if k not in ['target', 'dataset']}
        output_sizes = {}
        for layer_name, size in layer_sizes.items():
            if self.n_samples > size:
                self.indices[layer_name] = torch.arange(size)
                output_sizes[layer_name] = size
            else:
                if self.set_seed:
                    seed = int(hashlib.sha256(layer_name.encode('utf-8')).hexdigest(), 16) % (2**32) # Get seed corresponding to layer
                    self.seeds[layer_name] = seed
                    with utils.set_seed(seed): 
                        self.indices[layer_name] = torch.from_numpy(np.random.choice(size,size=self.n_samples,replace=False)).long()
                else:
                    self.indices[layer_name] = torch.from_numpy(np.random.choice(size,size=self.n_samples,replace=False)).long()
                output_sizes[layer_name] = self.n_samples
        self.configured = True
        output_sizes.update({'target': sizes['target'], 'dataset': sizes['dataset']})
        return output_sizes
        
    def process(self, tensor, layer_name, **kwargs):
        """
        tensor - (batch_size, N)
        """
        assert self.configured
        assert tensor.ndim == 2
        layer_indices = self.indices[layer_name]
        if tensor.is_cuda:
            layer_indices.to(tensor.get_device())
        return tensor[:,layer_indices]
