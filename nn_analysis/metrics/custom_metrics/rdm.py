import numpy as np

from .metric import Metric
from nn_analysis import metrics as me
from nn_analysis import acts as ac

class RDM(Metric):
    def __init__(self, dataset_configs, rdm_kwargs, rdm_neural_fits_kwargs):
        self.dataset_configs = dataset_configs
        self.rdm_kwargs = rdm_kwargs
        self.rdm_neural_fits_kwargs = rdm_neural_fits_kwargs
        
    @property
    def required_acts(self):
        acts = []
        for _, dataset_config in self.dataset_configs.items():
            acts.append(dataset_config['acts'])
        return acts
    
    def _evaluate_single(self, model_name, epoch, layer_name, base_dataset_name, acts, n_pcs, n_images, regions):
        X = ac.utils.load_data(model_name, epoch, acts['name'], acts['version'], layer_name=layer_name)[...,:n_pcs] # (...,n_pcs)
        X = X.reshape(-1,n_pcs)[:n_images] # (n_images,n_pcs)
        
        model_rdm = me.core.compute_rdm(X, **self.rdm_kwargs)
            
        result = {}
        
        # TODO: precompute neural rdms and save them as data, since the neural rdms are reused across models
        for region in regions:
            X = me.utils.load_neural_data(base_dataset_name, region, standardize=True)[:n_images] # (n_images, neurons, trials)
            neural_rdm = me.core.compute_rdm(X, **self.rdm_kwargs)
            score = me.core.compute_rdm_neural_fits(model_rdm, neural_rdm, **self.rdm_neural_fits_kwargs)
            result[f'{base_dataset_name}-{region}'] = score
            
        return result
    
    def evaluate(self, model_name, epoch, layer_name):
        result = {}
        for base_dataset_name, dataset_config in self.dataset_configs.items():
            print(f"Evaluating on dataset {base_dataset_name}")
            single_result = self._evaluate_single(model_name, epoch, layer_name, base_dataset_name, **dataset_config)
            result.update(single_result)
        return result
    