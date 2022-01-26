import numpy as np

from .metric import Metric
from nn_analysis import metrics as me
from nn_analysis import acts as ac

class NeuralFits(Metric):
    def __init__(self, dataset_configs):
        self.dataset_configs = dataset_configs
        
    @property
    def required_acts(self):
        acts = []
        for _, dataset_config in self.dataset_configs.items():
            acts.append(dataset_config['acts'])
        return acts
    
    def _evaluate_single(self, model_name, epoch, layer_name, base_dataset_name, acts, n_pcs, regions, kwargs):
        X = ac.utils.load_data(model_name, epoch, acts['name'], acts['version'], layer_name=layer_name)[...,:n_pcs] # (...,n_pcs)
            
        result = {}
        if 'clf' in kwargs:
            cv_alphas = kwargs['clf']['cv_alphas']
            alpha_per_target = kwargs['clf']['alpha_per_target']
            if cv_alpha_log_steps:
                cv_alphas = np.logspace(cv_alphas[0], cv_alphas[1], cv_alphas[2])
            else:
                cv_alphas = np.linspace(cv_alphas[0], cv_alphas[1], cv_alphas[2])
            kwargs['clf'] = me.core.RidgeCV(alphas=cv_alphas, alpha_per_target=alpha_per_target)

        for region in regions:
            y = me.utils.load_neural_data(base_dataset_name, region, standardize=True) # (images, neurons, trials)
            score, _ = me.core.compute_neural_fits(X, y, **kwargs)
            result[f'{base_dataset_name}-{region}'] = score
            
        return result
    
    def evaluate(self, model_name, epoch, layer_name):
        result = {}
        for base_dataset_name, dataset_config in self.dataset_configs.items():
            print(f"Evaluating on dataset {base_dataset_name}")
            single_result = self._evaluate_single(model_name, epoch, layer_name, base_dataset_name, **dataset_config)
            result.update(single_result)
        return result
    