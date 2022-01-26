import numpy as np

from .metric import Metric
from nn_analysis import metrics as me
from nn_analysis import acts as ac

class Factorization(Metric):
    def __init__(self, variable_configs):
        self.variable_configs = variable_configs
        
    @property
    def required_acts(self):
        acts = []
        for _, variable_config in self.variable_configs.items():
            acts.append(variable_config['acts'])
        return acts
    
    def _evaluate_single(self, model_name, epoch, layer_name, variable, acts, n_pcs, metric_types, kwargs):
        result = {}
        
        X = ac.utils.load_data(model_name, epoch, acts['name'], acts['version'], layer_name=layer_name)[...,:n_pcs] # (...,n_pcs)
        scores = me.core.compute_factorization_metrics(X, **kwargs)

        for metric_type in metric_types:
            result[f'{metric_type}-{variable}'] = scores[metric_type]
            
        return result
    
    def evaluate(self, model_name, epoch, layer_name):
        result = {}
        for variable, variable_config in self.variable_configs.items():
            print(f"Evaluating variable {variable}")
            single_result = self._evaluate_single(model_name, epoch, layer_name, variable, **variable_config)
            result.update(single_result)
                
        return result