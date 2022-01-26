import numpy as np

from .metric import Metric
from nn_analysis import metrics as me
from nn_analysis import acts as ac
from nn_analysis import utils
from nn_analysis.constants import ACTS_CONFIGS_PATH

class Generalization(Metric):
    def __init__(self, variable_configs, averages, metric_types):
        self.variable_configs = variable_configs
        self.averages = averages
        self.metric_types = metric_types
        
    @property
    def required_acts(self):
        acts = []
        for _, variable_config in self.variable_configs.items():
            acts.append(variable_config['acts'])
        return acts
    
    def _evaluate_single(self, model_name, epoch, layer_name, variable, acts, n_pcs, kwargs):
        result = {}
        
        acts_config = utils.load_config(ACTS_CONFIGS_PATH)[acts['name']][f"{acts['version']:02d}"]
        X = ac.utils.load_data(model_name, epoch, acts['name'], acts['version'], layer_name=layer_name)[...,:n_pcs] # (...,n_pcs)
        y = ac.utils.load_data(model_name, epoch, acts['name'], acts['version'], layer_name=layer_name, data_type='x') # (...,n_targets)
        y = y[...,acts_config['target_names'].index(variable)]
        
        if 'cv_regs' in kwargs:
            cv_regs = kwargs['cv_regs']
            kwargs['cv_regs'] = np.logspace(cv_regs[0], cv_regs[1], cv_regs[2])
                
        scores, _ = me.core.compute_ccg_scores(X, y, self.metric_types, **kwargs, suppress=True)
        for metric_type in self.metric_types:
            result[f'{metric_type}-{variable}'] = scores[metric_type]
            
        return result
    
    def evaluate(self, model_name, epoch, layer_name):
        result = {}
        
        for variable, variable_config in self.variable_configs.items():
            print(f"Evaluating variable {variable}")
            single_result = self._evaluate_single(model_name, epoch, layer_name, variable, **variable_config)
            result.update(single_result)
            
        for variable, sub_variables in self.averages.items():
            print(f"Evaluating average variable {variable}")
            for metric_type in self.metric_types:
                result[f'{metric_type}-{variable}'] = np.mean([result[f'{metric_type}-{sub_variable}'] for sub_variable in sub_variables])
                
        return result