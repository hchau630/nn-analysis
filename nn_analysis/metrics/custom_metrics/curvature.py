import numpy as np

from .metric import Metric
from nn_analysis import metrics as me
from nn_analysis import acts as ac
from nn_analysis import utils
from nn_analysis.constants import ACTS_CONFIGS_PATH

class Curvature(Metric):
    def __init__(self, variable_configs, averages):
        self.variable_configs = variable_configs
        self.averages = averages
        
    @property
    def required_acts(self):
        acts = []
        for _, variable_config in self.variable_configs.items():
            acts.append(variable_config['acts'])
        return acts
    
    def _evaluate_single(self, model_name, epoch, layer_name, variable, acts):
        result = {}
        
        acts_config = utils.load_config(ACTS_CONFIGS_PATH)[acts['name']][f"{acts['version']:02d}"]
        X = ac.utils.load_data(model_name, epoch, acts['name'], acts['version'], layer_name=layer_name) # (n_target_names, n_frames, n_pcs)
        X = X[acts_config['target_names'].index(variable)] # (n_frames, n_pcs)
        
        scores = me.core.compute_curvature(X)
        result[f'{variable}-detailed'] = scores
        result[variable] = np.mean(scores)
            
        return result
    
    def evaluate(self, model_name, epoch, layer_name):
        result = {}
        
        for variable, variable_config in self.variable_configs.items():
            print(f"Evaluating variable {variable}")
            single_result = self._evaluate_single(model_name, epoch, layer_name, variable, **variable_config)
            result.update(single_result)
            
        for variable, sub_variables in self.averages.items():
            print(f"Evaluating average variable {variable}")
            result[variable] = np.mean([result[sub_variable] for sub_variable in sub_variables])
                
        return result