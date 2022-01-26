import numpy as np

from .metric import Metric
from nn_analysis import metrics as me
from nn_analysis import acts as ac
from nn_analysis import utils
from nn_analysis.constants import ACTS_CONFIGS_PATH

class Trajectory(Metric):
    def __init__(self, variable_configs):
        self.variable_configs = variable_configs
        
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
        
        X, evr = utils.get_pcs(X, 2) # get the first 2 PCs for visualizing trajectory
        result[variable] = X
        result[f'{variable}-evr'] = evr
            
        return result
    
    def evaluate(self, model_name, epoch, layer_name):
        result = {}
        
        for variable, variable_config in self.variable_configs.items():
            print(f"Evaluating variable {variable}")
            single_result = self._evaluate_single(model_name, epoch, layer_name, variable, **variable_config)
            result.update(single_result)
                
        return result