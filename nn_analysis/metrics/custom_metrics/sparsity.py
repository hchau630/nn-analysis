import numpy as np

from .metric import Metric
from nn_analysis import metrics as me
from nn_analysis import acts as ac

class Sparsity(Metric):
    def __init__(self, acts, metric_types, kwargs):
        self.acts = acts
        self.metric_types = metric_types
        self.kwargs = kwargs
        
    @property
    def required_acts(self):
        return [self.acts]
    
    def evaluate(self, model_name, epoch, layer_name):
        X = ac.utils.load_data(model_name, epoch, self.acts['name'], self.acts['version'], layer_name=layer_name)
        scores = me.core.compute_sparsity(X, **self.kwargs)
        result = {metric_type: scores[metric_type] for metric_type in self.metric_types}
        return result