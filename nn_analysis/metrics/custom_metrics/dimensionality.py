import numpy as np

from .metric import Metric
from nn_analysis import metrics as me
from nn_analysis import acts as ac

class Dimensionality(Metric):
    def __init__(self, acts):
        self.acts = acts
        
    @property
    def required_acts(self):
        return [self.acts]
    
    def evaluate(self, model_name, epoch, layer_name):
        evr = ac.utils.load_data(model_name, epoch, self.acts['name'], self.acts['version'], layer_name=layer_name, data_type='evr')
        result = me.core.compute_dimensionality(evr)
        return result