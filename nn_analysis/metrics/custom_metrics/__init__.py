import inspect

from .metric import Metric
from .decoding import Decoding
from .dimensionality import Dimensionality
from .factorization import Factorization
from .generalization import Generalization
from .neural_fits import NeuralFits
from .rdm import RDM
from .sparsity import Sparsity
from .curvature import Curvature
from .trajectory import Trajectory

metrics_dict = {k: v for k, v in globals().items() if inspect.isclass(v) and issubclass(v, Metric) and v != Metric}