import typing
from abc import ABC, abstractmethod

class Metric(ABC):
    @property
    @abstractmethod
    def required_acts(self) -> list[dict]:
        pass
        
    @abstractmethod
    def evaluate(self, model_name: str, epoch: typing.Optional[int], layer_name: str) -> typing.Any:
        pass
        