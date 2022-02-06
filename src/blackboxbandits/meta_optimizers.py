from abc import ABC, abstractmethod
import pandas as pd

class AbstractMetaOptimizer(ABC):

    def __init__(self):
        raise NotImplementedError()

    @abstractmethod
    def run(self, base_comparison_results: pd.DataFrame) -> None:
        raise NotImplementedError()

    @abstractmethod
    def get_results(self) -> pd.DataFrame:
        raise NotImplementedError()

class BestFixedTAlgos(AbstractMetaOptimizer):

    def __init__(self, T: int):
        self.T = T

    def run(self, base_comparison_results: pd.DataFrame) -> None:
        pass


class FollowPerturbedMultipleLeaders(AbstractMetaOptimizer):
    pass

class StreeterOGAlgo(AbstractMetaOptimizer):
    pass

